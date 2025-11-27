from __future__ import annotations

import resource
import time

import tyro
import lightning as L
import torch as th
from tqdm import tqdm

from splat_init.data.datamodule_360 import SceneSampleLazy
from splat_init.data.threesixty_loc import ThreeSixtyLocDataset
from splat_init.data.stanford_2d_3d import Stanford2d3dDataset
from splat_init.models.sequence_chunker import SequenceChunker
from splat_init.models.vggt_perspective_transform import VggtPerspectiveTransform
from splat_init.models.vggt_naive_equirectangular import VggtNaiveEquirectangular
from configs.evaluation_args import Args
from utilities.pose import (
    geodesic_so3,
    pointing_and_roll_errors,
    relative_centers,
    relative_rotations,
)


def _start_metrics() -> dict[str, float]:
    """Capture timing and memory baselines at the start of a scene evaluation."""

    if th.cuda.is_available():
        th.cuda.reset_peak_memory_stats()
        gpu_alloc = float(th.cuda.memory_allocated())
    else:
        gpu_alloc = 0.0

    return {
        "t_start": time.perf_counter(),
        "gpu_alloc_start": gpu_alloc,
    }


def _end_metrics(
    start: dict[str, float],
    pose_gt: th.Tensor,
    pose_pred: th.Tensor,
    sequence_length: int,
    stride: int,
    model_name: str,
) -> dict[str, th.Tensor | str]:
    """Compute pose errors plus runtime, memory, and metadata for one scene."""

    gt_rot_rel = relative_rotations(pose_gt)
    pred_rot_rel = relative_rotations(pose_pred)

    gt_centers_rel = relative_centers(pose_gt)
    pred_centers_rel = relative_centers(pose_pred)

    translation_error = (gt_centers_rel - pred_centers_rel).norm(dim=-1)
    geodesic_error = geodesic_so3(gt_rot_rel, pred_rot_rel)
    pointing_error, roll_error = pointing_and_roll_errors(gt_rot_rel, pred_rot_rel)

    elapsed_seconds = time.perf_counter() - start["t_start"]

    gpu_alloc = th.cuda.memory_allocated() if th.cuda.is_available() else 0
    gpu_peak = th.cuda.max_memory_allocated() if th.cuda.is_available() else 0
    cpu_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    return {
        "translation_error_mean": translation_error.mean(),
        "translation_error_std": translation_error.std(unbiased=False),
        "rotation_geodesic_mean": geodesic_error.mean(),
        "rotation_geodesic_std": geodesic_error.std(unbiased=False),
        "rotation_pointing_mean": pointing_error.mean(),
        "rotation_pointing_std": pointing_error.std(unbiased=False),
        "rotation_roll_mean": roll_error.mean(),
        "rotation_roll_std": roll_error.std(unbiased=False),
        "elapsed_seconds": th.tensor(elapsed_seconds, dtype=th.float32),
        "num_images": th.tensor(sequence_length, dtype=th.int32),
        "dataset_stride": th.tensor(stride, dtype=th.int32),
        "gpu_memory_allocated": th.tensor(gpu_alloc, dtype=th.int64),
        "gpu_memory_peak": th.tensor(gpu_peak, dtype=th.int64),
        "cpu_memory_rss": th.tensor(cpu_rss_bytes, dtype=th.int64),
        "model_name": model_name,
    }


def _build_dataset(args: Args) -> ThreeSixtyLocDataset[SceneSampleLazy] | Stanford2d3dDataset[SceneSampleLazy]:
    if args.data.dataset_name == "stanford_2d_3d":
        return Stanford2d3dDataset(
            SceneSampleLazy,
            args.data.dataset_dir,
            image_size=args.data.dataset_image_size,
            perspective_loader_threads=args.data.dataloader_workers
        )
    elif args.data.dataset_name == "360_loc":
        return ThreeSixtyLocDataset(
            SceneSampleLazy,
            args.data.dataset_dir,
            stride=args.data.dataset_stride,
            depth_required=False,
            image_size=args.data.dataset_image_size,
            worker_count=args.data.dataloader_workers
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data.dataset_name}")


def _build_model(args: Args) -> SequenceChunker:
    models = {
        "vggt_perspective_transform": VggtPerspectiveTransform,
        "vggt_naive_equirectangular": VggtNaiveEquirectangular,
    }

    return SequenceChunker(
        model=models[args.model.model](),
        chunk_size=args.model.chunker_chunk_size,
        chunk_overlap=args.model.chunker_chunk_overlap,
        verbose=True,
    )

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    args = tyro.cli(Args)
    th.set_float32_matmul_precision("medium")
    L.seed_everything(args.seed)

    dataset = _build_dataset(args)

    dtype = th.bfloat16 if args.model.dtype == "bfloat16" else th.float32
    device = "cuda" if th.cuda.is_available() else "cpu"
    model = _build_model(args).to(device=device, dtype=dtype)

    for scene in (pbar := tqdm(dataset, desc="Evaluating scenes")):
        # Metrics start
        pbar.set_postfix({"scene_id": scene.id})
        metrics_runtime = _start_metrics()

        # Inference
        with th.no_grad(), th.inference_mode():
            poses, _, _ = model.forward([scene])
            poses_cpu = poses.detach().cpu()

        # Metrics end
        metrics = _end_metrics(
            metrics_runtime,
            pose_gt=scene.poses[:],
            pose_pred=poses_cpu,
            sequence_length=len(scene),
            stride=args.data.dataset_stride,
            model_name=args.model.model,
        )

        # Store
        output_dir = args.output_dir / scene.id
        output_dir.mkdir(parents=True, exist_ok=True)

        th.save({"poses": poses_cpu}, output_dir / "model_output.pt")
        th.save(metrics, output_dir / "metrics.pt")


if __name__ == "__main__":
    main()

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
from splat_init.models.da3_perspective_transform import Da3PerspectiveTransform
from splat_init.models.pycolmap_perspective_transform import PycolmapPerspectiveTransform
from splat_init.models.sequence_chunker import SequenceChunker
from splat_init.models.vggt_perspective_transform import VggtPerspectiveTransform
from splat_init.models.vggt_naive_equirectangular import VggtNaiveEquirectangular
from splat_init.models.vipe_panorama import VipePanorama
from configs.evaluation_args import Args
from utilities.pose import (
    camera_centers,
    geodesic_so3,
    pointing_and_roll_errors,
    procrustes_transform,
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
    scene_idx: int,
    sequence_length: int,
    dataset_stride: int,
    dataset_fps: float,
    chunker_chunk_size: int,
    chunker_chunk_overlap: int,
    model_name: str,
) -> dict[str, str | int | float]:
    """Compute pose errors plus runtime, memory, and metadata for one scene."""

    pose_gt_f32 = pose_gt.to(dtype=th.float32)
    pose_pred_f32 = pose_pred.to(dtype=th.float32)
    pose_pred_aligned = procrustes_transform(pose_pred_f32, pose_gt_f32, pose_pred_f32, allow_scale=True)

    gt_centers = camera_centers(pose_gt_f32)
    pred_centers = camera_centers(pose_pred_aligned)
    translation_error = (gt_centers - pred_centers).norm(dim=-1)

    gt_rot = pose_gt_f32[..., :3, :3]
    pred_rot = pose_pred_aligned[..., :3, :3]
    geodesic_error = geodesic_so3(gt_rot, pred_rot)
    pointing_error, roll_error = pointing_and_roll_errors(gt_rot, pred_rot)

    elapsed_seconds = time.perf_counter() - start["t_start"]

    gpu_alloc = th.cuda.memory_allocated() if th.cuda.is_available() else 0
    gpu_peak = th.cuda.max_memory_allocated() if th.cuda.is_available() else 0
    cpu_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    return {
        "translation_error_mean": translation_error.mean().item(),
        "translation_error_std": translation_error.std(unbiased=False).item(),
        "rotation_geodesic_mean": geodesic_error.mean().item(),
        "rotation_geodesic_std": geodesic_error.std(unbiased=False).item(),
        "rotation_pointing_mean": pointing_error.mean().item(),
        "rotation_pointing_std": pointing_error.std(unbiased=False).item(),
        "rotation_roll_mean": roll_error.mean().item(),
        "rotation_roll_std": roll_error.std(unbiased=False).item(),
        "elapsed_seconds": elapsed_seconds,
        "scene_idx": scene_idx,
        "sequence_length": sequence_length,
        "dataset_stride": dataset_stride,
        "dataset_fps": dataset_fps,
        "chunker_chunk_size": chunker_chunk_size,
        "chunker_chunk_overlap": chunker_chunk_overlap,
        "gpu_memory_allocated": gpu_alloc,
        "gpu_memory_peak": gpu_peak,
        "cpu_memory_rss": cpu_rss_bytes,
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
            depth_required=True,
            image_size=args.data.dataset_image_size,
            worker_count=args.data.dataloader_workers
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data.dataset_name}")


def _build_model(args: Args) -> SequenceChunker:
    models = {
        "vggt_perspective_transform": VggtPerspectiveTransform,
        "vggt_naive_equirectangular": VggtNaiveEquirectangular,
        "vipe_panorama": lambda: VipePanorama(fps=args.data.dataset_fps),
        "da3_perspective_transform": Da3PerspectiveTransform,
        "pycolmap_perspective_transform": PycolmapPerspectiveTransform,
    }

    return SequenceChunker(
        model=models[args.model.model](),
        chunking=args.model.chunker,
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
    model = _build_model(args).eval().to(device=device, dtype=dtype)

    for scene_idx, scene in (pbar := tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating scenes")):
        # Metrics start
        pbar.set_postfix({"scene_id": scene.id})
        metrics_runtime = _start_metrics()

        # Inference
        with th.no_grad(), th.inference_mode():
            poses, _, _ = model.forward([scene])
            poses_cpu = poses.detach().cpu()

        # Metrics end
        chunker_chunk_size, chunker_chunk_overlap = args.model.chunker or (0, 0)
        metrics = _end_metrics(
            metrics_runtime,
            pose_gt=scene.poses[:],
            pose_pred=poses_cpu,
            scene_idx=scene_idx,
            sequence_length=len(scene),
            dataset_stride=args.data.dataset_stride,
            dataset_fps=args.data.dataset_fps,
            chunker_chunk_size=chunker_chunk_size,
            chunker_chunk_overlap=chunker_chunk_overlap,
            model_name=args.model.model,
        )

        # Store
        output_dir = args.output_dir / scene.id
        output_dir.mkdir(parents=True, exist_ok=True)

        th.save({"poses": poses_cpu}, output_dir / "model_output.pt")
        th.save(metrics, output_dir / "metrics.pt")


if __name__ == "__main__":
    main()

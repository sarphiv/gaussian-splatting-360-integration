from __future__ import annotations

from pathlib import Path
from typing import cast

from joblib import Parallel, delayed
import torch as th
import tyro
from torchvision.io import decode_image, ImageReadMode
from tqdm import tqdm
from torchmetrics.functional.image import (
    learned_perceptual_image_patch_similarity,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from configs.evaluate_splats_args import Args
from utilities.cube_projector import CubeProjector

EVAL_DIR_PREFIX = "eval_step_"
METRICS_FILE = "metrics.pt"
SPLIT_GAP = 4


EvalRecord = tuple[str, int, Path]
MetricRecord = tuple[str, int, dict[str, float]]


def _process_batch(
    records: list[EvalRecord],
    *,
    split_gap: int,
    image_workers: int,
) -> list[MetricRecord]:
    """Load a batch, compute metrics, and attach metadata."""
    def _load_and_split(path: Path) -> tuple[th.Tensor, th.Tensor]:
        image = decode_image(str(path), mode=ImageReadMode.RGB).float().div(255.0)
        height, width = image.shape[1], image.shape[2]
        left_width = (width - split_gap) // 2
        assert left_width > 0, "Split gap too large for panorama width."
        assert width == left_width * 2 + split_gap, "Unexpected panorama layout."
        gt = image[:, :, :left_width]
        pred = image[:, :, left_width + split_gap :]
        assert gt.shape == pred.shape, "Ground truth and prediction shape mismatch."
        assert gt.shape[1] == height, "Unexpected panorama height."
        return gt, pred

    parallel = Parallel(n_jobs=image_workers, backend="threading")
    pairs = cast(
        list[tuple[th.Tensor, th.Tensor]],
        parallel(delayed(_load_and_split)(path) for _, _, path in records),
    )
    assert pairs, "No images loaded for batch."
    gt_list, pred_list = zip(*pairs)
    gt_rgb = th.stack(gt_list)
    pred_rgb = th.stack(pred_list)
    with th.inference_mode():
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        gt_rgb = gt_rgb.to(device=device, dtype=th.float32)
        pred_rgb = pred_rgb.to(device=device, dtype=th.float32)

        face_size = gt_rgb.shape[2] // 2
        assert face_size * 2 == gt_rgb.shape[2], "Expected panorama height divisible by 2."
        projector = CubeProjector(face_size=face_size)

        alpha = th.ones((gt_rgb.shape[0], 1, gt_rgb.shape[2], gt_rgb.shape[3]), device=device)
        gt_rgba = th.cat([gt_rgb, alpha], dim=1)
        pred_rgba = th.cat([pred_rgb, alpha], dim=1)
        gt_faces, gt_alpha, _ = projector(gt_rgba, depth=None)
        pred_faces, pred_alpha, _ = projector(pred_rgba, depth=None)
        gt_faces = gt_faces * gt_alpha
        pred_faces = pred_faces * pred_alpha

        batch, num_faces, _, face_h, face_w = gt_faces.shape
        assert pred_faces.shape == gt_faces.shape, "Face tensor shape mismatch."
        flat_gt = gt_faces.reshape(batch * num_faces, 3, face_h, face_w)
        flat_pred = pred_faces.reshape(batch * num_faces, 3, face_h, face_w)

        psnr_faces = peak_signal_noise_ratio(
            flat_pred,
            flat_gt,
            data_range=1.0,
            reduction="none",
            dim=(1, 2, 3),
        )
        ssim_faces = cast(
            th.Tensor,
            structural_similarity_index_measure(
                flat_pred,
                flat_gt,
                data_range=1.0,
                reduction="none",
            ),
        )
        lpips_faces = learned_perceptual_image_patch_similarity(
            flat_pred,
            flat_gt,
            net_type="alex",
            reduction="none",
            normalize=True,
        )

        psnr = psnr_faces.view(batch, num_faces).mean(dim=1)
        ssim = ssim_faces.view(batch, num_faces).mean(dim=1)
        lpips = lpips_faces.view(batch, num_faces).mean(dim=1)

    output: list[MetricRecord] = []
    for idx, (step, image_id, _) in enumerate(records):
        output.append(
            (
                step,
                image_id,
                {
                    "psnr": float(psnr[idx].item()),
                    "ssim": float(ssim[idx].item()),
                    "lpips": float(lpips[idx].item()),
                },
            )
        )
    return output


def _run_scene(
    scene_dir: Path,
    *,
    split_gap: int,
    batch_size: int,
    jobs: int,
    image_workers: int,
) -> None:
    """Evaluate all splat images for a single scene."""
    splat_dir = scene_dir / "splat"
    records: list[EvalRecord] = []
    for path in sorted(splat_dir.glob(f"{EVAL_DIR_PREFIX}*/*.png")):
        step = path.parent.name.removeprefix("eval_")
        records.append((step, int(path.stem), path))
    assert records, f"No evaluation images found in {scene_dir}"

    batches = [records[idx : idx + batch_size] for idx in range(0, len(records), batch_size)]
    backend = "threading" if th.cuda.is_available() else "loky"
    parallel = Parallel(n_jobs=jobs, backend=backend)
    results = cast(
        list[list[MetricRecord]],
        parallel(
            delayed(_process_batch)(
                batch,
                split_gap=split_gap,
                image_workers=image_workers,
            )
            for batch in batches
        ),
    )

    flat_results = [item for batch in results for item in batch]
    metrics_by_step: dict[str, dict[int, dict[str, float]]] = {}
    for step, image_id, metrics_out in flat_results:
        metrics_by_step.setdefault(step, {})[image_id] = metrics_out

    metrics: dict[str, dict[str, list[float]]] = {}
    for step, by_image in metrics_by_step.items():
        ordered = [by_image[image_id] for image_id in sorted(by_image)]
        metrics[step] = {
            "psnr": [item["psnr"] for item in ordered],
            "ssim": [item["ssim"] for item in ordered],
            "lpips": [item["lpips"] for item in ordered],
        }

    output_dir = scene_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    th.save(metrics, output_dir / METRICS_FILE)


def main() -> None:
    """CLI entrypoint for splat evaluation."""
    args = tyro.cli(Args)

    assert args.parallel.jobs > 0, "jobs must be positive."
    assert args.parallel.images_per_job > 0, "images_per_job must be positive."
    assert args.parallel.image_workers > 0, "image_workers must be positive."
    scene_dirs = sorted([path for path in args.results_dir.iterdir() if path.is_dir()])
    assert scene_dirs, f"No scene directories found in {args.results_dir}"

    for scene_dir in tqdm(scene_dirs, desc="Evaluating splats"):
        _run_scene(
            scene_dir,
            split_gap=SPLIT_GAP,
            batch_size=args.parallel.images_per_job,
            jobs=args.parallel.jobs,
            image_workers=args.parallel.image_workers,
        )


if __name__ == "__main__":
    main()

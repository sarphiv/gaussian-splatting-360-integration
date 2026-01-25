from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone

import torch as th
import tyro
from tqdm import tqdm

from configs.train_splats_args import Args
from splat_init.data.datamodule_360 import SceneSampleLazy
from splat_init.data.threesixty_loc import ThreeSixtyLocDataset
from utilities.colmap_export import export_colmap_scene
from utilities.pose import procrustes_transform

POSES_DIRNAME = "poses"
POSES_FILE = "poses.pt"
KEYPOINTS_FILE = "keypoints.pt"
METRICS_FILE = "metrics.pt"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LICHTFELD_BINARY = PROJECT_ROOT / "vendor" / "lichtfeld-studio" / "build" / "LichtFeld-Studio"
LICHTFELD_CONFIG = PROJECT_ROOT / "src" / "configs" / "splat.json"
BASIS_OFFSET = th.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
)


def main() -> None:
    args = tyro.cli(Args)

    assert LICHTFELD_BINARY.is_file(), f"Lichtfeld Studio binary missing at {LICHTFELD_BINARY}"
    assert LICHTFELD_CONFIG.is_file(), f"Lichtfeld Studio config missing at {LICHTFELD_CONFIG}"

    shm = Path("/dev/shm")
    assert shm.is_dir() and os.access(shm, os.W_OK), "RAM-backed temp dir unavailable"

    scene_dirs = sorted([path for path in args.results_dir.iterdir() if path.is_dir()])
    assert scene_dirs, f"No scene directories found in {args.results_dir}"

    base_metrics = th.load(scene_dirs[0] / POSES_DIRNAME / METRICS_FILE, map_location="cpu")
    old_stride = int(base_metrics["dataset_stride"])
    old_offset = int(base_metrics["dataset_offset"])
    image_size = (int(base_metrics["dataset_image_width"]), int(base_metrics["dataset_image_height"]))
    new_stride = old_stride // 2
    assert old_offset == new_stride

    dataset = ThreeSixtyLocDataset(
        SceneSampleLazy,
        args.data.dataset_dir,
        stride=new_stride,
        offset=0,
        depth_required=True,
        image_size=image_size,
        worker_count=args.data.dataloader_workers,
    )

    for scene_dir in (pbar := tqdm(scene_dirs, desc="Training splats")):
        pred_metrics = th.load(scene_dir / POSES_DIRNAME / METRICS_FILE, map_location="cpu")
        pred_poses = th.load(scene_dir / POSES_DIRNAME / POSES_FILE, map_location="cpu")
        pred_keypoints = th.load(scene_dir / POSES_DIRNAME / KEYPOINTS_FILE, map_location="cpu")

        scene_idx = int(pred_metrics["scene_idx"])
        scene = dataset[scene_idx]
        assert scene.id == scene_dir.name, "Scene ID mismatch between dataset and results directory."
        pbar.set_postfix({"scene_id": scene.id})

        gt_poses = scene.poses[:]
        gt_poses = gt_poses @ BASIS_OFFSET.to(gt_poses) # Make flat in LichtFeld
        pred_poses, align = procrustes_transform(pred_poses, gt_poses[1::2], pred_poses)

        output_poses = gt_poses.clone()
        output_poses[1 + 2 * th.arange(len(pred_poses))] = pred_poses

        output_keypoints = [
            (th.empty((2, 0), dtype=th.float32), th.empty((3, 0), dtype=th.float32))
            for _ in range(len(gt_poses))
        ]
        for i in range(len(pred_keypoints)):
            output_keypoints[1 + 2 * i] = (pred_keypoints[i][0], align(pred_keypoints[i][1].T, None)[0].T)

        with tempfile.TemporaryDirectory(dir=shm) as temp_dir:
            temp_path = Path(temp_dir)
            export_colmap_scene(temp_path, scene, output_poses, output_keypoints)

            output_path = args.results_dir / scene.id / "splat"
            if output_path.exists():
                os.rename(output_path, output_path.parent / f"splat-backed-up-at-{datetime.now(timezone.utc).strftime("outputs/%Y-%m-%dT%H:%M:%S")}")
            output_path.mkdir(parents=True, exist_ok=True)

            start_time = time.perf_counter()
            cmd = [
                str(LICHTFELD_BINARY),
                # "--headless",
                "--test-every", "2",
                "--config", str(LICHTFELD_CONFIG),
                "--data-path", str(temp_path),
                "--output-path", str(output_path),
            ]
            subprocess.run(cmd, check=True)
            elapsed_seconds = time.perf_counter() - start_time

            with (output_path / "metrics.csv").open(newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                metrics_out: dict[str, int] = {
                    f"num_gaussians_{int(row["iteration"])}": int(row["num_gaussians"])
                    for row in reader
                }
            metrics_out["elapsed_seconds"] = int(elapsed_seconds)

            th.save(metrics_out, output_path / METRICS_FILE)


if __name__ == "__main__":
    main()

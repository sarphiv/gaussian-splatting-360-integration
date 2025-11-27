from pathlib import Path
from typing import cast
from os import environ
from math import degrees

import rerun as rr
import rerun.blueprint as rrb
import torch as th
import numpy as np

from splat_init.data.threesixty_loc import ThreeSixtyLocDataset, SceneSample
from utilities.pose import procrustes_transform


# PRED_PATH = Path("outputs/2025-11-27T03:39:14") # VGGT Perspective barely works
# PRED_PATH = Path("outputs/2025-11-27T04:23:25")
PRED_PATH = Path("outputs/2025-11-27T05:34:31") # ViPE 4
# PRED_PATH = Path("outputs/2025-11-27T07:26:26") # VGGT Perspective 8
PRED_IDX = 0

RECONSTRUCT_STRIDE = 20
POINTS_STRIDE = 8
DATASET_WORKERS = 4

ERROR_LABELS_ENABLED = False
EQUIRECT_SHAPE = (800, 400)  # Width, height
SIZE_GT = 0.03
SIZE_PRED = 0.03
SIZE_ERROR = 0.01
COLOR_GT = [0.0, 1.0, 0.0]
COLOR_PRED = [1.0, 1.0, 0.0]
COLOR_ERROR = [1.0, 0.0, 0.0]



# TODO: Refactor perspective directions to x right, y up, z backward

pred_scene_path = sorted(p for p in PRED_PATH.iterdir() if p.is_dir())[PRED_IDX]
pred_metrics: dict[str, str | float | int] = th.load(pred_scene_path / "metrics.pt", map_location="cpu")
scene_idx = int(pred_metrics["scene_idx"])
pred_poses_w2c = cast(th.Tensor, th.load(pred_scene_path / "model_output.pt", map_location="cpu")["poses"])

# NOTE: Depth is required, so many scenes are filtered out
dataset_reconstruct = ThreeSixtyLocDataset(SceneSample, Path(environ.get("DATASET_360_LOC_ROOT", "")), stride=RECONSTRUCT_STRIDE, worker_count=DATASET_WORKERS)
dataset_validation = ThreeSixtyLocDataset(SceneSample, Path(environ.get("DATASET_360_LOC_ROOT", "")), stride=cast(int, pred_metrics["dataset_stride"]), worker_count=DATASET_WORKERS)
sample_reconstruct = dataset_reconstruct[scene_idx]
gt_poses_w2c = dataset_validation.load_poses(scene_idx)


# Setup rerun
rr.init("rerun_threesixty_explorer", spawn=True)
rr.send_blueprint(rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(
            overrides={
                
            },
            defaults=[
                rr.Pinhole.from_fields(image_plane_distance=0.01),
            ]
        ),
        rrb.TextDocumentView(origin="info"),
        column_shares=[8, 1]
    ),
    rrb.BlueprintPanel(expanded=False),
    rrb.SelectionPanel(expanded=False),
    rrb.TimePanel(expanded=False)
))

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
rr.set_time("time", timestamp=0)


# Reconstruct environment
for seq_idx in range(len(sample_reconstruct.pose)):
    # Retrieve data
    pose = sample_reconstruct.pose[seq_idx].inverse()
    pos, rot = pose[:3, 3], pose[:3, :3]

    rgb = sample_reconstruct.rgba[seq_idx].permute(1, 2, 0).numpy()
    height, width = rgb.shape[:2]
    depth = sample_reconstruct.depth[seq_idx, 0].numpy()

    # Create point cloud
    d = depth[::POINTS_STRIDE, ::POINTS_STRIDE].reshape(-1)
    v = (np.arange(0, height, POINTS_STRIDE, dtype=np.float32) + 0.5) / height
    u = (np.arange(0, width, POINTS_STRIDE, dtype=np.float32) + 0.5) / width
    lat, lon = np.meshgrid(
        -np.pi / 2 + np.pi * v,
        -np.pi + 2 * np.pi * u,
        indexing="ij",
    )
    lat, lon = lat.reshape(-1), lon.reshape(-1)
    x = d * np.cos(lat) * np.sin(lon)
    y = d * np.sin(lat)
    z = d * np.cos(lat) * np.cos(lon)

    points = np.stack((x, y, z), axis=-1)
    colors = rgb[::POINTS_STRIDE, ::POINTS_STRIDE, :].reshape(-1, 4)

    # Log environment
    rr.log(f"world/env/{seq_idx}", rr.Transform3D(translation=pos, mat3x3=rot))
    # rr.log(f"world/env/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_GT, radii=SIZE_GT))
    # rr.log(f"world/env/{seq_idx}/image", rr.Pinhole(resolution=EQUIRECT_SHAPE, focal_length=EQUIRECT_SHAPE[0], image_plane_distance=SIZE_GT * 10))
    # rr.log(f"world/env/{seq_idx}/image/rgb", rr.Image(cv2.resize(rgb, dsize=EQUIRECT_SHAPE, interpolation=cv2.INTER_LINEAR), color_model=rr.ColorModel.RGBA)) # type: ignore[reportArgumentType]
    rr.log(f"world/env/{seq_idx}/points", rr.Points3D(points, colors=colors))

sequence_len = min(len(pred_poses_w2c), len(gt_poses_w2c))
pred_poses_w2c = pred_poses_w2c[:sequence_len]
gt_poses_w2c = gt_poses_w2c[:sequence_len]

pred_aligned_w2c = procrustes_transform(
    pred_poses_w2c,
    gt_poses_w2c,
    pred_poses_w2c,
    allow_scale=True,
)
pred_aligned_c2w = pred_aligned_w2c.inverse()
gt_poses_c2w = gt_poses_w2c.inverse()

pos_gt_prev = None
pos_pred_prev = None

for seq_idx in range(sequence_len):
    pose_gt = gt_poses_c2w[seq_idx]
    pos_gt, rot_gt = pose_gt[:3, 3], pose_gt[:3, :3]
    # rgb = sample_gt.rgba[seq_idx].permute(1, 2, 0).numpy()
    # rgb = dataset_full.load_rgba(SCENE_IDX, seq_idx).permute(1, 2, 0).numpy()

    # Log ground truth
    rr.log(f"world/gt/{seq_idx}", rr.Transform3D(translation=pos_gt, mat3x3=rot_gt))
    rr.log(f"world/gt/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_GT, radii=SIZE_GT))
    rr.log(f"world/gt/{seq_idx}/image", rr.Pinhole(resolution=EQUIRECT_SHAPE, focal_length=EQUIRECT_SHAPE[0], image_plane_distance=SIZE_GT * 10))
    # rr.log(f"world/gt/{seq_idx}/image/rgb", rr.Image(cv2.resize(rgb, dsize=EQUIRECT_SHAPE, interpolation=cv2.INTER_LINEAR), color_model=rr.ColorModel.RGBA)) # type: ignore[reportArgumentType]
    rr.log(f"world/gt/{seq_idx}/image/rgb", rr.Image(np.tile(np.array(COLOR_GT), (EQUIRECT_SHAPE[1], EQUIRECT_SHAPE[0], 1)), color_model=rr.ColorModel.RGB))

    if pos_gt_prev is not None:
        rr.log(f"world/gt/traj/{seq_idx-1}-{seq_idx}", rr.Arrows3D(vectors=pos_gt - pos_gt_prev, origins=pos_gt_prev, colors=COLOR_GT, radii=SIZE_GT / 2))
    pos_gt_prev = pos_gt


#     # Log main prediction
    pose_main = pred_aligned_c2w[seq_idx]
    pos_main, rot_main = pose_main[:3, 3], pose_main[:3, :3]
    pos_error = th.linalg.norm(pos_gt - pos_main).item()
#     pos_error_total += pos_error

    rr.log(f"world/pred/main/{seq_idx}", rr.Transform3D(translation=pos_main, mat3x3=rot_main))
    rr.log(f"world/pred/main/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_PRED, radii=SIZE_PRED))
    rr.log(f"world/pred/main/{seq_idx}/image", rr.Pinhole(resolution=EQUIRECT_SHAPE, focal_length=EQUIRECT_SHAPE[0], image_plane_distance=SIZE_PRED * 10))
    rr.log(f"world/pred/main/{seq_idx}/image/rgb", rr.Image(np.tile(np.array(COLOR_PRED), (EQUIRECT_SHAPE[1], EQUIRECT_SHAPE[0], 1)), color_model=rr.ColorModel.RGB))
    
    if pos_pred_prev is not None:
        rr.log(f"world/pred/main/traj/{seq_idx-1}-{seq_idx}", rr.Arrows3D(vectors=pos_main - pos_pred_prev, origins=pos_pred_prev, colors=COLOR_PRED, radii=SIZE_PRED / 2))
    pos_pred_prev = pos_main

    rr.log(f"world/error/main/{seq_idx}", rr.Arrows3D(vectors=pos_gt - pos_main, origins=pos_main, colors=COLOR_ERROR, radii=SIZE_ERROR / 2, labels=[f"{pos_error:.3f}m"], show_labels=ERROR_LABELS_ENABLED))

# Log metrics
rr.log(
    "info",
    rr.TextDocument(
        f"- Model:\n"
        f"  - Name: {pred_metrics["model_name"]}\n"
        f"  - Chunker size: {pred_metrics["chunker_chunk_size"]}\n"
        f"  - Chunker overlap: {pred_metrics["chunker_chunk_overlap"]}\n"
        f"- Scene: {sample_reconstruct.id}\n"
        f"  - Index: {scene_idx}\n"
        f"  - Stride: {pred_metrics["dataset_stride"]}\n"
        f"  - Length: {pred_metrics["sequence_length"]}\n"
        f"- Compute:\n"
        f"  - FPS: {cast(float, pred_metrics["sequence_length"]) / cast(float, pred_metrics["elapsed_seconds"]):.2f}\n"
        f"  - Time (s): {cast(float, pred_metrics["elapsed_seconds"]):.2f}\n"
        f"  - GPU max (GB): {cast(float, pred_metrics["gpu_memory_peak"]) / 1024**3:.2f}\n"
        f"  - CPU RSS (GB): {cast(float, pred_metrics["cpu_memory_rss"]) / 1024**3:.2f}\n"
        f"- Errors:\n"
        f"  - Translation (m): {cast(float, pred_metrics["translation_error_mean"]):.2f} ± {cast(float, pred_metrics["translation_error_std"]):.2f}\n"
        f"  - Rotation Geodesic (°): {degrees(cast(float, pred_metrics["rotation_geodesic_mean"])):.2f} ± {degrees(cast(float, pred_metrics["rotation_geodesic_std"])):.2f}\n"
        f"  - Rotation Pointing (°): {degrees(cast(float, pred_metrics["rotation_pointing_mean"])):.2f} ± {degrees(cast(float, pred_metrics["rotation_pointing_std"])):.2f}\n"
        f"  - Rotation Roll (°): {degrees(cast(float, pred_metrics["rotation_roll_mean"])):.2f} ± {degrees(cast(float, pred_metrics["rotation_roll_std"])):.2f}\n",
        media_type=rr.MediaType.MARKDOWN
    )
)

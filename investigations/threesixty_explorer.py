from pathlib import Path
from typing import cast
from os import environ
from math import degrees

import rerun as rr
import rerun.blueprint as rrb
import torch as th
import numpy as np
from loguru import logger

from splat_init.data.threesixty_loc import ThreeSixtyLocDataset, SceneSample
from utilities.pose import mean_rotation_karcher, procrustes_analysis


PRED_PATH = Path("outputs/2025-11-27T02:08:06")
PRED_IDX = 0

RECONSTRUCT_STRIDE = 20
POINTS_STRIDE = 8
DATASET_WORKERS = 4

EQUIRECT_SHAPE = (800, 400)  # Width, height
SIZE_GT = 0.03
SIZE_PRED = 0.03
SIZE_PERSP = 0.01
COLOR_GT = [0.0, 1.0, 0.0]
COLOR_PRED = [1.0, 1.0, 0.0]
COLOR_PERSP = [0.5, 0.0, 0.0]



# TODO: Refactor perspective directions to x right, y up, z backward

# projector = OTCProjector(face_size=VGGT_TARGET_SIZE, alpha=1e-9)

pred_scene_path = sorted(p for p in PRED_PATH.iterdir() if p.is_dir())[PRED_IDX]
pred_metrics: dict[str, str | float | int] = th.load(pred_scene_path / "metrics.pt", map_location="cpu")
scene_idx = int(pred_metrics["scene_idx"])
pred_poses = cast(th.Tensor, th.load(pred_scene_path / "model_output.pt", map_location="cpu")["poses"]).inverse()

# NOTE: Depth is required, so many scenes are filtered out
dataset_reconstruct = ThreeSixtyLocDataset(SceneSample, Path(environ.get("DATASET_360_LOC_ROOT", "")), stride=RECONSTRUCT_STRIDE, worker_count=DATASET_WORKERS)
dataset_validation = ThreeSixtyLocDataset(SceneSample, Path(environ.get("DATASET_360_LOC_ROOT", "")), stride=cast(int, pred_metrics["dataset_stride"]), worker_count=DATASET_WORKERS)
sample_reconstruct = dataset_reconstruct[scene_idx]
sample_validation_poses = dataset_validation.load_poses(scene_idx).inverse()

# preds = cast(dict[str, th.Tensor], th.load(PRED_PATH / f"{sample_gt.id}.pt" if PRED_PATH.is_dir() else PRED_PATH, map_location="cpu"))
# procrustes_align = procrustes_analysis(preds["poses"][:, :3, 3], sample_gt.pose.inverse()[:, :3, 3])

# pos_error_total = 0.0
# pos_error_local_total = 0.0


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
    


# # Draw poses
# rgb_faces, alpha_faces, _ = projector(sample_gt.rgba.to("cuda", th.bfloat16), None)
# rgba_faces = th.concat([rgb_faces, alpha_faces], dim=2).permute(0, 1, 3, 4, 2).to("cpu", th.float32).numpy() # [S,6,H,W,4]
# rgba_faces = rgba_faces[:, [0, 1, 4, 5], ...] # Discard top and bottom faces


# # TODO: Remove temporary full pose logging code
# import numpy as np
# pred_data = np.load("outputs/vipe_results_atrium/pose/atrium.npz")
# # NOTE: Assuming correct order in pred_data["inds"]
# preds = th.tensor(pred_data["data"])


procrustes_align = procrustes_analysis(pred_poses[:, :3, 3], sample_validation_poses[:len(pred_poses), :3, 3])


pos_gt_prev = None
# NOTE: Used to correct for constant rotation offset in predictions
rot_delta = mean_rotation_karcher(
    sample_validation_poses[:len(pred_poses), :3, :3] @ procrustes_align(pred_poses[:, :3, 3], pred_poses[:, :3, :3])[1].inverse()
)

for seq_idx in range(len(pred_poses)):
    # Get ground truth pose
    # pose_gt = sample_gt.pose[seq_idx].inverse()
    pose_gt = sample_validation_poses[seq_idx]
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
        rr.log(f"world/traj/{seq_idx-1}-{seq_idx}", rr.Arrows3D(vectors=pos_gt - pos_gt_prev, origins=pos_gt_prev, colors=COLOR_GT, radii=SIZE_GT / 2))
    pos_gt_prev = pos_gt

#     # # TEMP: Checking for correct cube face rotations
#     # rgb, alpha, depth = projector(sample_gt.rgba.to(th.device("cuda"), th.bfloat16), sample_gt.depth.to(th.device("cuda"), th.bfloat16))
#     # rgb, alpha, depth = rgb.to(th.float32).cpu(), alpha.to(th.float32).cpu(), depth.to(th.float32).cpu()
#     # rgba = th.concat([rgb, alpha], dim=2)
#     # rot_gt_tmp = th.tensor(rot_gt) @ cube_face_relative_rotations()
#     # for i in range(6):
#     #     rr.log(f"world/gt/{seq_idx}-{i}/image", rr.Pinhole(resolution=(VGGT_TARGET_SIZE, VGGT_TARGET_SIZE), focal_length=VGGT_TARGET_SIZE/2, image_plane_distance=0.3))
#     #     rr.log(f"world/gt/{seq_idx}-{i}", rr.Transform3D(translation=pos_gt, mat3x3=rot_gt_tmp[i].numpy()))
#     #     rr.log(f"world/gt/{seq_idx}-{i}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=[0.0, 1.0, 1.0], radii=0.03))
#     #     rr.log(f"world/gt/{seq_idx}-{i}/image/rgb", rr.Image(rgba[seq_idx, i].permute(1, 2, 0).numpy(), color_model=rr.ColorModel.RGBA))


#     # Log main prediction
    pose_main = pred_poses[seq_idx]
    pos_main, rot_main = pose_main[:3, 3], pose_main[:3, :3]
    pos_main, rot_main = procrustes_align(pos_main, rot_main)
    rot_main = rot_delta @ rot_main
    pos_error = th.linalg.norm(pos_gt - pos_main).item()
#     pos_error_total += pos_error

    rr.log(f"world/pred/main/{seq_idx}", rr.Transform3D(translation=pos_main, mat3x3=rot_main))
    rr.log(f"world/pred/main/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_PRED, radii=SIZE_PRED))
    rr.log(f"world/pred/main/{seq_idx}/image", rr.Pinhole(resolution=EQUIRECT_SHAPE, focal_length=EQUIRECT_SHAPE[0], image_plane_distance=SIZE_PRED * 10))
    rr.log(f"world/pred/main/{seq_idx}/image/rgb", rr.Image(np.tile(np.array(COLOR_PRED), (EQUIRECT_SHAPE[1], EQUIRECT_SHAPE[0], 1)), color_model=rr.ColorModel.RGB))

    rr.log(f"world/error/main/{seq_idx}", rr.Arrows3D(vectors=pos_gt - pos_main, origins=pos_main, colors=COLOR_PRED, radii=SIZE_PRED / 10, labels=[f"{pos_error:.3f}m"]))





#     # Log perspective prediction
#     if "poses_faces" not in preds:
#         continue

#     if "images_faces" not in preds:
#         rgba = rgba_faces[seq_idx]
#     else:
#         rgba = [th.cat((img.permute(1, 2, 0), th.full((*img.shape[1:], 1), 1.0)), dim=-1).numpy() for img in preds["images_faces"][seq_idx]]

#     pos_error_local = 0.0

#     n_faces = preds["poses_faces"].shape[1]
#     for i in range(n_faces):
#         pos_persp, rot_persp = procrustes_align(preds["poses_faces"][seq_idx, i, :3, 3], preds["poses_faces"][seq_idx, i, :3, :3])
#         pos_error_local += th.linalg.norm(pos_persp - pos_main).item()

#         rr.log(f"world/pred/persp/{seq_idx}/{i}/image", rr.Pinhole(resolution=(VGGT_TARGET_SIZE, VGGT_TARGET_SIZE), focal_length=VGGT_TARGET_SIZE/2, image_plane_distance=SIZE_PERSP * 10))
#         rr.log(f"world/pred/persp/{seq_idx}/{i}", rr.Transform3D(translation=pos_persp, mat3x3=rot_persp))
#         rr.log(f"world/pred/persp/{seq_idx}/{i}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_PERSP, radii=SIZE_PERSP))
#         rr.log(f"world/pred/persp/{seq_idx}/{i}/image/rgb", rr.Image(rgba[i], color_model=rr.ColorModel.RGBA))

#         rr.log(f"world/error/persp/{seq_idx}/{i}", rr.Arrows3D(vectors=pos_main - pos_persp, origins=pos_persp, colors=COLOR_PERSP, radii=SIZE_PERSP / 10))

#     pos_error_local_total += pos_error_local / n_faces

# Log metrics
# Metrics format
# {
#         "translation_error_mean": translation_error.mean().item(),
#         "translation_error_std": translation_error.std(unbiased=False).item(),
#         "rotation_geodesic_mean": geodesic_error.mean().item(),
#         "rotation_geodesic_std": geodesic_error.std(unbiased=False).item(),
#         "rotation_pointing_mean": pointing_error.mean().item(),
#         "rotation_pointing_std": pointing_error.std(unbiased=False).item(),
#         "rotation_roll_mean": roll_error.mean().item(),
#         "rotation_roll_std": roll_error.std(unbiased=False).item(),
#           "elapsed_seconds": elapsed_seconds,
#           "scene_idx": scene_idx,
#           "sequence_length": sequence_length,
#           "dataset_stride": stride,
#         "gpu_memory_allocated": gpu_alloc,
#           "gpu_memory_peak": gpu_peak,
#           "cpu_memory_rss": cpu_rss_bytes,
#           "model_name": model_name,
#     }

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
        f"- Time (s):\n"
        f"  - Elapsed: {cast(float, pred_metrics["elapsed_seconds"]):.2f}\n"
        f"  - FPS: {cast(float, pred_metrics["sequence_length"]) / cast(float, pred_metrics["elapsed_seconds"]):.2f}\n"
        f"- Compute (GB):\n"
        f"  - GPU max: {cast(float, pred_metrics["gpu_memory_peak"]) / 1024**3:.2f}\n"
        f"  - CPU RSS: {cast(float, pred_metrics["cpu_memory_rss"]) / 1024**3:.2f}\n"
        f"- Errors:\n"
        f"  - Translation (m): {cast(float, pred_metrics["translation_error_mean"]):.2f} ± {cast(float, pred_metrics["translation_error_std"]):.2f}\n"
        f"  - Rotation Geodesic (°): {degrees(cast(float, pred_metrics["rotation_geodesic_mean"])):.2f} ± {degrees(cast(float, pred_metrics["rotation_geodesic_std"])):.2f}\n"
        f"  - Rotation Pointing (°): {degrees(cast(float, pred_metrics["rotation_pointing_mean"])):.2f} ± {degrees(cast(float, pred_metrics["rotation_pointing_std"])):.2f}\n"
        f"  - Rotation Roll (°): {degrees(cast(float, pred_metrics["rotation_roll_mean"])):.2f} ± {degrees(cast(float, pred_metrics["rotation_roll_std"])):.2f}\n",
        media_type=rr.MediaType.MARKDOWN
    )
)

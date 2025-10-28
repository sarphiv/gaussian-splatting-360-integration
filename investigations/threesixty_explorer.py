from pathlib import Path
from dataclasses import dataclass
from typing import cast, Callable
from os import environ
from math import pi, ceil

import rerun as rr
import rerun.blueprint as rrb
import torch as th
import torchvision.transforms.functional as tvf
import numpy as np
import cv2

from splat_init.data.threesixty_loc import ThreeSixtyLocDataset
from splat_init.models.vggt_perspective_transform import OTCProjector, cube_face_relative_rotations
from configs.training_args import Args
from configs.constants import VGGT_TARGET_SIZE


# PRED_PATH = Path("outputs/2025-10-05T17:30:48")
# PRED_PATH = Path("outputs/2025-10-05T18:20:01") # Perspective transform
# PRED_PATH = Path("outputs/2025-10-08T22:57:20") # Equirectangular
# PRED_PATH = Path("outputs/2025-10-08T23:01:09") # Perspective transform
# PRED_PATH = Path("outputs/2025-10-09T01:09:55/NON-COMPLIANT-FORMAT.area_4.conferenceRoom_3.pt") # Perspective direct
SCENE_IDX = 0

POINTS_STRIDE = 8

EQUIRECT_SHAPE = (800, 400)  # Width, height
SIZE_GT = 0.03
SIZE_PRED = 0.03
SIZE_PERSP = 0.01
COLOR_GT = [0.0, 1.0, 0.0]
COLOR_PRED = [1.0, 1.0, 0.0]
COLOR_PERSP = [0.5, 0.0, 0.0]



# TODO: Refactor perspective directions to x right, y up, z backward


# def procrustes_analysis(pred: th.Tensor, target: th.Tensor) -> Callable[[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]:
#     assert pred.shape == target.shape, "Predicted and target arrays must match in shape."
#     pred_centroid = pred.mean(dim=0)
#     target_centroid = target.mean(dim=0)

#     pred_centered = pred - pred_centroid
#     target_centered = target - target_centroid

#     covariance = pred_centered.T @ target_centered
#     u, singular_values, vt = th.linalg.svd(covariance)

#     align_rotation = u @ vt
#     if th.linalg.det(align_rotation) < 0:
#         vt[-1, :] *= -1
#         align_rotation = u @ vt

#     scale_numerator = singular_values.sum()
#     scale_denominator = th.sum(pred_centered ** 2)
#     assert scale_denominator > 0.0, "Predicted scene must span more than a single point."
#     scale = scale_numerator / scale_denominator

#     def procrustes_align(position: th.Tensor, rotation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
#         aligned_pos = scale * (position - pred_centroid) @ align_rotation + target_centroid
#         aligned_rot = rotation @ align_rotation
#         return aligned_pos, aligned_rot

#     return procrustes_align




args_main = Args()

# projector = OTCProjector(face_size=VGGT_TARGET_SIZE, alpha=1e-9)

dataset = ThreeSixtyLocDataset(Path(environ.get("DATASET_360_LOC_ROOT", "")), stride=20, worker_count=4)
sample_gt = dataset[SCENE_IDX]

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
pos_prev = None
for seq_idx in range(len(sample_gt.pose)):
    # Retrieve data
    pose = sample_gt.pose[seq_idx].inverse()
    pos, rot = pose[:3, 3], pose[:3, :3]

    rgb = sample_gt.rgba[seq_idx].permute(1, 2, 0).numpy()
    height, width = rgb.shape[:2]
    depth = sample_gt.depth[seq_idx, 0].numpy()

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
    rr.log(f"world/env/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_GT, radii=SIZE_GT))
    rr.log(f"world/env/{seq_idx}/image", rr.Pinhole(resolution=EQUIRECT_SHAPE, focal_length=EQUIRECT_SHAPE[0], image_plane_distance=SIZE_GT * 10))
    rr.log(f"world/env/{seq_idx}/image/rgb", rr.Image(cv2.resize(rgb, dsize=EQUIRECT_SHAPE, interpolation=cv2.INTER_LINEAR), color_model=rr.ColorModel.RGBA)) # type: ignore[reportArgumentType]
    rr.log(f"world/env/{seq_idx}/points", rr.Points3D(points, colors=colors))
    
    if pos_prev is not None:
        rr.log(f"world/traj/{seq_idx-1}-{seq_idx}", rr.Arrows3D(vectors=pos - pos_prev, origins=pos_prev, colors=COLOR_GT, radii=SIZE_GT / 2, labels=[f"{th.linalg.norm(pos - pos_prev).item():.3f}m"]))
    pos_prev = pos


# # Draw poses
# rgb_faces, alpha_faces, _ = projector(sample_gt.rgba.to("cuda", th.bfloat16), None)
# rgba_faces = th.concat([rgb_faces, alpha_faces], dim=2).permute(0, 1, 3, 4, 2).to("cpu", th.float32).numpy() # [S,6,H,W,4]
# rgba_faces = rgba_faces[:, [0, 1, 4, 5], ...] # Discard top and bottom faces



for seq_idx in range(len(sample_gt.pose)):
    # Get ground truth pose
    pose_gt = sample_gt.pose[seq_idx].inverse()
    pos_gt, rot_gt = pose_gt[:3, 3], pose_gt[:3, :3]

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
#     pos_main, rot_main = procrustes_align(preds["poses"][seq_idx, :3, 3], preds["poses"][seq_idx, :3, :3])
#     pos_error = th.linalg.norm(pos_gt - pos_main).item()
#     pos_error_total += pos_error

#     rr.log(f"world/pred/main/{seq_idx}/image", rr.Pinhole(resolution=EQUIRECT_SHAPE, focal_length=EQUIRECT_SHAPE[0], image_plane_distance=SIZE_PRED * 10))
#     rr.log(f"world/pred/main/{seq_idx}", rr.Transform3D(translation=pos_main, mat3x3=rot_main))
#     rr.log(f"world/pred/main/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_PRED, radii=SIZE_PRED))

#     rr.log(f"world/error/main/{seq_idx}", rr.Arrows3D(vectors=pos_gt - pos_main, origins=pos_main, colors=COLOR_PRED, radii=SIZE_PRED / 10, labels=[f"{pos_error:.3f}m"]))


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

# # Log information
# rr.log(
#     "info",
#     rr.TextDocument(
#         f"{sample_env.id}\n\n"
#         f"- Sequence length: {len(sample_gt.pose)}\n"
#         f"- Mean position error: {pos_error_total/len(sample_gt.pose):.2f}m\n"
#         f"- Mean local error: {pos_error_local_total/len(sample_gt.pose):.2f}m\n",
#         media_type=rr.MediaType.MARKDOWN
#     )
# )
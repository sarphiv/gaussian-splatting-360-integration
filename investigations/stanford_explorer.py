from pathlib import Path
from dataclasses import dataclass
from typing import cast, Callable

import rerun as rr
import rerun.blueprint as rrb
import torch as th
import numpy as np

from splat_init.data.stanford_2d_3d import Stanford2D3DDataset
from splat_init.models.vggt_perspective_transform import OTCProjector, cube_face_relative_rotations
from configs.training_args import Args
from configs.constants import VGGT_TARGET_SIZE


# PRED_PATH = Path("outputs/2025-10-05T17:30:48")
# PRED_PATH = Path("outputs/2025-10-05T18:20:01") # Perspective transform
# PRED_PATH = Path("outputs/2025-10-08T22:57:20") # Equirectangular
# PRED_PATH = Path("outputs/2025-10-08T23:01:09") # Perspective transform
PRED_PATH = Path("outputs/2025-10-09T01:09:55/NON-COMPLIANT-FORMAT.area_4.conferenceRoom_3.pt") # Perspective direct
DATASET_IDX = 1
SCENE_IDX = 5

DEPTH_MAX_DISTANCE = 6.0
POINTS_STRIDE = 10

EQUIRECT_SHAPE = (200, 100)  # Width, height
SIZE_GT = 0.03
SIZE_PRED = 0.03
SIZE_PERSP = 0.01
COLOR_GT = [0.0, 1.0, 0.0]
COLOR_PRED = [1.0, 1.0, 0.0]
COLOR_PERSP = [0.5, 0.0, 0.0]



# TODO: Refactor perspective directions to x right, y up, z backward


def procrustes_analysis(pred: th.Tensor, target: th.Tensor) -> Callable[[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]:
    assert pred.shape == target.shape, "Predicted and target arrays must match in shape."
    pred_centroid = pred.mean(dim=0)
    target_centroid = target.mean(dim=0)

    pred_centered = pred - pred_centroid
    target_centered = target - target_centroid

    covariance = pred_centered.T @ target_centered
    u, singular_values, vt = th.linalg.svd(covariance)

    align_rotation = u @ vt
    if th.linalg.det(align_rotation) < 0:
        vt[-1, :] *= -1
        align_rotation = u @ vt

    scale_numerator = singular_values.sum()
    scale_denominator = th.sum(pred_centered ** 2)
    assert scale_denominator > 0.0, "Predicted scene must span more than a single point."
    scale = scale_numerator / scale_denominator

    def procrustes_align(position: th.Tensor, rotation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        aligned_pos = scale * (position - pred_centroid) @ align_rotation + target_centroid
        aligned_rot = rotation @ align_rotation
        return aligned_pos, aligned_rot

    return procrustes_align




args_main = Args()

projector = OTCProjector(face_size=VGGT_TARGET_SIZE, alpha=1e-9)

datasets = [Stanford2D3DDataset(p, args_main.data.max_sequence_length, 12) for p in args_main.data.stanford_val_areas]
sample_env = datasets[DATASET_IDX].get_perspective(SCENE_IDX)
sample_gt = datasets[DATASET_IDX][SCENE_IDX]
assert sample_env.focal_length is not None

preds = cast(dict[str, th.Tensor], th.load(PRED_PATH / f"{sample_gt.id}.pt" if PRED_PATH.is_dir() else PRED_PATH, map_location="cpu"))
procrustes_align = procrustes_analysis(preds["poses"][:, :3, 3], sample_gt.pose.inverse()[:, :3, 3])

pos_error_total = 0.0
pos_error_local_total = 0.0


# Setup rerun
rr.init("rerun_stanford_explorer", spawn=True)
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
for seq_idx in range(len(sample_env.pose)):
    # Retrieve data
    f = sample_env.focal_length[seq_idx].item()
    pose = sample_env.pose[seq_idx].inverse()
    pos, rot = pose[:3, 3], pose[:3, :3]

    rgb = sample_env.rgba[seq_idx].numpy()
    height, width = rgb.shape[1], rgb.shape[2]
    depth = sample_env.depth[seq_idx].numpy()
    depth = depth * (depth < DEPTH_MAX_DISTANCE)

    # Create point cloud
    x, y = np.meshgrid(
        np.linspace(-height/2, height/2, height // POINTS_STRIDE), 
        np.linspace(-width/2, width/2, width // POINTS_STRIDE), 
        indexing="ij"
    )
    z = depth[0, ::POINTS_STRIDE, ::POINTS_STRIDE].reshape(-1)
    points = np.stack((y.reshape(-1) / f * z, x.reshape(-1) / f * z, z), axis=-1)
    colors = rgb[:, ::POINTS_STRIDE, ::POINTS_STRIDE].transpose(1, 2, 0).reshape(-1, 3)

    # Log environment
    rr.log(f"world/env/{seq_idx}/image", rr.Pinhole(resolution=(width, height), focal_length=f))
    rr.log(f"world/env/{seq_idx}", rr.Transform3D(translation=pos, mat3x3=rot))

    rr.log(f"world/env/{seq_idx}/points", rr.Points3D(points, colors=colors))
    # rr.log(f"world/env/{seq_idx}/{face_idx}/image/rgb", rr.Image(rgba[seq_idx, face_idx].permute(1, 2, 0).numpy(), color_model=ColorModel.RGBA))
    # rr.log(f"world/env/{seq_idx}/image/depth", rr.DepthImage(depth.numpy(), meter=1.0))
    

# Draw poses
rgb_faces, alpha_faces, _ = projector(sample_gt.rgba.to("cuda", th.bfloat16), None)
rgba_faces = th.concat([rgb_faces, alpha_faces], dim=2).permute(0, 1, 3, 4, 2).to("cpu", th.float32).numpy() # [S,6,H,W,4]
rgba_faces = rgba_faces[:, [0, 1, 4, 5], ...] # Discard top and bottom faces

for seq_idx in range(len(sample_gt.pose)):
    # Log ground truth
    pose_gt = sample_gt.pose[seq_idx].inverse()
    pos_gt, rot_gt = pose_gt[:3, 3], pose_gt[:3, :3]

    rgb = sample_env.rgba[seq_idx].numpy()
    height, width = rgb.shape[1], rgb.shape[2]
    rr.log(f"world/gt/{seq_idx}/image", rr.Pinhole(resolution=EQUIRECT_SHAPE, focal_length=EQUIRECT_SHAPE[0], image_plane_distance=SIZE_GT * 10))
    rr.log(f"world/gt/{seq_idx}", rr.Transform3D(translation=pos_gt, mat3x3=rot_gt))
    rr.log(f"world/gt/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_GT, radii=SIZE_GT))


    # # TEMP: Checking for correct cube face rotations
    # rgb, alpha, depth = projector(sample_gt.rgba.to(th.device("cuda"), th.bfloat16), sample_gt.depth.to(th.device("cuda"), th.bfloat16))
    # rgb, alpha, depth = rgb.to(th.float32).cpu(), alpha.to(th.float32).cpu(), depth.to(th.float32).cpu()
    # rgba = th.concat([rgb, alpha], dim=2)
    # rot_gt_tmp = th.tensor(rot_gt) @ cube_face_relative_rotations()
    # for i in range(6):
    #     rr.log(f"world/gt/{seq_idx}-{i}/image", rr.Pinhole(resolution=(VGGT_TARGET_SIZE, VGGT_TARGET_SIZE), focal_length=VGGT_TARGET_SIZE/2, image_plane_distance=0.3))
    #     rr.log(f"world/gt/{seq_idx}-{i}", rr.Transform3D(translation=pos_gt, mat3x3=rot_gt_tmp[i].numpy()))
    #     rr.log(f"world/gt/{seq_idx}-{i}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=[0.0, 1.0, 1.0], radii=0.03))
    #     rr.log(f"world/gt/{seq_idx}-{i}/image/rgb", rr.Image(rgba[seq_idx, i].permute(1, 2, 0).numpy(), color_model=rr.ColorModel.RGBA))


    # Log main prediction
    pos_main, rot_main = procrustes_align(preds["poses"][seq_idx, :3, 3], preds["poses"][seq_idx, :3, :3])
    pos_error = th.linalg.norm(pos_gt - pos_main).item()
    pos_error_total += pos_error

    rr.log(f"world/pred/main/{seq_idx}/image", rr.Pinhole(resolution=EQUIRECT_SHAPE, focal_length=EQUIRECT_SHAPE[0], image_plane_distance=SIZE_PRED * 10))
    rr.log(f"world/pred/main/{seq_idx}", rr.Transform3D(translation=pos_main, mat3x3=rot_main))
    rr.log(f"world/pred/main/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_PRED, radii=SIZE_PRED))

    rr.log(f"world/error/main/{seq_idx}", rr.Arrows3D(vectors=pos_gt - pos_main, origins=pos_main, colors=COLOR_PRED, radii=SIZE_PRED / 10, labels=[f"{pos_error:.3f}m"]))


    # Log perspective prediction
    if "poses_faces" not in preds:
        continue

    if "images_faces" not in preds:
        rgba = rgba_faces[seq_idx]
    else:
        rgba = [th.cat((img.permute(1, 2, 0), th.full((*img.shape[1:], 1), 1.0)), dim=-1).numpy() for img in preds["images_faces"][seq_idx]]

    pos_error_local = 0.0

    n_faces = preds["poses_faces"].shape[1]
    for i in range(n_faces):
        pos_persp, rot_persp = procrustes_align(preds["poses_faces"][seq_idx, i, :3, 3], preds["poses_faces"][seq_idx, i, :3, :3])
        pos_error_local += th.linalg.norm(pos_persp - pos_main).item()

        rr.log(f"world/pred/persp/{seq_idx}/{i}/image", rr.Pinhole(resolution=(VGGT_TARGET_SIZE, VGGT_TARGET_SIZE), focal_length=VGGT_TARGET_SIZE/2, image_plane_distance=SIZE_PERSP * 10))
        rr.log(f"world/pred/persp/{seq_idx}/{i}", rr.Transform3D(translation=pos_persp, mat3x3=rot_persp))
        rr.log(f"world/pred/persp/{seq_idx}/{i}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=COLOR_PERSP, radii=SIZE_PERSP))
        rr.log(f"world/pred/persp/{seq_idx}/{i}/image/rgb", rr.Image(rgba[i], color_model=rr.ColorModel.RGBA))

        rr.log(f"world/error/persp/{seq_idx}/{i}", rr.Arrows3D(vectors=pos_main - pos_persp, origins=pos_persp, colors=COLOR_PERSP, radii=SIZE_PERSP / 10))

    pos_error_local_total += pos_error_local / n_faces

# Log information
rr.log(
    "info",
    rr.TextDocument(
        f"{sample_env.id}\n\n"
        f"- Sequence length: {len(sample_gt.pose)}\n"
        f"- Mean position error: {pos_error_total/len(sample_gt.pose):.2f}m\n"
        f"- Mean local error: {pos_error_local_total/len(sample_gt.pose):.2f}m\n",
        media_type=rr.MediaType.MARKDOWN
    )
)
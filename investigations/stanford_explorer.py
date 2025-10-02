from math import tan, pi

import rerun as rr
import rerun.blueprint as rrb
import torch as th
import torch.nn.functional as F
import numpy as np
from loguru import logger

from splat_init.data.stanford_2d_3d import Stanford2D3DDataset
from splat_init.models.vggt_perspective_transform import OTCProjector
from configs.training_args import Args
from configs.constants import VGGT_TARGET_SIZE


def cube_face_relative_rotations(R: th.Tensor) -> th.Tensor:
    """
    R: [...,3,3] rotation (camera->world) for the equirect panorama.
    Returns R_face: [...,6,3,3]
    Frames are right-handed with +X right, +Y down, +Z forward.
    """
    assert R.shape[-2:] == (3, 3)
    device, dtype = R.device, R.dtype

    ex = th.tensor([1.,0.,0.], device=device, dtype=dtype)
    ey = th.tensor([0.,1.,0.], device=device, dtype=dtype)
    ez = th.tensor([0.,0.,1.], device=device, dtype=dtype)

    def M(c1, c2, c3):
        return th.stack((c1, c2, c3), dim=-1)  # [3,3] with columns r,u,f

    faces = [
        M(-ez,  ey,  ex),  # +X
        M( ez,  ey, -ex),  # -X
        M( ex, -ez,  ey),  # +Y
        M( ex,  ez, -ey),  # -Y
        M( ex,  ey,  ez),  # +Z
        M(-ex,  ey, -ez),  # -Z
    ]
    R_i = th.stack(faces, dim=-3)  # [6,3,3]
    R_face = R @ R_i
    return R_face # [...,6,3,3]


DEPTH_MAX_DISTANCE = 6.0
POINTS_STRIDE = 10

rr.init("rerun_stanford_explorer", spawn=True)
rr.send_blueprint(rrb.Blueprint(
    rrb.Spatial3DView(
        overrides={
            
        },
        defaults=[
            rr.Pinhole.from_fields(image_plane_distance=0.01),
        ]
    ),
    rrb.BlueprintPanel(expanded=False),
    rrb.SelectionPanel(expanded=False),
    rrb.TimePanel(expanded=False)
))


args_training = Args()
datasets = [Stanford2D3DDataset(p, args_training.data.max_sequence_length, 12) for p in args_training.data.val_areas]

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
rr.set_time("time", timestamp=0)
projector = OTCProjector(face_size=VGGT_TARGET_SIZE, alpha=1e-9)


# TODO: In VGGT, make perspective predictions point in main direction assuming they were correctly predicted
# TODO: Save averaged and perspective pose predictions again
# TODO: LOad predictions
# TODO: Procrustus alignment

# TODO: Output error statistics again
# TODO: Refactor camera visualization numbers into constants
# TODO: Refactor perspective directions to x right, y up, z backward


dataset_idx = 0
scene_idx = 2
sample_env = datasets[dataset_idx].get_perspective(scene_idx)
sample_gt = datasets[dataset_idx][scene_idx]
assert sample_env.focal_length is not None

# TODO: Make this part of the rerun blueprint instead
logger.info(sample_env.id)


for seq_idx in range(len(sample_env.pose)):
    # Retrieve data
    f = sample_env.focal_length[seq_idx].item()
    pose = sample_env.pose[seq_idx].inverse()
    rot = pose[:3, :3].numpy()
    pos = pose[:3, 3].numpy()

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
    

for seq_idx in range(len(sample_gt.pose)):
    pose_gt = sample_gt.pose[seq_idx].inverse()
    rot_gt = pose_gt[:3, :3]
    pos_gt = pose_gt[:3, 3]
    
    # Log ground truth
    rr.log(f"world/gt/{seq_idx}/image", rr.Pinhole(resolution=(200, 100), focal_length=200.0, image_plane_distance=0.3))
    rr.log(f"world/gt/{seq_idx}", rr.Transform3D(translation=pos_gt, mat3x3=rot_gt))
    rr.log(f"world/gt/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=[0.0, 1.0, 0.0], radii=0.03))


    # # TEMP: Checking for correct cube face rotations
    # rgb, alpha, depth = projector(sample_gt.rgba.to(th.device("cuda"), th.bfloat16), sample_gt.depth.to(th.device("cuda"), th.bfloat16))
    # rgb, alpha, depth = rgb.to(th.float32).cpu(), alpha.to(th.float32).cpu(), depth.to(th.float32).cpu()
    # rgba = th.concat([rgb, alpha], dim=2)
    # rot_gt_tmp = cube_face_relative_rotations(th.tensor(rot_gt))
    # for i in range(6):
    #     rr.log(f"world/gt/{seq_idx}-{i}/image", rr.Pinhole(resolution=(VGGT_TARGET_SIZE, VGGT_TARGET_SIZE), focal_length=VGGT_TARGET_SIZE/2, image_plane_distance=0.3))
    #     rr.log(f"world/gt/{seq_idx}-{i}", rr.Transform3D(translation=pos_gt, mat3x3=rot_gt_tmp[i].numpy()))
    #     rr.log(f"world/gt/{seq_idx}-{i}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=[0.0, 1.0, 1.0], radii=0.03))
    #     rr.log(f"world/gt/{seq_idx}-{i}/image/rgb", rr.Image(rgba[seq_idx, i].permute(1, 2, 0).numpy(), color_model=rr.ColorModel.RGBA))


    # TODO: Use real data
    # Log main prediction
    pos_main = pos_gt + np.random.randn(3) / 10
    rr.log(f"world/pred/main/{seq_idx}/image", rr.Pinhole(resolution=(200, 100), focal_length=200.0, image_plane_distance=0.3))
    rr.log(f"world/pred/main/{seq_idx}", rr.Transform3D(translation=pos_main, mat3x3=rot_gt))
    rr.log(f"world/pred/main/{seq_idx}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=[1.0, 1.0, 0.0], radii=0.03))
    
    rr.log(f"world/error/main/{seq_idx}", rr.Arrows3D(vectors=pos_gt - pos_main, origins=pos_main, colors=[1.0, 1.0, 0.0], radii=0.005))
    
    # TODO: Use real data, note real data only has 4 perspectives
    # Log perspective prediction
    rgb, alpha, depth = projector(sample_gt.rgba.to("cuda", th.bfloat16), sample_gt.depth.to("cuda", th.bfloat16))
    rgb, alpha, depth = rgb.to("cpu", th.float32).numpy(), alpha.to("cpu", th.float32).numpy(), depth.to("cpu", th.float32).numpy()
    rgba = np.concatenate([rgb, alpha], axis=2).transpose(0, 1, 3, 4, 2)  # [S,6,H,W,4]

    for i in range(6):
        pos_persp = pos_main + np.random.randn(3) / 5
        rr.log(f"world/pred/persp/{seq_idx}/{i}/image", rr.Pinhole(resolution=(VGGT_TARGET_SIZE, VGGT_TARGET_SIZE), focal_length=VGGT_TARGET_SIZE/2, image_plane_distance=0.1))
        rr.log(f"world/pred/persp/{seq_idx}/{i}", rr.Transform3D(translation=pos_persp, mat3x3=rot_gt))
        rr.log(f"world/pred/persp/{seq_idx}/{i}/pos", rr.Points3D(positions=[0.0, 0.0, 0.0], colors=[1.0, 0.0, 0.0], radii=0.01))
        rr.log(f"world/pred/persp/{seq_idx}/{i}/image/rgb", rr.Image(rgba[seq_idx, i], color_model=rr.ColorModel.RGBA))

        rr.log(f"world/error/persp/{seq_idx}/{i}", rr.Arrows3D(vectors=pos_main - pos_persp, origins=pos_persp, colors=[1.0, 0.0, 0.0], radii=0.005))
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
# ============================================
# =============== CHANGE START ===============
# ============================================
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
from typing import cast
from pathlib import Path
from os import environ
from itertools import groupby
from datetime import datetime, timezone
import re
import numpy as np

area_name = "area_4"
room_name = "conferenceRoom_3"
data_dir = Path(environ.get("DATASET_STANFORD_2D_3D_ROOT", "")) / area_name / "data" / "rgb"
image_paths = list(data_dir.glob(f"camera_*_{room_name}_*.png"))
image_paths.sort(key=lambda p: p.name)

samples_per_camera = 8
rng = np.random.default_rng(0)
image_paths_by_camera = [
    (cam, list(paths))
    for cam, paths
    in groupby(image_paths, key=lambda p: cast(re.Match[str], re.search(r"camera_([a-z0-9]+)_", p.name)).group(1))
]
image_paths_by_camera = [
    (
        cast(str, cam),
        cast(list[Path], list(
            rng.choice(
                np.array(paths),
                size=samples_per_camera,
                replace=False
            )
        ))
    )
    for cam, paths 
    in image_paths_by_camera
]
camera_count = len(image_paths_by_camera)

image_paths = [p for _, paths in image_paths_by_camera for p in paths]
# ============================================
# =============== CHANGE ENDED ===============
# ============================================
images = load_and_preprocess_images(image_paths).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        
# ============================================
# =============== CHANGE START ===============
# ============================================
        trans = predictions["pose_enc"][..., :3]
        quats = predictions["pose_enc"][..., 3:7]


def quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(quat.dtype).eps)
    x, y, z, w = torch.unbind(quat, dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m22 = 1.0 - 2.0 * (xx + yy)
    m01 = 2.0 * (xy - wz)
    m10 = 2.0 * (xy + wz)
    m02 = 2.0 * (xz + wy)
    m20 = 2.0 * (xz - wy)
    m12 = 2.0 * (yz - wx)
    m21 = 2.0 * (yz + wx)

    row0 = torch.stack((m00, m01, m02), dim=-1)
    row1 = torch.stack((m10, m11, m12), dim=-1)
    row2 = torch.stack((m20, m21, m22), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


trans = trans[0].to("cpu", torch.float32)
rots = quat_to_mat(quats)[0].to("cpu", torch.float32)

output_dir = Path(datetime.now(timezone.utc).strftime("outputs/%Y-%m-%dT%H:%M:%S"))
output_dir.mkdir(parents=True, exist_ok=True)

start_idx = 0
trans_by_camera = torch.empty((camera_count, samples_per_camera, 3), dtype=torch.float32)
rots_by_camera = torch.empty((camera_count, samples_per_camera, 3, 3), dtype=torch.float32)
image_data_by_camera: list[list[torch.Tensor]] = []
for i, (_, paths) in enumerate(image_paths_by_camera):
    idx = list(range(start_idx, start_idx + len(paths)))
    start_idx += len(paths)
    
    trans_by_camera[i, :samples_per_camera] = trans[idx]
    rots_by_camera[i, :samples_per_camera] = rots[idx]
    image_data_by_camera.append([images[j].to("cpu", torch.float32) for j in idx])


output_file = output_dir / f"NON-COMPLIANT-FORMAT.{area_name}.{room_name}.pt"

scale = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
poses = torch.empty([camera_count, 4, 4], dtype=torch.float32)
# NOTE: Setting identity rotation to avoid loading in poses and having to average them
poses[:, :3, :3] = torch.eye(3, dtype=torch.float32)
poses[:, :3, 3] = trans_by_camera.mean(dim=1)
poses[:, 3, :] = scale

poses_faces = torch.empty([camera_count, samples_per_camera, 4, 4], dtype=torch.float32)
poses_faces[:, :, :3, :3] = rots_by_camera
poses_faces[:, :, :3, 3] = trans_by_camera
poses_faces[:, :, 3, :] = scale

# Store as dictionary
torch.save(
    {
        "poses": poses,
        "poses_faces": poses_faces,
        "images_faces": image_data_by_camera,
    },
    output_file
)
# ============================================
# =============== CHANGE ENDED ===============
# ============================================

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch as th
from loguru import logger
from torchvision.io import write_png

from splat_init.data.datamodule_360 import SceneSampleLazy


def _write_bytes(fid: BinaryIO, data: float | int | bytes | list[float | int], fmt: str) -> None:
    if isinstance(data, list):
        fid.write(struct.pack(f"<{fmt}", *data))
    else:
        fid.write(struct.pack(f"<{fmt}", data))


def _rotmat2qvec(rot: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rot.flat
    k_mat = (
        np.array(
            [
                [rxx - ryy - rzz, 0, 0, 0],
                [ryx + rxy, ryy - rxx - rzz, 0, 0],
                [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
                [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
            ],
            dtype=np.float64,
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(k_mat)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def _rgba_to_rgb_uint8(rgba: th.Tensor) -> th.Tensor:
    assert rgba.shape[0] == 4, "Expected RGBA frame shaped [4,H,W]"
    rgb = rgba[:3] * rgba[3:4]
    return rgb.clamp(0.0, 1.0).mul(255.0).to(dtype=th.uint8)


def _load_rgba_frame(scene: SceneSampleLazy, frame_idx: int) -> th.Tensor:
    sample = scene[frame_idx]
    rgba = sample.rgba
    if rgba.dim() == 4:
        rgba = rgba[0]
    assert rgba.dim() == 3, "Expected RGBA frame shaped [4,H,W]"
    return rgba


def _validate_keypoints(
    keypoints: list[tuple[th.Tensor, th.Tensor]], sequence_length: int
) -> None:
    assert len(keypoints) == sequence_length, "Keypoints length mismatch"
    for xy, xyz in keypoints:
        assert xy.shape[0] == 2, "Keypoints xy must be [2,N]"
        assert xyz.shape[0] == 3, "Keypoints xyz must be [3,N]"
        assert xy.shape[1] == xyz.shape[1], "Keypoint xy/xyz count mismatch"


def _write_cameras_bin(path: Path, width: int, height: int) -> None:
    model_id = 12 # NOTE: Hardcoded custom camera model ID for equirectangular
    focal = 0.5 * (width - 1)
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    params = [focal, cx, cy]

    with open(path, "wb") as fid:
        _write_bytes(fid, 1, "Q")
        _write_bytes(fid, [1, model_id, width, height], "iiQQ")
        for param in params:
            _write_bytes(fid, float(param), "d")


def _frame_name(frame_idx: int) -> str:
    return f"frame_{frame_idx:06d}.png"


def export_colmap_scene(scene: SceneSampleLazy, eval_dir: Path, output_dir: Path) -> Path:
    """Export predicted poses/keypoints for a scene into COLMAP binary format."""
    scene_dir = output_dir / scene.id
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    poses_path = eval_dir / scene.id / "poses" / "poses.pt"
    keypoints_path = eval_dir / scene.id / "poses" / "keypoints.pt"
    assert poses_path.is_file(), f"Missing poses at {poses_path}"
    assert keypoints_path.is_file(), f"Missing keypoints at {keypoints_path}"

    poses = th.load(poses_path, map_location="cpu")
    keypoints = th.load(keypoints_path, map_location="cpu")

    assert isinstance(poses, th.Tensor), "Expected poses tensor"
    assert isinstance(keypoints, list), "Expected keypoints list"
    if poses.ndim == 4:
        assert poses.shape[0] == 1, "Unexpected batch dimension in poses"
        poses = poses[0]
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4), "Expected poses shaped [S,4,4]"

    sequence_length = poses.shape[0]
    assert len(scene) == sequence_length, "Scene length mismatch with poses"
    _validate_keypoints(keypoints, sequence_length)

    first_rgba = _load_rgba_frame(scene, 0)
    height, width = first_rgba.shape[1:]
    _write_cameras_bin(sparse_dir / "cameras.bin", width, height)

    points_per_image = [xy.shape[1] for xy, _ in keypoints]
    total_points = int(sum(points_per_image))
    logger.info(
        "Exporting scene {} with {} images and {} points",
        scene.id,
        sequence_length,
        total_points,
    )

    images_path = sparse_dir / "images.bin"
    points_path = sparse_dir / "points3D.bin"
    next_point_id = 1

    with open(images_path, "wb") as images_fid, open(points_path, "wb") as points_fid:
        _write_bytes(images_fid, sequence_length, "Q")
        _write_bytes(points_fid, total_points, "Q")

        for frame_idx in range(sequence_length):
            rgba = first_rgba if frame_idx == 0 else _load_rgba_frame(scene, frame_idx)
            rgb_uint8 = _rgba_to_rgb_uint8(rgba)
            image_name = _frame_name(frame_idx)
            write_png(rgb_uint8, str(images_dir / image_name))

            pose = poses[frame_idx].to(dtype=th.float64).numpy()
            rot = pose[:3, :3]
            tvec = pose[:3, 3]
            qvec = -_rotmat2qvec(rot)

            image_id = frame_idx + 1
            _write_bytes(images_fid, image_id, "i")
            _write_bytes(images_fid, qvec.tolist(), "dddd")
            _write_bytes(images_fid, tvec.tolist(), "ddd")
            _write_bytes(images_fid, 1, "i")
            images_fid.write(image_name.encode("utf-8") + b"\x00")

            xy, xyz = keypoints[frame_idx]
            xy = xy.to(dtype=th.float64)
            xyz = xyz.to(dtype=th.float64)
            num_points = int(xy.shape[1])
            _write_bytes(images_fid, num_points, "Q")

            if num_points == 0:
                continue

            x_idx = xy[0].round().to(dtype=th.long)
            y_idx = xy[1].round().to(dtype=th.long)
            assert (x_idx >= 0).all() and (x_idx < width).all(), "Keypoint x out of bounds"
            assert (y_idx >= 0).all() and (y_idx < height).all(), "Keypoint y out of bounds"

            colors = rgb_uint8[:, y_idx, x_idx].permute(1, 0).contiguous()
            for point2d_idx in range(num_points):
                point_id = next_point_id
                next_point_id += 1

                x = float(xy[0, point2d_idx].item())
                y = float(xy[1, point2d_idx].item())
                _write_bytes(images_fid, [x, y, point_id], "ddq")

                xyz_point = xyz[:, point2d_idx].tolist()
                rgb = colors[point2d_idx].tolist()
                _write_bytes(points_fid, point_id, "Q")
                _write_bytes(points_fid, xyz_point, "ddd")
                _write_bytes(points_fid, rgb, "BBB")
                _write_bytes(points_fid, 0.0, "d")
                _write_bytes(points_fid, 1, "Q")
                _write_bytes(points_fid, [image_id, point2d_idx], "ii")

    assert next_point_id == total_points + 1, "Point count mismatch"
    logger.info("Finished COLMAP export for {}", scene.id)
    return scene_dir


__all__ = ["export_colmap_scene"]

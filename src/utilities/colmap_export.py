from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch as th
from joblib import Parallel, delayed
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


def _write_cameras_bin(path: Path, width: int, height: int) -> None:
    model_id = 12  # NOTE: Hardcoded custom camera model ID for equirectangular
    focal = cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    params = [focal, cx, cy]

    with open(path, "wb") as fid:
        _write_bytes(fid, 1, "Q")
        _write_bytes(fid, [1, model_id, width, height], "iiQQ")
        for param in params:
            _write_bytes(fid, float(param), "d")


def export_colmap_scene(scene: SceneSampleLazy, eval_dir: Path, output_dir: Path) -> None:
    """Export predicted poses/keypoints for a scene into COLMAP binary format."""
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    poses = th.load(eval_dir / scene.id / "poses" / "poses.pt", map_location="cpu").numpy()
    keypoints = th.load(eval_dir / scene.id / "poses" / "keypoints.pt", map_location="cpu")

    sequence_length = poses.shape[0]
    assert len(scene) == sequence_length, "Scene length mismatch with poses"
    assert len(keypoints) == sequence_length, "Keypoints length mismatch"

    first_rgba = scene[0].rgba[0]
    assert first_rgba.shape[0] == 4, "Expected RGBA frame shaped [4,H,W]"
    height, width = first_rgba.shape[1:]
    _write_cameras_bin(sparse_dir / "cameras.bin", width, height)

    total_points = sum(xy.shape[1] for xy, _ in keypoints)
    logger.info(
        "Exporting scene {} with {} images and {} points",
        scene.id,
        sequence_length,
        total_points,
    )

    image_workers = min(4, os.cpu_count() or 1)
    chunk_size = image_workers * 4
    parallel = Parallel(n_jobs=image_workers, backend="threading")

    next_point_id = 1

    with open(sparse_dir / "images.bin", "wb") as images_fid, open(sparse_dir / "points3D.bin", "wb") as points_fid:
        _write_bytes(images_fid, sequence_length, "Q")
        _write_bytes(points_fid, total_points, "Q")

        for start in range(0, sequence_length, chunk_size):
            end = min(start + chunk_size, sequence_length)
            rgba_batch = scene[start:end].rgba
            rgb_uint8_batch = (
                (rgba_batch[:, :3] * rgba_batch[:, 3:4])
                .clamp(0.0, 1.0)
                .mul(255.0)
                .to(dtype=th.uint8)
                .cpu()
            )

            parallel(
                delayed(write_png)(
                    rgb_uint8_batch[idx],
                    str(images_dir / f"frame_{start + idx:06d}.png"),
                )
                for idx in range(rgb_uint8_batch.shape[0])
            )

            for batch_idx, frame_idx in enumerate(range(start, end)):
                rgb_uint8 = rgb_uint8_batch[batch_idx]
                image_name = f"frame_{frame_idx:06d}.png"

                pose = poses[frame_idx]
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
                num_points = xy.shape[1]
                _write_bytes(images_fid, num_points, "Q")

                if num_points == 0:
                    continue

                idx = xy.round().to(dtype=th.long)
                colors = rgb_uint8[:, idx[1], idx[0]].permute(1, 0)
                for point2d_idx in range(num_points):
                    point_id = next_point_id
                    next_point_id += 1

                    x, y = xy[:, point2d_idx].tolist()
                    _write_bytes(images_fid, [x, y, point_id], "ddq")

                    _write_bytes(points_fid, point_id, "Q")
                    _write_bytes(points_fid, xyz[:, point2d_idx].tolist(), "ddd")
                    _write_bytes(points_fid, colors[point2d_idx].tolist(), "BBB")
                    _write_bytes(points_fid, 0.0, "d")
                    _write_bytes(points_fid, 1, "Q")
                    _write_bytes(points_fid, [image_id, point2d_idx], "ii")

    assert next_point_id == total_points + 1, "Point count mismatch"
    logger.info("Finished COLMAP export for {}", scene.id)


__all__ = ["export_colmap_scene"]

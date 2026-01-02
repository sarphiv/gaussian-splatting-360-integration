"""DA3 perspective-transform panorama wrapper for pose estimation from 360 panoramas."""

from __future__ import annotations

import copy
import gc
import os
import tempfile
from pathlib import Path
from typing import Any

from lightning.pytorch import LightningModule
import torch as th
from torchvision.io import write_png
from depth_anything_3.da3_streaming import DA3_Streaming

from utilities.da3_assets import (
    DA3StreamingAssets,
    ensure_da3_streaming_assets,
    load_da3_streaming_config,
)
from utilities.otc_projector import OTCProjector, cube_face_relative_rotations
from utilities.pose import mat_to_quat_xyzw, mean_quaternion_markley, pose_from_center_and_rotation, quat_to_mat_xyzw

PATCH_SIZE = 14
FACE_ORDER = ("+X", "-X", "+Z", "-Z")
FACE_INDICES = (0, 1, 4, 5)


class Da3PerspectiveTransform(LightningModule):
    """Project panoramas to faces, run DA3 per face, and merge poses."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        face_size: int = 504,
        temporary_storage_in_ram: bool = True,
    ) -> None:
        """Initialize the DA3 wrapper and projection settings."""
        super().__init__()
        self.save_hyperparameters()

        assert face_size > 0, "Face size must be positive."

        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "depth_anything_3.yaml"
        self.config_path: Path = Path(config_path)
        self.face_size: int = max(PATCH_SIZE, int(round(face_size / PATCH_SIZE) * PATCH_SIZE))

        self._projector: OTCProjector = OTCProjector(face_size=self.face_size)

        self._face_rots: th.Tensor
        self.register_buffer(
            "_face_rots",
            cube_face_relative_rotations()[[0, 1, 4, 5]],
            persistent=False,
        )

        self.temporary_storage_in_ram: bool = temporary_storage_in_ram
        self._assets: DA3StreamingAssets | None = None

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    def _temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        """Create a temporary directory, preferring RAM-backed storage when available."""
        if not self.temporary_storage_in_ram:
            return tempfile.TemporaryDirectory()
        shm = Path("/dev/shm")
        assert shm.is_dir() and os.access(shm, os.W_OK), "RAM-backed temp dir unavailable"
        return tempfile.TemporaryDirectory(dir=shm)

    # ------------------------------------------------------------------
    # DA3-Streaming runner
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_memory() -> None:
        """Clear cached GPU memory and collect garbage to reduce peak usage."""
        if th.cuda.is_available():
            th.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _write_image_sequence(images: th.Tensor, output_dir: Path) -> None:
        """Write a sequence of RGB images to disk as zero-padded PNGs."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(images.shape[0]):
            frame = images[idx].detach().cpu().clamp(0.0, 1.0).mul(255.0).to(dtype=th.uint8)
            filename = output_dir / f"frame_{idx:06d}.png"
            write_png(frame, str(filename))

    @staticmethod
    def _load_camera_poses(path: Path, expected_len: int) -> th.Tensor:
        """Load flattened 4x4 camera poses from a DA3-Streaming output file."""
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        pose_values = [list(map(float, line.split())) for line in lines]
        poses = th.tensor(pose_values, dtype=th.float32)
        if poses.ndim == 1:
            poses = poses.unsqueeze(0)
        assert poses.shape[0] == expected_len, "Pose count does not match frame count."
        assert poses.shape[1] == 16, "Expected flattened 4x4 pose matrices."
        return poses.view(-1, 4, 4)

    def _run_da3(self, images: th.Tensor, config: dict[str, Any]) -> th.Tensor:
        """Run DA3 on a face sequence and return w2c poses."""
        num_frames = images.shape[0]
        with self._temporary_directory() as image_dir, tempfile.TemporaryDirectory() as output_dir:
            self._write_image_sequence(images, Path(image_dir))
            del images
            self._clean_memory()

            config_local = copy.deepcopy(config)

            runner = DA3_Streaming(image_dir, output_dir, config_local)
            runner.run()
            runner.close()

            pose_path = Path(output_dir) / "camera_poses.txt"
            c2w = self._load_camera_poses(pose_path, num_frames)

        del runner
        self._clean_memory()

        w2c = th.linalg.inv(c2w)
        return w2c

    # ------------------------------------------------------------------
    # Projection + merge
    # ------------------------------------------------------------------

    def _project_face(self, images: th.Tensor, face_idx: int) -> th.Tensor:
        """Project panoramas into a single perspective face."""
        assert images.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        batch, seq_len, channels, height, width = images.shape
        assert batch == 1, "Batch size > 1 not supported"
        assert channels == 4, "Expected RGBA input shaped [B, S, 4, H, W]"
        assert 0 <= face_idx < len(FACE_INDICES), "Unexpected face index"

        images = images.to(device=self.device, dtype=th.float32)
        rgb = images[:, :, :3]
        alpha = images[:, :, 3:4]
        rgba = th.cat((rgb * alpha, alpha), dim=2)

        flat_rgba = rgba.reshape(batch * seq_len, 4, height, width)
        rgb_faces, alpha_faces, _ = self._projector(flat_rgba, depth=None)
        rgb_faces = rgb_faces * alpha_faces
        face = rgb_faces[:, FACE_INDICES[face_idx]].clone()
        face_size = face.shape[-1]
        face = face.reshape(seq_len, 3, face_size, face_size)
        return face

    def _merge_face_poses(self, w2c_faces: th.Tensor) -> th.Tensor:
        """Merge per-face world-to-camera poses into a single panorama pose."""
        device, dtype = w2c_faces.device, w2c_faces.dtype
        face_rot = self._face_rots.to(device=device, dtype=dtype)

        face_rot_mats = th.eye(4, device=device, dtype=dtype).repeat(face_rot.shape[0], 1, 1)
        face_rot_mats[:, :3, :3] = face_rot

        w2c_faces = face_rot_mats.unsqueeze(0) @ w2c_faces

        rotations = w2c_faces[:, :, :3, :3]
        translations = w2c_faces[:, :, :3, 3]

        centers = -(rotations.transpose(-1, -2) @ translations.unsqueeze(-1)).squeeze(-1)

        centers_merged = centers.mean(dim=1)

        quats = mat_to_quat_xyzw(rotations)
        merged_quat = mean_quaternion_markley(quats)
        merged_rot = quat_to_mat_xyzw(merged_quat)

        ref_rot = merged_rot[:1]
        rel_rot = merged_rot @ ref_rot.transpose(-1, -2)
        rel_centers = centers_merged - centers_merged[:1]
        return pose_from_center_and_rotation(rel_centers, rel_rot)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, images: th.Tensor) -> tuple[th.Tensor, None, dict[str, th.Tensor]]:
        """Estimate panorama poses by projecting faces and fusing DA3 outputs."""
        if self._assets is None:
            self._assets = ensure_da3_streaming_assets()
        config = load_da3_streaming_config(self.config_path, self._assets)

        face_poses = []
        for idx in range(len(FACE_ORDER)):
            face_pose = self._run_da3(self._project_face(images, idx), config)
            face_poses.append(face_pose)

        w2c_faces = th.stack(face_poses, dim=1).to(device=self.device, dtype=th.float32)
        merged = self._merge_face_poses(w2c_faces)

        return merged.unsqueeze(0), None, {
            "poses_faces_w2c": w2c_faces.unsqueeze(0),
        }

    def configure_optimizers(self):
        """Lightning hook for compatibility; DA3-Streaming is inference-only."""
        return []

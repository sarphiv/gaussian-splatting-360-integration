"""DA3 perspective-transform panorama wrapper for pose estimation from 360 panoramas."""

from __future__ import annotations

import copy
import gc
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from lightning.pytorch import LightningModule
from joblib import Parallel, delayed
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
        projection_batch_size: int = 32,
        image_workers: int = min(4, os.cpu_count() or 1),
        images_per_worker: int = 4,
    ) -> None:
        """Initialize the DA3 wrapper and projection settings."""
        super().__init__()
        self.save_hyperparameters()

        assert face_size > 0, "Face size must be positive."
        assert image_workers > 0, "n_workers must be positive."
        assert projection_batch_size > 0, "projection_batch_size must be positive."

        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "depth_anything_3.yaml"
        self.config_path: Path = Path(config_path)
        self.face_size: int = max(PATCH_SIZE, int(round(face_size / PATCH_SIZE) * PATCH_SIZE))

        self._projector: OTCProjector = OTCProjector(face_size=self.face_size, alpha=1e-9)

        self._face_rots: th.Tensor
        self.register_buffer(
            "_face_rots",
            cube_face_relative_rotations()[[0, 1, 4, 5]],
            persistent=False,
        )

        self.temporary_storage_in_ram: bool = temporary_storage_in_ram
        self.projection_batch_size: int = projection_batch_size
        self.image_workers = image_workers
        self.images_per_worker = images_per_worker
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
    def _write_png_frame(frame: th.Tensor, filename: Path) -> None:
        write_png(frame, str(filename))

    def _write_image_sequence(self, images: th.Tensor, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        num_frames = images.shape[0]
        chunk_size = self.image_workers * self.images_per_worker

        parallel = Parallel(n_jobs=self.image_workers, backend="threading")
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            chunk = images[start:end].detach().clamp(0.0, 1.0).mul(255.0).to(device=th.device("cpu"), dtype=th.uint8)
            parallel(
                delayed(self._write_png_frame)(chunk[idx], output_dir / f"frame_{start + idx:06d}.png")
                for idx in range(chunk.shape[0])
            )

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

            self._clean_memory()
            runner = DA3_Streaming(image_dir, output_dir, copy.deepcopy(config))
            runner.run()
            runner.close()

            pose_path = Path(output_dir) / "camera_poses.txt"
            c2w = self._load_camera_poses(pose_path, num_frames)

        del runner
        self._clean_memory()

        w2c = th.linalg.inv(c2w)
        return w2c.to(images)

    # ------------------------------------------------------------------
    # Projection + merge
    # ------------------------------------------------------------------

    def _project_faces(self, images: th.Tensor) -> th.Tensor:
        """Project panoramas into the four perspective faces used for DA3."""
        assert images.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        batch, seq_len, channels, height, width = images.shape
        assert batch == 1, "Batch size > 1 not supported"
        assert channels == 4, "Expected RGBA input shaped [B, S, 4, H, W]"

        flat_rgba = images.reshape(batch * seq_len, 4, height, width)
        proj_batch = self.projection_batch_size
        face_chunks: list[th.Tensor] = []
        for start in range(0, flat_rgba.shape[0], proj_batch):
            end = min(start + proj_batch, flat_rgba.shape[0])
            batch = flat_rgba[start:end].to(device=self.device, dtype=cast(th.dtype, self.dtype))
            rgb_faces, alpha_faces, _ = self._projector(batch, depth=None)
            face_chunks.append((rgb_faces * alpha_faces)[:, FACE_INDICES].to(images))

        face_stack = th.cat(face_chunks, dim=0)
        face_size = face_stack.shape[-1]
        faces = face_stack.reshape(seq_len, len(FACE_INDICES), 3, face_size, face_size)
        return faces

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
        rel_centers = (ref_rot @ rel_centers.unsqueeze(-1)).squeeze(-1)
        return pose_from_center_and_rotation(rel_centers, rel_rot)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, images: th.Tensor) -> tuple[th.Tensor, None, dict[str, th.Tensor]]:
        """Estimate panorama poses by projecting faces and fusing DA3 outputs."""
        if self._assets is None:
            self._assets = ensure_da3_streaming_assets()
        config = load_da3_streaming_config(self.config_path, self._assets)

        faces = self._project_faces(images)
        face_poses = [
            self._run_da3(faces[:, i, ...], config)
            for i in range(len(FACE_ORDER))
        ]

        w2c_faces = th.stack(face_poses, dim=1)
        merged = self._merge_face_poses(w2c_faces)

        return merged.unsqueeze(0).to(images), None, {
            "pose_faces": w2c_faces.unsqueeze(0).to(images)
        }

    def configure_optimizers(self):
        """Lightning hook for compatibility; DA3-Streaming is inference-only."""
        return []

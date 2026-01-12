"""pycolmap wrapper that projects panoramas to cube faces and merges pose estimates."""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import cast

from joblib import Parallel, delayed
from lightning.pytorch import LightningModule
from loguru import logger
import pycolmap
import torch as th
from torchvision.io import write_png

from utilities.otc_projector import OTCProjector, cube_face_relative_rotations
from utilities.pose import mat_to_quat_xyzw, mean_quaternion_markley, pose_from_center_and_rotation, quat_to_mat_xyzw


FACE_INDICES = (0, 1, 4, 5)


class PycolmapPerspectiveTransform(LightningModule):
    """Project panoramas to perspective faces, run pycolmap, and merge poses."""

    def __init__(
        self,
        *,
        face_size: int = 504,
        temporary_storage_in_ram: bool = True,
        projection_batch_size: int = 32,
        image_workers: int = min(4, os.cpu_count() or 1),
        images_per_worker: int = 4,
        use_gpu: bool = True,
        gpu_index: str = "0",
    ) -> None:
        """Configure projection, COLMAP matching, and temporary storage settings."""
        super().__init__()
        self.save_hyperparameters()

        assert face_size > 0, "face_size must be positive."
        assert projection_batch_size > 0, "projection_batch_size must be positive."
        assert image_workers > 0, "image_workers must be positive."
        assert images_per_worker > 0, "images_per_worker must be positive."
        self.face_size = int(face_size)
        self._projector = OTCProjector(face_size=self.face_size, alpha=1e-9)

        self.temporary_storage_in_ram = temporary_storage_in_ram
        self.projection_batch_size = projection_batch_size
        self.image_workers = image_workers
        self.images_per_worker = images_per_worker
        self.use_gpu = use_gpu
        self.gpu_index = str(gpu_index)

        self.face_indices = FACE_INDICES
        self._face_to_index = {face_idx: idx for idx, face_idx in enumerate(self.face_indices)}

        self._face_rots: th.Tensor
        self.register_buffer(
            "_face_rots",
            cube_face_relative_rotations()[list(self.face_indices)],
            persistent=False,
        )

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
    # Projection + IO
    # ------------------------------------------------------------------

    def _project_faces(self, rgba: th.Tensor) -> th.Tensor:
        """Project panoramas into the four perspective faces used for COLMAP."""
        assert rgba.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        batch, seq_len, channels, height, width = rgba.shape
        assert batch == 1, "Batch size > 1 not supported"
        assert channels == 4, "Expected RGBA input shaped [B, S, 4, H, W]"

        flat_rgba = rgba.reshape(batch * seq_len, 4, height, width)
        proj_batch = self.projection_batch_size
        face_chunks: list[th.Tensor] = []
        for start in range(0, flat_rgba.shape[0], proj_batch):
            end = min(start + proj_batch, flat_rgba.shape[0])
            chunk = flat_rgba[start:end].to(device=self.device, dtype=cast(th.dtype, self.dtype))
            rgb_faces, alpha_faces, _ = self._projector(chunk, depth=None)
            face_chunks.append((rgb_faces * alpha_faces)[:, self.face_indices].to(rgba))

        face_stack = th.cat(face_chunks, dim=0)
        face_size = face_stack.shape[-1]
        return face_stack.reshape(seq_len, len(self.face_indices), 3, face_size, face_size)

    @staticmethod
    def _write_png_frame(frame: th.Tensor, filename: Path) -> None:
        """Write a single RGB frame to disk as PNG."""
        write_png(frame, str(filename))

    def _write_face_images(self, faces: th.Tensor, output_dir: Path) -> list[str]:
        """Write cubemap face images to disk and return relative image names."""
        output_dir.mkdir(parents=True, exist_ok=True)
        num_frames = faces.shape[0]
        chunk_size = self.image_workers * self.images_per_worker
        parallel = Parallel(n_jobs=self.image_workers, backend="threading")

        image_names: list[str] = []
        for face_slot, face_idx in enumerate(self.face_indices):
            face_dir = output_dir / f"face_{face_idx}"
            face_dir.mkdir(parents=True, exist_ok=True)

            for start in range(0, num_frames, chunk_size):
                end = min(start + chunk_size, num_frames)
                chunk = (
                    faces[start:end, face_slot]
                    .detach()
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .to(device=th.device("cpu"), dtype=th.uint8)
                    .contiguous()
                )
                parallel(
                    delayed(self._write_png_frame)(chunk[idx], face_dir / f"frame_{start + idx:06d}.png")
                    for idx in range(chunk.shape[0])
                )

            image_names.extend(
                str(Path(f"face_{face_idx}") / f"frame_{frame_idx:06d}.png")
                for frame_idx in range(num_frames)
            )

        return image_names

    # ------------------------------------------------------------------
    # pycolmap helpers
    # ------------------------------------------------------------------

    def _camera_params(self) -> tuple[float, float, float, float]:
        """Compute pinhole intrinsics for cube-map faces (90° FOV)."""
        fx = 0.5 * (self.face_size - 1)
        fy = fx
        cx = 0.5 * (self.face_size - 1)
        cy = 0.5 * (self.face_size - 1)
        return fx, fy, cx, cy

    def _camera_params_str(self) -> str:
        """Return COLMAP camera params string for the projected faces."""
        fx, fy, cx, cy = self._camera_params()
        return f"{fx},{fy},{cx},{cy}"

    def _build_reader_options(self) -> pycolmap.ImageReaderOptions:
        """Create reader options with fixed intrinsics for projected faces."""
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "PINHOLE"
        reader_options.camera_params = self._camera_params_str()
        return reader_options

    def _build_extraction_options(self) -> pycolmap.FeatureExtractionOptions:
        """Create feature extraction options for pycolmap."""
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.use_gpu = self.use_gpu
        extraction_options.gpu_index = self.gpu_index
        if self.use_gpu:
            extraction_options.num_threads = 1
        return extraction_options

    def _build_matching_options(self) -> pycolmap.FeatureMatchingOptions:
        """Create feature matching options for pycolmap."""
        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.use_gpu = self.use_gpu
        matching_options.gpu_index = self.gpu_index
        return matching_options

    def _colmap_device(self) -> pycolmap.Device:
        """Return the pycolmap device enum for the requested backend."""
        return pycolmap.Device.cuda if self.use_gpu else pycolmap.Device.cpu

    def _run_colmap(self, image_dir: Path, image_names: list[str]) -> pycolmap.Reconstruction | None:
        """Run COLMAP feature extraction, matching, and incremental mapping."""
        if not image_names:
            return None

        reader_options = self._build_reader_options()
        extraction_options = self._build_extraction_options()
        matching_options = self._build_matching_options()
        verification_options = pycolmap.TwoViewGeometryOptions()
        device = self._colmap_device()

        if self.use_gpu and (not extraction_options.check() or not matching_options.check()):
            build_info = str(pycolmap.COLMAP_build)
            raise RuntimeError(
                "pycolmap GPU SIFT is unavailable. "
                f"COLMAP build: {build_info}. "
                "Install a CUDA-enabled pycolmap build (e.g., ensure the pycolmap-cuda12 wheel "
                "overwrites any CPU-only pycolmap install) and verify CUDA/OpenGL runtime support."
            )

        with tempfile.TemporaryDirectory() as colmap_dir:
            colmap_path = Path(colmap_dir)
            database_path = colmap_path / "database.db"
            output_path = colmap_path / "sparse"
            output_path.mkdir(parents=True, exist_ok=True)

            start = time.perf_counter()
            logger.info("Extracting features for {} images", len(image_names))
            pycolmap.extract_features(
                str(database_path),
                str(image_dir),
                image_names=image_names,
                camera_mode=pycolmap.CameraMode.SINGLE,
                camera_model="RADIAL",
                reader_options=reader_options,
                extraction_options=extraction_options,
                device=device,
            )
            logger.info("Feature extraction finished in {:.2f}s", time.perf_counter() - start)

            logger.info("Matching exhaustively (COLMAP default CLI matcher)")
            pycolmap.match_exhaustive(
                str(database_path),
                matching_options=matching_options,
                pairing_options=pycolmap.ExhaustivePairingOptions(),
                verification_options=verification_options,
                device=device,
            )

            logger.info("Running incremental mapping")
            options = pycolmap.IncrementalPipelineOptions()
            options.image_names = list(image_names)
            reconstructions = pycolmap.incremental_mapping(
                str(database_path),
                str(image_dir),
                str(output_path),
                options=options,
            )

        if not reconstructions:
            logger.warning("No reconstructions returned by COLMAP.")
            return None

        best_id, best_reconstruction = max(
            reconstructions.items(),
            key=lambda item: item[1].num_points3D(),
        )
        logger.info(
            "COLMAP produced {} reconstructions; keeping model {} with {} points and {} images",
            len(reconstructions),
            best_id,
            best_reconstruction.num_points3D(),
            best_reconstruction.num_images(),
        )
        return best_reconstruction

    def _parse_image_name(self, name: str) -> tuple[int, int]:
        """Parse a COLMAP image name into (face_idx, frame_idx)."""
        path = Path(name)
        face_token = path.parent.name
        frame_token = path.stem
        assert face_token.startswith("face_"), f"Unexpected face token: {face_token}"
        assert frame_token.startswith("frame_"), f"Unexpected frame token: {frame_token}"
        face_idx = int(face_token.split("_", 1)[1])
        frame_idx = int(frame_token.split("_", 1)[1])
        return face_idx, frame_idx

    def _poses_from_reconstruction(
        self,
        reconstruction: pycolmap.Reconstruction,
        seq_len: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Extract per-face w2c poses and a validity mask from a reconstruction."""
        num_faces = len(self.face_indices)
        w2c_faces = th.eye(4, dtype=th.float32).repeat(seq_len, num_faces, 1, 1)
        valid_mask = th.zeros((seq_len, num_faces), dtype=th.bool)

        for image in reconstruction.images.values():
            if not image.has_pose:
                continue
            face_idx, frame_idx = self._parse_image_name(image.name)
            if face_idx not in self._face_to_index:
                continue
            assert 0 <= frame_idx < seq_len, f"Frame index {frame_idx} out of range"
            face_slot = self._face_to_index[face_idx]

            w2c_3x4 = th.from_numpy(image.cam_from_world().matrix()).to(dtype=th.float32)
            w2c = th.eye(4, dtype=th.float32)
            w2c[:3, :4] = w2c_3x4

            w2c_faces[frame_idx, face_slot] = w2c
            valid_mask[frame_idx, face_slot] = True

        return w2c_faces, valid_mask

    # ------------------------------------------------------------------
    # Pose merging
    # ------------------------------------------------------------------

    def _merge_face_poses(self, w2c_faces: th.Tensor, valid_mask: th.Tensor) -> th.Tensor:
        """Merge per-face world-to-camera poses into panorama poses."""
        device, dtype = w2c_faces.device, w2c_faces.dtype
        face_rot = self._face_rots.to(device=device, dtype=dtype)

        face_rot_mats = th.eye(4, device=device, dtype=dtype).repeat(face_rot.shape[0], 1, 1)
        face_rot_mats[:, :3, :3] = face_rot

        w2c_faces = face_rot_mats.unsqueeze(0) @ w2c_faces

        rotations = w2c_faces[:, :, :3, :3]
        translations = w2c_faces[:, :, :3, 3]

        centers = -(rotations.transpose(-1, -2) @ translations.unsqueeze(-1)).squeeze(-1)
        weights = valid_mask.to(dtype)
        counts = weights.sum(dim=1)

        if th.count_nonzero(counts) == 0:
            return th.eye(4, device=device, dtype=dtype).repeat(w2c_faces.shape[0], 1, 1)

        missing = counts == 0
        if missing.any():
            weights = weights.clone()
            weights[missing, 0] = 1.0

        weights = weights / weights.sum(dim=1, keepdim=True)
        centers_merged = (centers * weights.unsqueeze(-1)).sum(dim=1)

        quats = mat_to_quat_xyzw(rotations)
        merged_quat = mean_quaternion_markley(quats, weights=weights)
        merged_rot = quat_to_mat_xyzw(merged_quat)

        ref_idx = int(th.nonzero(counts > 0, as_tuple=False)[0].item())
        ref_rot = merged_rot[ref_idx : ref_idx + 1]
        rel_rot = merged_rot @ ref_rot.transpose(-1, -2)
        rel_centers = centers_merged - centers_merged[ref_idx : ref_idx + 1]
        rel_centers = (ref_rot @ rel_centers.unsqueeze(-1)).squeeze(-1)
        merged = pose_from_center_and_rotation(rel_centers, rel_rot)

        if missing.any():
            merged[missing] = th.eye(4, device=device, dtype=dtype)

        return merged

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, images: th.Tensor) -> tuple[th.Tensor, None, dict[str, th.Tensor]]:
        """Estimate panorama poses by projecting faces and running COLMAP."""
        assert images.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        batch, seq_len, channels, height, width = images.shape
        assert batch == 1, "Batch size > 1 not supported"
        assert channels in (3, 4), "Expected RGB or RGBA inputs"

        img = images.to(device=self.device, dtype=cast(th.dtype, self.dtype))
        if channels == 3:
            alpha = th.ones((batch, seq_len, 1, height, width), device=img.device, dtype=img.dtype)
            rgba = th.cat((img, alpha), dim=2)
        else:
            rgba = img

        faces = self._project_faces(rgba)

        with self._temporary_directory() as image_dir:
            image_dir_path = Path(image_dir)
            image_names = self._write_face_images(faces, image_dir_path)
            reconstruction = self._run_colmap(image_dir_path, image_names)

        if reconstruction is None:
            identity = th.eye(4, device=img.device, dtype=img.dtype).repeat(seq_len, 1, 1)
            w2c_faces = identity.unsqueeze(1).repeat(1, len(self.face_indices), 1, 1)
            valid_mask = th.zeros((seq_len, len(self.face_indices)), device=img.device, dtype=th.bool)
            merged = identity
        else:
            w2c_faces, valid_mask = self._poses_from_reconstruction(reconstruction, seq_len)
            w2c_faces = w2c_faces.to(device=img.device, dtype=cast(th.dtype, self.dtype))
            valid_mask = valid_mask.to(device=img.device)
            merged = self._merge_face_poses(w2c_faces, valid_mask)

        return merged.unsqueeze(0).to(images), None, {
            "pose_faces": w2c_faces.unsqueeze(0).to(images)
        }

    def configure_optimizers(self):
        """Lightning hook for compatibility; pycolmap runs inference-only."""
        return []

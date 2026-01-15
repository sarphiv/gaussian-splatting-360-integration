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
import numpy as np
import pycolmap
import torch as th
from torchvision.io import write_png

from utilities.cube_projector import CubeProjector
from utilities.otc_projector import cube_face_relative_rotations
from utilities.pose import mat_to_quat_xyzw, mean_quaternion_markley, pose_from_center_and_rotation, quat_to_mat_xyzw


FACE_ORDER = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
FACE_INDICES = (0, 1, 4, 5)
FACE_TO_INDEX = {face_idx: idx for idx, face_idx in enumerate(FACE_INDICES)}


class PycolmapPerspectiveTransform(LightningModule):
    """Project panoramas to perspective faces, run pycolmap, and merge poses."""

    def __init__(
        self,
        *,
        face_size: int = 512,
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
        self._projector = CubeProjector(face_size=self.face_size)

        self.temporary_storage_in_ram = temporary_storage_in_ram
        self.projection_batch_size = projection_batch_size
        self.image_workers = image_workers
        self.images_per_worker = images_per_worker
        self.use_gpu = use_gpu
        self.gpu_index = str(gpu_index)

        self._face_rots: th.Tensor
        self.register_buffer(
            "_face_rots",
            cube_face_relative_rotations()[list(FACE_INDICES)],
            persistent=False,
        )

        fx = 0.5 * (self.face_size - 1)
        cx = 0.5 * (self.face_size - 1)
        self._camera_params_str = f"{fx},{fx},{cx},{cx}"

        self._reader_options = pycolmap.ImageReaderOptions()
        self._reader_options.camera_model = "PINHOLE"
        self._reader_options.camera_params = self._camera_params_str

        self._extraction_options = pycolmap.FeatureExtractionOptions()
        self._extraction_options.use_gpu = self.use_gpu
        self._extraction_options.gpu_index = self.gpu_index
        if self.use_gpu:
            self._extraction_options.num_threads = 1

        self._matching_options = pycolmap.FeatureMatchingOptions()
        self._matching_options.use_gpu = self.use_gpu
        self._matching_options.gpu_index = self.gpu_index

        self._colmap_device = pycolmap.Device.cuda if self.use_gpu else pycolmap.Device.cpu

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

    def _project_faces(self, images: th.Tensor) -> th.Tensor:
        """Project panoramas into the four perspective faces used for COLMAP."""
        assert images.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        batch, seq_len, channels, height, width = images.shape
        assert batch == 1, "Batch size > 1 not supported"
        assert channels == 4, "Expected RGBA input shaped [B, S, 4, H, W]"

        flat_rgba = images.reshape(batch * seq_len, 4, height, width)
        proj_batch = self.projection_batch_size
        face_chunks: list[th.Tensor] = []
        for start in range(0, flat_rgba.shape[0], proj_batch):
            end = min(start + proj_batch, flat_rgba.shape[0])
            chunk = flat_rgba[start:end].to(device=self.device, dtype=cast(th.dtype, self.dtype))
            rgb_faces, alpha_faces, _ = self._projector(chunk, depth=None)
            face_chunks.append((rgb_faces * alpha_faces)[:, FACE_INDICES].to(images))

        face_stack = th.cat(face_chunks, dim=0)
        face_size = face_stack.shape[-1]
        return face_stack.reshape(seq_len, len(FACE_INDICES), 3, face_size, face_size)

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
        for face_slot, face_idx in enumerate(FACE_INDICES):
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

    def _run_colmap(self, image_dir: Path, image_names: list[str]) -> pycolmap.Reconstruction:
        """Run COLMAP feature extraction, matching, and incremental mapping."""
        assert image_names, "No images to run COLMAP on."

        verification_options = pycolmap.TwoViewGeometryOptions()

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
                camera_model="PINHOLE",
                reader_options=self._reader_options,
                extraction_options=self._extraction_options,
                device=self._colmap_device,
            )
            logger.info("Feature extraction finished in {:.2f}s", time.perf_counter() - start)

            logger.info("Matching exhaustively (COLMAP default CLI matcher)")
            pycolmap.match_exhaustive(
                str(database_path),
                matching_options=self._matching_options,
                pairing_options=pycolmap.ExhaustivePairingOptions(),
                verification_options=verification_options,
                device=self._colmap_device,
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
                raise RuntimeError("COLMAP produced no reconstructions.")

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

            dense_path = colmap_path / "dense"
            dense_path.mkdir(parents=True, exist_ok=True)
            logger.info("Undistorting images for COLMAP model {}", best_id)
            pycolmap.undistort_images(
                str(dense_path),
                str(output_path / str(best_id)),
                str(image_dir),
                image_names=image_names,
                output_type="COLMAP",
            )

            model_path = dense_path / "sparse"

            assert model_path.is_dir(), f"Missing undistorted model at {model_path}"
            reconstruction = pycolmap.Reconstruction()
            reconstruction.read_binary(str(model_path))
            return reconstruction

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

    def _poses_from_model(
        self,
        reconstruction: pycolmap.Reconstruction,
        seq_len: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Extract per-face w2c poses and a validity mask from a COLMAP model."""
        num_faces = len(FACE_INDICES)
        w2c_faces = th.eye(4, dtype=th.float32).repeat(seq_len, num_faces, 1, 1)
        valid_mask = th.zeros((seq_len, num_faces), dtype=th.bool)

        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.images[image_id]
            assert image.has_pose, f"Image {image_id} missing pose."
            face_idx, frame_idx = self._parse_image_name(image.name)
            assert face_idx in FACE_TO_INDEX, f"Unexpected face index: {face_idx}"
            assert 0 <= frame_idx < seq_len, f"Frame index {frame_idx} out of range"
            face_slot = FACE_TO_INDEX[face_idx]

            cam_from_world = image.cam_from_world()
            w2c = th.eye(4, dtype=th.float32)
            w2c[:3, :4] = th.from_numpy(cam_from_world.matrix()).to(dtype=th.float32)

            w2c_faces[frame_idx, face_slot] = w2c
            valid_mask[frame_idx, face_slot] = True

        return w2c_faces, valid_mask

    def _map_face_to_equirect(
        self,
        face_idx: int,
        xys: th.Tensor,
        output_size: tuple[int, int],
    ) -> th.Tensor:
        """Map face pixel coordinates to equirectangular pixel coordinates."""
        face_size = self.face_size
        out_height, out_width = output_size
        assert face_size > 1, "Face images must be at least 2x2"
        assert out_width > 1 and out_height > 1, "Panorama images must be at least 2x2"

        u_lin = 2.0 * xys[:, 0] / (face_size - 1.0) - 1.0
        v_lin = 2.0 * xys[:, 1] / (face_size - 1.0) - 1.0
        direction = self._projector._dir_for_face(u_lin, v_lin, FACE_ORDER[face_idx])

        lon = th.atan2(direction[0], direction[2])
        lat = th.atan2(direction[1], th.sqrt(direction[0] * direction[0] + direction[2] * direction[2]))

        x = (lon / math.pi + 1.0) * 0.5 * (out_width - 1.0)
        y = (-2.0 * lat / math.pi + 1.0) * 0.5 * (out_height - 1.0)
        return th.stack((x, y), dim=1)

    def _keypoints_from_model(
        self,
        reconstruction: pycolmap.Reconstruction,
        seq_len: int,
        output_size: tuple[int, int],
        align: tuple[th.Tensor, th.Tensor],
    ) -> list[tuple[th.Tensor, th.Tensor]]:
        """Extract equirectangular keypoints and world-space points from COLMAP."""
        xy_accum: list[list[th.Tensor]] = [[] for _ in range(seq_len)]
        xyz_accum: list[list[th.Tensor]] = [[] for _ in range(seq_len)]

        ref_rot, ref_center = align
        ref_rot = ref_rot.to(device=th.device("cpu"), dtype=th.float32)
        ref_center = ref_center.to(device=th.device("cpu"), dtype=th.float32)

        points3d = reconstruction.points3D

        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.images[image_id]
            face_idx, frame_idx = self._parse_image_name(image.name)
            assert face_idx in FACE_TO_INDEX, f"Unexpected face index: {face_idx}"
            assert 0 <= frame_idx < seq_len, f"Frame index {frame_idx} out of range"

            points2d = [point for point in image.points2D if point.has_point3D()]
            if not points2d:
                continue

            point_ids = [int(point.point3D_id) for point in points2d]
            xys = th.from_numpy(np.stack([point.xy for point in points2d], axis=0)).to(dtype=th.float32)
            eq_xy = self._map_face_to_equirect(
                face_idx,
                xys,
                output_size,
            ).transpose(0, 1)

            xyz = np.stack([points3d[point_id].xyz for point_id in point_ids], axis=0)
            xyz = th.from_numpy(xyz).to(dtype=th.float32).transpose(0, 1)
            xyz = ref_rot @ (xyz - ref_center[:, None])

            xy_accum[frame_idx].append(eq_xy)
            xyz_accum[frame_idx].append(xyz)

        results: list[tuple[th.Tensor, th.Tensor]] = []
        for frame_idx in range(seq_len):
            if xy_accum[frame_idx]:
                xy = th.cat(xy_accum[frame_idx], dim=1)
                xyz = th.cat(xyz_accum[frame_idx], dim=1)
            else:
                xy = th.empty((2, 0), dtype=th.float32)
                xyz = th.empty((3, 0), dtype=th.float32)
            results.append((xy, xyz))

        return results

    # ------------------------------------------------------------------
    # Pose merging
    # ------------------------------------------------------------------

    def _merge_face_poses(
        self, w2c_faces: th.Tensor, valid_mask: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Merge per-face world-to-camera poses into panorama poses."""
        device, dtype = w2c_faces.device, w2c_faces.dtype
        face_rot = self._face_rots.to(device=device, dtype=dtype)

        face_rot_mats = th.eye(4, device=device, dtype=dtype).repeat(face_rot.shape[0], 1, 1)
        face_rot_mats[:, :3, :3] = face_rot

        w2c_faces = face_rot_mats.unsqueeze(0) @ w2c_faces

        rotations = w2c_faces[:, :, :3, :3]
        translations = w2c_faces[:, :, :3, 3]

        centers = -(rotations.transpose(-1, -2) @ translations.unsqueeze(-1)).squeeze(-1)
        counts = valid_mask.sum(dim=1)
        assert th.all(counts > 0), "No valid face poses for one or more frames."
        weights = valid_mask.to(dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        centers_merged = (centers * weights.unsqueeze(-1)).sum(dim=1)

        quats = mat_to_quat_xyzw(rotations)
        merged_quat = mean_quaternion_markley(quats, weights=weights)
        merged_rot = quat_to_mat_xyzw(merged_quat)

        ref_idx = int(th.nonzero(counts > 0, as_tuple=False)[0].item())
        ref_rot = merged_rot[ref_idx]
        ref_center = centers_merged[ref_idx]
        rel_rot = merged_rot @ ref_rot.transpose(-1, -2)
        rel_centers = centers_merged - ref_center
        rel_centers = (ref_rot @ rel_centers.unsqueeze(-1)).squeeze(-1)
        merged = pose_from_center_and_rotation(rel_centers, rel_rot)

        return merged, ref_rot, ref_center

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: th.Tensor,
    ) -> tuple[th.Tensor, list[tuple[th.Tensor, th.Tensor]], dict[str, th.Tensor]]:
        """Estimate panorama poses by projecting faces and running COLMAP."""
        assert images.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        batch, seq_len, channels, height, width = images.shape
        assert batch == 1, "Batch size > 1 not supported"
        assert channels == 4, "Expected RGBA inputs"

        faces = self._project_faces(images)

        with self._temporary_directory() as image_dir:
            image_dir_path = Path(image_dir)
            image_names = self._write_face_images(faces, image_dir_path)
            model = self._run_colmap(image_dir_path, image_names)

        w2c_faces, valid_mask = self._poses_from_model(model, seq_len)
        merged, ref_rot, ref_center = self._merge_face_poses(w2c_faces, valid_mask)
        keypoints = self._keypoints_from_model(
            model,
            seq_len,
            (height, width),
            align=(ref_rot, ref_center),
        )
        keypoints = [(xy.to(device=images.device, dtype=th.int32), xyz.to(images)) for xy, xyz in keypoints]

        return merged.unsqueeze(0).to(images), keypoints, {
            "pose_faces": w2c_faces.unsqueeze(0).to(images)
        }

    def configure_optimizers(self):
        """Lightning hook for compatibility; pycolmap runs inference-only."""
        return []

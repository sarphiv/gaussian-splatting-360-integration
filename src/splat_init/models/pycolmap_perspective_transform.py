"""pycolmap wrapper that projects panoramas to cube faces and merges pose estimates."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import struct
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

_CAMERA_MODEL_IDS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass(frozen=True)
class ColmapPoint3D:
    id: int
    xyz: np.ndarray


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points3D: dict[int, ColmapPoint3D]


def _read_next_bytes(handle, num_bytes: int, format_char_sequence: str, endian_character: str = "<") -> tuple:
    data = handle.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def _read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            model_name, num_params = _CAMERA_MODEL_IDS[model_id]
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
    return cameras


def _read_images_binary(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            (
                image_id,
                qw,
                qx,
                qy,
                qz,
                tx,
                ty,
                tz,
                camera_id,
            ) = _read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            qvec = np.array((qw, qx, qy, qz))
            tvec = np.array((tx, ty, tz))
            name_bytes = b""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                name_bytes += current_char
                current_char = _read_next_bytes(fid, 1, "c")[0]
            image_name = name_bytes.decode("utf-8")
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = _read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack((x_y_id_s[0::3], x_y_id_s[1::3]))
            point3D_ids = np.array(x_y_id_s[2::3], dtype=np.int64)
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def _read_points3d_binary(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    with path.open("rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point3d_id, x, y, z, _, _, _, _ = _read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            track_length = _read_next_bytes(fid, 8, "Q")[0]
            _ = _read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            points[point3d_id] = ColmapPoint3D(
                id=point3d_id,
                xyz=np.array((x, y, z)),
            )
    return points


def _read_colmap_model(model_path: Path) -> ColmapModel:
    cameras = _read_cameras_binary(model_path / "cameras.bin")
    images = _read_images_binary(model_path / "images.bin")
    points3d = _read_points3d_binary(model_path / "points3D.bin")
    return ColmapModel(cameras=cameras, images=images, points3D=points3d)


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ]
    )


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

    def _run_colmap(self, image_dir: Path, image_names: list[str]) -> ColmapModel | None:
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
                camera_model="PINHOLE",
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
            return _read_colmap_model(model_path)

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
        model: ColmapModel,
        seq_len: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Extract per-face w2c poses and a validity mask from a COLMAP model."""
        num_faces = len(self.face_indices)
        w2c_faces = th.eye(4, dtype=th.float32).repeat(seq_len, num_faces, 1, 1)
        valid_mask = th.zeros((seq_len, num_faces), dtype=th.bool)

        for image in model.images.values():
            face_idx, frame_idx = self._parse_image_name(image.name)
            if face_idx not in self._face_to_index:
                continue
            assert 0 <= frame_idx < seq_len, f"Frame index {frame_idx} out of range"
            face_slot = self._face_to_index[face_idx]

            w2c = th.eye(4, dtype=th.float32)
            w2c[:3, :3] = th.from_numpy(_qvec_to_rotmat(image.qvec)).to(dtype=th.float32)
            w2c[:3, 3] = th.from_numpy(image.tvec).to(dtype=th.float32)

            w2c_faces[frame_idx, face_slot] = w2c
            valid_mask[frame_idx, face_slot] = True

        return w2c_faces, valid_mask

    def _map_face_to_equirect(
        self,
        face_idx: int,
        xys: th.Tensor,
        face_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> th.Tensor:
        """Map face pixel coordinates to equirectangular pixel coordinates."""
        face_width, face_height = face_size
        out_height, out_width = output_size
        assert face_width > 1 and face_height > 1, "Face images must be at least 2x2"
        assert out_width > 1 and out_height > 1, "Panorama images must be at least 2x2"

        u_lin = 2.0 * xys[:, 0] / (face_width - 1.0) - 1.0
        v_lin = 2.0 * xys[:, 1] / (face_height - 1.0) - 1.0
        direction = self._projector._dir_for_face(u_lin, v_lin, FACE_ORDER[face_idx])

        lon = th.atan2(direction[0], direction[2])
        lat = th.atan2(direction[1], th.sqrt(direction[0] * direction[0] + direction[2] * direction[2]))

        x = (lon / math.pi + 1.0) * 0.5 * (out_width - 1.0)
        y = (-2.0 * lat / math.pi + 1.0) * 0.5 * (out_height - 1.0)
        return th.stack((x, y), dim=1)

    def _keypoints_from_model(
        self,
        model: ColmapModel,
        seq_len: int,
        output_size: tuple[int, int],
        align: tuple[th.Tensor, th.Tensor] | None = None,
    ) -> list[tuple[th.Tensor, th.Tensor]]:
        """Extract equirectangular keypoints and world-space points from COLMAP."""
        xy_accum: list[list[th.Tensor]] = [[] for _ in range(seq_len)]
        xyz_accum: list[list[th.Tensor]] = [[] for _ in range(seq_len)]

        ref_rot = None
        ref_center = None
        if align is not None:
            ref_rot, ref_center = align
            ref_rot = ref_rot.to(device=th.device("cpu"), dtype=th.float32)
            ref_center = ref_center.to(device=th.device("cpu"), dtype=th.float32)

        for image in model.images.values():
            face_idx, frame_idx = self._parse_image_name(image.name)
            if face_idx not in self._face_to_index:
                continue
            assert 0 <= frame_idx < seq_len, f"Frame index {frame_idx} out of range"

            if image.point3D_ids.size == 0:
                continue
            valid = image.point3D_ids >= 0
            if not np.any(valid):
                continue

            camera = model.cameras[image.camera_id]
            xys = th.from_numpy(image.xys[valid]).to(dtype=th.float32)
            eq_xy = self._map_face_to_equirect(
                face_idx,
                xys,
                (camera.width, camera.height),
                output_size,
            ).transpose(0, 1)

            point_ids = image.point3D_ids[valid]
            xyz = np.stack([model.points3D[int(pid)].xyz for pid in point_ids], axis=0)
            xyz = th.from_numpy(xyz).to(dtype=th.float32).transpose(0, 1)
            if ref_rot is not None and ref_center is not None:
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
        weights = valid_mask.to(dtype)
        counts = weights.sum(dim=1)

        if th.count_nonzero(counts) == 0:
            identity = th.eye(4, device=device, dtype=dtype).repeat(w2c_faces.shape[0], 1, 1)
            ref_rot = th.eye(3, device=device, dtype=dtype)
            ref_center = th.zeros((3,), device=device, dtype=dtype)
            return identity, ref_rot, ref_center

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
        ref_rot = merged_rot[ref_idx]
        ref_center = centers_merged[ref_idx]
        rel_rot = merged_rot @ ref_rot.transpose(-1, -2)
        rel_centers = centers_merged - ref_center
        rel_centers = (ref_rot @ rel_centers.unsqueeze(-1)).squeeze(-1)
        merged = pose_from_center_and_rotation(rel_centers, rel_rot)

        if missing.any():
            merged[missing] = th.eye(4, device=device, dtype=dtype)

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
            model = self._run_colmap(image_dir_path, image_names)

        if model is None:
            identity = th.eye(4, device=img.device, dtype=img.dtype).repeat(seq_len, 1, 1)
            w2c_faces = identity.unsqueeze(1).repeat(1, len(self.face_indices), 1, 1)
            valid_mask = th.zeros((seq_len, len(self.face_indices)), device=img.device, dtype=th.bool)
            merged = identity
            keypoints = [
                (
                    th.empty((2, 0), device=img.device, dtype=th.int32),
                    th.empty((3, 0), device=img.device, dtype=images.dtype),
                )
                for _ in range(seq_len)
            ]
        else:
            w2c_faces, valid_mask = self._poses_from_model(model, seq_len)
            w2c_faces = w2c_faces.to(device=img.device, dtype=cast(th.dtype, self.dtype))
            valid_mask = valid_mask.to(device=img.device)
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

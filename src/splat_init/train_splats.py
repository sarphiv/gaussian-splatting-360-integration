from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import torch as th
import tyro
from joblib import Parallel, delayed
from loguru import logger
from torchvision.io import write_png

from configs.train_splats_args import Args, ProjectionArgs
from splat_init.data.datamodule_360 import SceneSampleLazy
from splat_init.data.stanford_2d_3d import Stanford2d3dDataset
from splat_init.data.threesixty_loc import ThreeSixtyLocDataset
from utilities.otc_projector import OTCProjector, cube_face_relative_rotations


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OTF_NVS_ROOT = _PROJECT_ROOT / "vendor" / "otf-nvs"
sys.path.append(str(_OTF_NVS_ROOT))

from dataloaders.read_write_model import BaseImage, Camera, rotmat2qvec, write_model  # noqa: E402


_FACE_INDICES = (0, 1, 4, 5)
_FORWARD_FACE = 4
_LEFT_FACE = 1
_RIGHT_FACE = 0
_BACK_FACE = 5


def _temporary_directory() -> tempfile.TemporaryDirectory[str]:
    """Create a RAM-backed temporary directory under /dev/shm."""
    shm = Path("/dev/shm")
    assert shm.is_dir() and os.access(shm, os.W_OK), "RAM-backed temp dir unavailable"
    return tempfile.TemporaryDirectory(dir=shm)


def _load_dataset_stride(output_dir: Path) -> int:
    """Read dataset_stride from the first available poses/metrics.pt."""
    assert output_dir.is_dir(), f"Missing output directory: {output_dir}"
    metrics_paths = sorted(output_dir.glob("*/poses/metrics.pt"))
    assert metrics_paths, f"No metrics.pt found under {output_dir}"
    metrics = th.load(metrics_paths[0], map_location="cpu")
    dataset_stride = int(metrics["dataset_stride"])
    return dataset_stride


def _build_dataset(
    args: Args, dataset_stride: int
) -> ThreeSixtyLocDataset[SceneSampleLazy] | Stanford2d3dDataset[SceneSampleLazy]:
    """Construct the dataset using the same loader types as evaluate_poses."""
    if args.data.dataset_name == "stanford_2d_3d":
        return Stanford2d3dDataset(
            SceneSampleLazy,
            args.data.dataset_dir,
            image_size=args.data.dataset_image_size,
            perspective_loader_threads=args.data.dataloader_workers,
        )
    if args.data.dataset_name == "360_loc":
        return ThreeSixtyLocDataset(
            SceneSampleLazy,
            args.data.dataset_dir,
            stride=dataset_stride,
            depth_required=True,
            image_size=args.data.dataset_image_size,
            worker_count=args.data.dataloader_workers,
        )
    raise ValueError(f"Unknown dataset: {args.data.dataset_name}")


def _face_order(sequence_length: int) -> list[tuple[int, int]]:
    """Return ordered (face_idx, frame_idx) pairs for the otf-nvs dataset."""
    order: list[tuple[int, int]] = []
    order.extend((_FORWARD_FACE, idx) for idx in range(sequence_length))
    order.extend((_LEFT_FACE, idx) for idx in reversed(range(sequence_length)))
    order.extend((_RIGHT_FACE, idx) for idx in range(sequence_length))
    order.extend((_BACK_FACE, idx) for idx in reversed(range(sequence_length)))
    return order


def _image_name(ordinal: int, face_idx: int, frame_idx: int) -> str:
    """Generate a filename that encodes the required ordering."""
    return f"ord_{ordinal:06d}_face_{face_idx}_frame_{frame_idx:06d}.png"


def _ordinal_lookup(order: Sequence[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Map (face_idx, frame_idx) to the ordinal position in the ordering."""
    return {pair: ordinal for ordinal, pair in enumerate(order)}


def _write_png_frame(frame: th.Tensor, filename: Path) -> None:
    """Write a single PNG image to disk."""
    write_png(frame, str(filename))


def _write_image_tasks(
    tasks: Sequence[tuple[th.Tensor, Path]],
    projection: ProjectionArgs,
) -> None:
    """Write images in chunks to avoid holding large batches in memory."""
    if not tasks:
        return
    if projection.image_workers <= 1:
        for frame, filename in tasks:
            _write_png_frame(frame, filename)
        return

    chunk_size = projection.image_workers * projection.images_per_worker
    parallel = Parallel(n_jobs=projection.image_workers, backend="threading")
    for start in range(0, len(tasks), chunk_size):
        end = min(start + chunk_size, len(tasks))
        chunk = tasks[start:end]
        parallel(delayed(_write_png_frame)(frame, filename) for frame, filename in chunk)


def _project_and_write_scene(
    scene: SceneSampleLazy,
    projector: OTCProjector,
    device: th.device,
    projection: ProjectionArgs,
    images_dir: Path,
    ordinal_lookup: dict[tuple[int, int], int],
) -> None:
    """Project panorama frames to cube faces and write them to disk."""
    images_dir.mkdir(parents=True, exist_ok=True)
    face_index_to_slot = {face_idx: slot for slot, face_idx in enumerate(_FACE_INDICES)}
    face_indices = list(_FACE_INDICES)

    for start in range(0, len(scene), projection.projection_batch_size):
        end = min(start + projection.projection_batch_size, len(scene))
        sample = scene[start:end]
        rgba = sample.rgba.to(device=device, dtype=th.float32)
        rgb_faces, alpha_faces, _ = projector(rgba, depth=None)
        rgb_faces = rgb_faces * alpha_faces
        rgb_faces = rgb_faces[:, face_indices]
        rgb_faces = (
            rgb_faces.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(device=th.device("cpu"), dtype=th.uint8)
        )

        tasks: list[tuple[th.Tensor, Path]] = []
        for local_idx, frame_idx in enumerate(range(start, end)):
            for face_idx in face_indices:
                face_slot = face_index_to_slot[face_idx]
                ordinal = ordinal_lookup[(face_idx, frame_idx)]
                name = _image_name(ordinal, face_idx, frame_idx)
                tasks.append((rgb_faces[local_idx, face_slot], images_dir / name))
        _write_image_tasks(tasks, projection)


def _build_colmap_model(
    poses_w2c: th.Tensor,
    face_size: int,
    order: Sequence[tuple[int, int]],
) -> tuple[dict[int, Camera], dict[int, BaseImage], float]:
    """Create COLMAP cameras/images for the ordered cube faces."""
    assert poses_w2c.ndim == 3 and poses_w2c.shape[1:] == (4, 4), "Expected poses [S,4,4]"

    face_indices = list(_FACE_INDICES)
    face_rot = cube_face_relative_rotations()[face_indices]
    face_rot_t = face_rot.transpose(-1, -2)

    rotations = poses_w2c[:, :3, :3]
    translations = poses_w2c[:, :3, 3]

    face_rotations = (face_rot_t[:, None] @ rotations[None]).permute(1, 0, 2, 3)
    face_translations = (
        face_rot_t[:, None] @ translations[None, :, :, None]
    ).squeeze(-1).permute(1, 0, 2)

    focal = 0.5 * (face_size - 1)
    principal = 0.5 * (face_size - 1)
    cameras = {
        1: Camera(
            id=1,
            model="SIMPLE_PINHOLE",
            width=face_size,
            height=face_size,
            params=[focal, principal, principal],
        )
    }

    face_index_to_slot = {face_idx: slot for slot, face_idx in enumerate(face_indices)}
    images: dict[int, BaseImage] = {}
    for ordinal, (face_idx, frame_idx) in enumerate(order):
        face_slot = face_index_to_slot[face_idx]
        rot = face_rotations[frame_idx, face_slot].cpu().numpy()
        trans = face_translations[frame_idx, face_slot].cpu().numpy()
        qvec = -rotmat2qvec(rot)
        name = _image_name(ordinal, face_idx, frame_idx)
        images[ordinal + 1] = BaseImage(
            id=ordinal + 1,
            qvec=qvec,
            tvec=trans,
            camera_id=1,
            name=name,
            xys=[],
            point3D_ids=[],
        )

    return cameras, images, focal


def _run_otf_nvs(source_path: Path, output_path: Path, focal: float) -> None:
    """Launch otf-nvs training using COLMAP initialization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        str(_OTF_NVS_ROOT / "train.py"),
        "-s",
        str(source_path),
        "-m",
        str(output_path),
        "--num_iterations",
        "18",
        "--min_num_inliers",
        "0",
        "--use_colmap_poses",
        "--init_focal",
        f"{focal}",
        "--fix_focal",
    ]
    logger.info("Running otf-nvs: {}", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_PROJECT_ROOT)


def _load_scene_metrics(metrics_path: Path) -> dict[str, float | int | str]:
    """Load per-scene metrics stored by evaluate_poses."""
    metrics = th.load(metrics_path, map_location="cpu")
    assert isinstance(metrics, dict), "Expected metrics to be a dict."
    return metrics


def _load_scene_poses(poses_path: Path) -> th.Tensor:
    """Load predicted panorama poses from model_output.pt."""
    payload = th.load(poses_path, map_location="cpu")
    poses = payload["poses"]
    if poses.ndim == 4:
        assert poses.shape[0] == 1, "Unexpected batch dimension in poses"
        poses = poses[0]
    assert poses.ndim == 3, "Expected poses shaped [S,4,4]"
    return poses.to(dtype=th.float32)


def _iter_scenes(dataset: Iterable[SceneSampleLazy]) -> Iterable[SceneSampleLazy]:
    """Yield scenes from an iterable dataset."""
    for scene in dataset:
        yield scene


def main() -> None:
    args = tyro.cli(Args)

    dataset_stride = _load_dataset_stride(args.output_dir)
    logger.info("Using dataset stride {}", dataset_stride)

    dataset = _build_dataset(args, dataset_stride)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    projector = OTCProjector(face_size=args.projection.face_size, alpha=1e-9)

    for scene in _iter_scenes(dataset):
        poses_dir = args.output_dir / scene.id / "poses"
        poses_path = poses_dir / "model_output.pt"
        metrics_path = poses_dir / "metrics.pt"
        if not poses_path.is_file():
            logger.info("Skipping {} (missing poses at {})", scene.id, poses_path)
            continue

        metrics = _load_scene_metrics(metrics_path)
        sequence_length = int(metrics["sequence_length"])
        metrics_stride = int(metrics["dataset_stride"])
        assert metrics_stride == dataset_stride, "Dataset stride mismatch with metrics."
        assert len(scene) == sequence_length, "Scene length does not match metrics."
        assert sequence_length >= 6, "Need at least 6 frames for otf-nvs alignment."

        poses_w2c = _load_scene_poses(poses_path)
        assert poses_w2c.shape[0] == sequence_length, "Pose count mismatch."

        order = _face_order(sequence_length)
        ordinal_lookup = _ordinal_lookup(order)

        with _temporary_directory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / "images"
            sparse_dir = tmp_path / "sparse" / "0"
            sparse_dir.mkdir(parents=True, exist_ok=True)

            _project_and_write_scene(
                scene,
                projector,
                device,
                args.projection,
                images_dir,
                ordinal_lookup,
            )

            cameras, images, focal = _build_colmap_model(
                poses_w2c,
                args.projection.face_size,
                order,
            )
            write_model(cameras, images, {}, sparse_dir, ext=".bin")
            
            if device.type == "cuda":
                th.cuda.empty_cache()

            splat_dir = args.output_dir / scene.id / "splat"
            _run_otf_nvs(tmp_path, splat_dir, focal)

        metadata_path = splat_dir / "metadata.json"
        logger.info("Splat metadata at {}", metadata_path)


if __name__ == "__main__":
    main()

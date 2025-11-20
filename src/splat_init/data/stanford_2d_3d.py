"""Stanford 2D-3D Panorama Dataset utilities (data-oriented).

This module provides a dataset that groups all panoramas from the same room
within a given Stanford 2D-3D ``area_*`` directory. It indexes once at init and
returns per-room batches on demand.

Dataset specifics (as seen in area_1_no_xyz):
- Directories: ``pano/{rgb,depth,pose}``.
- Filenames share the same prefix up to ``_domain_``.
- Modalities:
  - RGBA:  ``..._domain_rgb.png``
  - Depth: ``..._domain_depth.png``
  - Pose:  ``..._domain_pose.json`` with key ``camera_rt_matrix`` (3x4).
- Room grouping uses the token in the filename, e.g. ``conferenceRoom_1``.

Returned sample
- origin_name: ``stanford-2d-3d/area-1/conference-room-1``
- rgba:  torch tensor ``[S, 4, H, W]`` (uint8, RGBA where A is a cutout mask)
- depth: torch tensor ``[S, 1, H, W]`` (float32)
- pose:  torch tensor ``[S, 4, 4]`` (float32)
"""
from __future__ import annotations

from typing import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import re

import torch
from torch import Tensor
from torchvision.io import read_image

from splat_init.data.datamodule_360 import SceneSample, SceneSampleLazy


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


_ROOM_REGEX = re.compile(r"camera_[^_]+_(?P<room>.+?)_frame_")


def _extract_room_id(name: str) -> str:
    """Extract the room token from a file name.

    Example input: "camera_..._conferenceRoom_1_frame_equirectangular_domain_rgb.png"
    or "camera_..._office_12_frame_4_domain_rgb.png".
    Returns: "conferenceRoom_1"

    Assumes the naming pattern contains a single room token between the camera
    hash and the "_frame_" marker.
    """

    m = _ROOM_REGEX.search(name)
    assert m is not None, f"room token not found in file name: {name}"
    return m.group("room")


def _prefix_up_to_domain(name: str) -> str:
    """Return filename prefix up to the "_domain_" token (excluded).

    This prefix is shared across modalities (rgb/depth/pose) for a given view.
    """

    head, sep, _ = name.partition("_domain_")
    assert sep == "_domain_", f"'_domain_' not present in file name: {name}"
    return head


def _kebab_case_area(area_dir: Path) -> str:
    """Convert an area directory name to kebab-case (e.g., area_1 -> area-1)."""

    m = re.search(r"area[_-]?(\d+)", area_dir.name, flags=re.IGNORECASE)
    assert m, f"area directory should include an index: {area_dir}"
    return f"area-{m.group(1)}"


_CAMEL_TO_KEBAB = re.compile(r"(?<!^)(?=[A-Z])")


def _room_to_kebab(room_id: str) -> str:
    """Convert room token like 'conferenceRoom_1' to 'conference-room-1'."""

    if "_" in room_id:
        base, num = room_id.split("_", 1)
    else:
        base, num = room_id, ""
    base_kebab = _CAMEL_TO_KEBAB.sub("-", base).lower()
    return f"{base_kebab}-{num}" if num else base_kebab


def _canonical_origin(area_dir: Path, room_id: str) -> str:
    """Compose the canonical origin string for a room in an area."""

    return f"stanford-2d-3d.{_kebab_case_area(area_dir)}.{_room_to_kebab(room_id)}"


def _stack_with_channel(imgs: list[Tensor]) -> Tensor:
    """Stack a list of CxHxW tensors into SxC x H x W, preserving channel dim."""

    assert len(imgs) > 0, "expected at least one image to stack"
    c, h, w = imgs[0].shape
    for t in imgs:
        assert t.ndim == 3, "each image must be [C,H,W]"
        assert tuple(t.shape) == (c, h, w), "all images must share shape"
    return torch.stack(imgs, dim=0)


def _ensure_4x4(mat3x4: Tensor) -> Tensor:
    """Create a [4,4] homogeneous matrix from a [3,4] matrix."""

    pad = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=mat3x4.dtype)
    return torch.cat([mat3x4, pad], dim=0)

def _load_pose_json(path: Path) -> tuple[Tensor, float]:
    """Load a pose and focal length from Stanford pose JSON as [4,4] float32.

    Expected schema contains key ``camera_rt_matrix`` with shape [3,4].
    """

    data = json.loads(path.read_text())
    mat3x4 = torch.tensor(data["camera_rt_matrix"], dtype=torch.float32)
    focal_length = float(data["camera_k_matrix"][0][0])
    return _ensure_4x4(mat3x4), focal_length


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class Stanford2D3DDataset[T: (SceneSample | SceneSampleLazy)](torch.utils.data.Dataset[T]):
    """Data-oriented dataset grouping all views per room in one Stanford area.

    Specific to pano/{rgb,depth,pose} with JSON pose format containing
    ``camera_rt_matrix`` (3x4). Indexing is done once; lookups are direct.

    Parameters
    - area_dir: Path to an ``area_*`` directory with ``pano`` subfolders.
    - max_sequence_length: Optional cap on the number of views per room. Rooms
      exceeding this length are skipped entirely when indexing.
    - perspective_workers: Optional thread count for loading perspective views
      in parallel. Defaults to ``min(4, max(1, cpu_count // 2))``.
    """

    def __init__(
        self,
        area_dir: Path,
        max_sequence_length: int | None = None,
        perspective_loader_threads: int = 1,
    ) -> None:
        super().__init__()
        self.area_dir = area_dir
        self.max_sequence_length = max_sequence_length
        self.perspective_loader_threads = perspective_loader_threads

        pano_dir = area_dir / "pano"
        rgba_dir = pano_dir / "rgb"
        depth_dir = pano_dir / "depth"
        pose_dir = pano_dir / "pose"

        # Enumerate RGB files once, derive matching depth/pose names.
        rgba_files = sorted(rgba_dir.glob("*.png"))

        self._rgba_paths: list[Path] = []
        self._depth_paths: list[Path] = []
        self._pose_paths: list[Path] = []
        self._room_per_view: list[str] = []

        for p in rgba_files:
            name = p.name
            prefix = _prefix_up_to_domain(name)
            room_id = _extract_room_id(name)
            depth_path = depth_dir / f"{prefix}_domain_depth.png"
            pose_path = pose_dir / f"{prefix}_domain_pose.json"

            # Happy path: files exist with exact names.
            assert depth_path.is_file(), f"missing depth: {depth_path}"
            assert pose_path.is_file(), f"missing pose: {pose_path}"

            self._rgba_paths.append(p)
            self._depth_paths.append(depth_path)
            self._pose_paths.append(pose_path)
            self._room_per_view.append(room_id)

        # Build room index: dataset items correspond to unique rooms.
        room_to_indices: dict[str, list[int]] = {}
        for i, room in enumerate(self._room_per_view):
            room_to_indices.setdefault(room, []).append(i)

        # Sort rooms and build a new global order so each room is contiguous.
        rooms_sorted: list[str] = sorted(room_to_indices.keys())
        new_order: list[int] = []
        new_room_indices: list[list[int]] = []
        kept_rooms: list[str] = []
        limit = self.max_sequence_length

        for room in rooms_sorted:
            # Stable order within a room by filename for determinism.
            indices = room_to_indices[room]
            indices.sort(key=lambda i: self._rgba_paths[i].name)
            if limit is not None and len(indices) > limit:
                continue
            start = len(new_order)
            new_order.extend(indices)
            end = len(new_order)
            new_room_indices.append(list(range(start, end)))
            kept_rooms.append(room)

        # Reorder per-view arrays to make rooms contiguous.
        self._rgba_paths = [self._rgba_paths[i] for i in new_order]
        self._depth_paths = [self._depth_paths[i] for i in new_order]
        self._pose_paths = [self._pose_paths[i] for i in new_order]
        self._room_per_view = [self._room_per_view[i] for i in new_order]

        # Freeze room lists and indices post-reordering.
        self._rooms: list[str] = kept_rooms
        self._room_indices: list[list[int]] = new_room_indices

        # Index perspective assets under area_dir / "data" mirroring pano layout.
        data_dir = area_dir / "data"
        rgb_dir = data_dir / "rgb"
        depth_dir = data_dir / "depth"
        pose_dir = data_dir / "pose"

        assert rgb_dir.is_dir(), f"missing perspective rgb directory: {rgb_dir}"
        assert depth_dir.is_dir(), f"missing perspective depth directory: {depth_dir}"
        assert pose_dir.is_dir(), f"missing perspective pose directory: {pose_dir}"

        self._persp_rgba_paths: list[Path] = []
        self._persp_depth_paths: list[Path] = []
        self._persp_pose_paths: list[Path] = []

        kept_room_set = set(self._rooms)
        persp_room_to_indices: dict[str, list[int]] = {}

        for p in sorted(rgb_dir.glob("*.png")):
            room_id = _extract_room_id(p.name)
            if room_id not in kept_room_set:
                continue

            prefix = _prefix_up_to_domain(p.name)
            depth_path = depth_dir / f"{prefix}_domain_depth.png"
            pose_path = pose_dir / f"{prefix}_domain_pose.json"

            assert depth_path.is_file(), f"missing perspective depth: {depth_path}"
            assert pose_path.is_file(), f"missing perspective pose: {pose_path}"

            global_idx = len(self._persp_rgba_paths)
            self._persp_rgba_paths.append(p)
            self._persp_depth_paths.append(depth_path)
            self._persp_pose_paths.append(pose_path)
            persp_room_to_indices.setdefault(room_id, []).append(global_idx)

        self._persp_room_indices: list[list[int]] = []
        for room in self._rooms:
            indices = persp_room_to_indices.get(room)
            assert indices, f"no perspective frames found for room: {room}"
            sorted_indices = sorted(indices, key=lambda i: self._persp_rgba_paths[i].name)
            self._persp_room_indices.append(sorted_indices)


    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._rooms)

    @staticmethod
    def _make_item_getter(scene_id: str, rgba_paths: list[Path], depth_paths: list[Path], pose_paths: list[Path]) -> Callable[[Sequence[int]], SceneSample]:
        def getter(indices: Sequence[int]) -> SceneSample:
            rgba_imgs: list[Tensor] = []
            depth_imgs: list[Tensor] = []
            poses: list[Tensor] = []

            for i in indices:
                rgba = read_image(str(rgba_paths[i]))  # [4,H,W] RGBA
                rgba = rgba.to(torch.float32) / 255.0
                rgba_imgs.append(rgba)

                depth = read_image(str(depth_paths[i]))  # [1,H,W]
                # NOTE: Each depth unit is 1/512 meters.
                depth = depth.to(torch.float32) / 512.0
                depth_imgs.append(depth)

                pose, _ = _load_pose_json(pose_paths[i])
                poses.append(pose)

            rgba_batch = _stack_with_channel(rgba_imgs)   # [S,4,H,W]
            depth_batch = _stack_with_channel(depth_imgs)  # [S,1,H,W]
            pose_batch = torch.stack(poses, dim=0)  # [S,4,4]

            return SceneSample(scene_id, rgba_batch, depth_batch, pose_batch, None)

        return getter


    def __getitem__(self, idx: int) -> T:
        room_id = self._rooms[idx]
        view_indices = self._room_indices[idx]
        scene_id = _canonical_origin(self.area_dir, room_id)
        
        loader = self._make_item_getter(
            scene_id,
            [self._rgba_paths[i] for i in view_indices],
            [self._depth_paths[i] for i in view_indices],
            [self._pose_paths[i] for i in view_indices]
        )

        if T is SceneSampleLazy:
            return SceneSampleLazy(
                id=scene_id,
                get_item_range=loader,
                item_count=len(self.poses[scene_id])
            )
        elif T is SceneSample:
            return loader(range(len(self.poses[scene_id])))
        else:
            raise TypeError(f"Unsupported dataset item type: {T}")


    def get_perspective(self, idx: int) -> SceneSample:
        """Load all perspective views from the room at index ``idx``.

        Returns a ``SceneSample`` dataclass with origin name and stacked tensors:
            - rgb   [S, 4, H, W]
            - depth [S, 1, H, W]
            - pose  [S, 4, 4]
        """
        room_id = self._rooms[idx]
        view_indices = self._persp_room_indices[idx]
        origin = _canonical_origin(self.area_dir, room_id)

        def _load_view(vi: int) -> tuple[Tensor, Tensor, Tensor, float]:
            rgba = read_image(str(self._persp_rgba_paths[vi]))
            depth = read_image(str(self._persp_depth_paths[vi]))
            pose, focal_length = _load_pose_json(self._persp_pose_paths[vi])
            return (
                rgba.to(torch.float32) / 255.0,
                depth.to(torch.float32) / 512.0,
                pose,
                focal_length
            )

        worker_count = min(self.perspective_loader_threads, len(view_indices))
        if worker_count > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                loaded = list(pool.map(_load_view, view_indices))
        else:
            loaded = [_load_view(vi) for vi in view_indices]

        rgba_batch = _stack_with_channel([rgba for rgba, _, _, _ in loaded])
        depth_batch = _stack_with_channel([depth for _, depth, _, _ in loaded])
        pose_batch = torch.stack([pose for _, _, pose, _ in loaded], dim=0)
        focal_length_batch = torch.tensor([focal for *_, focal in loaded])

        return SceneSample(origin, rgba_batch, depth_batch, pose_batch, focal_length_batch)


__all__ = [
    "Stanford2D3DDataset",
]

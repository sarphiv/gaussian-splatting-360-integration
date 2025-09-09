"""Stanford 2D-3D Panorama Dataset utilities (data-oriented).

This module provides a dataset that groups all panoramas from the same room
within a given Stanford 2D-3D ``area_*`` directory. It indexes once at init and
returns per-room batches on demand.

Dataset specifics (as seen in area_1_no_xyz):
- Directories: ``pano/{rgb,depth,pose}``.
- Filenames share the same prefix up to ``_domain_``.
- Modalities:
  - RGB:   ``..._domain_rgb.png``
  - Depth: ``..._domain_depth.png``
  - Pose:  ``..._domain_pose.json`` with key ``camera_rt_matrix`` (3x4).
- Room grouping uses the token in the filename, e.g. ``conferenceRoom_1``.

Returned sample
- origin_name: ``stanford-2d-3d/area-1/conference-room-1``
- rgb:   torch tensor ``[S, 4, H, W]`` (uint8, RGBA where A is a cutout mask)
- depth: torch tensor ``[S, 1, H, W]`` (float32)
- pose:  torch tensor ``[S, 4, 4]`` (float32)
"""
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Iterator

import torch
from torch import Tensor
from torchvision.io import read_image
from .datamodule_360 import RoomSample360


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


_ROOM_REGEX = re.compile(r"camera_[^_]+_(?P<room>.+?)_frame_equirectangular")


def _extract_room_id(name: str) -> str:
    """Extract the room token from a file name.

    Example input: "camera_..._conferenceRoom_1_frame_equirectangular_domain_rgb.png"
    Returns: "conferenceRoom_1"

    Assumes the naming pattern contains a single room token between the camera
    hash and "frame_equirectangular".
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

    return f"stanford-2d-3d/{_kebab_case_area(area_dir)}/{_room_to_kebab(room_id)}"


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


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class Stanford2D3DDataset(torch.utils.data.Dataset[RoomSample360]):
    """Data-oriented dataset grouping all views per room in one Stanford area.

    Specific to pano/{rgb,depth,pose} with JSON pose format containing
    ``camera_rt_matrix`` (3x4). Indexing is done once; lookups are direct.

    Parameters
    - area_dir: Path to an ``area_*`` directory with ``pano`` subfolders.
    """

    def __init__(self, area_dir: Path) -> None:
        super().__init__()
        self.area_dir = area_dir

        pano_dir = area_dir / "pano"
        rgb_dir = pano_dir / "rgb"
        depth_dir = pano_dir / "depth"
        pose_dir = pano_dir / "pose"

        # Enumerate RGB files once, derive matching depth/pose names.
        rgb_files = sorted(rgb_dir.glob("*.png"))

        self._rgb_paths: list[Path] = []
        self._depth_paths: list[Path] = []
        self._pose_paths: list[Path] = []
        self._room_per_view: list[str] = []

        for p in rgb_files:
            name = p.name
            prefix = _prefix_up_to_domain(name)
            room_id = _extract_room_id(name)
            depth_path = depth_dir / f"{prefix}_domain_depth.png"
            pose_path = pose_dir / f"{prefix}_domain_pose.json"

            # Happy path: files exist with exact names.
            assert depth_path.is_file(), f"missing depth: {depth_path}"
            assert pose_path.is_file(), f"missing pose: {pose_path}"

            self._rgb_paths.append(p)
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

        for room in rooms_sorted:
            # Stable order within a room by filename for determinism.
            indices = room_to_indices[room]
            indices.sort(key=lambda i: self._rgb_paths[i].name)
            start = len(new_order)
            new_order.extend(indices)
            end = len(new_order)
            new_room_indices.append(list(range(start, end)))

        # Reorder per-view arrays to make rooms contiguous.
        self._rgb_paths = [self._rgb_paths[i] for i in new_order]
        self._depth_paths = [self._depth_paths[i] for i in new_order]
        self._pose_paths = [self._pose_paths[i] for i in new_order]
        self._room_per_view = [self._room_per_view[i] for i in new_order]

        # Freeze room lists and indices post-reordering.
        self._rooms: list[str] = rooms_sorted
        self._room_indices: list[list[int]] = new_room_indices

        # Assumptions verified during development: RGB is 3/4-ch (we keep 3),
        # Depth is single-channel, and RGB/Depth resolutions match.

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._rooms)

    def __getitem__(self, idx: int) -> RoomSample360:
        """Load all views from the room at index ``idx``.

        Returns a dataclass with origin name and stacked tensors:
        - rgb   [S, 4, H, W]
        - depth [S, 1, H, W]
        - pose  [S, 4, 4]
        """

        room_id = self._rooms[idx]
        view_indices = self._room_indices[idx]
        origin = _canonical_origin(self.area_dir, room_id)

        rgb_imgs: list[Tensor] = []
        depth_imgs: list[Tensor] = []
        poses: list[Tensor] = []

        for vi in view_indices:
            rgb = read_image(str(self._rgb_paths[vi]))  # [4,H,W] RGBA
            rgb_imgs.append(rgb)

            depth = read_image(str(self._depth_paths[vi]))  # [1,H,W]
            depth = depth.to(torch.float32)
            depth_imgs.append(depth)

            pose = _load_pose_json(self._pose_paths[vi])
            poses.append(pose)

        rgb_batch = _stack_with_channel(rgb_imgs)   # [S,4,H,W]
        depth_batch = _stack_with_channel(depth_imgs)  # [S,1,H,W]
        pose_batch = torch.stack(poses, dim=0)  # [S,4,4]

        return RoomSample360(origin, rgb_batch, depth_batch, pose_batch)

    


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def _load_pose_json(path: Path) -> Tensor:
    """Load a pose from Stanford pose JSON as [4,4] float32.

    Expected schema contains key ``camera_rt_matrix`` with shape [3,4].
    """

    data = json.loads(path.read_text())
    mat3x4 = torch.tensor(data["camera_rt_matrix"], dtype=torch.float32)
    return _ensure_4x4(mat3x4)


# -----------------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------------


def iter_rooms(dataset: Stanford2D3DDataset) -> Iterator[RoomSample360]:
    """Iterate over full-room batches for convenience (one sample per room)."""

    for i in range(len(dataset)):
        yield dataset[i]


__all__ = [
    "Stanford2D3DDataset",
    "iter_rooms",
]

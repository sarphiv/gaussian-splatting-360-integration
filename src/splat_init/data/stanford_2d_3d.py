"""Stanford 2D-3D Panorama Dataset utilities (data-oriented).

This module provides:
- ``Stanford2D3DAreaDataset`` for grouping all panoramas from the same room
  within a single ``area_*`` directory.
- ``Stanford2d3dDataset`` for combining multiple areas into one dataset while
  retaining area-aware indexing.

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

from typing import Callable, Iterator, Sequence, cast
from bisect import bisect_right
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


_CAMEL_TO_KEBAB = re.compile(r"(?<!^)(?=[A-Z])")


def _is_area_dir(path: Path) -> bool:
    """Check whether ``path`` looks like a Stanford 2D-3D area directory."""

    return path.is_dir() and (path / "pano").is_dir()


def _discover_area_dirs(dataset_root: Path) -> list[Path]:
    """List area directories under the dataset root, sorted by name."""

    return sorted([p for p in dataset_root.iterdir() if _is_area_dir(p)], key=lambda p: p.name)


def _load_pose_json(path: Path) -> tuple[Tensor, float]:
    """Load a pose and focal length from Stanford pose JSON as [4,4] float32."""

    data = json.loads(path.read_text())
    mat3x4 = torch.tensor(data["camera_rt_matrix"], dtype=torch.float32)
    pad = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=mat3x4.dtype)
    pose = torch.cat([mat3x4, pad], dim=0)
    focal_length = float(data["camera_k_matrix"][0][0])
    return pose, focal_length


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class Stanford2D3DAreaDataset[T: (SceneSample | SceneSampleLazy)](torch.utils.data.IterableDataset[T]):
    """Data-oriented dataset grouping all views per room in one Stanford area.

    Specific to pano/{rgb,depth,pose} with JSON pose format containing
    ``camera_rt_matrix`` (3x4). Indexing is done once; lookups are direct.

    Parameters
    - area_dir: Path to an ``area_*`` directory with ``pano`` subfolders.
    - max_sequence_length: Optional cap on the number of views per room. Rooms
      exceeding this length are skipped entirely when indexing.
    - perspective_workers: Optional thread count for loading perspective views
      in parallel. Defaults to 1 (single-threaded).
    """

    def __init__(
        self,
        output_type: type[T],
        area_dir: Path,
        max_sequence_length: int | None = None,
        perspective_loader_threads: int = 1,
    ) -> None:
        super().__init__()

        self.output_type = output_type
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
            prefix, _, _ = name.partition("_domain_")
            room_match = cast(re.Match[str], _ROOM_REGEX.search(name))
            room_id = room_match.group("room")
            depth_path = depth_dir / f"{prefix}_domain_depth.png"
            pose_path = pose_dir / f"{prefix}_domain_pose.json"

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
        area_match = cast(re.Match[str], re.search(r"area[_-]?(\d+)", area_dir.name, flags=re.IGNORECASE))
        area_slug = f"area-{area_match.group(1)}"
        self._scene_ids: list[str] = []
        for room in self._rooms:
            if "_" in room:
                base, suffix = room.split("_", 1)
            else:
                base, suffix = room, ""
            kebab_base = _CAMEL_TO_KEBAB.sub("-", base).lower()
            suffix_part = f"-{suffix}" if suffix else ""
            self._scene_ids.append(f"stanford-2d-3d.{area_slug}.{kebab_base}{suffix_part}")

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
            room_match = cast(re.Match[str], _ROOM_REGEX.search(p.name))
            room_id = room_match.group("room")
            if room_id not in kept_room_set:
                continue

            prefix, _, _ = p.name.partition("_domain_")
            depth_path = depth_dir / f"{prefix}_domain_depth.png"
            pose_path = pose_dir / f"{prefix}_domain_pose.json"

            global_idx = len(self._persp_rgba_paths)
            self._persp_rgba_paths.append(p)
            self._persp_depth_paths.append(depth_path)
            self._persp_pose_paths.append(pose_path)
            persp_room_to_indices.setdefault(room_id, []).append(global_idx)

        self._persp_room_indices: list[list[int]] = []
        for room in self._rooms:
            indices = cast(list[int], persp_room_to_indices.get(room))
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

            rgba_batch = torch.stack(rgba_imgs, dim=0)   # [S,4,H,W]
            depth_batch = torch.stack(depth_imgs, dim=0)  # [S,1,H,W]
            pose_batch = torch.stack(poses, dim=0)  # [S,4,4]

            return SceneSample(scene_id, rgba_batch, depth_batch, pose_batch, None)

        return getter


    def __getitem__(self, idx: int) -> T:
        view_indices = self._room_indices[idx]
        view_count = len(view_indices)
        scene_id = self._scene_ids[idx]

        loader = self._make_item_getter(
            scene_id,
            [self._rgba_paths[i] for i in view_indices],
            [self._depth_paths[i] for i in view_indices],
            [self._pose_paths[i] for i in view_indices]
        )

        if self.output_type is SceneSampleLazy:
            output = SceneSampleLazy(
                id=scene_id,
                loader=loader,
                length=view_count
            )
        elif self.output_type is SceneSample:
            output = loader(range(view_count))
        else:
            raise TypeError(f"Unsupported dataset item type: {self.output_type}")

        return cast(T, output)


    def __iter__(self) -> Iterator[T]:
        for idx in range(len(self)):
            yield self[idx]


    def get_perspective(self, idx: int) -> SceneSample:
        """Load all perspective views from the room at index ``idx``.

        Returns a ``SceneSample`` dataclass with origin name and stacked tensors:
            - rgb   [S, 4, H, W]
            - depth [S, 1, H, W]
            - pose  [S, 4, 4]
        """
        view_indices = self._persp_room_indices[idx]
        scene_id = self._scene_ids[idx]

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

        rgba_batch = torch.stack([rgba for rgba, _, _, _ in loaded], dim=0)
        depth_batch = torch.stack([depth for _, depth, _, _ in loaded], dim=0)
        pose_batch = torch.stack([pose for _, _, pose, _ in loaded], dim=0)
        focal_length_batch = torch.tensor([focal for *_, focal in loaded])

        return SceneSample(scene_id, rgba_batch, depth_batch, pose_batch, focal_length_batch)


class Stanford2d3dDataset[T: (SceneSample | SceneSampleLazy)](torch.utils.data.IterableDataset[T]):
    """Unified Stanford 2D-3D dataset spanning all ``area_*`` folders.

    Combines all discovered ``Stanford2D3DAreaDataset`` instances under a
    dataset root. Indexing is global across all rooms, while
    ``get_perspective`` remains area-aware via ``(area_idx, room_idx)``.
    Areas are kept in sorted order by directory name and can be restricted via
    ``area_names``.
    """

    def __init__(
        self,
        output_type: type[T],
        dataset_root: Path,
        max_sequence_length: int | None = None,
        perspective_loader_threads: int = 1,
        area_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()

        area_dirs = _discover_area_dirs(dataset_root)
        assert len(area_dirs) > 0, f"No area directories found under {dataset_root}"

        if area_names is not None:
            allowed = set(area_names)
            area_dirs = [p for p in area_dirs if p.name in allowed]
            assert len(area_dirs) == len(allowed), "Missing requested area directories"

        self._areas: list[Stanford2D3DAreaDataset[T]] = [
            Stanford2D3DAreaDataset(
                output_type=output_type,
                area_dir=area_dir,
                max_sequence_length=max_sequence_length,
                perspective_loader_threads=perspective_loader_threads,
            )
            for area_dir in area_dirs
        ]

        self._area_dirs = area_dirs
        self._area_lengths: list[int] = [len(area_ds) for area_ds in self._areas]
        self._area_offsets: list[int] = []
        total = 0
        for length in self._area_lengths:
            self._area_offsets.append(total)
            total += length
        self._length = total
        self._area_end_offsets: list[int] = [start + length for start, length in zip(self._area_offsets, self._area_lengths)]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length

    def index_to_area_room(self, idx: int) -> tuple[int, int]:
        """Map global ``idx`` to ``(area_idx, room_idx)`` within that area."""

        assert 0 <= idx < self._length

        area_idx = bisect_right(self._area_end_offsets, idx)
        start = 0 if area_idx == 0 else self._area_end_offsets[area_idx - 1]
        return area_idx, idx - start

    def area_room_to_index(self, area_idx: int, room_idx: int) -> int:
        """Map an ``(area_idx, room_idx)`` pair to the global dataset index."""

        assert 0 <= area_idx < len(self._areas)
        assert 0 <= room_idx < self._area_lengths[area_idx]

        return self._area_offsets[area_idx] + room_idx

    def __getitem__(self, idx: int) -> T:
        area_idx, room_idx = self.index_to_area_room(idx)
        return self._areas[area_idx][room_idx]

    def __iter__(self) -> Iterator[T]:
        for area in self._areas:
            for room_sample in area:
                yield room_sample

    def get_perspective(self, area_idx: int, room_idx: int) -> SceneSample:
        """Load perspective views for ``room_idx`` inside ``area_idx``."""

        assert 0 <= area_idx < len(self._areas)
        return self._areas[area_idx].get_perspective(room_idx)


__all__ = [
    "Stanford2D3DAreaDataset",
    "Stanford2d3dDataset",
]

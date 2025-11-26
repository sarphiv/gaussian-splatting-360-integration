from __future__ import annotations

from typing import Iterator, cast, Sequence, Callable
from pathlib import Path
import json

import torch as th
from torch.utils.data import IterableDataset
from torchvision.io import decode_image, ImageReadMode
from joblib import Parallel, delayed
from loguru import logger

from splat_init.data.datamodule_360 import SceneSample, SceneSampleLazy


class ThreeSixtyLocDataset[T: (SceneSample, SceneSampleLazy)](IterableDataset[T]):
    def __init__(self, output_type: type[T], data_dir: Path, stride: int = 1, depth_required: bool = True, worker_count: int = 1) -> None:
        super().__init__()

        self.output_type = output_type
        self.data_dir = data_dir
        self.stride = stride
        self.depth_only = depth_required
        self.worker_count = worker_count

        search_dirs = [Path("query_360"), Path("mapping")]
        area_dirs = [f for f in self.data_dir.iterdir() if f.is_dir()]
        scene_dirs = [seq_dir for a in area_dirs for s in search_dirs if (a / s).is_dir() for seq_dir in (a / s).iterdir() if seq_dir.is_dir()]
        scene_dirs = [seq_dir for seq_dir in scene_dirs if (seq_dir / "depth").is_dir() or not self.depth_only]

        self.scene_ids = [f"360-loc.{seq_dir.parent.parent.name}.{seq_dir.name.replace('_', '-')}" for seq_dir in scene_dirs]
        self.rgb_paths = {
            id: sorted((seq_dir / "image").glob("*.jpg"))[::self.stride]
            for id, seq_dir
            in zip(self.scene_ids, scene_dirs)
        }
        self.depth_paths = {
            id: (sorted((seq_dir / "depth").glob("*.png")) if (seq_dir / "depth").is_dir() else [None] * len(self.rgb_paths[id]))[::self.stride]
            for id, seq_dir
            in zip(self.scene_ids, scene_dirs)
        }
        # Read poses as world to camera matrices
        self.poses = {
            id: [th.tensor(v).inverse() for _, v in sorted(json.loads((seq_dir / "camera_pose.json").read_text()).items(), key=lambda kv: kv[0])][::self.stride]
            for id, seq_dir
            in zip(self.scene_ids, scene_dirs)
        }


    def __len__(self) -> int:
        return len(self.scene_ids)


    @staticmethod
    def _load_rgba(path: Path) -> th.Tensor:
        rgb = decode_image(str(path), mode=ImageReadMode.RGB).float() / 255.0
        alpha = th.ones((1, rgb.shape[1], rgb.shape[2]), dtype=rgb.dtype)
        return th.cat([rgb, alpha], dim=0)

    @staticmethod
    def _load_depth(path: Path, default_shape: tuple[int, int, int]) -> th.Tensor:
        if path is not None:
            # Convert depth to be consistent with the pose units.
            # NOTE: The paper does not document this factor anywhere,
            #  so the factor was found through trial and error of aligning of point clouds.
            #  A red container was found in scene index 2, and approximate measurements lead to a width of 6.0,
            #  while the true container is likely 5.9 meters long. The units are therefore likely meters.
            return decode_image(str(path), mode=ImageReadMode.GRAY).float() * 0.01
        else:
            return th.full(default_shape, float("inf"))

    @staticmethod
    def _load_to_tensor(tasks, worker_count) -> th.Tensor:
        return th.stack(cast(list[th.Tensor], Parallel(n_jobs=worker_count, backend="threading")(tasks)))
    
    @staticmethod
    def _make_item_getter(scene_id: str, rgb_paths: list[Path], depth_paths: list[Path] | list[None], poses: list[th.Tensor], worker_count: int) -> Callable[[Sequence[int]], SceneSample]:
        def getter(indices: Sequence[int]) -> SceneSample:
            load_rgba_tasks = (delayed(ThreeSixtyLocDataset._load_rgba)(p) for p in [rgb_paths[i] for i in indices])
            rgba = ThreeSixtyLocDataset._load_to_tensor(load_rgba_tasks, worker_count)

            load_depth_tasks = (delayed(ThreeSixtyLocDataset._load_depth)(p, (1, *rgba.shape[2:])) for p in [depth_paths[i] for i in indices])
            depth = ThreeSixtyLocDataset._load_to_tensor(load_depth_tasks, worker_count)

            pose = th.stack([poses[i] for i in indices])

            return SceneSample(
                id=scene_id,
                rgba=rgba,
                depth=depth,
                pose=pose,
                focal_length=None
        )

        return getter


    def __getitem__(self, idx: int) -> T:
        scene_id = self.scene_ids[idx]

        loader = self._make_item_getter(
            scene_id,
            self.rgb_paths[scene_id],
            self.depth_paths[scene_id],
            self.poses[scene_id],
            self.worker_count
        )

        if self.output_type is SceneSampleLazy:
            output = SceneSampleLazy(
                id=scene_id,
                loader=loader,
                length=len(self.poses[scene_id])
            )
        elif self.output_type is SceneSample:
            output = loader(range(len(self.poses[scene_id])))
        else:
            raise TypeError(f"Unsupported dataset item type: {T}")

        return cast(T, output)

    
    def __iter__(self) -> Iterator[T]:
        for idx in range(len(self)):
            yield self[idx]


    def load_poses(self, idx: int) -> th.Tensor:
        return th.stack(self.poses[self.scene_ids[idx]])
    
    def load_rgba(self, idx: int, seq_idx: int) -> th.Tensor:
        return self._load_rgba(self.rgb_paths[self.scene_ids[idx]][seq_idx])

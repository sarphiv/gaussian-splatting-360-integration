from __future__ import annotations

from typing import cast
from pathlib import Path
import json
from os import environ

import torch as th
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from joblib import Parallel, delayed
from loguru import logger

from splat_init.data.datamodule_360 import SceneSample


class ThreeSixtyLocDataset(Dataset[SceneSample]):
    def __init__(self, data_dir: Path, stride: int = 1, depth_required: bool = True, worker_count: int = 1) -> None:
        super().__init__()

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

    def _load_to_tensor(self, tasks) -> th.Tensor:
        return th.stack(cast(list[th.Tensor], Parallel(n_jobs=self.worker_count, backend="threading")(tasks)))

    def __getitem__(self, idx: int) -> SceneSample:
        scene_id = self.scene_ids[idx]

        load_rgba_tasks = (delayed(self._load_rgba)(p) for p in self.rgb_paths[scene_id])
        rgba = self._load_to_tensor(load_rgba_tasks)

        load_depth_tasks = (delayed(self._load_depth)(p, (1, *rgba.shape[2:])) for p in self.depth_paths[scene_id])
        depth = self._load_to_tensor(load_depth_tasks)

        pose = th.stack(self.poses[scene_id])

        return SceneSample(
            id=scene_id,
            rgba=rgba,
            depth=depth,
            pose=pose,
            focal_length=None
        )

    def load_poses(self, idx: int) -> th.Tensor:
        return th.stack(self.poses[self.scene_ids[idx]])
    
    def load_rgba(self, idx: int, seq_idx: int) -> th.Tensor:
        return self._load_rgba(self.rgb_paths[self.scene_ids[idx]][seq_idx])



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logger.info("Initializing dataset")
    ds = ThreeSixtyLocDataset(Path(environ.get("DATASET_360_LOC_ROOT", "")), stride=20, worker_count=4)

    logger.info("Loading data")
    img = ds[0].depth[0].permute(1, 2, 0).numpy()
    
    logger.info("Plotting image")
    # plt.imshow(img)
    plt.hist(img.flatten())
    plt.show()
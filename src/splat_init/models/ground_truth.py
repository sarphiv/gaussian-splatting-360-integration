"""Ground-truth pose model for evaluation."""

from __future__ import annotations

import torch as th
from lightning.pytorch import LightningModule

from splat_init.data.datamodule_360 import SceneSampleLazy
from splat_init.data.stanford_2d_3d import Stanford2d3dDataset
from splat_init.data.threesixty_loc import ThreeSixtyLocDataset


class GroundTruthPose(LightningModule):
    """Return per-scene ground-truth poses in evaluation order."""

    def __init__(
        self,
        dataset: ThreeSixtyLocDataset[SceneSampleLazy] | Stanford2d3dDataset[SceneSampleLazy],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self.scene_lengths: list[int] = []
        self.scene_poses: list[th.Tensor] = []

        for scene in dataset:
            self.scene_lengths.append(len(scene))
            self.scene_poses.append(scene.poses[:])
            assert self.scene_poses[-1].shape[0] == self.scene_lengths[-1], "Pose count must match scene length"

        self._forward_calls: int = 0

    def forward(self, images: th.Tensor) -> tuple[th.Tensor, None, dict[str, th.Tensor]]:
        """Return the stored poses for the next scene in evaluation order."""
        batch, sequence_length = images.shape[:2]
        scene_length = self.scene_lengths[self._forward_calls]

        assert self._forward_calls < len(self.scene_poses), "GroundTruthPose called more times than available scenes"
        assert images.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        assert batch == 1, "Only supports batch size 1"
        assert sequence_length == scene_length

        self._forward_calls += 1
        return self.scene_poses[self._forward_calls - 1].to(images).unsqueeze(0), None, {}

    def configure_optimizers(self):
        return []

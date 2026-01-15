"""Ground-truth pose model for evaluation."""

from __future__ import annotations

from typing import cast

import torch as th
from lightning.pytorch import LightningModule

from splat_init.data.datamodule_360 import SceneSample, SceneSampleLazy
from splat_init.data.stanford_2d_3d import Stanford2d3dDataset
from splat_init.data.threesixty_loc import ThreeSixtyLocDataset
from utilities.keypoints import sample_keypoints_from_depth


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
        self.scene_keypoints: list[list[tuple[th.Tensor, th.Tensor]]] = []

        for scene in dataset:
            if isinstance(scene, SceneSampleLazy):
                scene_sample = scene[:]
            else:
                scene_sample = cast(SceneSample, scene)

            self.scene_lengths.append(len(scene_sample))
            self.scene_poses.append(scene_sample.pose)
            self.scene_keypoints.append(
                sample_keypoints_from_depth(
                    scene_sample.pose,
                    scene_sample.rgba,
                    scene_sample.depth,
                    sample_ratio=0.001,
                )
            )
            assert self.scene_poses[-1].shape[0] == self.scene_lengths[-1], "Pose count must match scene length"

        self._forward_calls: int = 0

    def forward(
        self,
        images: th.Tensor,
    ) -> tuple[th.Tensor, list[tuple[th.Tensor, th.Tensor]], dict[str, th.Tensor]]:
        """Return the stored poses for the next scene in evaluation order."""
        batch, sequence_length = images.shape[:2]
        scene_length = self.scene_lengths[self._forward_calls]

        assert self._forward_calls < len(self.scene_poses), "GroundTruthPose called more times than available scenes"
        assert images.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        assert batch == 1, "Only supports batch size 1"
        assert sequence_length == scene_length

        scene_idx = self._forward_calls
        self._forward_calls += 1
        device = images.device
        keypoints = [
            (xy.to(device=device), xyz.to(device=device))
            for xy, xyz in self.scene_keypoints[scene_idx]
        ]
        return self.scene_poses[scene_idx].to(images).unsqueeze(0), keypoints, {}

    def configure_optimizers(self):
        return []

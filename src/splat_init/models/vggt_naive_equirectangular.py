"""Naive VGGT variant that ingests equirectangular panoramas directly.

This module reproduces VGGT's standard image preprocessing in-memory so that
SceneSample tensors can be fed straight into the pretrained VGGT model
without touching the filesystem. The resulting predictions are evaluated with
the same rotation, translation, and depth losses used by the perspective
baseline while also dumping position traces for downstream analysis.
"""

from __future__ import annotations

from typing import cast
from pathlib import Path

import torch as th
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from vggt.models.vggt import VGGT

from configs.constants import VGGT_TARGET_SIZE
from utilities.keypoints import keypoints_from_depth
from utilities.pose import quat_to_mat_xyzw



class VggtNaiveEquirectangular(LightningModule):
    """LightningModule wrapping VGGT for direct equirectangular supervision."""

    def __init__(
        self,
        model_url: Path = Path("facebook/VGGT-1B")
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = VGGT.from_pretrained(
            model_url,
            enable_point=False, # NOTE: Using depth head to get keypoints instead, similar to other models (except Colmap)
            enable_track=False,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _center_crop_height(tensor: th.Tensor, target_height: int) -> th.Tensor:
        """Centre crop the height if the tensor is taller than ``target_height``."""

        excess = tensor.shape[-2] - target_height
        if excess <= 0:
            return tensor
        top = excess // 2
        bottom = top + target_height
        return tensor[..., top:bottom, :]

    def _preprocess_rgba_tensor(self, rgba: th.Tensor) -> th.Tensor:
        """Apply alpha masking, resizing, and cropping to RGBA tensors.

        Accepts tensors shaped ``[..., 4, H, W]`` where leading dimensions capture
        batch and sequence axes.
        """

        assert rgba.shape[-3] == 4, "RGBA tensor must contain four channels"

        *leading, _, height, width = rgba.shape
        leading_shape = tuple(leading)

        scale = VGGT_TARGET_SIZE / float(width)
        new_height = int(max(14, round(height * scale / 14.0) * 14))

        rgb = rgba[..., :3, :, :] * rgba[..., 3:4, :, :]

        rgb = rgb.reshape(-1, 3, height, width)
        rgb = F.interpolate(rgb, size=(new_height, VGGT_TARGET_SIZE), mode="bilinear", align_corners=False)
        rgb = self._center_crop_height(rgb, VGGT_TARGET_SIZE)
        rgb = rgb.clamp(0.0, 1.0)

        proc_height, proc_width = rgb.shape[-2:]
        return rgb.reshape(*leading_shape, 3, proc_height, proc_width)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _gather_predictions(
        self,
        preds: dict[str, th.Tensor]
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Slice VGGT outputs to match the number of views."""

        pose = preds["pose_enc"]  # [B, S, 9]
        depth = preds["depth"].squeeze(-1).unsqueeze(2)  # [B, S, 1, H, W]
        depth_conf = preds["depth_conf"].unsqueeze(2)  # [B, S, 1, H, W]
        return pose, depth, depth_conf

    @staticmethod
    def _from_pose_encoding(pose_enc: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Decode VGGT pose encoding into quaternions, translations, and FoVs."""

        quat = pose_enc[..., 3:7]
        translation = pose_enc[..., :3]
        fov = pose_enc[..., 7:9]
        return quat, translation, fov

    def _pose_matrices_from_encoding(self, pose: th.Tensor) -> th.Tensor:
        """Convert pose encoding into homogeneous transformation matrices."""

        quat, translation, _ = self._from_pose_encoding(pose)
        rotation = quat_to_mat_xyzw(quat)

        mats = th.zeros((*rotation.shape[:-2], 4, 4), device=pose.device, dtype=pose.dtype)
        mats[..., :3, :3] = rotation
        mats[..., :3, 3] = translation
        mats[..., 3, 3] = 1.0
        return mats

    def forward(
        self, images: th.Tensor
    ) -> tuple[th.Tensor, list[tuple[th.Tensor, th.Tensor]] | None, dict[str, th.Tensor]]:
        """Forward pass returning pose matrices and depth-derived keypoints."""

        preds = self.model(self._preprocess_rgba_tensor(images).to(device=self.device, dtype=cast(th.dtype, self.dtype)))
        pose_enc, depth_pred, depth_conf = self._gather_predictions(preds)
        pose_mats_pred = self._pose_matrices_from_encoding(pose_enc).to(images)

        keypoints = keypoints_from_depth(
            pose_mats_pred,
            depth_pred.to(images),
            depth_conf.to(images),
            image_shape=tuple(images.shape[-2:]), # type: ignore[reportArgumentType]
            confidence_threshold=1.0,
            sample_ratio=0.0001
        )

        return pose_mats_pred, keypoints, {}

    def configure_optimizers(self):
        return []

"""Naive VGGT variant that ingests equirectangular panoramas directly.

This module reproduces VGGT's standard image preprocessing in-memory so that
SceneSample tensors can be fed straight into the pretrained VGGT model
without touching the filesystem. The resulting predictions are evaluated with
the same rotation, translation, and depth losses used by the perspective
baseline while also dumping position traces for downstream analysis.
"""

from __future__ import annotations

from typing import cast
from dataclasses import dataclass
from pathlib import Path

import torch as th
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, STEP_OUTPUT
from vggt.models.vggt import VGGT

from configs.constants import TRAIN_PREFIX, VALIDATION_PREFIX, VGGT_TARGET_SIZE
from splat_init.data.datamodule_360 import SceneSample
from utilities.pose import camera_centers, quat_to_mat_xyzw


@dataclass
class _ProcessedSample:
    """Container holding tensors that share the same spatial resolution."""

    rgb: th.Tensor    # [S, 3, H, W]
    depth: th.Tensor  # [S, 1, H, W]
    alpha: th.Tensor  # [S, 1, H, W]
    pose: th.Tensor   # [S, 4, 4]


class VggtNaiveEquirectangular(LightningModule):
    """LightningModule wrapping VGGT for direct equirectangular supervision."""

    def __init__(
        self,
        model_url: Path = Path("facebook/VGGT-1B"),
        output_dir: Path | None = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = VGGT.from_pretrained(
            model_url,
            enable_point=False,
            enable_track=False,
        )
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_height(height: int, width: int, target_width: int) -> int:
        """Return the height after resizing the width to ``target_width``.

        VGGT's loader ensures dimensions remain divisible by 14; the same rule is
        applied here to mimic its behaviour.
        """

        scale = target_width / float(width)
        new_height = max(14, round(height * scale / 14.0) * 14)
        return int(new_height)

    @staticmethod
    def _center_crop_height(tensor: th.Tensor, target_height: int) -> th.Tensor:
        """Centre crop the height if the tensor is taller than ``target_height``."""

        excess = tensor.shape[-2] - target_height
        if excess <= 0:
            return tensor
        top = excess // 2
        bottom = top + target_height
        return tensor[..., top:bottom, :]

    def _preprocess_rgba_tensor(self, rgba: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Apply alpha masking, resizing, and cropping to RGBA tensors.

        Accepts tensors shaped ``[..., 4, H, W]`` where leading dimensions capture
        batch and sequence axes.
        """

        assert rgba.shape[-3] == 4, "RGBA tensor must contain four channels"

        *leading, _, height, width = rgba.shape
        leading_shape = tuple(leading)
        rgba = rgba.to(device=self.device, dtype=cast(th.dtype, self.dtype))

        rgb = rgba[..., :3, :, :]
        alpha = rgba[..., 3:4, :, :]
        rgb = rgb * alpha

        new_height = self._resize_height(height, width, VGGT_TARGET_SIZE)

        rgb = rgb.reshape(-1, 3, height, width)
        alpha = alpha.reshape(-1, 1, height, width)

        rgb = F.interpolate(rgb, size=(new_height, VGGT_TARGET_SIZE), mode="bilinear", align_corners=False)
        alpha = F.interpolate(alpha, size=(new_height, VGGT_TARGET_SIZE), mode="bilinear", align_corners=False)

        rgb = self._center_crop_height(rgb, VGGT_TARGET_SIZE)
        alpha = self._center_crop_height(alpha, VGGT_TARGET_SIZE)

        rgb = rgb.clamp(0.0, 1.0)
        alpha = alpha.clamp(0.0, 1.0)

        proc_height, proc_width = rgb.shape[-2:]
        rgb = rgb.reshape(*leading_shape, 3, proc_height, proc_width)
        alpha = alpha.reshape(*leading_shape, 1, proc_height, proc_width)
        return rgb, alpha

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _prepare_forward_inputs(self, images: th.Tensor) -> th.Tensor:
        """Normalize raw RGBA batches to the format expected by VGGT."""

        assert images.dim() == 5, "Expected [B, S, C, H, W] input"
        _, _, channels, height, width = images.shape
        assert channels in (3, 4), "Input must have three (RGB) or four (RGBA) channels"

        if channels == 4:
            rgb, _ = self._preprocess_rgba_tensor(images)
        else:
            assert height == VGGT_TARGET_SIZE and width == VGGT_TARGET_SIZE, "RGB inputs must already be preprocessed"
            rgb = images.to(device=self.device, dtype=cast(th.dtype, self.dtype)).clamp(0.0, 1.0)

        return rgb

    def _gather_predictions(
        self,
        preds: dict[str, th.Tensor]
    ) -> tuple[th.Tensor, th.Tensor]:
        """Slice VGGT outputs to match the number of views."""

        pose = preds["pose_enc"]  # [B, S, 9]
        depth = preds["depth"].squeeze(-1).unsqueeze(2)  # [B, S, 1, H, W]
        return pose, depth

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

    def forward(self, images: th.Tensor) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """Forward pass returning pose matrices and depth predictions."""

        rgb_inputs = self._prepare_forward_inputs(images)

        preds = self.model(rgb_inputs)
        pose_enc, depth_pred = self._gather_predictions(preds)
        pose_mats_pred = self._pose_matrices_from_encoding(pose_enc)

        return pose_mats_pred.to(images), depth_pred.to(images), {}

    def configure_optimizers(self):
        return []

"""Naive VGGT variant that ingests equirectangular panoramas directly.

This module reproduces VGGT's standard image preprocessing in-memory so that
RoomSample360 tensors can be fed straight into the pretrained VGGT model
without touching the filesystem. The resulting predictions are evaluated with
the same rotation, translation, and depth losses used by the perspective
baseline while also dumping position traces for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch as th
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, STEP_OUTPUT
from vggt.models.vggt import VGGT

from configs.constants import TRAIN_PREFIX, VALIDATION_PREFIX, VGGT_TARGET_SIZE
from splat_init.data.datamodule_360 import RoomSample360


@dataclass
class _ProcessedSample:
    """Container holding tensors that share the same spatial resolution."""

    rgb: th.Tensor  # [S, 3, H, W]
    depth: th.Tensor  # [S, 1, H, W]
    alpha: th.Tensor  # [S, 1, H, W]
    pose: th.Tensor  # [S, 4, 4]


class VggtNaiveEquirectangular(LightningModule):
    """LightningModule wrapping VGGT for direct equirectangular supervision."""

    def __init__(self, model_url: Path = Path("facebook/VGGT-1B")) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = VGGT.from_pretrained(model_url)
        self.model.eval()

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

    def _preprocess_sample(self, sample: RoomSample360) -> _ProcessedSample:
        """Apply VGGT's crop/pad preprocessing in-memory to one room sample."""

        rgba = sample.rgba.to(device=self.device, dtype=th.float32)
        depth = sample.depth.to(device=self.device, dtype=th.float32)
        pose = sample.pose.to(device=self.device, dtype=th.float32)

        assert rgba.dim() == 4 and depth.dim() == 4, "Expected [S,C,H,W] tensors"
        assert rgba.shape[0] == depth.shape[0] == pose.shape[0]
        assert rgba.shape[1] == 4, "RGBA tensor must contain four channels"
        assert depth.shape[1] == 1, "Depth tensor must contain a single channel"

        rgb = rgba[:, :3]
        alpha = rgba[:, 3:4]

        rgb = rgb * alpha

        _, _, height, width = rgb.shape
        new_height = self._resize_height(height, width, VGGT_TARGET_SIZE)

        rgb = F.interpolate(rgb, size=(new_height, VGGT_TARGET_SIZE), mode="bilinear", align_corners=False)
        alpha = F.interpolate(alpha, size=(new_height, VGGT_TARGET_SIZE), mode="bilinear", align_corners=False)
        depth = F.interpolate(depth, size=(new_height, VGGT_TARGET_SIZE), mode="bilinear", align_corners=False)

        rgb = self._center_crop_height(rgb, VGGT_TARGET_SIZE)
        alpha = self._center_crop_height(alpha, VGGT_TARGET_SIZE)
        depth = self._center_crop_height(depth, VGGT_TARGET_SIZE)

        rgb = rgb.clamp(0.0, 1.0)
        alpha = alpha.clamp(0.0, 1.0)

        return _ProcessedSample(rgb=rgb, depth=depth, alpha=alpha, pose=pose)

    @staticmethod
    def _pad_to_height(tensor: th.Tensor, target_height: int, pad_value: float) -> th.Tensor:
        """Symmetrically pad a tensor along the height dimension."""

        height = tensor.shape[-2]
        if height == target_height:
            return tensor
        pad_total = target_height - height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padding = (0, 0, pad_top, pad_bottom)
        return F.pad(tensor, padding, value=pad_value)

    def _prepare_batch(
        self, batch: list[RoomSample360]
    ) -> tuple[th.Tensor, list[_ProcessedSample], list[int]]:
        """Pads panoramas to a common shape and stacks them for VGGT."""

        processed: list[_ProcessedSample] = [self._preprocess_sample(sample) for sample in batch]
        max_height = max(item.rgb.shape[-2] for item in processed)
        max_views = max(item.rgb.shape[0] for item in processed)

        for item in processed:
            if item.rgb.shape[-2] != max_height:
                item.rgb = self._pad_to_height(item.rgb, max_height, pad_value=1.0)
                item.depth = self._pad_to_height(item.depth, max_height, pad_value=0.0)
                item.alpha = self._pad_to_height(item.alpha, max_height, pad_value=0.0)

        batch_size = len(processed)
        device = processed[0].rgb.device
        dtype = processed[0].rgb.dtype
        depth_dtype = processed[0].depth.dtype
        alpha_dtype = processed[0].alpha.dtype

        rgb_batch = th.ones(
            (batch_size, max_views, 3, max_height, VGGT_TARGET_SIZE), device=device, dtype=dtype
        )
        depth_batch = th.zeros(
            (batch_size, max_views, 1, max_height, VGGT_TARGET_SIZE), device=device, dtype=depth_dtype
        )
        alpha_batch = th.zeros(
            (batch_size, max_views, 1, max_height, VGGT_TARGET_SIZE), device=device, dtype=alpha_dtype
        )

        view_counts: list[int] = []
        for idx, item in enumerate(processed):
            views = item.rgb.shape[0]
            rgb_batch[idx, :views] = item.rgb
            depth_batch[idx, :views] = item.depth
            alpha_batch[idx, :views] = item.alpha
            view_counts.append(views)

        for idx, item in enumerate(processed):
            item.rgb = rgb_batch[idx, : view_counts[idx]]
            item.depth = depth_batch[idx, : view_counts[idx]]
            item.alpha = alpha_batch[idx, : view_counts[idx]]

        return rgb_batch, processed, view_counts

    # ------------------------------------------------------------------
    # Pose utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_mat_xyzw(quat: th.Tensor) -> th.Tensor:
        """Convert quaternions (x, y, z, w) with scalar-last layout to rotation matrices."""

        eps = th.finfo(quat.dtype).eps
        quat_norm = quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)
        x, y, z, w = th.unbind(quat_norm, dim=-1)

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        m00 = 1.0 - 2.0 * (yy + zz)
        m11 = 1.0 - 2.0 * (xx + zz)
        m22 = 1.0 - 2.0 * (xx + yy)

        m01 = 2.0 * (xy - wz)
        m10 = 2.0 * (xy + wz)

        m02 = 2.0 * (xz + wy)
        m20 = 2.0 * (xz - wy)

        m12 = 2.0 * (yz - wx)
        m21 = 2.0 * (yz + wx)

        row0 = th.stack((m00, m01, m02), dim=-1)
        row1 = th.stack((m10, m11, m12), dim=-1)
        row2 = th.stack((m20, m21, m22), dim=-1)
        return th.stack((row0, row1, row2), dim=1)

    @staticmethod
    def _geodesic_so3(R_gt: th.Tensor, R_pred: th.Tensor) -> th.Tensor:
        """Return the geodesic distance (radians) between rotation matrices."""

        delta = R_gt.transpose(-1, -2) @ R_pred
        trace = th.diagonal(delta, dim1=-2, dim2=-1).sum(dim=-1)
        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        return th.acos(cos_theta)

    @staticmethod
    def _from_pose_encoding(pose_enc: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Decode VGGT pose encoding into quaternions, translations, and FoVs."""

        quat = pose_enc[..., 3:7]
        translation = pose_enc[..., :3]
        fov = pose_enc[..., 7:9]
        return quat, translation, fov

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def forward(self, images: th.Tensor) -> dict[str, th.Tensor]:
        """Forward pass through VGGT."""

        return self.model(images)

    def _compute_depth_loss(self, depth_gt: th.Tensor, depth_pred: th.Tensor, alpha: th.Tensor) -> th.Tensor:
        """Mean squared error on depth masked by alpha and saturation threshold."""

        mask = (alpha > 0.5).float() * (depth_gt < 0.99).float()
        residual = (depth_gt - depth_pred) ** 2
        return (residual * mask).mean()

    def _gather_predictions(
        self,
        preds: dict[str, th.Tensor],
        view_counts: Iterable[int],
    ) -> tuple[list[th.Tensor], list[th.Tensor]]:
        """Slice VGGT outputs to match the original view counts."""

        pose = preds["pose_enc"]  # [B, V, 9]
        depth = preds["depth"]    # [B, V, H, W, 1]

        pose_list = [pose[idx, :count] for idx, count in enumerate(view_counts)]
        depth_list = [depth[idx, :count].squeeze(-1).unsqueeze(1) for idx, count in enumerate(view_counts)]
        return pose_list, depth_list

    def _pose_matrices_from_encoding(self, pose: th.Tensor) -> th.Tensor:
        """Convert pose encoding into homogeneous transformation matrices."""

        quat, translation, _ = self._from_pose_encoding(pose)
        rotation = self._quat_to_mat_xyzw(quat)

        mats = th.zeros((*rotation.shape[:-2], 4, 4), device=rotation.device, dtype=rotation.dtype)
        mats[..., :3, :3] = rotation
        mats[..., :3, 3] = translation
        mats[..., 3, 3] = 1.0
        return mats

    @staticmethod
    def _relative_rotations(mats: th.Tensor) -> th.Tensor:
        """Return rotations relative to the first pose."""

        ref_inv = th.linalg.inv(mats[0])
        relative = mats @ ref_inv
        return relative[:, :3, :3]

    @staticmethod
    def _camera_centers(mats: th.Tensor) -> th.Tensor:
        """Compute camera centres in world coordinates from pose matrices."""

        inv = th.linalg.inv(mats)
        return inv[..., :3, 3]

    def _write_positions(self, stage: str, preds_t: th.Tensor, target_t: th.Tensor) -> None:
        """Append predicted and target translations to disk for inspection."""

        pred_path = Path(f"positions_pred_{stage}.txt")
        target_path = Path(f"positions_target_{stage}.txt")
        for path, tensor in ((pred_path, preds_t), (target_path, target_t)):
            assert tensor.shape[-1] == 3, "Expected 3D translation vectors"
            flat = tensor.detach().cpu().reshape(-1, 3)
            with path.open("a", encoding="utf-8") as handle:
                for vector in flat:
                    x, y, z = vector.tolist()
                    handle.write(f"{x:.6f}, {y:.6f}, {z:.6f}\n")
                handle.write("---\n")

    def _shared_step(self, batch: list[RoomSample360], stage: str) -> dict[str, th.Tensor]:
        """Compute losses and auxiliary metrics for one step."""

        assert len(batch) > 0, "Batch must not be empty"

        rgb_inputs, processed, view_counts = self._prepare_batch(batch)
        preds = self.forward(rgb_inputs)

        pose_sequences, depth_sequences = self._gather_predictions(preds, view_counts)

        depth_losses, rot_losses, trans_losses = [], [], []
        pred_logs: list[th.Tensor] = []
        target_logs: list[th.Tensor] = []

        for pose_enc, depth_out, item, count in zip(pose_sequences, depth_sequences, processed, view_counts):
            gt_depth = item.depth[:count]
            alpha = item.alpha[:count]

            loss_depth = self._compute_depth_loss(gt_depth, depth_out, alpha)
            depth_losses.append(loss_depth)

            pose_mats = item.pose[:count]
            target_rot = self._relative_rotations(pose_mats)
            target_centers = self._camera_centers(pose_mats)
            target_centers_rel = target_centers - target_centers[:1]

            pose_mats_pred = self._pose_matrices_from_encoding(pose_enc)
            pred_rot_rel = self._relative_rotations(pose_mats_pred)
            pred_centers = self._camera_centers(pose_mats_pred)
            pred_centers_rel = pred_centers - pred_centers[:1]

            if count > 1:
                geodesic = self._geodesic_so3(target_rot[1:], pred_rot_rel[1:]).mean()
                translation_loss = ((target_centers_rel[1:] - pred_centers_rel[1:]) ** 2).mean()
            else:
                zero = loss_depth.new_zeros(())
                geodesic = zero
                translation_loss = zero

            rot_losses.append(geodesic)
            trans_losses.append(translation_loss)

            pred_logs.append(pred_centers_rel)
            target_logs.append(target_centers_rel)

        loss_depth = th.stack(depth_losses).mean()
        loss_rot = th.stack(rot_losses).mean()
        loss_trans = th.stack(trans_losses).mean()

        if pred_logs:
            pred_concat = th.cat(pred_logs, dim=0)
            target_concat = th.cat(target_logs, dim=0)
            self._write_positions(stage, pred_concat, target_concat)

        loss = 0.2 * loss_depth + 0.4 * loss_rot + 0.4 * loss_trans

        prefix = TRAIN_PREFIX if stage == "train" else VALIDATION_PREFIX
        metrics = {
            "loss": loss,
            f"{prefix}_loss": loss.detach(),
            f"{prefix}_loss_depth": loss_depth.detach(),
            f"{prefix}_loss_r": loss_rot.detach(),
            f"{prefix}_loss_t": loss_trans.detach(),
        }
        return metrics

    def training_step(self, batch: list[RoomSample360], batch_idx: int) -> STEP_OUTPUT:
        metrics = self._shared_step(batch, stage="train")
        self.log_dict({k: v for k, v in metrics.items() if k != "loss"}, prog_bar=True, on_step=True)
        return metrics["loss"]

    def validation_step(self, batch: list[RoomSample360], batch_idx: int) -> STEP_OUTPUT:
        metrics = self._shared_step(batch, stage="val")
        self.log_dict({k: v for k, v in metrics.items() if k != "loss"}, prog_bar=True, on_step=False)
        return metrics["loss"]

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = th.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = th.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        return OptimizerLRSchedulerConfig(optimizer=optimizer, lr_scheduler=scheduler)

"""VGGT wrapper that projects equirectangular panoramas to perspective faces.

The module mirrors the structure of the naive equirectangular implementation
but inserts a cubemap projection stage before feeding images into VGGT. Depth
and pose supervision follow the same loss stack and logging conventions for
consistency across initialisation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch as th
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, STEP_OUTPUT
from vggt.models.vggt import VGGT

from configs.constants import TRAIN_PREFIX, VALIDATION_PREFIX, VGGT_TARGET_SIZE
from splat_init.data.datamodule_360 import SceneSample
from utilities.otc_projector import OTCProjector, cube_face_relative_rotations
from utilities.pose import mat_to_quat_xyzw, mean_quaternion_markley, pose_to_mat, quat_to_mat_xyzw


@dataclass
class _ProjectedSample:
    """Perspective faces derived from an equirectangular panorama."""

    rgb: th.Tensor  # [V, 6, 3, F, F]
    depth: th.Tensor  # [V, 6, 1, F, F]
    alpha: th.Tensor  # [V, 6, 1, F, F]
    pose: th.Tensor  # [V, 4, 4]


class VggtPerspectiveTransform(LightningModule):
    """LightningModule wrapping VGGT for perspective face supervision."""

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
        self.model.eval()
        self.model.requires_grad_(False)
        
        self.output_dir = output_dir

        self._projector = OTCProjector(face_size=VGGT_TARGET_SIZE, alpha=1e-9)
        face_weights = th.tensor([0.25, 0.25, 0.25, 0.25], dtype=th.float32)
        self.face_weights: th.Tensor
        self.register_buffer("face_weights", face_weights, persistent=False)
        
        self._face_rots: th.Tensor
        self.register_buffer("_face_rots", cube_face_relative_rotations()[[0, 1, 4, 5]], persistent=False)

        self.depth_frames_chunk_size = 2

    def _ensure_model_dtype(self, images: th.Tensor) -> th.dtype:
        """Move VGGT to the image device and cast to inference precision."""
        self.model.to(device=images.device, dtype=th.bfloat16)
        return th.bfloat16

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def _project_sample(self, sample: SceneSample) -> _ProjectedSample:
        rgba = sample.rgba.to(device=self.device, dtype=th.float32)
        depth = sample.depth.to(device=self.device, dtype=th.float32)
        pose = sample.pose.to(device=self.device, dtype=th.float32)

        rgb_faces, alpha_faces, depth_faces = self._projector(rgba, depth)
        rgb_faces = rgb_faces * alpha_faces
        depth_faces = depth_faces * alpha_faces

        return _ProjectedSample(
            rgb=rgb_faces,
            depth=depth_faces,
            alpha=alpha_faces,
            pose=pose,
        )

    def _prepare_vggt_input(self, projected: _ProjectedSample) -> th.Tensor:
        projected.rgb = projected.rgb[:, [0, 1, 4, 5], ...]
        projected.depth = projected.depth[:, [0, 1, 4, 5], ...]
        projected.alpha = projected.alpha[:, [0, 1, 4, 5], ...]
        views = projected.rgb.shape[0]
        faces = projected.rgb.reshape(1, views * len(self.face_weights), 3, VGGT_TARGET_SIZE, VGGT_TARGET_SIZE)
        if faces.is_cuda:
            faces = faces.to(dtype=th.bfloat16)
        return faces

    def _forward_vggt(self, images: th.Tensor) -> dict[str, th.Tensor]:
        assert self.model.camera_head is not None, "VGGT missing camera head"
        assert self.model.depth_head is not None, "VGGT missing depth head"
        
        target_dtype = self._ensure_model_dtype(images)
        autocast_dtype = target_dtype
        device_type = images.device.type

        with th.inference_mode():
            with th.autocast(device_type=device_type, dtype=autocast_dtype, enabled=True):
                token_sequences, patch_start_idx = self.model.aggregator(images)

                pose_list = self.model.camera_head(token_sequences)
                pose = pose_list[-1]
                del pose_list

                depth, depth_conf = self.model.depth_head(
                    token_sequences,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=self.depth_frames_chunk_size,
                )
                del depth_conf

        del token_sequences

        pose = pose.float()
        depth = depth.float()
        return {"pose_enc": pose, "depth": depth}

    # ------------------------------------------------------------------
    # Pose utilities
    # ------------------------------------------------------------------


    @staticmethod
    def _geodesic_so3(target: th.Tensor, predicted: th.Tensor) -> th.Tensor:
        delta = target.transpose(-1, -2) @ predicted
        trace = th.diagonal(delta, dim1=-2, dim2=-1).sum(dim=-1)
        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        return th.acos(cos_theta)

    @staticmethod
    def _from_pose_encoding(pose_enc: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        translation = pose_enc[..., :3]
        quat = pose_enc[..., 3:7]
        fov = pose_enc[..., 7:9]
        return quat, translation, fov

    def _mean_rotation_markley(self, quat: th.Tensor) -> th.Tensor:
        # Rotate face to canonical orientation
        quat = mat_to_quat_xyzw(
            quat_to_mat_xyzw(quat) @ self._face_rots.permute(0, 2, 1)[None, ...].to(quat)
        )
        weights = (self.face_weights / self.face_weights.sum()).to(quat)
        dominant = mean_quaternion_markley(quat, weights=weights)
        return quat_to_mat_xyzw(dominant)

    def _write_poses(
        self,
        id: str,
        rots: th.Tensor,
        trans: th.Tensor, 
        rots_faces: th.Tensor,
        trans_faces: th.Tensor
    ) -> None:
        # If no output, skip writing poses
        if self.output_dir is None:
            return

        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f"{id}.pt"

        # Prepare pose matrices
        seq_len = rots.shape[0]
        face_len = rots_faces.shape[1]
        scale = th.tensor([0, 0, 0, 1], dtype=th.float32)

        # TODO: Verify that this way of indexing is correct
        poses = th.empty([seq_len, 4, 4], dtype=th.float32)
        poses[:, :3, :3] = rots.to("cpu", th.float32)
        poses[:, :3, 3] = trans.to("cpu", th.float32)
        poses[:, 3, :] = scale

        # TODO: Verify that this way of indexing is correct
        poses_faces = th.empty([seq_len, face_len, 4, 4], dtype=th.float32)
        poses_faces[:, :, :3, :3] = rots_faces.to("cpu", th.float32)
        poses_faces[:, :, :3, 3] = trans_faces.to("cpu", th.float32)
        poses_faces[:, :, 3, :] = scale
        
        order_persp = ("+X", "-X", "+Z", "-Z")
        assert len(order_persp) == len(self.face_weights), "Unexpected amount of perspective faces"

        # Store as dictionary
        th.save(
            {
                "poses": poses,
                "poses_faces": poses_faces,
                "order_faces": order_persp,
            },
            output_file
        )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def forward(self, images: th.Tensor) -> tuple[th.Tensor, None, dict[str, th.Tensor]]:
        """Project panoramas, run VGGT, and merge face poses into camera transforms."""

        assert images.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        batch, views, channels, height, width = images.shape
        assert channels in (3, 4), "Channels must be 3 (RGB) or 4 (RGBA)"
        assert views > 1, "Need at least two views per sample"
        assert batch == 1, "Batch size > 1 not supported"

        images = images.to(device=self.device, dtype=th.float32)

        if channels == 3:
            alpha = th.ones((batch, views, 1, height, width), device=images.device, dtype=images.dtype)
            rgba = th.cat((images, alpha), dim=2)
        else:
            rgba = images

        flat_rgba = rgba.reshape(batch * views, 4, height, width)

        rgb_faces, alpha_faces, depth_faces = self._projector(flat_rgba, depth=None)
        rgb_faces = rgb_faces * alpha_faces
        depth_faces = depth_faces * alpha_faces

        projected = _ProjectedSample(
            rgb=rgb_faces,
            depth=depth_faces,
            alpha=alpha_faces,
            pose=th.zeros((batch * views, 4, 4), device=rgb_faces.device, dtype=rgb_faces.dtype),
        )
        vggt_input = self._prepare_vggt_input(projected)

        preds = self._forward_vggt(vggt_input)
        num_faces = len(self.face_weights)

        pose_faces = preds["pose_enc"][0].view(views, num_faces, -1)
        depth_pred_faces = preds["depth"][0].view(views, num_faces, VGGT_TARGET_SIZE, VGGT_TARGET_SIZE)

        quat_faces, translation_faces, _ = self._from_pose_encoding(pose_faces)
        rotation_faces = quat_to_mat_xyzw(quat_faces)

        rotation_merged = self._mean_rotation_markley(quat_faces)
        weights = self.face_weights.to(translation_faces).view(1, num_faces, 1)

        centers_faces = -(rotation_faces.transpose(-1, -2) @ translation_faces.unsqueeze(-1)).squeeze(-1)
        centers_merged = (centers_faces * weights).sum(dim=1)
        translation_merged = -(rotation_merged @ centers_merged.unsqueeze(-1)).squeeze(-1)

        ref_rot = rotation_merged[0]
        rotation_rel = rotation_merged @ ref_rot.transpose(-1, -2)
        centers_rel = centers_merged - centers_merged[:1]
        translation_rel = -(rotation_merged @ centers_rel.unsqueeze(-1)).squeeze(-1)

        mats_pred_rel = pose_to_mat(rotation_rel, translation_rel)

        return mats_pred_rel.unsqueeze(0), None, {
            "depth_faces": depth_pred_faces.unsqueeze(0),
            "pose_faces": pose_faces.unsqueeze(0),
            "centers_rel": centers_rel.unsqueeze(0),
            "rotation_merged": rotation_merged.unsqueeze(0),
            "translation_merged": translation_merged.unsqueeze(0),
        }

    def _shared_step(self, batch: list[SceneSample], stage: str) -> dict[str, th.Tensor]:
        assert len(batch) == 1, "Batch size > 1 not supported yet"
        sample = batch[0]

        projected = self._project_sample(sample)

        assert sample.rgba.ndim == 4, "Expected sample RGBA shaped [V, C, H, W]"
        assert sample.rgba.shape[0] > 1, "Need at least two views per sample"

        mats_pred, _, outputs = self.forward(sample.rgba.unsqueeze(0))
        assert mats_pred.shape[0] == 1, "Forward expected to return batch dimension"

        depth_faces_batch = outputs["depth_faces"]
        pose_faces_batch = outputs.get("pose_faces")
        centers_rel_batch = outputs.get("centers_rel")
        rotation_merged_batch = outputs.get("rotation_merged")
        translation_merged_batch = outputs.get("translation_merged")
        assert pose_faces_batch is not None, "forward must return pose_faces for training"
        assert centers_rel_batch is not None, "forward must return centers_rel for training"
        assert rotation_merged_batch is not None and translation_merged_batch is not None, "forward must return merged poses for training"

        depth_faces = depth_faces_batch[0]
        pose_faces = pose_faces_batch[0]
        centers_pred_rel = centers_rel_batch[0]
        rotation_merged = rotation_merged_batch[0]
        translation_merged = translation_merged_batch[0]
        views = pose_faces.shape[0]
        assert views > 1, "Need at least two views per sample"
        assert depth_faces.shape[0] == views, "Depth predictions view mismatch"

        quat_faces, translation_faces, _ = self._from_pose_encoding(pose_faces)
        rotation_faces = quat_to_mat_xyzw(quat_faces)

        pose_ref_inv = th.linalg.inv(projected.pose[0])
        pose_rel = projected.pose @ pose_ref_inv
        target_rot = pose_rel[:, :3, :3]

        target_centers = th.linalg.inv(projected.pose)[..., :3, 3]
        centers_target_rel = target_centers - target_centers[:1]

        depth_gt = projected.depth[:, [0, 1, 4, 5], ...]
        alpha = projected.alpha[:, [0, 1, 4, 5], ...]
        mask = (alpha > 0.5).float() * (depth_gt < 0.99).float()
        depth_residual = (depth_gt - depth_faces.unsqueeze(2)) ** 2
        depth_loss = (depth_residual * mask).mean()

        rot_loss = self._geodesic_so3(target_rot[1:], rotation_merged[1:]).mean()
        trans_loss = ((centers_target_rel[1:] - centers_pred_rel[1:]) ** 2).mean()

        total_loss = 0.2 * depth_loss + 0.4 * rot_loss + 0.4 * trans_loss

        self._write_poses(sample.id, rotation_merged, translation_merged, rotation_faces, translation_faces)

        prefix = TRAIN_PREFIX if stage == "train" else VALIDATION_PREFIX
        metrics = {
            "loss": total_loss,
            f"{prefix}_loss": total_loss.detach(),
            f"{prefix}_loss_depth": depth_loss.detach(),
            f"{prefix}_loss_r": rot_loss.detach(),
            f"{prefix}_loss_t": trans_loss.detach(),
        }
        return metrics

    def training_step(self, batch: list[SceneSample], batch_idx: int) -> STEP_OUTPUT:
        metrics = self._shared_step(batch, stage="train")
        assert len(batch) == 1, "Batch size > 1 not supported yet"
        self.log_dict({k: v for k, v in metrics.items() if k != "loss"}, prog_bar=True, on_step=True, batch_size=1)
        return metrics["loss"]

    def validation_step(self, batch: list[SceneSample], batch_idx: int) -> STEP_OUTPUT:
        metrics = self._shared_step(batch, stage="val")
        assert len(batch) == 1, "Batch size > 1 not supported yet"
        self.log_dict({k: v for k, v in metrics.items() if k != "loss"}, prog_bar=True, on_step=False, batch_size=1)
        return metrics["loss"]

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = th.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = th.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        return OptimizerLRSchedulerConfig(optimizer=optimizer, lr_scheduler=scheduler)

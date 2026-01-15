"""VGGT wrapper that projects equirectangular panoramas to perspective faces.

The module mirrors the structure of the naive equirectangular implementation
but inserts a cubemap projection stage before feeding images into VGGT. Depth
and pose supervision follow the same loss stack and logging conventions for
consistency across initialisation experiments.
"""

from __future__ import annotations

from typing import cast
from pathlib import Path

import torch as th
from lightning.pytorch import LightningModule
from vggt.models.vggt import VGGT

from configs.constants import VGGT_TARGET_SIZE
from utilities.cube_projector import CubeProjector
from utilities.keypoints import keypoints_from_depth
from utilities.otc_projector import cube_face_relative_rotations
from utilities.pose import mat_to_quat_xyzw, mean_quaternion_markley, pose_to_mat, quat_to_mat_xyzw

FACE_INDICES = (0, 1, 4, 5)


class VggtPerspectiveTransform(LightningModule):
    """LightningModule wrapping VGGT for perspective face supervision."""

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

        self._projector = CubeProjector(face_size=VGGT_TARGET_SIZE, face_up=False, face_down=False)
        face_weights = th.tensor([0.25, 0.25, 0.25, 0.25], dtype=cast(th.dtype, self.dtype))
        self.face_weights: th.Tensor
        self.register_buffer("face_weights", face_weights, persistent=False)
        
        self._face_rots: th.Tensor
        self.register_buffer("_face_rots", cube_face_relative_rotations()[list(FACE_INDICES)], persistent=False)

        self.depth_frames_chunk_size = 2

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def _prepare_vggt_input(self, rgb: th.Tensor, depth: th.Tensor, alpha: th.Tensor) -> th.Tensor:
        """Select the cubemap faces used by VGGT and pack them into a batch tensor."""

        assert rgb.shape[:2] == depth.shape[:2] == alpha.shape[:2], "RGB/depth/alpha faces must align"
        assert rgb.shape[1] == len(self.face_weights), "Expected four cubemap faces"

        views = rgb.shape[0]
        faces = rgb.reshape(1, views * len(self.face_weights), 3, VGGT_TARGET_SIZE, VGGT_TARGET_SIZE)
        faces = faces.to(dtype=th.bfloat16) if faces.is_cuda else faces
        return faces

    def _forward_vggt(self, images: th.Tensor) -> dict[str, th.Tensor]:
        assert self.model.camera_head is not None, "VGGT missing camera head"
        assert self.model.depth_head is not None, "VGGT missing depth head"
        
        autocast_dtype = th.bfloat16 if images.is_cuda else images.dtype
        self.model.to(device=images.device, dtype=autocast_dtype)
        device_type = images.device.type

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

        del token_sequences

        pose = pose.float()
        depth = depth.float().squeeze(-1).unsqueeze(2)
        depth_conf = depth_conf.float().unsqueeze(2)
        return {"pose_enc": pose, "depth": depth, "depth_conf": depth_conf}

    # ------------------------------------------------------------------
    # Pose utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _from_pose_encoding(pose_enc: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        translation = pose_enc[..., :3]
        quat = pose_enc[..., 3:7]
        fov = pose_enc[..., 7:9]
        return quat, translation, fov

    def _mean_rotation_markley(self, quat: th.Tensor) -> th.Tensor:
        # Rotate face to canonical orientation
        quat = mat_to_quat_xyzw(
            self._face_rots[None, ...].to(quat) @ quat_to_mat_xyzw(quat)
        )
        weights = (self.face_weights / self.face_weights.sum()).to(quat)
        dominant = mean_quaternion_markley(quat, weights=weights)
        return quat_to_mat_xyzw(dominant)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def forward(
        self, images: th.Tensor
    ) -> tuple[th.Tensor, list[tuple[th.Tensor, th.Tensor]] | None, dict[str, th.Tensor]]:
        """Project panoramas, run VGGT, and merge face poses into camera transforms."""

        assert images.dim() == 5, "Expected images shaped [B, S, C, H, W]"
        batch, views, channels, height, width = images.shape
        assert channels in (3, 4), "Channels must be 3 (RGB) or 4 (RGBA)"
        assert views > 1, "Need at least two views per sample"
        assert batch == 1, "Batch size > 1 not supported"

        flat_rgba = images.reshape(batch * views, 4, height, width).to(device=self.device, dtype=cast(th.dtype, self.dtype))
        rgb_faces, alpha_faces, depth_faces = self._projector(flat_rgba, depth=None)
        rgb_faces = rgb_faces * alpha_faces
        depth_faces = depth_faces * alpha_faces

        vggt_input = self._prepare_vggt_input(
            rgb_faces,  # [V, 4, 3, F, F]
            depth_faces,  # [V, 4, 1, F, F]
            alpha_faces,  # [V, 4, 1, F, F]
        )

        preds = self._forward_vggt(vggt_input)
        num_faces = len(self.face_weights)

        pose_faces = preds["pose_enc"][0] \
            .view(views, num_faces, -1) \
            .to(images)
        depth_pred_faces = preds["depth"][0] \
            .view(views, num_faces, 1, VGGT_TARGET_SIZE, VGGT_TARGET_SIZE) \
            .to(images)
        depth_conf_faces = preds["depth_conf"][0] \
            .view(views, num_faces, 1, VGGT_TARGET_SIZE, VGGT_TARGET_SIZE) \
            .to(images)

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

        depth_equi, conf_equi = self._projector.inverse(
            depth_pred_faces.to(device=self.device, dtype=cast(th.dtype, self.dtype)),
            depth_conf_faces.to(device=self.device, dtype=cast(th.dtype, self.dtype)),
            output_size=(height, width)
        )
        depth_equi, conf_equi = depth_equi.to(images), conf_equi.to(images)

        keypoints = keypoints_from_depth(
            mats_pred_rel.unsqueeze(0),
            depth_equi.unsqueeze(0),
            conf_equi.unsqueeze(0),
            image_shape=tuple(images.shape[-2:]), # type: ignore[reportArgumentType]
            confidence_threshold=2.4,
            sample_ratio=0.01
        )

        depth_pred_faces_out = depth_pred_faces.squeeze(2)

        return mats_pred_rel.unsqueeze(0), keypoints, {
            "depth_faces": depth_pred_faces_out.unsqueeze(0),
            "pose_faces": pose_faces.unsqueeze(0),
            "centers_rel": centers_rel.unsqueeze(0),
            "rotation_merged": rotation_merged.unsqueeze(0),
            "translation_merged": translation_merged.unsqueeze(0),
        }

    def configure_optimizers(self):
        return []

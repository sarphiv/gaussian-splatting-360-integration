"""VGGT wrapper that projects equirectangular panoramas to perspective faces.

The module mirrors the structure of the naive equirectangular implementation
but inserts a cubemap projection stage before feeding images into VGGT. Depth
and pose supervision follow the same loss stack and logging conventions for
consistency across initialisation experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch as th
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, STEP_OUTPUT
from vggt.models.vggt import VGGT

from configs.constants import TRAIN_PREFIX, VALIDATION_PREFIX, VGGT_TARGET_SIZE
from splat_init.data.datamodule_360 import SceneSample

_FACE_ORDER = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def cube_face_relative_rotations() -> th.Tensor:
    """
    Frames are right-handed with +X right, +Y down, +Z forward.
    Returns:
        R_i [6, 3, 3] such that R_i_face = R @ R_i,
        where R is the cannonical orientation and R_i_face is the face orientation.
    """
    ex = th.tensor([1.,0.,0.])
    ey = th.tensor([0.,1.,0.])
    ez = th.tensor([0.,0.,1.])

    def M(c1, c2, c3):
        return th.stack((c1, c2, c3), dim=-1)  # [3,3] with columns r,u,f

    faces = [
        M(-ez,  ey,  ex),  # +X
        M( ez,  ey, -ex),  # -X
        M( ex, -ez,  ey),  # +Y
        M( ex,  ez, -ey),  # -Y
        M( ex,  ey,  ez),  # +Z
        M(-ex,  ey, -ez),  # -Z
    ]

    return th.stack(faces, dim=-3)  # [6,3,3]


@dataclass
class _ProjectedSample:
    """Perspective faces derived from an equirectangular panorama."""

    rgb: th.Tensor  # [V, 6, 3, F, F]
    depth: th.Tensor  # [V, 6, 1, F, F]
    alpha: th.Tensor  # [V, 6, 1, F, F]
    pose: th.Tensor  # [V, 4, 4]


class OTCProjector:
    """Project equirectangular tensors to optimized tangens cube faces."""

    def __init__(self, face_size: int, alpha: float = 0.8687) -> None:
        self.face_size = int(face_size)
        self.alpha = float(alpha)
        self._grid: th.Tensor | None = None
        self._grid_device: th.device | None = None
        self._grid_dtype: th.dtype | None = None

    def _dir_for_face(self, u: th.Tensor, v: th.Tensor, face: str) -> th.Tensor:
        one = th.ones_like(u)
        if face == "+X":
            x, y, z = one, -v, -u
        elif face == "-X":
            x, y, z = -one, -v, u
        elif face == "+Y":
            x, y, z = u, -one, -v
        elif face == "-Y":
            x, y, z = u, one, v
        elif face == "+Z":
            x, y, z = u, -v, one
        elif face == "-Z":
            x, y, z = -u, -v, -one
        else:  # pragma: no cover - defensive against typos
            raise ValueError(f"Unknown face '{face}'")
        stack = th.stack((x, y, z), dim=0)
        return stack / stack.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def _dirs_to_lonlat(self, direction: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
        lon = th.atan2(x, z)
        lat = th.atan2(y, th.sqrt(x * x + z * z))
        return lon, lat

    @staticmethod
    def _wrap_periodic(x: th.Tensor) -> th.Tensor:
        return x - 2.0 * th.floor((x + 1.0) / 2.0)

    def _ensure_grid(self, device: th.device, dtype: th.dtype) -> th.Tensor:
        if self._grid is not None and self._grid_device == device and self._grid_dtype == dtype:
            return self._grid

        face = self.face_size
        g = th.linspace(-1.0, 1.0, face, device=device, dtype=dtype)
        v_lin, u_lin = th.meshgrid(g, g, indexing="ij")
        tan_alpha = math.tan(self.alpha)
        u = th.tan(self.alpha * u_lin) / tan_alpha
        v = th.tan(self.alpha * v_lin) / tan_alpha

        directions = th.stack(
            [self._dir_for_face(u, v, face_name) for face_name in _FACE_ORDER], dim=0
        )  # [6, 3, F, F]
        lon, lat = self._dirs_to_lonlat(directions)
        x = self._wrap_periodic(lon / math.pi)
        y = -2.0 * lat / math.pi
        grid = th.stack((x, y), dim=-1)  # [6, F, F, 2]

        self._grid = grid
        self._grid_device = device
        self._grid_dtype = dtype
        return grid

    def __call__(
        self,
        rgba: th.Tensor,
        depth: th.Tensor | None,
        *,
        alpha_mode: str = "nearest",
        depth_mode: str = "bilinear",
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if rgba.dim() != 4 or rgba.shape[1] < 4:
            raise ValueError("Expected RGBA tensor [B,4,H,W]")
        batch = rgba.shape[0]
        device, dtype = rgba.device, rgba.dtype
        grid = self._ensure_grid(device=device, dtype=dtype)

        alpha_idx = rgba.shape[1] - 1
        rgb = rgba[:, :alpha_idx]
        alpha = rgba[:, alpha_idx : alpha_idx + 1]

        def _sample(tensor: th.Tensor, mode: str) -> th.Tensor:
            faces = []
            for face_idx in range(6):
                face_grid = grid[face_idx].unsqueeze(0).expand(batch, -1, -1, -1)
                faces.append(
                    F.grid_sample(
                        tensor,
                        face_grid,
                        mode=mode,
                        padding_mode="border",
                        align_corners=True,
                    )
                )
            return th.stack(faces, dim=1)

        rgb_faces = _sample(rgb, mode="bilinear")
        alpha_faces = _sample(alpha, mode="nearest" if alpha_mode == "nearest" else "bilinear")

        depth_faces = None
        if depth is not None:
            depth_tensor = depth.to(device=device, dtype=dtype)
            depth_faces = _sample(
                depth_tensor,
                mode="nearest" if depth_mode == "nearest" else "bilinear",
            )
        else:
            depth_faces = th.zeros(
                (batch, 6, 1, self.face_size, self.face_size),
                device=device,
                dtype=dtype,
            )

        return rgb_faces, alpha_faces, depth_faces


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
    def _quat_to_mat(quat: th.Tensor) -> th.Tensor:
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(th.finfo(quat.dtype).eps)
        x, y, z, w = th.unbind(quat, dim=-1)

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
        return th.stack((row0, row1, row2), dim=-2)

    @staticmethod
    def _mat_to_quat_xyzw(mat: th.Tensor) -> th.Tensor:
        # mat: (..., 3, 3) -> (..., 4) quaternion in (x, y, z, w)
        m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
        m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
        m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

        eps = th.finfo(mat.dtype).eps
        t0 = 1.0 + m00 - m11 - m22
        t1 = 1.0 - m00 + m11 - m22
        t2 = 1.0 - m00 - m11 + m22
        t3 = 1.0 + m00 + m11 + m22
        t2 = 1.0 - m00 - m11 + m22

        t = th.stack((t0, t1, t2, t3), dim=-1).clamp_min(eps)        # (..., 4)
        idx = t.argmax(dim=-1)                                       # (...)

        s = 2.0 * th.sqrt(t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)).clamp_min(eps)  # (...)

        s01 = m01 + m10; s02 = m02 + m20; s12 = m12 + m21
        d21 = m21 - m12; d20 = m02 - m20; d10 = m10 - m01

        q0 = th.stack((0.25 * s,  s01 / s,  s02 / s,  d21 / s), dim=-1)  # x largest
        q1 = th.stack((s01 / s,  0.25 * s,  s12 / s,  d20 / s), dim=-1)  # y largest
        q2 = th.stack((s02 / s,  s12 / s,  0.25 * s,  d10 / s), dim=-1)  # z largest
        q3 = th.stack((d21 / s,  d20 / s,  d10 / s,  0.25 * s), dim=-1)  # w largest

        oh = th.nn.functional.one_hot(idx, num_classes=4).to(mat.dtype)
        quat = (
            q0 * oh[..., 0].unsqueeze(-1)
          + q1 * oh[..., 1].unsqueeze(-1)
          + q2 * oh[..., 2].unsqueeze(-1)
          + q3 * oh[..., 3].unsqueeze(-1)
        )

        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)
        return quat


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
        quat = self._mat_to_quat_xyzw(self._quat_to_mat(quat) @ self._face_rots.permute(0, 2, 1)[None, ...].to(quat))
        weights = (self.face_weights / self.face_weights.sum()).to(quat)
        weight_view = weights.view(1, weights.shape[0], 1)
        weighted = quat * weight_view
        k_mat = th.einsum("vni,vnj->vij", quat, weighted)
        eigvals, eigvecs = th.linalg.eigh(k_mat.float())
        dominant = eigvecs[..., -1].to(dtype=quat.dtype)
        return self._quat_to_mat(dominant)

    @staticmethod
    def _assemble_se3(rotation: th.Tensor, translation: th.Tensor) -> th.Tensor:
        mats = rotation.new_zeros((*rotation.shape[:-2], 4, 4))
        mats[..., :3, :3] = rotation
        mats[..., :3, 3] = translation
        mats[..., 3, 3] = 1.0
        return mats

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

        assert images.dim() == 5, "Expected images shaped [B, V, C, H, W]"
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
        rotation_faces = self._quat_to_mat(quat_faces)

        rotation_merged = self._mean_rotation_markley(quat_faces)
        weights = self.face_weights.to(translation_faces).view(1, num_faces, 1)

        centers_faces = -(rotation_faces.transpose(-1, -2) @ translation_faces.unsqueeze(-1)).squeeze(-1)
        centers_merged = (centers_faces * weights).sum(dim=1)
        translation_merged = -(rotation_merged @ centers_merged.unsqueeze(-1)).squeeze(-1)

        ref_rot = rotation_merged[0]
        rotation_rel = rotation_merged @ ref_rot.transpose(-1, -2)
        centers_rel = centers_merged - centers_merged[:1]
        translation_rel = -(rotation_merged @ centers_rel.unsqueeze(-1)).squeeze(-1)

        mats_pred_rel = self._assemble_se3(rotation_rel, translation_rel)

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
        assert depth_faces_batch.shape[0] == 1 and pose_faces_batch.shape[0] == 1, "Batch size mismatch"
        assert centers_rel_batch.shape[0] == 1 and rotation_merged_batch.shape[0] == 1, "Batch size mismatch"

        depth_faces = depth_faces_batch[0]
        pose_faces = pose_faces_batch[0]
        centers_pred_rel = centers_rel_batch[0]
        rotation_merged = rotation_merged_batch[0]
        translation_merged = translation_merged_batch[0]
        views = pose_faces.shape[0]
        assert views > 1, "Need at least two views per sample"
        assert depth_faces.shape[0] == views, "Depth predictions view mismatch"

        quat_faces, translation_faces, _ = self._from_pose_encoding(pose_faces)
        rotation_faces = self._quat_to_mat(quat_faces)

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

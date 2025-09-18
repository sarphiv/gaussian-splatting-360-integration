from typing import cast
import math
from pathlib import Path

import torch as th
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from vggt.models.vggt import VGGT

from configs.constants import TRAIN_PREFIX, VALIDATION_PREFIX, VGGT_TARGET_SIZE
from splat_init.data.datamodule_360 import RoomSample360



# TODO: Refactor projector, clean up
class OTCProjectorFast:
    """
    ERP -> OTC faces. Processes RGBA and/or depth.
    - RGB bilinear; alpha & depth: nearest or bilinear.
    - channels_last, expand-reshape, optional fp16/bf16.
    """
    def __init__(self, face_size, alpha=0.8687, device: th.device = th.device("cuda"), dtype=th.float16):
        self.F = int(face_size)
        self.alpha = float(alpha)
        self.device = th.device(device)
        self.dtype = dtype
        self.grid = self._build_grid().contiguous()
        
    def _dir_for_face(self, U, V, face):
        one = th.ones_like(U)
        X, Y, Z = one, one, one
        if   face == "+X": X, Y, Z =  one, -V, -U
        elif face == "-X": X, Y, Z = -one, -V,  U
        elif face == "+Y": X, Y, Z =   U,  one,  V
        elif face == "-Y": X, Y, Z =   U, -one, -V
        elif face == "+Z": X, Y, Z =   U, -V,  one
        elif face == "-Z": X, Y, Z =  -U, -V, -one
        D = th.stack([X, Y, Z], 0)
        return D / th.linalg.norm(D, dim=0, keepdim=True).clamp_min(1e-12)
    
    def _dirs_to_lonlat(self, D):
        X, Y, Z = D[:,0], D[:,1], D[:,2]
        lon = th.atan2(X, Z)                          # [-pi,pi]
        lat = th.atan2(Y, th.sqrt(X*X + Z*Z))      # [-pi/2,pi/2]
        return lon, lat

    def _wrap_norm(self, x):
        # Wrap to [-1,1)
        return x - 2.0 * th.floor((x + 1.0) / 2.0)

    def _build_grid(self):
        
        F = self.F
        g = th.linspace(-1, 1, F, device=self.device, dtype=self.dtype)
        v_lin, u_lin = th.meshgrid(g, g, indexing="ij")
        ta = math.tan(self.alpha)
        Upre = th.tan(self.alpha * u_lin) / ta
        Vpre = th.tan(self.alpha * v_lin) / ta
        face_order = ["+X","-X","+Y","-Y","+Z","-Z"]
        D = th.stack([self._dir_for_face(Upre, Vpre, f) for f in face_order], 0)  # 6x3xF xF
        lon, lat = self._dirs_to_lonlat(D)
        x = self._wrap_norm(lon / math.pi)            # [-1,1] periodic
        y = -2.0 * lat / math.pi                 # [-1,1] clamped
        return th.stack([x, y], dim=-1)       # 6xF xF x2


    @th.no_grad()
    def __call__(self, erp_rgba, erp_depth,
                alpha_mode="nearest", depth_mode="bilinear", alpha_index=-1
                ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        B = None
        Ff = self.F
        g6 = self.grid  # [6, F, F, 2] on device,dtype from __init__
        out_rgb = out_a = out_d = None

        def _prep(x):
            x = x.to(self.device)
            if not x.dtype.is_floating_point:
                x = x.float().div_(255 if x.dtype == th.uint8 else 1)
            # keep NCHW; avoid channels_last to prevent large contiguous copies
            return x.to(self.dtype, copy=False).contiguous()

        if erp_rgba is not None:
            x = _prep(erp_rgba)                          # [B,C,H,W], fp16/bf16
            B, C, H, W = x.shape
            a_ch = alpha_index % C
            keep_idx = [i for i in range(C) if i != a_ch]
            rgb = x[:, keep_idx, :, :]                   # view or light copy
            a   = x[:, a_ch:a_ch+1, :, :]                # view

            Cr = rgb.shape[1]
            out_rgb = th.empty((B, 6, Cr, Ff, Ff), dtype=self.dtype, device=self.device)
            out_a   = th.empty((B, 6, 1,  Ff, Ff), dtype=self.dtype, device=self.device)

            a_mode = "nearest" if alpha_mode == "nearest" else "bilinear"

            # process each face without expanding the input
            for i in range(6):
                gi = g6[i].unsqueeze(0).expand(B, -1, -1, -1)   # [B,F,F,2], cheap expand
                out_rgb[:, i] = F.grid_sample(
                    rgb, gi, mode="bilinear", padding_mode="border", align_corners=True
                )
                out_a[:, i] = F.grid_sample(
                    a, gi, mode=a_mode, padding_mode="border", align_corners=True
                )

        if erp_depth is not None:
            d = _prep(erp_depth)                        # [B,1,H,W]
            Bd, Cd, Hd, Wd = d.shape
            if B is None: B = Bd
            out_d = th.empty((Bd, 6, 1, Ff, Ff), dtype=self.dtype, device=self.device)
            d_mode = "nearest" if depth_mode == "nearest" else "bilinear"
            for i in range(6):
                gi = g6[i].unsqueeze(0).expand(Bd, -1, -1, -1)
                out_d[:, i] = F.grid_sample(
                    d, gi, mode=d_mode, padding_mode="border", align_corners=True
                )

        return cast(th.Tensor, out_rgb), cast(th.Tensor, out_a), cast(th.Tensor, out_d)



class VggtPerspectiveTransform(LightningModule):
    def __init__(self, model_url: Path = Path("facebook/VGGT-1B")) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.perspective_projector: OTCProjectorFast | None = None
        self.model = VGGT.from_pretrained(model_url).eval()


    def from_equirectangular(self, rgb: th.Tensor, depth: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if not self.perspective_projector or self.perspective_projector.device != rgb.device or self.perspective_projector.dtype != rgb.dtype:
            self.perspective_projector = OTCProjectorFast(
                face_size=VGGT_TARGET_SIZE,
                device=rgb.device,
                dtype=rgb.dtype,
            )
        
        rgb_faces, alpha_faces, depth_faces = self.perspective_projector(erp_rgba=rgb,erp_depth=depth)
        
        # NOTE: Ignored regions are set to 0 so they can serve as input to VGGT
        return rgb_faces * alpha_faces, depth_faces * alpha_faces, alpha_faces
    
    
    # TODO: Refactor to follow convention
    def _quat_to_mat_xyzw(self, quat: th.Tensor) -> th.Tensor:
        """
        Convert quaternions (x, y, z, w) with scalar last to rotation matrices.
        Args:
            quat: (..., 4) tensor [qx, qy, qz, qw]
        Returns:
            (..., 3, 3) rotation matrices
        """
        assert quat.shape[-1] == 4
        x, y, z, w = th.unbind(quat, dim=-1)
        tiny = th.finfo(quat.dtype).tiny

        # shape (...,)
        n = (quat * quat).sum(dim=-1).clamp_min(tiny)
        s = 2.0 / n

        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        m00 = 1.0 - s * (yy + zz)
        m11 = 1.0 - s * (xx + zz)
        m22 = 1.0 - s * (xx + yy)

        m01 = s * (xy - wz)
        m10 = s * (xy + wz)

        m02 = s * (xz + wy)
        m20 = s * (xz - wy)

        m12 = s * (yz - wx)
        m21 = s * (yz + wx)

        R = th.stack([m00, m01, m02,
                    m10, m11, m12,
                    m20, m21, m22], dim=-1)
        return R.reshape(*quat.shape[:-1], 3, 3)


    # TODO: Refactor to follow convention
    def mean_rotation_markley(self, q: th.Tensor, w: th.Tensor | None = None) -> th.Tensor:
        """
        Markley quaternion mean across N.
        Args:
            q:    [B,S,N,4] quaternions (x, y, z, w) with scalar last
            w:    optional [B,S,N] or [N] nonnegative weights
        Returns:
            R_mean:   [B,S,3,3]
        """
        if w is None:
            w = th.ones(q.shape[:3], dtype=q.dtype, device=q.device)

        if w.dim() == 1:  # [N] -> [B,S,N]
            w = w[None, None, ...].expand(q.shape[:3])

        # K = sum_i w_i q_i q_i^T
        K = th.einsum('...ni,...nj->...ij', q, q * w[..., None])  # [B,S,4,4]
        # principal eigenvector
        evals, evecs = th.linalg.eigh(K.float())                 # ascending
        q_bar = evecs[..., -1].to(dtype=q.dtype)                              # [B,S,4], unit
        return self._quat_to_mat_xyzw(q_bar)


    # TODO: Refactor to follow convention
    def geodesic_so3_atan2(self, R_gt: th.Tensor,
                        R_pred: th.Tensor,
                        return_degrees: bool = False) -> th.Tensor:
        """
        atan2-based geodesic angle between rotation matrices.

        Inputs:
            R_gt:   [B, S, 3, 3] ground-truth rotations (SO(3))
            R_pred: [B, S, 3, 3] predicted rotations (SO(3))
            return_degrees: if True, return angles in degrees

        Output:
            theta:  [B, S] angles in [0, pi] (radians by default)
        """
        assert R_gt.shape == R_pred.shape and R_gt.shape[-2:] == (3, 3), "inputs must be [B,S,3,3]"

        # Relative rotation Δ = R_gt^T R_pred
        Delta = R_gt.transpose(-1, -2) @ R_pred  # [B,S,3,3]

        # vee( (Δ - Δ^T)/2 ) -> vector part
        skew = 0.5 * (Delta - Delta.transpose(-1, -2))
        v = th.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)  # [B,S,3]

        y = th.linalg.norm(v, dim=-1)  # ||v||
        x = 0.5 * (th.diagonal(Delta, dim1=-2, dim2=-1).sum(dim=-1) - 1.0)  # (tr(Δ) - 1)/2

        theta = th.atan2(y, x)  # [B,S], robust near 0 and pi
        if return_degrees:
            theta = theta * (180.0 / th.pi)
        return theta


    # TODO: Refactor to follow convention
    def from_pose_encoding(self, pose_enc: th.Tensor):
        """
        Decode VGGT pose encoding to rotation, translation, and FoV.

        Args:
            pose_enc: [B, S, 9] with layout
                    [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]
                    where quaternion is scalar-last and FoVs are radians.

        Returns:
            # R:    [B, S, 3, 3] rotation matrices
            quat: [B, S, 4]    quaternions (x, y, z, w) with scalar last
            T:    [B, S, 3]    translations
            fov:  [B, S, 2]    [fov_h, fov_w] in radians
        """
        T   = pose_enc[..., :3]
        quat= pose_enc[..., 3:7]
        fov = pose_enc[..., 7:9]
        # R   = self._quat_to_mat_xyzw(quat)
        # return R, T, fov
        return quat, T, fov


    def forward(self, images: th.Tensor) -> dict[str, th.Tensor]:
        with th.inference_mode(), th.autocast(device_type=str(self.device), dtype=cast(th.dtype, self.dtype)):
            return self.model.forward(images)


    def _helper_step(self, batch: list[RoomSample360], batch_idx: int, stage: str) -> dict:
        assert len(batch) == 1, "Batch size > 1 not supported yet."
        b = len(batch)
        s = len(batch[0].rgba)


        # TODO: Ensure the funky operations below yield expected outputs

        # Convert to perspective images
        rgb, depth, alpha = self.from_equirectangular(batch[0].rgba, batch[0].depth)
        depth = depth.view(b, s, 6, *depth.shape[-2:])
        alpha = alpha.view(b, s, 6, *alpha.shape[-2:])

        # Predict with VGGT
        preds = self.forward(rgb.view(b, -1, 3, rgb.shape[-2], rgb.shape[-1]))
        preds_quat, preds_t, _ = self.from_pose_encoding(preds["pose_enc"])

        # Merge predictions for each perspective
        # NOTE: Downweighting estimates for floor and ceiling faces
        weights = th.tensor([0.23, 0.23, 0.04, 0.04, 0.23, 0.23], device=self.device)
        preds_depth = preds["depth"].view(b, s, 6, VGGT_TARGET_SIZE, VGGT_TARGET_SIZE)
        preds_r_merged = self.mean_rotation_markley(preds_quat.view(b, s, 6, 4), w=weights)
        preds_t_merged = (preds_t.view(b, s, 6, 3) * weights[None, None, :, None]).sum(dim=2)

        # Make relative to first camera
        preds_r_rel = preds_r_merged @ preds_r_merged[:, :1, :].transpose(-2, -1)
        preds_t_rel = preds_t_merged - preds_t_merged[:, :1, :]

        target_rel = batch[0].pose @ batch[0].pose[:1].inverse()
        target_r_rel = target_rel[None, :, :3, :3]
        target_t_rel = target_rel[None, :, :3, 3]
        
        # TODO: Figure out how to deal with the outlier depth values
        # NOTE: Using the perspective projections for the depth loss
        # NOTE: Depth loss ignores alpha masked regions and saturated depth regions
        loss_depth = th.mean((depth - preds_depth)**2 * (alpha > 0.50) * (depth < 0.99))
        loss_r = th.mean(self.geodesic_so3_atan2(target_r_rel[:, 1:, ...], preds_r_rel[:, 1:, ...]))
        loss_t = th.mean((target_t_rel[:, 1:, ...] - preds_t_rel[:, 1:, ...])**2)
        loss = 0.2*loss_depth + 0.4*loss_r + 0.4*loss_t
        
        # Write position to file for debugging
        with open(f"positions_pred_{stage}.txt", "a") as f:
            for i in range(s):
                t = preds_t_rel[0, i].cpu().numpy()
                f.write(f"{t[0]}, {t[1]}, {t[2]}\n")
            f.write("---\n")
        with open(f"positions_target_{stage}.txt", "a") as f:
            for i in range(s):
                t = target_t_rel[0, i].cpu().numpy()
                f.write(f"{t[0]}, {t[1]}, {t[2]}\n")
            f.write("---\n")

        match stage:
            case "train": stage_prefix = TRAIN_PREFIX
            case "val":   stage_prefix = VALIDATION_PREFIX
            case _:       raise ValueError(f"Unsupported stage: {stage}")

        return {
            "loss": loss,
            "log": {
                f"{stage_prefix}_loss": loss,
                f"{stage_prefix}_loss_depth": loss_depth,
                f"{stage_prefix}_loss_r": loss_r,
                f"{stage_prefix}_loss_t": loss_t,
            },
        }

    def training_step(self, batch: list[RoomSample360], batch_idx: int) -> STEP_OUTPUT:
        output = self._helper_step(batch, batch_idx, stage="train")
        output["loss"] = None

        return output["loss"]
    
    
    def validation_step(self, batch: list[RoomSample360], batch_idx: int) -> STEP_OUTPUT:
        output = self._helper_step(batch, batch_idx, stage="val")
        output["loss"] = None
        self.log_dict(output["log"], prog_bar=True, on_step=True)

        return output["loss"]

    
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = th.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = th.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        
        
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )


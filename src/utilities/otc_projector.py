"""Project equirectangular tensors to optimized tangens cube faces."""

from __future__ import annotations

import math

import torch as th
import torch.nn.functional as F

__all__ = ["OTCProjector", "cube_face_relative_rotations"]

_FACE_ORDER = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def cube_face_relative_rotations() -> th.Tensor:
    """Return canonical rotations for cube faces (right-handed frames)."""

    ex = th.tensor([1.0, 0.0, 0.0])
    ey = th.tensor([0.0, 1.0, 0.0])
    ez = th.tensor([0.0, 0.0, 1.0])

    def M(c1: th.Tensor, c2: th.Tensor, c3: th.Tensor) -> th.Tensor:
        return th.stack((c1, c2, c3), dim=-1)  # [3,3] with columns r,u,f

    faces = [
        M(-ez, ey, ex),  # +X
        M(ez, ey, -ex),  # -X
        M(ex, -ez, ey),  # +Y
        M(ex, ez, -ey),  # -Y
        M(ex, ey, ez),  # +Z
        M(-ex, ey, -ez),  # -Z
    ]

    return th.stack(faces, dim=-3)  # [6,3,3]


class OTCProjector:
    """Project equirectangular tensors to optimized tangens cube faces."""

    def __init__(self, face_size: int, alpha: float = 0.8687) -> None:
        self.face_size = int(face_size)
        self.alpha = float(alpha)
        self._grid: th.Tensor | None = None
        self._grid_device: th.device | None = None
        self._grid_dtype: th.dtype | None = None

    def _dir_for_face(self, u: th.Tensor, v: th.Tensor, face: str) -> th.Tensor:
        """Return normalized directions for a single cube face given face plane grids."""
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

    def _dirs_to_lonlat(self, direction: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Convert direction vectors to longitude and latitude angles."""
        x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
        lon = th.atan2(x, z)
        lat = th.atan2(y, th.sqrt(x * x + z * z))
        return lon, lat

    @staticmethod
    def _wrap_periodic(x: th.Tensor) -> th.Tensor:
        """Wrap coordinates into the [-1, 1) interval for sampling."""
        return x - 2.0 * th.floor((x + 1.0) / 2.0)

    def _ensure_grid(self, device: th.device, dtype: th.dtype) -> th.Tensor:
        """Cache and return the sampling grid for the current device and dtype."""
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
        """Project RGBA (and optional depth) panoramas to cube faces.

        Args:
            rgba: Batch of RGBA panoramas shaped ``[B, 4, H, W]``.
            depth: Optional depth maps matching ``rgba`` shape.
            alpha_mode: Resampling mode for the alpha channel.
            depth_mode: Resampling mode for the depth channel.
        Returns:
            Tuple of projected ``(rgb_faces, alpha_faces, depth_faces)`` tensors.
        """
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

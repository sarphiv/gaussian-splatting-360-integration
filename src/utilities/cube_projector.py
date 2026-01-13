"""Project equirectangular tensors to standard cubemap faces and back."""

from __future__ import annotations

import math

import torch as th
import torch.nn.functional as F

__all__ = ["CubeProjector"]

_FACE_ORDER = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


class CubeProjector:
    """Project equirectangular tensors to cubemap faces with optional inverse."""

    def __init__(
        self,
        face_size: int,
        face_forward: bool = True,
        face_left: bool = True,
        face_right: bool = True,
        face_up: bool = True,
        face_down: bool = True,
        face_back: bool = True,
    ) -> None:
        self.face_size = int(face_size)
        self._face_enabled = {
            "+X": bool(face_right),
            "-X": bool(face_left),
            "+Y": bool(face_up),
            "-Y": bool(face_down),
            "+Z": bool(face_forward),
            "-Z": bool(face_back),
        }
        self._face_indices = [
            idx for idx, name in enumerate(_FACE_ORDER) if self._face_enabled[name]
        ]
        self._grid: th.Tensor | None = None
        self._grid_device: th.device | None = None
        self._grid_dtype: th.dtype | None = None
        self._inverse_cache: dict[tuple[int, int, th.device, th.dtype], tuple[list[th.Tensor], list[th.Tensor]]] = {}

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

        directions = th.stack(
            [self._dir_for_face(u_lin, v_lin, face_name) for face_name in _FACE_ORDER],
            dim=0,
        )  # [6, 3, F, F]
        lon, lat = self._dirs_to_lonlat(directions)
        x = self._wrap_periodic(lon / math.pi)
        y = -2.0 * lat / math.pi
        grid = th.stack((x, y), dim=-1)  # [6, F, F, 2]

        self._grid = grid
        self._grid_device = device
        self._grid_dtype = dtype
        return grid

    def _ensure_inverse_cache(
        self, device: th.device, dtype: th.dtype, output_size: tuple[int, int]
    ) -> tuple[list[th.Tensor], list[th.Tensor]]:
        """Cache and return inverse grids/masks for the requested output size."""
        height, width = output_size
        key = (height, width, device, dtype)
        cached = self._inverse_cache.get(key)
        if cached is not None:
            return cached

        x_lin = th.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        y_lin = th.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        y_grid, x_grid = th.meshgrid(y_lin, x_lin, indexing="ij")
        lon = math.pi * x_grid
        lat = -0.5 * math.pi * y_grid

        x_dir = th.sin(lon) * th.cos(lat)
        y_dir = th.sin(lat)
        z_dir = th.cos(lon) * th.cos(lat)

        abs_x = x_dir.abs()
        abs_y = y_dir.abs()
        abs_z = z_dir.abs()

        face_indices = th.full((height, width), -1, device=device, dtype=th.int64)
        unassigned = th.ones_like(face_indices, dtype=th.bool)

        def _assign(face_idx: int, cond: th.Tensor) -> None:
            nonlocal unassigned
            select = unassigned & cond
            face_indices[select] = face_idx
            unassigned = unassigned & ~select

        _assign(0, (x_dir >= 0) & (abs_x >= abs_y) & (abs_x >= abs_z))
        _assign(1, (x_dir < 0) & (abs_x >= abs_y) & (abs_x >= abs_z))
        _assign(2, (y_dir >= 0) & (abs_y >= abs_x) & (abs_y >= abs_z))
        _assign(3, (y_dir < 0) & (abs_y >= abs_x) & (abs_y >= abs_z))
        _assign(4, (z_dir >= 0) & (abs_z >= abs_x) & (abs_z >= abs_y))
        _assign(5, (z_dir < 0) & (abs_z >= abs_x) & (abs_z >= abs_y))

        assert (face_indices >= 0).all(), "Face assignment failed for inverse mapping."

        grids: list[th.Tensor] = []
        masks: list[th.Tensor] = []
        for face_idx, face_name in enumerate(_FACE_ORDER):
            mask = face_indices == face_idx
            u = th.zeros_like(x_dir)
            v = th.zeros_like(x_dir)
            if mask.any():
                if face_name == "+X":
                    u[mask] = -z_dir[mask] / x_dir[mask]
                    v[mask] = -y_dir[mask] / x_dir[mask]
                elif face_name == "-X":
                    u[mask] = -z_dir[mask] / x_dir[mask]
                    v[mask] = y_dir[mask] / x_dir[mask]
                elif face_name == "+Y":
                    u[mask] = -x_dir[mask] / y_dir[mask]
                    v[mask] = z_dir[mask] / y_dir[mask]
                elif face_name == "-Y":
                    u[mask] = x_dir[mask] / y_dir[mask]
                    v[mask] = z_dir[mask] / y_dir[mask]
                elif face_name == "+Z":
                    u[mask] = x_dir[mask] / z_dir[mask]
                    v[mask] = -y_dir[mask] / z_dir[mask]
                elif face_name == "-Z":
                    u[mask] = x_dir[mask] / z_dir[mask]
                    v[mask] = y_dir[mask] / z_dir[mask]
                else:  # pragma: no cover - defensive against typos
                    raise ValueError(f"Unknown face '{face_name}'")

            grids.append(th.stack((u, v), dim=-1))
            masks.append(mask.to(dtype=dtype).unsqueeze(0).unsqueeze(0))

        self._inverse_cache[key] = (grids, masks)
        return grids, masks

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
            if not self._face_indices:
                return th.zeros(
                    (batch, 0, tensor.shape[1], self.face_size, self.face_size),
                    device=device,
                    dtype=dtype,
                )
            faces = []
            for face_idx in self._face_indices:
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

        if depth is not None:
            depth_tensor = depth.to(device=device, dtype=dtype)
            depth_faces = _sample(
                depth_tensor,
                mode="nearest" if depth_mode == "nearest" else "bilinear",
            )
        else:
            depth_faces = th.zeros(
                (batch, len(self._face_indices), 1, self.face_size, self.face_size),
                device=device,
                dtype=dtype,
            )

        return rgb_faces, alpha_faces, depth_faces

    def inverse(
        self, *face_tensors: th.Tensor, output_size: tuple[int, int]
    ) -> tuple[th.Tensor, ...]:
        """Project cubemap faces back to equirectangular tensors.

        Args:
            face_tensors: Face stacks shaped ``[B, num_faces, C, F, F]``.
            output_size: Output (height, width) of the equirectangular tensor.
        Returns:
            One equirectangular tensor per input in ``face_tensors``.
        """
        if not face_tensors:
            return tuple()

        num_faces = len(self._face_indices)
        first = face_tensors[0]
        assert first.dim() == 5, "Expected face tensors shaped [B, num_faces, C, F, F]."
        assert first.shape[1] == num_faces, "Inverse expects all enabled faces."
        assert first.shape[-1] == self.face_size and first.shape[-2] == self.face_size

        batch = first.shape[0]
        device, dtype = first.device, first.dtype

        for tensor in face_tensors[1:]:
            assert tensor.device == device
            assert tensor.dtype == dtype
            assert tensor.shape[0] == batch
            assert tensor.shape[1] == num_faces
            assert tensor.shape[-1] == self.face_size and tensor.shape[-2] == self.face_size

        grids, masks = self._ensure_inverse_cache(
            device=device, dtype=dtype, output_size=output_size
        )
        height, width = output_size

        outputs: list[th.Tensor] = []
        for faces in face_tensors:
            channels = faces.shape[2]
            out = th.zeros((batch, channels, height, width), device=device, dtype=dtype)
            for face_pos, face_idx in enumerate(self._face_indices):
                face = faces[:, face_pos]
                grid = grids[face_idx].unsqueeze(0).expand(batch, -1, -1, -1)
                sample = F.grid_sample(
                    face,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                out = out + sample * masks[face_idx]
            outputs.append(out)

        return tuple(outputs)

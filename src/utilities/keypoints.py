"""Utilities for extracting keypoints from depth panoramas."""

from __future__ import annotations

import math

import torch as th
import torch.nn.functional as F

__all__ = ["keypoints_from_depth", "sample_keypoints_from_depth"]


def _ensure_channel_dim(tensor: th.Tensor, name: str) -> th.Tensor:
    """Ensure ``tensor`` has a singleton channel dimension before H/W."""

    assert tensor.dim() >= 2, f"{name} must be at least 2D [H,W]"
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.shape[-3] == 1:
        return tensor
    return tensor.unsqueeze(-2)


def _flatten_leading(tensor: th.Tensor, trailing_dims: int, name: str) -> tuple[th.Tensor, tuple[int, ...]]:
    """Flatten leading dimensions into one axis, keeping the trailing shape."""

    assert tensor.dim() >= trailing_dims, f"{name} must have at least {trailing_dims} dims"
    leading = tensor.shape[:-trailing_dims]
    flat = tensor.reshape(-1, *tensor.shape[-trailing_dims:])
    return flat, leading


def _resize_map(tensor: th.Tensor, height: int, width: int) -> th.Tensor:
    """Resize a single-channel map to the target spatial resolution."""

    if tensor.shape[-2:] == (height, width):
        return tensor
    resized = F.interpolate(
        tensor.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _points_from_indices(
    pose: th.Tensor,
    depth: th.Tensor,
    x_idx: th.Tensor,
    y_idx: th.Tensor,
    height: int,
    width: int,
) -> tuple[th.Tensor, th.Tensor]:
    """Convert selected pixel indices and depths into world-space points."""

    if x_idx.numel() == 0:
        empty_xy = th.empty((2, 0), device=depth.device, dtype=depth.dtype)
        empty_xyz = th.empty((3, 0), device=depth.device, dtype=depth.dtype)
        return empty_xy, empty_xyz

    assert height > 0 and width > 0, "Equirectangular images must be at least 1x1."

    v = (y_idx.to(dtype=depth.dtype) + 0.5) / height
    u = (x_idx.to(dtype=depth.dtype) + 0.5) / width

    lat = -0.5 * math.pi + math.pi * v
    lon = -math.pi + 2.0 * math.pi * u

    directions = th.stack(
        (
            th.cos(lat) * th.sin(lon),
            th.sin(lat),
            th.cos(lat) * th.cos(lon),
        ),
        dim=0,
    )

    depths = depth[y_idx, x_idx]
    xyz_cam = directions * depths.unsqueeze(0)

    pose = pose.to(device=depth.device, dtype=depth.dtype)
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    xyz_world = rot.transpose(0, 1) @ (xyz_cam - trans[:, None])

    xy = th.stack((x_idx, y_idx), dim=0)
    return xy, xyz_world


def keypoints_from_depth(
    poses: th.Tensor,
    rgb: th.Tensor,
    depth: th.Tensor,
    depth_confidence: th.Tensor,
    confidence_threshold: float,
) -> list[tuple[th.Tensor, th.Tensor]]:
    """Return per-image pixel coordinates and world points filtered by depth confidence."""

    assert rgb.dim() >= 3, "Expected RGB/RGBA input with channel and spatial dims"
    assert rgb.shape[-3] in (3, 4), "Expected RGB/RGBA input"
    assert poses.shape[-2:] == (4, 4), "Pose matrices must be 4x4"

    depth = _ensure_channel_dim(depth, "depth")
    depth_confidence = _ensure_channel_dim(depth_confidence, "depth_confidence")

    flat_rgb, leading = _flatten_leading(rgb, 3, "rgb")
    flat_depth, depth_leading = _flatten_leading(depth, 3, "depth")
    flat_conf, conf_leading = _flatten_leading(depth_confidence, 3, "depth_confidence")
    flat_pose, pose_leading = _flatten_leading(poses, 2, "poses")

    assert leading == depth_leading, "Depth must match RGB leading dimensions"
    assert leading == conf_leading, "Depth confidence must match RGB leading dimensions"
    assert leading == pose_leading, "Poses must match RGB leading dimensions"

    threshold = float(confidence_threshold)
    results: list[tuple[th.Tensor, th.Tensor]] = []
    for idx in range(flat_rgb.shape[0]):
        rgb_i = flat_rgb[idx]
        depth_i = flat_depth[idx].to(device=rgb_i.device)
        conf_i = flat_conf[idx].to(device=rgb_i.device)
        pose_i = flat_pose[idx]

        height, width = rgb_i.shape[-2:]
        depth_i = _resize_map(depth_i, height, width).squeeze(0)
        conf_i = _resize_map(conf_i, height, width).squeeze(0)

        valid_depth = (depth_i > 0.0) & th.isfinite(depth_i)
        valid_conf = (conf_i >= threshold) & th.isfinite(conf_i)
        mask = valid_depth & valid_conf

        indices = mask.nonzero(as_tuple=False)
        y_idx = indices[:, 0]
        x_idx = indices[:, 1]
        xy, xyz = _points_from_indices(pose_i, depth_i, x_idx, y_idx, height, width)
        results.append((xy.to(device=rgb_i.device), xyz.to(device=rgb_i.device)))

    return results


def sample_keypoints_from_depth(
    poses: th.Tensor,
    rgb: th.Tensor,
    depth: th.Tensor,
    sample_ratio: float = 0.1,
) -> list[tuple[th.Tensor, th.Tensor]]:
    """Randomly sample keypoints from valid depth values per image."""

    assert rgb.dim() >= 3, "Expected RGB/RGBA input with channel and spatial dims"
    assert rgb.shape[-3] in (3, 4), "Expected RGB/RGBA input"
    assert poses.shape[-2:] == (4, 4), "Pose matrices must be 4x4"
    assert 0.0 < sample_ratio <= 1.0, "Sample ratio must be in (0, 1]"

    depth = _ensure_channel_dim(depth, "depth")

    flat_rgb, leading = _flatten_leading(rgb, 3, "rgb")
    flat_depth, depth_leading = _flatten_leading(depth, 3, "depth")
    flat_pose, pose_leading = _flatten_leading(poses, 2, "poses")

    assert leading == depth_leading, "Depth must match RGB leading dimensions"
    assert leading == pose_leading, "Poses must match RGB leading dimensions"

    results: list[tuple[th.Tensor, th.Tensor]] = []
    for idx in range(flat_rgb.shape[0]):
        rgb_i = flat_rgb[idx]
        depth_i = flat_depth[idx].to(device=rgb_i.device)
        pose_i = flat_pose[idx]

        height, width = rgb_i.shape[-2:]
        depth_i = _resize_map(depth_i, height, width).squeeze(0)

        mask = (depth_i > 0.0) & th.isfinite(depth_i)
        indices = mask.nonzero(as_tuple=False)
        if indices.numel() == 0:
            empty_xy = th.empty((2, 0), device=rgb_i.device, dtype=depth_i.dtype)
            empty_xyz = th.empty((3, 0), device=rgb_i.device, dtype=depth_i.dtype)
            results.append((empty_xy.to(device=rgb_i.device), empty_xyz.to(device=rgb_i.device)))
            continue

        num_valid = indices.shape[0]
        num_samples = min(num_valid, math.ceil(sample_ratio * num_valid))
        perm = th.randperm(num_valid, device=indices.device)[:num_samples]
        chosen = indices[perm]
        y_idx = chosen[:, 0]
        x_idx = chosen[:, 1]
        xy, xyz = _points_from_indices(pose_i, depth_i, x_idx, y_idx, height, width)
        results.append((xy.to(device=rgb_i.device), xyz.to(device=rgb_i.device)))

    return results

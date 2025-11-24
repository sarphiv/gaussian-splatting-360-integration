from __future__ import annotations

import torch as th


def camera_centers(pose: th.Tensor) -> th.Tensor:
    """Compute camera centres from pose matrices."""
    inv = th.linalg.inv(pose)
    return inv[..., :3, 3]


def quat_to_mat_xyzw(quat: th.Tensor) -> th.Tensor:
    """Convert quaternions in (x, y, z, w) format to rotation matrices."""
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
    return th.stack((row0, row1, row2), dim=-2)


def mat_to_quat_xyzw(mat: th.Tensor) -> th.Tensor:
    """Convert rotation matrices to quaternions with (x, y, z, w) ordering."""
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

    eps = th.finfo(mat.dtype).eps
    t0 = 1.0 + m00 - m11 - m22
    t1 = 1.0 - m00 + m11 - m22
    t2 = 1.0 - m00 - m11 + m22
    t3 = 1.0 + m00 + m11 + m22

    t = th.stack((t0, t1, t2, t3), dim=-1).clamp_min(eps)
    idx = t.argmax(dim=-1)

    s = 2.0 * th.sqrt(t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)).clamp_min(eps)

    s01 = m01 + m10
    s02 = m02 + m20
    s12 = m12 + m21
    d21 = m21 - m12
    d20 = m02 - m20
    d10 = m10 - m01

    q0 = th.stack((0.25 * s, s01 / s, s02 / s, d21 / s), dim=-1)
    q1 = th.stack((s01 / s, 0.25 * s, s12 / s, d20 / s), dim=-1)
    q2 = th.stack((s02 / s, s12 / s, 0.25 * s, d10 / s), dim=-1)
    q3 = th.stack((d21 / s, d20 / s, d10 / s, 0.25 * s), dim=-1)

    oh = th.nn.functional.one_hot(idx, num_classes=4).to(mat.dtype)
    quat = (
        q0 * oh[..., 0].unsqueeze(-1)
        + q1 * oh[..., 1].unsqueeze(-1)
        + q2 * oh[..., 2].unsqueeze(-1)
        + q3 * oh[..., 3].unsqueeze(-1)
    )

    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)
    return quat


def pose_to_mat(rotation: th.Tensor, translation: th.Tensor) -> th.Tensor:
    """Assemble SE(3) transformation matrices from rotation and translation."""
    mats = rotation.new_zeros((*rotation.shape[:-2], 4, 4))
    mats[..., :3, :3] = rotation
    mats[..., :3, 3] = translation
    mats[..., 3, 3] = 1.0
    return mats


def mean_quaternion_markley(quat: th.Tensor, weights: th.Tensor | None = None) -> th.Tensor:
    """Compute the Markley mean quaternion for a stack of quaternions."""
    if weights is None:
        weights = th.ones(quat.shape[:-1], device=quat.device, dtype=quat.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    weighted = quat * weights.unsqueeze(-1)
    k_mat = th.einsum("...ni,...nj->...ij", quat, weighted)
    eigvals, eigvecs = th.linalg.eigh(k_mat.float())
    dominant = eigvecs[..., -1].to(dtype=quat.dtype)
    return dominant


def mean_rotation_markley(rotations: th.Tensor, weights: th.Tensor | None = None) -> th.Tensor:
    """Compute the Markley mean rotation matrix from a stack of rotation matrices."""
    quat = mat_to_quat_xyzw(rotations)
    dominant = mean_quaternion_markley(quat, weights)
    return quat_to_mat_xyzw(dominant)

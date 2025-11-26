from __future__ import annotations

from typing import Callable

import torch as th
from kornia.geometry.conversions import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from tqdm import tqdm

def _normalize_vector(vec: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    norm = vec.norm(dim=-1, keepdim=True).clamp_min(eps)
    return vec / norm


def _orthonormal_vector(vec: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    basis_x = th.tensor((1.0, 0.0, 0.0), device=vec.device, dtype=vec.dtype).expand_as(vec)
    basis_y = th.tensor((0.0, 1.0, 0.0), device=vec.device, dtype=vec.dtype).expand_as(vec)

    cross_x = th.cross(vec, basis_x)
    cross_y = th.cross(vec, basis_y)

    use_x = cross_x.norm(dim=-1, keepdim=True) >= cross_y.norm(dim=-1, keepdim=True)
    ortho = th.where(use_x, cross_x, cross_y)
    return _normalize_vector(ortho, eps)


def _rotate_vectors(vectors: th.Tensor, axes: th.Tensor, angles: th.Tensor) -> th.Tensor:
    """Rotate ``vectors`` by ``angles`` around ``axes`` using Rodrigues' formula."""

    angles = angles.unsqueeze(-1)
    sin_theta = th.sin(angles)
    cos_theta = th.cos(angles)

    cross_term = th.cross(axes, vectors)
    dot_term = (axes * vectors).sum(dim=-1, keepdim=True)

    return vectors * cos_theta + cross_term * sin_theta + axes * dot_term * (1.0 - cos_theta)


def _project_to_plane(vec: th.Tensor, normal: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    """Project ``vec`` onto the plane orthogonal to ``normal`` and renormalize."""

    normal = _normalize_vector(normal, eps)
    projection = vec - (vec * normal).sum(dim=-1, keepdim=True) * normal
    return _normalize_vector(projection, eps)


def camera_centers(pose: th.Tensor) -> th.Tensor:
    """Compute camera centres from pose matrices."""
    inv = th.linalg.inv(pose)
    return inv[..., :3, 3]


def pose_from_center_and_rotation(center: th.Tensor, rotation: th.Tensor) -> th.Tensor:
    """Assemble world-to-camera pose matrices from camera centres and rotations."""

    pose = rotation.new_zeros((*rotation.shape[:-2], 4, 4))
    pose[..., :3, :3] = rotation
    pose[..., :3, 3] = -(rotation @ center.unsqueeze(-1)).squeeze(-1)
    pose[..., 3, 3] = 1.0
    return pose


def pose_to_mat(rotation: th.Tensor, translation: th.Tensor) -> th.Tensor:
    """Assemble SE(3) transformation matrices from rotation and translation."""
    mats = rotation.new_zeros((*rotation.shape[:-2], 4, 4))
    mats[..., :3, :3] = rotation
    mats[..., :3, 3] = translation
    mats[..., 3, 3] = 1.0
    return mats


def relative_rotations(pose: th.Tensor) -> th.Tensor:
    """Rotation matrices relative to the first world-to-camera pose."""

    ref_inv = th.linalg.inv(pose[0])
    relative = pose @ ref_inv
    return relative[..., :3, :3]


def relative_centers(pose: th.Tensor) -> th.Tensor:
    """Camera centres relative to the first view."""

    centres = camera_centers(pose)
    return centres - centres[:1]


def geodesic_so3(R_gt: th.Tensor, R_pred: th.Tensor) -> th.Tensor:
    """Geodesic distance (radians) between rotation matrices."""

    delta = R_gt.transpose(-1, -2) @ R_pred
    trace = th.diagonal(delta, dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return th.acos(cos_theta)


def pointing_and_roll_errors(gt_rot: th.Tensor, pred_rot: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Decompose rotation error into pointing (forward) and roll components.

    Both inputs are world-to-camera rotation matrices, typically relative to a
    shared reference view. The pointing error measures the angular difference
    between camera forward axes. The roll error measures the remaining rotation
    about that forward axis after aligning the pointing directions.
    """

    assert gt_rot.shape == pred_rot.shape, "Rotation tensors must match in shape"

    gt_cam_to_world = gt_rot.transpose(-1, -2)
    pred_cam_to_world = pred_rot.transpose(-1, -2)

    f_gt = _normalize_vector(gt_cam_to_world[..., :, 2])
    f_pred = _normalize_vector(pred_cam_to_world[..., :, 2])
    u_gt = _normalize_vector(gt_cam_to_world[..., :, 1])
    u_pred = _normalize_vector(pred_cam_to_world[..., :, 1])

    pointing_error = th.acos(th.clamp((f_gt * f_pred).sum(dim=-1), -1.0, 1.0))

    raw_axis = th.cross(f_pred, f_gt)
    axis_norm = raw_axis.norm(dim=-1, keepdim=True)
    unit_axis = th.where(axis_norm > 1e-8, raw_axis / axis_norm, _orthonormal_vector(f_gt))

    u_pred_aligned = _rotate_vectors(u_pred, unit_axis, pointing_error)

    u_gt_proj = _project_to_plane(u_gt, f_gt)
    u_pred_proj = _project_to_plane(u_pred_aligned, f_gt)

    roll_error = th.acos(th.clamp((u_gt_proj * u_pred_proj).sum(dim=-1), -1.0, 1.0))
    return pointing_error, roll_error


def _procrustes_components(
    source: th.Tensor, target: th.Tensor, allow_scale: bool
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    assert source.shape == target.shape, "Source and target arrays must match in shape."

    source_centroid = source.mean(dim=0)
    target_centroid = target.mean(dim=0)

    source_centered = source - source_centroid
    target_centered = target - target_centroid

    covariance = source_centered.transpose(-1, -2) @ target_centered
    u, singular_values, vt = th.linalg.svd(covariance)

    rotation = u @ vt
    if th.linalg.det(rotation) < 0:
        vt[..., -1, :] *= -1
        rotation = u @ vt

    if allow_scale:
        scale_denominator = th.sum(source_centered ** 2)
        assert scale_denominator > 0.0, "Source scene must span more than a single point."
        scale = singular_values.sum() / scale_denominator
    else:
        scale = th.ones((), device=source.device, dtype=source.dtype)

    return rotation, scale, source_centroid, target_centroid


def procrustes_analysis(
    source: th.Tensor, target: th.Tensor, allow_scale: bool = True
) -> Callable[[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]:
    """Return an aligner that rigidly (and optionally uniformly) scales ``source`` onto ``target``.

    The returned function expects position tensors shaped ``[..., 3]`` and rotation matrices shaped
    ``[..., 3, 3]`` and applies the same alignment computed from ``source`` and ``target``.
    """

    rotation, scale, source_centroid, target_centroid = _procrustes_components(source, target, allow_scale)

    def procrustes_align(position: th.Tensor, rotation_mats: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        aligned_pos = scale * (position - source_centroid) @ rotation + target_centroid
        aligned_rot = rotation_mats @ rotation
        return aligned_pos, aligned_rot

    return procrustes_align


def procrustes_transform(
    source_pose: th.Tensor, target_pose: th.Tensor, input_pose: th.Tensor, allow_scale: bool = True
) -> th.Tensor:
    """Align ``input_pose`` using a Procrustes alignment computed from ``source_pose`` to ``target_pose``.

    Alignment is estimated on camera centres of ``source_pose`` and ``target_pose`` and then applied to
    the positions and rotations of ``input_pose``.
    """

    assert source_pose.shape == target_pose.shape, "Source and target poses must match in shape."

    source_pos = camera_centers(source_pose)
    target_pos = camera_centers(target_pose)

    align = procrustes_analysis(source_pos, target_pos, allow_scale=allow_scale)
    aligned_pos, aligned_rot = align(camera_centers(input_pose), input_pose[..., :3, :3])

    return pose_from_center_and_rotation(aligned_pos, aligned_rot)


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


def mean_rotation_karcher(rot: th.Tensor, verbose: bool = False, max_iter: int = 200, tol: float = 1e-9) -> th.Tensor:
    """Compute Karcher mean of rotation matrices via Weiszfeld iterations."""

    assert rot.shape[0] > 0, "Rotation tensor must have at least one element."
    assert len(rot.shape) == 3, "Rotation tensor must be batched."

    mean = axis_angle_to_rotation_matrix(rotation_matrix_to_axis_angle(rot).mean(dim=0)[None])

    prev_norm = None
    iterator = tqdm(range(max_iter), desc="Karcher Mean Iterations", disable=not verbose)
    for _ in iterator:
        axis_angles = rotation_matrix_to_axis_angle(rot @ mean.permute(0, 2, 1))
        delta = axis_angles.mean(dim=0)[None]
        mean = axis_angle_to_rotation_matrix(0.5 * delta) @ mean

        curr_norm = th.sqrt(th.sum(delta**2)).item()
        iterator.set_postfix(loss=curr_norm)

        if prev_norm is not None and abs(curr_norm - prev_norm) < tol:
            break
        prev_norm = curr_norm

    iterator.close()
    return mean[0]

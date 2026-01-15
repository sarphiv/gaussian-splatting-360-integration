from __future__ import annotations
from typing import Callable, cast

import torch as th
from lightning.pytorch import LightningModule
from tqdm import tqdm

from splat_init.data.datamodule_360 import SceneSample, SceneSampleLazy
from utilities.pose import (
    camera_centers,
    mean_rotation_markley,
    pose_from_center_and_rotation,
    procrustes_analysis,
)


class SequenceChunker(LightningModule):
    """Run a sequence model in overlapping chunks and merge the outputs when enabled."""

    def __init__(self, model: LightningModule, chunking: tuple[int, int] | None, verbose: bool) -> None:
        """Wrap a sequence model with optional chunked processing."""
        super().__init__()
        self.model = model
        self.chunking = chunking
        self.chunk_size, self.chunk_overlap = chunking or (0, 0)
        self.verbose = verbose

    def forward(
        self,
        sample: list[SceneSample] | list[SceneSampleLazy],
    ) -> tuple[th.Tensor, list[tuple[th.Tensor, th.Tensor]] | None, dict[str, th.Tensor]]:
        """Chunk a long sequence, align overlapping pose predictions, and fuse them."""

        assert len(sample) == 1, "Batch size greater than 1 not supported"
        scene = sample[0]
        sequence_length = len(scene)

        if self.chunking is None:
            images = self._slice_sample(scene, range(sequence_length)).rgba.unsqueeze(0)
            pose_pred, keypoints, _ = self.model.forward(images)
            return pose_pred[0], keypoints, {}

        chunk_ranges, overlap_ranges = self._chunk_ranges(sequence_length, self.chunk_size, self.chunk_overlap)
        pose_dtype = cast(th.dtype, self.model.dtype)

        pose_pred = th.zeros((sequence_length, 4, 4), dtype=pose_dtype)
        keypoints_state: list[tuple[th.Tensor, th.Tensor] | None] | None = None

        for chunk_range, overlap_range in tqdm(
            zip(chunk_ranges, overlap_ranges),
            desc="Processing chunks",
            total=len(chunk_ranges),
            leave=False,
            disable=not self.verbose,
        ):
            chunk_sample = self._slice_sample(scene, chunk_range)
            images = chunk_sample.rgba.unsqueeze(0)
            pose_chunk, keypoints_chunk, _ = self.model.forward(images)
            pose_chunk = pose_chunk[0]

            aligned_pose, aligner, overlap_rel = self._align_chunk_poses(
                pose_chunk,
                pose_pred,
                overlap_range,
                chunk_range.start,
            )
            self._write_chunk_poses(pose_pred, aligned_pose, chunk_range, overlap_range, overlap_rel)

            keypoints_state = self._merge_chunk_keypoints(
                keypoints_state,
                keypoints_chunk,
                chunk_range,
                aligner,
                sequence_length,
            )

        merged_keypoints = None
        if keypoints_state is not None:
            merged_keypoints = [
                kp if kp is not None else self._empty_keypoints(pose_pred.device, pose_pred.dtype)
                for kp in keypoints_state
            ]

        return pose_pred, merged_keypoints, {}

    @staticmethod
    def _chunk_ranges(length: int, size: int, overlap: int) -> tuple[list[range], list[range]]:
        """Return chunk and overlap ranges for the given sequence length."""
        chunk_ranges: list[range] = []
        overlap_ranges: list[range] = []

        start = 0
        prev_stop = 0
        while prev_stop < length:
            end = min(length, start + size)
            chunk_ranges.append(range(start, end))
            overlap_ranges.append(range(start, prev_stop))

            start = end - overlap
            prev_stop = end

        return chunk_ranges, overlap_ranges

    @staticmethod
    def _slice_sample(scene: SceneSample | SceneSampleLazy, idx_range: range) -> SceneSample:
        """Slice a scene sample or lazily load the requested subset."""
        slice_idx = slice(idx_range.start, idx_range.stop)

        if isinstance(scene, SceneSampleLazy):
            return scene[slice_idx]

        focal = None if scene.focal_length is None else scene.focal_length[slice_idx]

        return SceneSample(
            id=scene.id,
            rgba=scene.rgba[slice_idx],
            depth=scene.depth[slice_idx],
            pose=scene.pose[slice_idx],
            focal_length=focal,
        )

    @staticmethod
    def _align_chunk_poses(
        pose_chunk: th.Tensor,
        pose_pred: th.Tensor,
        overlap_range: range,
        start: int,
    ) -> tuple[th.Tensor, Callable[[th.Tensor, th.Tensor | None], tuple[th.Tensor, th.Tensor]], slice | None]:
        """Align chunk pose predictions to the existing overlap when available."""
        if overlap_range.start == overlap_range.stop:
            return pose_chunk, SequenceChunker._identity_aligner, None

        overlap_rel = slice(overlap_range.start - start, overlap_range.stop - start)
        overlap_target = pose_pred[overlap_range.start:overlap_range.stop]
        aligner = procrustes_analysis(
            camera_centers(pose_chunk[overlap_rel]),
            camera_centers(overlap_target),
            allow_scale=True,
        )
        aligned_centers, aligned_rots = aligner(
            camera_centers(pose_chunk),
            pose_chunk[..., :3, :3],
        )
        aligned_pose = pose_from_center_and_rotation(aligned_centers, aligned_rots)
        return aligned_pose, aligner, overlap_rel

    @staticmethod
    def _identity_aligner(position: th.Tensor, rotation_mats: th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor]:
        """Return inputs unchanged to match the Procrustes aligner signature."""
        if rotation_mats is None:
            return position, th.empty((0,), device=position.device, dtype=position.dtype)
        return position, rotation_mats

    @staticmethod
    def _write_chunk_poses(
        pose_pred: th.Tensor,
        aligned_pose: th.Tensor,
        chunk_range: range,
        overlap_range: range,
        overlap_rel: slice | None,
    ) -> None:
        """Write aligned chunk poses into the output buffer with overlap fusion."""
        start, stop = chunk_range.start, chunk_range.stop
        if overlap_rel is None:
            pose_pred[start:stop] = aligned_pose
            return

        pre_len = overlap_range.start - start
        if pre_len > 0:
            pose_pred[start:overlap_range.start] = aligned_pose[:pre_len]

        pose_pred[overlap_range.start:overlap_range.stop] = SequenceChunker._fuse_poses(
            pose_pred[overlap_range.start:overlap_range.stop],
            aligned_pose[overlap_rel],
        )

        if overlap_range.stop < stop:
            pose_pred[overlap_range.stop:stop] = aligned_pose[overlap_rel.stop:]

    @staticmethod
    def _merge_chunk_keypoints(
        keypoints_state: list[tuple[th.Tensor, th.Tensor] | None] | None,
        keypoints_chunk: list[tuple[th.Tensor, th.Tensor]] | None,
        chunk_range: range,
        aligner: Callable[[th.Tensor, th.Tensor | None], tuple[th.Tensor, th.Tensor]],
        sequence_length: int,
    ) -> list[tuple[th.Tensor, th.Tensor] | None] | None:
        """Merge chunk keypoints or return None when keypoints are disabled."""
        if keypoints_chunk is None:
            assert keypoints_state is None, "Expected keypoints to be disabled for all chunks."
            return None
        if keypoints_state is None:
            keypoints_pred: list[tuple[th.Tensor, th.Tensor] | None] = [None for _ in range(sequence_length)]
            keypoints_state = keypoints_pred
        assert keypoints_state is not None

        assert len(keypoints_chunk) == len(chunk_range), "Keypoints must match chunk length"
        for frame_idx, (xy, xyz) in zip(chunk_range, keypoints_chunk):
            if xyz.numel() > 0:
                xyz = SequenceChunker._align_keypoints(xyz, aligner)
            keypoints_state[frame_idx] = SequenceChunker._merge_keypoints(keypoints_state[frame_idx], (xy, xyz))

        return keypoints_state

    @staticmethod
    def _fuse_poses(existing: th.Tensor, incoming: th.Tensor) -> th.Tensor:
        """Fuse overlapping pose predictions using mean rotation and translation."""
        rotations = th.stack((existing[..., :3, :3], incoming[..., :3, :3]), dim=1)
        translations = th.stack((existing[..., :3, 3], incoming[..., :3, 3]), dim=1)

        fused_rot = mean_rotation_markley(rotations)
        fused_trans = translations.mean(dim=1)

        fused = existing.clone()
        fused[..., :3, :3] = fused_rot
        fused[..., :3, 3] = fused_trans
        fused[..., 3, :] = 0.0
        fused[..., 3, 3] = 1.0
        return fused

    @staticmethod
    def _align_keypoints(
        xyz: th.Tensor,
        aligner: Callable[[th.Tensor, th.Tensor | None], tuple[th.Tensor, th.Tensor]],
    ) -> th.Tensor:
        """Apply a Procrustes alignment to keypoint coordinates."""
        points = xyz.transpose(0, 1)
        identity = th.eye(3, device=points.device, dtype=points.dtype).expand(points.shape[0], -1, -1)
        aligned, _ = aligner(points, identity)
        return aligned.transpose(0, 1)

    @staticmethod
    def _merge_keypoints(
        existing: tuple[th.Tensor, th.Tensor] | None,
        incoming: tuple[th.Tensor, th.Tensor],
    ) -> tuple[th.Tensor, th.Tensor]:
        """Merge keypoints by averaging 3D points that share integer pixel coordinates."""
        if existing is None or existing[0].numel() == 0:
            return SequenceChunker._as_int32_indices(incoming[0]), incoming[1]
        if incoming[0].numel() == 0:
            return SequenceChunker._as_int32_indices(existing[0]), existing[1]

        xy = th.cat((existing[0], incoming[0]), dim=1)
        xyz = th.cat((existing[1], incoming[1]), dim=1)
        if not xy.is_floating_point() or xy.dtype in (th.float16, th.bfloat16):
            xy = xy.to(dtype=th.float32)
        xy_int = SequenceChunker._as_int32_indices(xy).transpose(0, 1)

        unique, inverse = th.unique(xy_int, dim=0, return_inverse=True)
        counts = th.bincount(inverse, minlength=unique.shape[0]).to(xy.dtype)

        xy_sum = th.zeros((2, unique.shape[0]), device=xy.device, dtype=xy.dtype)
        xyz_sum = th.zeros((3, unique.shape[0]), device=xyz.device, dtype=xyz.dtype)
        xy_sum.scatter_add_(1, inverse.unsqueeze(0).expand(2, -1), xy)
        xyz_sum.scatter_add_(1, inverse.unsqueeze(0).expand(3, -1), xyz)

        xy_avg = xy_sum / counts.unsqueeze(0)
        xyz_avg = xyz_sum / counts.unsqueeze(0)
        return SequenceChunker._as_int32_indices(xy_avg), xyz_avg

    @staticmethod
    def _as_int32_indices(xy: th.Tensor) -> th.Tensor:
        """Round keypoint indices to int32 while preserving intent."""
        if xy.numel() == 0:
            return xy.to(dtype=th.int32)
        if xy.dtype in (th.float16, th.bfloat16):
            xy = xy.to(dtype=th.float32)
        if xy.is_floating_point():
            xy = xy.round()
        return xy.to(dtype=th.int32)

    @staticmethod
    def _empty_keypoints(device: th.device, dtype: th.dtype) -> tuple[th.Tensor, th.Tensor]:
        """Return an empty keypoint pair on the requested device."""
        return (
            th.empty((2, 0), device=device, dtype=th.int32),
            th.empty((3, 0), device=device, dtype=dtype),
        )

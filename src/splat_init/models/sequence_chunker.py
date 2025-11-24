from __future__ import annotations

from typing import Tuple

import torch as th
from lightning.pytorch import LightningModule

from splat_init.data.datamodule_360 import SceneSample, SceneSampleLazy
from utilities.pose import camera_centers, mean_rotation_markley


class SequenceChunker(LightningModule):
    """Run a sequence model in overlapping chunks and merge the outputs."""

    def __init__(self, model: th.nn.Module, chunk_size: int, chunk_overlap: int) -> None:
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def forward(self, sample: list[SceneSample] | list[SceneSampleLazy]) -> Tuple[th.Tensor, None, dict[str, th.Tensor]]:
        """Chunk a long sequence, align overlapping pose predictions, and fuse them."""

        assert len(sample) == 1, "Batch size greater than 1 not supported"
        scene = sample[0]

        idx_chunks, idx_overlap = self._chunk_ranges(len(scene), self.chunk_size, self.chunk_overlap)

        pose_pred = th.zeros((len(scene), 4, 4), device=self.device, dtype=th.float32)

        for chunk_range, overlap_range in zip(idx_chunks, idx_overlap):
            chunk_sample = self._slice_sample(scene, chunk_range)
            images = chunk_sample.rgba.to(device=self.device).unsqueeze(0)
            pose_chunk, _, _ = self.model(images)
            pose_chunk = pose_chunk[0]

            start, stop = chunk_range.start, chunk_range.stop
            if len(overlap_range) == 0:
                pose_pred[start:stop] = pose_chunk
                continue

            overlap_rel = slice(overlap_range.start - start, overlap_range.stop - start)
            overlap_target = pose_pred[overlap_range.start:overlap_range.stop]
            aligned_pose = self._align_chunk(pose_chunk, overlap_rel, overlap_target)

            pre_len = overlap_range.start - start
            if pre_len > 0:
                pose_pred[start:overlap_range.start] = aligned_pose[:pre_len]

            pose_pred[overlap_range.start:overlap_range.stop] = self._fuse_poses(
                pose_pred[overlap_range.start:overlap_range.stop],
                aligned_pose[overlap_rel],
            )

            if overlap_range.stop < stop:
                pose_pred[overlap_range.stop:stop] = aligned_pose[overlap_rel.stop:]

        return pose_pred, None, {}

    @staticmethod
    def _chunk_ranges(length: int, size: int, overlap: int) -> tuple[list[range], list[range]]:
        """Return chunk and overlap ranges for the given sequence length."""
        idx_chunks: list[range] = []
        idx_overlap: list[range] = []

        start = 0
        prev_stop = 0
        while start < length:
            end = min(length, start + size)
            idx_chunks.append(range(start, end))
            idx_overlap.append(range(start, prev_stop))

            if end == length:
                break

            prev_stop = end
            start = end - overlap

        return idx_chunks, idx_overlap

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
    def _align_chunk(pose_chunk: th.Tensor, overlap_rel: slice, target: th.Tensor) -> th.Tensor:
        """Rigidly align one chunk's poses to the accumulated predictions."""
        overlap_chunk = pose_chunk[overlap_rel]
        rotation, translation = SequenceChunker._rigid_transform(
            camera_centers(overlap_chunk),
            camera_centers(target),
        )
        return SequenceChunker._apply_transform(pose_chunk, rotation, translation)

    @staticmethod
    def _rigid_transform(source: th.Tensor, target: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Return the rotation and translation aligning ``source`` points to ``target``."""
        source_mean = source.mean(dim=0)
        target_mean = target.mean(dim=0)

        source_centered = source - source_mean
        target_centered = target - target_mean
        covariance = source_centered.transpose(-1, -2) @ target_centered

        u, _, v_t = th.linalg.svd(covariance)
        rot = v_t.transpose(-1, -2) @ u.transpose(-1, -2)
        if th.linalg.det(rot) < 0:
            v_t[..., -1, :] *= -1
            rot = v_t.transpose(-1, -2) @ u.transpose(-1, -2)

        translation = target_mean - rot @ source_mean
        return rot, translation

    @staticmethod
    def _apply_transform(pose: th.Tensor, rotation: th.Tensor, translation: th.Tensor) -> th.Tensor:
        """Apply a rigid transform to a set of pose matrices."""
        rot = rotation @ pose[..., :3, :3]
        trans = (rotation @ pose[..., :3, 3].unsqueeze(-1)).squeeze(-1) + translation

        transformed = pose.clone()
        transformed[..., :3, :3] = rot
        transformed[..., :3, 3] = trans
        return transformed

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

from __future__ import annotations

from typing import Tuple
from pathlib import Path

import torch as th
from lightning.pytorch import LightningModule
from tqdm import tqdm

from splat_init.data.datamodule_360 import SceneSample, SceneSampleLazy
from utilities.pose import (
    mean_rotation_markley,
    procrustes_transform,
)


class SequenceChunker(LightningModule):
    """Run a sequence model in overlapping chunks and merge the outputs."""

    def __init__(self, model: th.nn.Module, chunk_size: int, chunk_overlap: int, verbose: bool) -> None:
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose

    def forward(self, sample: list[SceneSample] | list[SceneSampleLazy]) -> Tuple[th.Tensor, None, dict[str, th.Tensor]]:
        """Chunk a long sequence, align overlapping pose predictions, and fuse them."""

        assert len(sample) == 1, "Batch size greater than 1 not supported"
        scene = sample[0]

        idx_chunks, idx_overlap = self._chunk_ranges(len(scene), self.chunk_size, self.chunk_overlap)

        pose_pred = th.zeros((len(scene), 4, 4), device=self.device, dtype=th.float32)

        iterator = tqdm(zip(idx_chunks, idx_overlap), desc="Processing chunks", total=len(idx_chunks), leave=False, disable=not self.verbose)
        for chunk_range, overlap_range in iterator:
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
            aligned_pose = procrustes_transform(pose_chunk[overlap_rel], overlap_target, pose_chunk, allow_scale=True)

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
        while prev_stop < length:
            end = min(length, start + size)
            idx_chunks.append(range(start, end))
            idx_overlap.append(range(start, prev_stop))

            start = end - overlap
            prev_stop = end

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

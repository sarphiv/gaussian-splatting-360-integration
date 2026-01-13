# Context
Review the multi-part implementation for cube projector replacement, keypoint generation, model output changes, evaluation output updates, and COLMAP export utility.


# Review
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- This file is an initial checklist only. Report outcomes (including skipped tests) in chat.


## Scope
### In scope
- `src/utilities/cube_projector.py` correctness (face order, enabled faces, inverse output size, missing-face asserts).
- `src/utilities/*` keypoint helper correctness and memory usage.
- Model forward signatures and keypoint outputs across all models.
- `sequence_chunker.py` keypoint merge behavior.
- `evaluate_poses.py` output updates (`keypoints.pt`, metrics width/height).
- New COLMAP export utility output structure and binary file formats.

### Out of scope
- Performance benchmarking or training quality evaluation.
- Changes to datasets or vendor libraries.


## Requirements
- [ ] Cube projector forward matches OTCProjector output order and shapes for enabled faces; mapping uses standard cubemap (no tangent warp); inverse uses explicit output size and asserts if enabled faces missing.
- [ ] All models return `(poses, keypoints_list, extras)` with keypoints in pixel coords + world coords.
- [ ] VGGT/DA3 keypoints use a fixed confidence threshold of `0.9`.
- [ ] SequenceChunker merges keypoints correctly for overlapping chunks (duplicates averaged, alignment applied).
- [ ] GroundTruthPose precomputes 10% random keypoints from depth GT.
- [ ] Random sampling helper returns 10% valid-depth keypoints and is reused by ViPE + GroundTruth.
- [ ] PycolmapPerspectiveTransform includes undistort and returns keypoints mapped to equirectangular using dense/sparse/0 outputs.
- [ ] evaluate_poses saves `keypoints.pt` and image width/height metrics.
- [ ] COLMAP export utility writes `images/` and `sparse/0/{cameras,images,points3D}.bin` with valid formatting.
- [ ] Metrics or performance gates: keypoint extraction avoids large GPU memory spikes.
- [ ] Scope assertions (must remain unchanged): no changes to dataset loaders, no vendor edits.


## Verification
- [ ] Manual check: spot-check cube projector face order with a small synthetic input.
- [ ] Manual check: verify keypoint list shapes for a small batch.
- [ ] Manual check: verify COLMAP binary files parse with a known reader (if available).
- [ ] Edge case or risk area: mismatched depth/RGB sizes and empty keypoints.
- [ ] Artifact or output inspection: ensure `keypoints.pt` saved alongside `poses.pt` in eval outputs.


# Questions
-- Important questions that require clarification.

1. None.

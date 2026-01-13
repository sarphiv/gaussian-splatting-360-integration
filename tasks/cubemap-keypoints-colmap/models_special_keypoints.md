# Context
Update special-case models and wrappers: `sequence_chunker.py` (merge keypoints), `ground_truth.py` (precompute random keypoints from depth GT), and `pycolmap_perspective_transform.py` (undistort + extract keypoints mapped to equirectangular).


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- `src/splat_init/models/sequence_chunker.py`:
  - Merge keypoints from overlapping chunks by concatenating and averaging duplicates with identical pixel coords.
  - Apply the same alignment transform used for poses so final keypoints remain in the world frame of the full sequence.
  - If underlying model returns `None`/empty keypoints, skip merge logic.
- `src/splat_init/models/ground_truth.py`:
  - Precompute random keypoints from dataset depth + poses in `__init__`.
  - Sample 10% of pixels per frame (shared helper with ViPE).
  - Return precomputed keypoints in `forward` as second element.
- `src/splat_init/models/pycolmap_perspective_transform.py`:
  - Add undistort stage to COLMAP pipeline (`pycolmap.undistort_images`).
  - Read COLMAP outputs from `dense/sparse/0` (binary files) using a local reader (similar to `vendor/otf-nvs/dataloaders/read_write_model.py`).
  - Extract 2D keypoints + 3D points; preserve COLMAP point3D IDs (do not deduplicate beyond what COLMAP already provides).
  - Do not apply extra confidence/inlier filtering beyond COLMAP's own point3D associations.
  - Map face pixel coordinates back into equirectangular pixel coordinates.
  - Return keypoints list as second output element.
  - Skip 2D observations whose `point3D_id` is `-1` (COLMAP no‑point sentinel).
  - Replace `OTCProjector(...)` with `CubeProjector(...)` and remove any `alpha=` argument.

### Out of scope
- Changes to cube projector or keypoint helper utilities.
- Changes to `evaluate_poses.py` or COLMAP export utility.


## Steps
- [x] Collect necessary information.
    - [x] Review `SequenceChunker.forward` chunk logic and overlap alignment.
    - [x] Review `PycolmapPerspectiveTransform` face ordering and COLMAP run pipeline.
    - [x] Review dataset depth/pose availability for ground-truth.
- [ ] Formulate overall approach to solve the task.
    - [ ] **SequenceChunker**:
        - Carry `keypoints` through the chunking path.
        - For each chunk, align keypoints into the global world frame using the same Procrustes transform applied to chunk poses (use `procrustes_analysis` to retrieve rotation/scale/translation, then apply to 3D points).
        - For overlapping frames: stack coords, convert xy to integer pixel coords for stable matching, use `torch.unique` + scatter-add to average 3D points for duplicate pixels.
    - [ ] **GroundTruthPose**:
        - During `__init__`, sample 10% of pixels per frame (shared utility with ViPE).
        - Sample only valid depth pixels (finite and > 0).
        - Use depth + pose to compute world-space keypoints, store in list aligned with scenes.
        - `forward` returns poses and precomputed keypoints for the current scene.
    - [ ] **PycolmapPerspectiveTransform**:
        - After reconstruction, run `pycolmap.undistort_images` (dense output).
        - Read `dense/sparse/0` binary outputs (cameras/images/points3D).
        - For each image, use `image.name` to parse `(face_idx, frame_idx)` via `_parse_image_name`.
        - For each `(x, y, point3D_id)` in `image.xys`/`image.point3D_ids`, skip if `point3D_id < 0`.
        - For each face image, map pixel coords to equirectangular pixel coords via cubemap direction -> lon/lat mapping:
            - Convert pixel `(x, y)` to normalized `u_lin, v_lin` in `[-1, 1]` using face size.
            - Use standard cubemap coordinates (no tangent warp): `u = u_lin`, `v = v_lin`.
            - Build direction using the same face conventions as `OTCProjector._dir_for_face`.
            - Convert to `(lon, lat)` and then to equirect pixel coords `(x, y)` in `[0, W-1] x [0, H-1]`.
        - Build per-frame list[(xy, xyz)] where xyz come from COLMAP `points3D[point3D_id].xyz`; include one entry per image observation (do not deduplicate across images).
- [ ] Append to the plan.
    - [ ] Update the plan if new information or issues arise.
    - [ ] Update assumptions and questions if necessary.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. The Procrustes alignment computed in SequenceChunker can be applied to 3D keypoints to place them in the global world frame.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

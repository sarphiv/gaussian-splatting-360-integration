# Context
Implement the DA3-Streaming panorama wrapper: project 360 panoramas to 4 perspective faces, run DA3-Streaming per face via `da3_streaming.py`, and merge face poses into a single panorama pose per frame with careful rotation/translation math.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- New model class under `src/splat_init/models/` implementing `forward` only (evaluation use).
- Projection of equirectangular panoramas to 4 perspective faces.
- Temporary image directory handling (prefer RAM-backed if available) and output parsing.
- Pose merging math with explicit pre-rotation of faces before averaging.

### Out of scope
- Training or loss computation.
- Modifying DA3-Streaming vendor code.
- Changing dataset loaders.


## Steps
- [x] Collect necessary information.
    - [x] Inspect `vggt_perspective_transform.py` for face projection + pose merge logic.
    - [x] Inspect `utilities/otc_projector.py` for cubemap projection and face rotations.
    - [x] Inspect `vendor/depth-anything-3/da3_streaming/da3_streaming.py` to confirm inputs (image dir) and outputs (camera_poses.txt).
    - [x] Describe a small experiment: create a synthetic identity pose per face and verify pre-rotation recovers identity after merge.
- [x] Formulate overall approach to solve the task.
    - [x] Add a new model file `src/splat_init/models/da3_perspective_transform.py` with a `Da3PerspectiveTransform` LightningModule wrapper.
    - [x] Implement a face projector for 4 faces (+X, -X, +Z, -Z):
        - [x] Use `OTCProjector` for face projection.
        - [x] Precompute face rotations via `cube_face_relative_rotations()[[0,1,4,5]]` and face weights (uniform).
    - [x] In `forward(images)`:
        - [x] Assert input shape `[B, S, C, H, W]`, `B == 1`, and `C in {3,4}`.
        - [x] Composite alpha if present (RGB * alpha) to match other models.
        - [x] Project to faces and slice to 4 faces; set `face_size` compatible with DA3 default `process_res=504` (rounded to PATCH_SIZE=14).
        - [x] Write per-face image sequences to temporary directories with stable filenames (e.g. `frame_000000.png`), using `/dev/shm` when RAM-backed storage is requested.
        - [x] For each face directory, call `ensure_da3_streaming_assets()` and `load_da3_streaming_config()` from `utilities/da3_assets.py` to build the config dict; run DA3-Streaming (import `DA3_Streaming`) with that config; use a temp output dir (disk-backed OK) and clean up afterwards.
        - [x] Parse `camera_poses.txt` (c2w) into a torch tensor, invert to w2c, and ensure dtype float32.
    - [x] Merge face poses per frame:
        - [x] Pre-rotate each face pose by the known face rotation (same convention as `vggt_perspective_transform`), applying the rotation to both R and t (i.e., pre-multiply w2c matrices).
        - [x] Compute Markley mean rotation across faces and average camera centers (weighted) to get merged pose.
        - [x] Optionally convert to poses relative to the first frame (match other evaluation outputs) using `pose_from_center_and_rotation` / `pose_to_mat` utilities.
    - [x] Return `(poses, None, extras)` where `poses` is `[1, S, 4, 4]` and `extras` may include per-face poses for debugging.
    - [x] Add clear docstrings, type hints, and minimal assertions to document assumptions.
    - [x] Expose the model name string `da3_perspective_transform` for evaluation wiring.
    - [x] Append to the plan.
    - [x] Update merge logic if the DA3 output is determined to be c2w/w2c with a different convention.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. DA3-Streaming `camera_poses.txt` stores c2w matrices (as documented in the script), pending visualization confirmation.
2. The face rotations from `cube_face_relative_rotations()` match the face projection directions used for the cubemap projector.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

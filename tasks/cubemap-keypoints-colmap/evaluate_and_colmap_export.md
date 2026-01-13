# Context
Update evaluation outputs to store keypoints and image size metrics, and add a new utility to export predicted poses/keypoints into a COLMAP-format directory (images + sparse/0). This utility should be memory-efficient and write COLMAP files directly in binary format.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- `src/splat_init/evaluate_poses.py`:
  - Capture keypoints returned by models.
  - Save to `keypoints.pt` next to `poses.pt` and `metrics.pt`.
  - Add dataset image width/height to `metrics` using `args.data.dataset_image_size`.
- Add a new utility helper (suggested: `src/utilities/colmap_export.py`) that:
  - Accepts `(scene: SceneSampleLazy, eval_dir: Path, output_dir: Path)`.
  - Chooses a scene subdirectory based on `scene.id` under `output_dir`.
  - Writes `output_dir/<scene-id>/images/*` (RGB, alpha-applied) and `output_dir/<scene-id>/sparse/0/{cameras,images,points3D}.bin`.
  - Uses predicted poses + keypoints from eval output to populate images and points3D.
  - Is memory efficient (iterate per frame; avoid loading all images at once).
  - Uses `EQUIRECTANGULAR` camera model (assume same params as SIMPLE_PINHOLE; if model IDs are unknown, write as SIMPLE_PINHOLE in binary with the same params).

### Out of scope
- Changes to model internals or keypoint helper.
- End-to-end CLI for the new utility (unless required by surrounding code).


## Steps
- [x] Collect necessary information.
    - [x] Review `evaluate_poses.py` output structure.
    - [x] Review COLMAP binary file formats in `vendor/otf-nvs/dataloaders/read_write_model.py`.
- [ ] Formulate overall approach to solve the task.
    - [ ] **evaluate_poses.py**:
        - Update inference to capture `(poses, keypoints, _)`.
        - Save `keypoints.pt` after moving tensors to CPU.
        - Extend metrics with `dataset_image_width` and `dataset_image_height` from `args.data.dataset_image_size`.
    - [ ] **COLMAP export utility**:
        - Load `poses.pt` + `keypoints.pt` from `eval_dir/<scene-id>/poses/`.
        - Write images incrementally to `output_dir/<scene-id>/images`.
        - Build and write `cameras.bin` using `EQUIRECTANGULAR` with SIMPLE_PINHOLE-style params.
        - Build and write `images.bin` per frame:
            - Convert w2c pose to `qvec` (qw,qx,qy,qz) + `tvec` using a rotmat->quat helper (mirror `read_write_model.py`).
            - Match the sign convention used in `train_splats_otf-nvs.py` (`qvec = -rotmat2qvec(rot)`).
            - Create `xys` from keypoint pixel coords (float32) and `point3D_ids` with matching length.
        - Build and write `points3D.bin` with unique point IDs per observation (track length 1); each keypoint becomes one point3D with a single `(image_id, point2D_idx)` track entry.
        - Sample point RGB from the corresponding image (nearest neighbor) for each keypoint.
        - Use streaming writes to keep memory usage low.
        - Binary format details (from COLMAP spec):
            - `cameras.bin`: `num_cameras (Q)`, then per camera: `id (int)`, `model_id (int)`, `width (Q)`, `height (Q)`, `params (num_params * double)`.
            - `images.bin`: `num_images (Q)`, then per image: `id (int)`, `qvec (4 * double)`, `tvec (3 * double)`, `camera_id (int)`, `name (chars + '\\0')`, `num_points2D (Q)`, then `num_points2D` records of `(x (double), y (double), point3D_id (q))`.
            - `points3D.bin`: `num_points3D (Q)`, then per point: `id (Q)`, `xyz (3 * double)`, `rgb (3 * uint8)`, `error (double)`, `track_length (Q)`, then `track_length` records of `(image_id (int), point2D_idx (int))`.
- [ ] Append to the plan.
    - [ ] Update the plan if new information or issues arise.
    - [ ] Update assumptions and questions if necessary.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. Treating each keypoint as a unique point3D (track length 1) is acceptable for non-pycolmap models.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

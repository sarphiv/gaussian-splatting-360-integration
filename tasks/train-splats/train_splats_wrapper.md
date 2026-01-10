# Context
Implement the new `src/splat_init/train_splats.py` entrypoint plus `src/configs/train_splats_args.py` to run otf-nvs training after `evaluate_poses.py`, using predicted poses stored under `args.output_dir/<scene-id>/poses/model_output.pt` and writing the trained splat to `args.output_dir/<scene-id>/splat`. The wrapper must project each equirectangular panorama to cube-map faces using `OTCProjector` (FOV is always 90 deg; alpha is not FOV), discard top/bottom faces, order faces in the specified forward/left/right/back pattern, and pass a dataset-derived focal length to otf-nvs via `--init_focal` with `--fix_focal`.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Create `src/splat_init/train_splats.py` that:
  - Builds the same dataset types as `evaluate_poses.py` (use `SceneSampleLazy`).
  - Skips any scene that does not have `args.output_dir/<scene-id>/poses/model_output.pt`.
  - Projects each panorama frame to cube faces using `OTCProjector` (alpha is not FOV; FOV is 90 deg), keeps faces indices `(0, 1, 4, 5)` only, and multiplies RGB by alpha once after projection (no premultiply before projection).
  - Orders faces as: forward faces in sequential frame order, left faces in sequential reverse frame order, right faces in sequential frame order, back faces in sequential reverse frame order, and applies the same order to COLMAP image names/ids and disk filenames.
  - Writes the projected RGB faces to a RAM-backed temporary dataset (`/dev/shm`) with the otf-nvs expected layout: `<tmp>/images/` and `<tmp>/sparse/0/`.
  - Converts predicted panorama poses to per-face COLMAP `images`/`cameras` files and writes them with `write_model(..., ext=".bin")` (points3D empty dict).
  - Runs otf-nvs via `uv run python vendor/otf-nvs/train.py -s <tmp> -m <out>/<scene-id>/splat --use_colmap_poses --init_focal <focal> --fix_focal`.
  - Leaves otf-nvs `metadata.json` as the metrics artifact (it already includes time and FPS).
- Create `src/configs/train_splats_args.py` with minimal wrapper arguments (dataset selection + output_dir for evaluate_poses outputs + projection settings, including `face_size`).

### Out of scope
- Modifying vendor code.
- Adding new datasets or changing otf-nvs training defaults beyond `--use_colmap_poses`, `--init_focal`, and `--fix_focal`.
- Post-processing or editing any existing output data on disk.


## Steps
- [x] Collect necessary information.
    - [x] Explore `vendor/otf-nvs` training entrypoints and COLMAP helpers.
    - [x] Confirm COLMAP points3D are unused in otf-nvs runtime (safe to leave empty).
    - [x] Note how `evaluate_poses.py` outputs poses and how datasets expose images.
- [ ] Formulate overall approach to solve the task.
    - [ ] Reuse dataset builders from `evaluate_poses.py` with `SceneSampleLazy` to avoid loading full sequences.
    - [ ] Read `poses/metrics.pt` and use `dataset_stride` to configure the dataset; assert `len(scene) == sequence_length`.
    - [ ] Assert `sequence_length >= 6` and fail fast otherwise.
    - [ ] Stage data in `/dev/shm` via a helper similar to `_temporary_directory` in `da3_perspective_transform.py`.
    - [ ] Project equirect frames using `OTCProjector(face_size=..., alpha=1e-9)`, keep faces `(0, 1, 4, 5)`, and compute `rgb_faces = rgb_faces * alpha_faces` once after projection.
    - [ ] Perform projection + PNG writing in batches (similar to `Da3PerspectiveTransform._write_image_sequence`) to avoid holding all frames/faces in RAM/GPU at once.
    - [ ] Define the face mapping explicitly using `OTCProjector`/`cube_face_relative_rotations` order (`_FACE_ORDER = (+X, -X, +Y, -Y, +Z, -Z)`): forward=+Z (index 4), back=-Z (index 5), right=+X (index 0), left=-X (index 1). This keeps “forward” consistent with existing cube-map usage in DA3/VGGT (indices `(0,1,4,5)` are `+X,-X,+Z,-Z`).
    - [ ] Implement the ordering explicitly (for frames labeled 1..N): [forward-1..N, left-N..1, right-1..N, back-N..1]
    - [ ] Encode the ordering into filenames so `get_image_names(...).sort()` yields the required sequence. Use a monotonic ordinal prefix, e.g. `ord_{ordinal:06d}_face_{face_idx}_frame_{frame_idx:06d}.png`, where `ordinal` increments in the exact ordering above. Construct COLMAP images with the same names/ids in that ordinal order.
    - [ ] Compute per-face w2c poses: with panorama `w2c` and `face_rot = cube_face_relative_rotations()[face_idx]` (face->pano), use `R_face = face_rot.T @ R_pano` and `t_face = face_rot.T @ t_pano`. Build `w2c_face` and export COLMAP `qvec = -rotmat2qvec(R_face)` and `tvec = t_face`.
    - [ ] Create a single COLMAP `Camera` entry (SIMPLE_PINHOLE) with `width=height=face_size`, `params=[focal, cx, cy]` using `cx=cy=0.5*(face_size-1)`, and re-use its `camera_id` for all images to match otf-nvs expectation of a single camera.
    - [ ] Compute focal length from the cube-face size (FOV 90): `focal = 0.5 * (face_size - 1)`. Pass it to otf-nvs with `--init_focal --fix_focal` and also embed it in the COLMAP camera params.
    - [ ] Run otf-nvs with `uv run` and only the minimal required args: `-s`, `-m`, `--use_colmap_poses`, `--init_focal`, `--fix_focal`.
    - [ ] Rely on otf-nvs `metadata.json` in `<scene-id>/splat` for metrics; ensure the wrapper does not delete it and logs its path.
    - [ ] Verify the changes behave as intended through a dry run or minimal invocation if feasible.
- [ ] Append to the plan.
    - [ ] Update the plan if new information or issues arise.
    - [ ] Update assumptions and questions if necessary.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. `/dev/shm` is available and writable in the target environment.
2. `uv run python vendor/otf-nvs/train.py ...` works with the project-managed environment and discovers otf-nvs modules correctly.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

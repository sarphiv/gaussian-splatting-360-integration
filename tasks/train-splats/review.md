# Context
Review the integration of `train_splats.py` + `train_splats_args.py` with otf-nvs training, and the updated `evaluate_poses.py` output path (`.../<scene-id>/poses`). Ensure the wrapper correctly stages data, projects to cube faces, writes COLMAP, invokes otf-nvs via `uv run`, and leaves `metadata.json` for metrics.


# Review
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- This file is an initial checklist only. Report outcomes (including skipped tests) in chat.


## Scope
### In scope
- `src/splat_init/train_splats.py`
- `src/configs/train_splats_args.py`
- `src/splat_init/evaluate_poses.py` output path update
- `investigations/threesixty_explorer.py` path update

### Out of scope
- otf-nvs vendor behavior changes
- Dataset format changes
- Editing any existing output data on disk


## Requirements
- [ ] `evaluate_poses.py` writes to `args.output_dir/<scene-id>/poses/` and readers use this path.
- [ ] `train_splats.py` writes splat outputs to `args.output_dir/<scene-id>/splat/` and uses predicted poses from `.../poses/model_output.pt`.
- [ ] Only scenes with `model_output.pt` are trained; others are skipped with a log message.
- [ ] Cube-map projection: uses `OTCProjector`, keeps faces `(0,1,4,5)`, multiplies RGB by alpha after projection, and does not premultiply before projection.
- [ ] Face ordering matches the required forward/left/right/back sequence and is consistent between image filenames and COLMAP images.
- [ ] Filenames encode ordering via a monotonic ordinal prefix so `get_image_names(...).sort()` returns forward-asc, left-desc, right-asc, back-desc.
- [ ] Face mapping matches `OTCProjector`/`cube_face_relative_rotations` order: forward=+Z (4), back=-Z (5), right=+X (0), left=-X (1).
- [ ] COLMAP export uses one SIMPLE_PINHOLE camera with `focal = 0.5*(face_size-1)` and per-face images with matching names; points3D empty.
- [ ] COLMAP principal point uses `cx=cy=0.5*(face_size-1)`.
- [ ] Dataset stride and expected length are read from `poses/metrics.pt`; assert `len(scene)==sequence_length` and `sequence_length>=6`.
- [ ] otf-nvs invoked with `uv run` and minimal args (`-s`, `-m`, `--use_colmap_poses`, `--init_focal`, `--fix_focal`).
- [ ] Metrics are preserved via otf-nvs `metadata.json` (contains time/FPS).
- [ ] Scope assertions (must remain unchanged): dataset loaders and otf-nvs vendor files are not modified.


## Verification
- [ ] Manual check: confirm temp dataset layout (`images/` and `sparse/0/`) is written in RAM-backed storage.
- [ ] Manual check: projection/writing is batched to avoid keeping all frames/faces in memory.
- [ ] Test: run one small scene through `train_splats.py` (or a dry-run that stops before training) and verify it locates `model_output.pt` under the new `poses/` path.
- [ ] Edge case or risk area: sequence length < 6 (otf-nvs `align_colmap_poses`) and correctness of per-face pose rotations.
- [ ] Artifact or output inspection: inspect `<scene-id>/splat/metadata.json` for `time` and `FPS` and confirm it’s retained.


# Questions
-- Important questions that require clarification.

1. Should we add a CLI flag to skip training and only stage the COLMAP/temp dataset for inspection?

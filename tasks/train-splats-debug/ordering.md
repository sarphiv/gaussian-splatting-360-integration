# Context
Diagnose whether cube-face ordering or ordering transitions are causing otf-nvs to reject most faces with "too few inliers", and implement a better ordering strategy or debug artifacts to confirm.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- `src/splat_init/train_splats.py` ordering logic and image naming
- `src/configs/train_splats_args.py` for ordering/debug flags
- Non-destructive debug artifacts (CSV/JSON of ordering) written under output or temp dir

### Out of scope
- Modifying any vendor code under `vendor/`
- Changing dataset loaders or pose evaluation output format


## Steps
- [x] Collect necessary information.
    - [x] Explore the relevant code.
    - [x] Explore the data for e.g. relevant metadata.
    - [x] Describe a small experiment to verify or deepen understanding.
    - [x] Identify where otf-nvs selects keyframes and how ordering impacts inlier counts.
- [ ] Formulate overall approach to solve the task.
    - [ ] Add a configurable `face_order_mode` (e.g., `face_blocks`, `face_index_blocks`, `frame_major`) so we can test if order affects inlier success.
    - [ ] Add an optional `--dump_image_order` flag that writes a CSV/JSON mapping `ordinal, face_idx, frame_idx, filename`.
    - [ ] Ensure image filenames and COLMAP image names remain consistent with the selected order.
    - [ ] Default to the order that yields best inlier counts once verified.
    - [ ] Verify the changes behave as intended through a dry-run that writes the ordering file.
- [ ] Append to the plan.
    - [ ] Update the plan if new information or issues arise.
    - [ ] Update assumptions and questions if necessary.
    - [ ] Record the ordering that produces the most keyframes in otf-nvs logs.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. Ordering transitions between orthogonal cube faces are a primary driver of low inlier counts.
2. A different ordering can materially improve inlier counts without changing otf-nvs internals.
3. Writing ordering artifacts is acceptable for debugging (non-destructive).


# Questions
-- Important questions about the task that cannot be answered without help.

1. Which ordering should be considered canonical for this pipeline: face blocks or frame-major?
2. Is it acceptable to add debug artifacts alongside outputs for analysis?

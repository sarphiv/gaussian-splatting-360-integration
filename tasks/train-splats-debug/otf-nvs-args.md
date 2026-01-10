# Context
Add wrapper-level controls for otf-nvs keyframe thresholds to address "too few inliers for pose initialization" without editing vendor code.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- `src/configs/train_splats_args.py` for new otf-nvs tuning args
- `src/splat_init/train_splats.py` to pass those args into the otf-nvs CLI

### Out of scope
- Modifying otf-nvs implementation under `vendor/`
- Changing dataset selection or pose export formats


## Steps
- [x] Collect necessary information.
    - [x] Explore the relevant code.
    - [x] Explore the data for e.g. relevant metadata.
    - [x] Describe a small experiment to verify or deepen understanding.
    - [x] Identify otf-nvs CLI flags that impact inlier thresholds (`min_num_inliers`, `min_displacement`, `match_max_error`, `num_keyframes_miniba_bootstrap`).
- [ ] Formulate overall approach to solve the task.
    - [ ] Add a new `OtfNvsArgs` dataclass with optional overrides for the relevant CLI flags.
    - [ ] Update `_run_otf_nvs` to include flags only when the override is set (keep default behavior otherwise).
    - [ ] Wire the new args into `Args` and the CLI via tyro.
    - [ ] Add a brief log line when overrides are active to aid debugging.
    - [ ] Verify the changes behave as intended by confirming the generated CLI includes the overrides.
- [ ] Append to the plan.
    - [ ] Update the plan if new information or issues arise.
    - [ ] Update assumptions and questions if necessary.
    - [ ] Capture recommended override values once inlier counts improve.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. Lowering inlier thresholds will allow more faces to register without destabilizing training.
2. The project is willing to expose otf-nvs tuning options in the wrapper CLI.


# Questions
-- Important questions about the task that cannot be answered without help.

1. Which otf-nvs thresholds should be the initial recommended overrides?
2. Should overrides be persisted in outputs for reproducibility?

# Context
Investigate the DA3 panorama-to-face projection refactor between 40a3f6dd and current, identify changes that plausibly degrade pose quality, and propose code-level adjustments (no implementation) to restore previous behavior while keeping batching/parallelism.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Analyze `src/splat_init/models/da3_perspective_transform.py` diffs that affect face projection, alpha handling, dtype/device, and DA3 I/O.
- Propose minimal fixes that preserve new batching/parallelism but restore the old numerical behavior.

### Out of scope
- Any changes to dataset loaders or evaluation scripts.
- Model training or tuning; focus only on deterministic preprocessing and pose merging behavior.


## Steps
- [x] Collect necessary information.
    - [x] Diff `da3_perspective_transform.py` against 40a3f6dd and record behavioral changes.
    - [x] Inspect `OTCProjector` to understand alpha handling and sampling modes.
    - [x] Check dataset RGBA conventions (alpha masks, potential non-zero RGB in masked regions).
    - [x] Diff `src/configs/depth_anything_3.yaml` for DA3 streaming parameter changes.
    - [x] Summarize hypotheses that could explain degraded predictions.
- [x] Collect additional regression clues.
    - [x] Verify git history for `da3_perspective_transform.py` (only commit is 40a3f6d + working tree).
    - [x] Inspect DA3 streaming chunk handling to understand sensitivity to chunk_size/overlap.
    - [x] Verify `_project_faces` matches old `_project_face` numerically for alpha=1 (no diff).
    - [x] Confirm loop optimizer off still yields bad trajectories (issue is pre-loop).
- [ ] Formulate overall approach to solve the task.
    - [ ] Restore pre-multiplied alpha behavior before projection while keeping batched projection.
    - [ ] Ensure projector inputs use float32 as before, independent of model dtype, unless proven safe.
    - [ ] Decide whether to keep returning faces on input dtype/device or to restore float32 for merging.
    - [ ] Specify a minimal patch to `_project_faces` and `_run_da3` for implementer.
    - [ ] Describe a quick validation to compare outputs against 40a3f6dd (e.g., A/B run on a short sequence).
    - [ ] Add an A/B check for PNG outputs between sequential vs threaded `_write_image_sequence`.
    - [ ] Add an A/B check for projection batching vs per-face projection (`_project_face`) equivalence.
    - [ ] Add a DA3 config sanity check: compare output with chunk_size >= sequence length to remove internal chunk alignment.
    - [ ] Propose a robust fix for DA3 sequential alignment drift (Sim3 gating or scale clamp).
    - [ ] Identify where to insert alignment quality checks (likely `da3_streaming.py` / `sim3utils.py`).
- [ ] Append to the plan.
    - [ ] Update assumptions/questions if dataset or alpha semantics differ from expectations.
    - [ ] Add any further hypotheses if new evidence arises (e.g., thread-safety of `write_png`).


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. Alpha masks in RGBA panoramas are not necessarily applied to RGB values (masked RGB may be non-zero).
2. DA3 quality is sensitive to boundary contamination introduced by post-projection alpha masking.
3. For DA3, float32 projection is materially better than bfloat16 when used.


# Questions
-- Important questions about the task that cannot be answered without help.

1. Which dataset is showing the regression (360_loc vs stanford_2d_3d), and are alpha masks non-trivial?
2. Are runs using float32 or bfloat16 for the DA3 model?
3. Is the regression observable when using the old sequential `_write_image_sequence` (i.e., could thread-safety be involved)?

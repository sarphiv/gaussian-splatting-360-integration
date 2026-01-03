# Context
Run a synthetic/controlled check to validate that `Da3PerspectiveTransform._merge_face_poses` reconstructs the expected panorama pose from per-face `w2c` inputs, and that the relative-to-first-view logic matches VGGT.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Synthetic math checks using `cube_face_relative_rotations` and `_merge_face_poses`
- Comparison against expected relative pose formula and VGGT’s translation logic

### Out of scope
- Running DA3/VGGT models on real data
- Altering training or projection code


## Steps
- [ ] Create a minimal synthetic test harness.
    - [ ] Build a small script (e.g., in `investigations/` or a temporary test file) that constructs a sequence of known `w2c` poses with non-identity `R_0` and non-zero `C_0`.
    - [ ] Generate per-face inputs by applying the inverse face rotations: `w2c_face = R_face^T @ w2c_pano` (using `cube_face_relative_rotations()[[0,1,4,5]]`).
    - [ ] Feed the stacked `w2c_face` into `_merge_face_poses` and capture the output.
- [ ] Compute the expected relative poses.
    - [ ] Use the reference formula `R_rel = R_i R_0^T`, `t_rel = t_i - R_rel t_0`, with `t_i = -R_i C_i`.
    - [ ] Verify whether DA3’s merged output matches the expected `R_rel` and `t_rel` for all frames.
- [ ] Compare against VGGT’s merge formula.
    - [ ] Re-implement VGGT’s translation logic on the same synthetic data and compare with DA3 output.
    - [ ] If DA3 deviates, quantify the error and confirm it matches the predicted `R_0^T` mismatch.
- [ ] Report results and cleanup.
    - [ ] Record whether the mismatch disappears when `R_0 = I` or `C_0 = 0`.
    - [ ] Remove any temporary script if it is not intended to live in the repo.
- [ ] Append to the plan.
    - [ ] Update the plan if new information or issues arise.
    - [ ] Update assumptions and questions if necessary.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. The synthetic test can safely treat `cube_face_relative_rotations` as the ground-truth mapping between face and panorama camera frames.


# Questions
-- Important questions about the task that cannot be answered without help.

1. Should the synthetic test live as a permanent unit test (if it exposes a bug) or be removed after manual verification?

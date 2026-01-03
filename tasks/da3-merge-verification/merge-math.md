# Context
Audit the math and coordinate-frame logic in `Da3PerspectiveTransform._merge_face_poses`, using `VggtPerspectiveTransform` as a reference. Focus on face-rotation application, pose representation (w2c vs c2w), center/translation formulas, and the relative-to-first-view conversion.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- `src/splat_init/models/da3_perspective_transform.py` merge math
- `src/splat_init/models/vggt_perspective_transform.py` merge math for comparison
- Coordinate frame utilities in `src/utilities/pose.py` and `src/utilities/otc_projector.py`
- DA3 output semantics in `vendor/depth-anything-3/da3_streaming/da3_streaming.py`

### Out of scope
- Training losses, optimization, or performance tuning
- Changes to cubemap projection itself
- Any modifications outside the DA3 merge logic


## Steps
- [x] Collect necessary information.
    - [x] Inspect `cube_face_relative_rotations()` to confirm rotations are `world_from_face` (columns are right/up/forward) and that `FACE_INDICES = (0,1,4,5)` map to `+X,-X,+Z,-Z`.
    - [x] Confirm DA3 outputs `camera_poses.txt` as `c2w` and that `_run_da3` inverts to `w2c`.
    - [x] Compare DA3 merge flow vs VGGT merge flow; note that VGGT computes `translation_rel = -(rotation_merged @ centers_rel)` while DA3 uses `pose_from_center_and_rotation(rel_centers, rel_rot)`.
    - [x] Note that camera centers are invariant to left-multiplying `w2c` by a camera-frame rotation (so pre-rotation should not change centers).
- [ ] Derive expected math for the relative transform.
    - [ ] Write out the relationship between `w2c`, camera centers `C`, and relative transforms with a reference view (`R_rel = R_i R_0^T`, `t_rel = t_i - R_rel t_0`).
    - [ ] Verify whether `pose_from_center_and_rotation(rel_centers, rel_rot)` produces `t_rel` only when `C_rel` is expressed in the reference-frame coordinates; otherwise flag mismatch.
    - [ ] Check if DA3 output poses are already normalized to the reference view (e.g., `R_0 ≈ I`, `t_0 ≈ 0`) by reading model/DA3 conventions.
- [ ] Compare with VGGT logic in detail.
    - [ ] Confirm VGGT’s left-multiply of face rotations (quats) matches DA3’s `face_rot @ w2c_faces`.
    - [ ] Verify that VGGT’s translation formula corresponds to `t_rel = t_i - R_rel t_0` even when `R_0 != I`.
    - [ ] Identify whether DA3 should mirror VGGT’s translation computation or rotate `centers_rel` before `pose_from_center_and_rotation`.
- [ ] Summarize findings and propose a concrete fix or confirm correctness.
    - [ ] If a discrepancy exists, specify the exact lines/operations to change and the expected corrected formula.
    - [ ] If no discrepancy, document the assumptions that make DA3’s math valid (e.g., DA3 outputs already normalized to the reference view).
- [ ] Append to the plan.
    - [ ] Update the plan if new information or issues arise.
    - [ ] Update assumptions and questions if necessary.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. DA3’s `prediction.extrinsics` are in a reference frame that might or might not be normalized to the reference view; this must be confirmed from model conventions or empirical checks.
2. The intended output of `Da3PerspectiveTransform.forward` is a relative pose sequence aligned to the first view, matching VGGT’s behavior.


# Questions
-- Important questions about the task that cannot be answered without help.

1. Should DA3’s output poses be in the same relative frame as VGGT (camera-0 at identity), or should they remain in DA3’s native world frame?
2. If DA3 outputs are already normalized to a reference view, which reference strategy is used in practice for the 360 pipeline (first/middle/saddle)?

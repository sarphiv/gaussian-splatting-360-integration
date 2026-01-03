# Context
Review the DA3 merge-math verification and any subsequent fix to ensure the pose merging is correct and consistent with VGGT/reference formulas.


# Review
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- This file is an initial checklist only. Report outcomes (including skipped tests) in chat.


## Scope
### In scope
- `src/splat_init/models/da3_perspective_transform.py` merge logic
- Any added test/sanity script used to validate the math

### Out of scope
- Model training or data preprocessing
- Projection code changes


## Requirements
- [ ] DA3 merge math matches the expected relative pose formula (`R_rel = R_i R_0^T`, `t_rel = t_i - R_rel t_0`) or explicitly documents assumptions that make the existing math valid.
- [ ] Face-rotation application (`face_rot @ w2c_face`) is validated against `cube_face_relative_rotations` and matches VGGT behavior.
- [ ] Scope assertions (must remain unchanged): No changes to cubemap projection, DA3 inference I/O, or unrelated model code.


## Verification
- [ ] Manual check: Walk through `da3_perspective_transform._merge_face_poses` and compare to VGGT merge logic line-by-line.
- [ ] Test: Run any synthetic check and confirm expected vs actual rotations/translations match within tolerance.
- [ ] Edge case or risk area: Confirm behavior when the first view has a non-identity rotation and non-zero center.
- [ ] Artifact or output inspection: If a visual sanity check is available, verify no obvious axis flips or sign errors.


# Questions
-- Important questions that require clarification.

1. If DA3 outputs are already reference-normalized, do we want to keep the additional relative conversion step or remove it to avoid double-normalization?

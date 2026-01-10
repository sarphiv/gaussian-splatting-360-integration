# Context
Review updates to the train_splats wrapper that aim to improve keyframe acceptance across cube faces via ordering changes and/or otf-nvs tuning args.


# Review
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- This file is an initial checklist only. Report outcomes (including skipped tests) in chat.


## Scope
### In scope
- `src/splat_init/train_splats.py`
- `src/configs/train_splats_args.py`
- Any new debug artifacts produced by the wrapper

### Out of scope
- Vendor changes under `vendor/`
- Dataset format changes


## Requirements
- [ ] Ordering mode changes keep image filenames and COLMAP image names consistent.
- [ ] Any new CLI args are optional and do not change defaults when unset.
- [ ] Metrics or performance gates: keyframe count or inlier warnings improve relative to baseline run.
- [ ] Scope assertions (must remain unchanged): no vendor edits; dataset loaders unchanged.


## Verification
- [ ] Manual check: ordering dump (if enabled) matches expected face/frame order.
- [ ] Test: run a small scene and confirm fewer "too few inliers" warnings.
- [ ] Edge case or risk area: sequences shorter than `num_keyframes_miniba_bootstrap`.
- [ ] Artifact or output inspection: `metadata.json` still produced in the splat output directory.


# Questions
-- Important questions that require clarification.

1. Which ordering mode should be treated as the new default if multiple are available?
2. What inlier threshold overrides are acceptable for the initial tuning pass?
3. ...

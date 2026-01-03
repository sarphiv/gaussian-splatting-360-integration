# Context
Review changes that aim to restore DA3/SequenceChunker pose quality after regression between 40a3f6dd and current.


# Review
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- This file is an initial checklist only. Report outcomes (including skipped tests) in chat.


## Scope
### In scope
- `src/splat_init/models/da3_perspective_transform.py` projection and DA3 runner changes.
- `src/splat_init/models/sequence_chunker.py` device/dtype/chunking logic.

### Out of scope
- Dataset loaders, training code, or unrelated model implementations.


## Requirements
- [ ] Pose quality matches or improves relative to 40a3f6dd on a short, fixed sequence.
- [ ] Metrics or performance gates: no regression in inference runtime beyond acceptable overhead for pre-multiplication.
- [ ] Scope assertions (must remain unchanged): external API signatures remain compatible and no dataset behavior changes.


## Verification
- [ ] Manual check: inspect a few projected face PNGs for correct masking (no halo/bleed).
- [ ] Test: run evaluation on 1-2 scenes and compare pose errors to 40a3f6dd baseline.
- [ ] Edge case or risk area: RGBA with non-trivial alpha; ensure pre-multiplication works.
- [ ] Artifact or output inspection: confirm output pose dtype/device align with expectations.


# Questions
-- Important questions that require clarification.

1. Which dataset/scene should be used as the baseline for comparison?
2. Are there known acceptable deltas in pose error or drift?

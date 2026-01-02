# Context
Review the DA3-Streaming integration: asset management, projection + pose merge model, and evaluation wiring. Ensure no messy downloads into cwd, correct pose conventions, and clean runtime behavior.


# Review
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- This file is an initial checklist only. Report outcomes (including skipped tests) in chat.


## Scope
### In scope
- `src/configs/depth_anything_3.yaml` and DA3 asset download helpers.
- New DA3-Streaming model wrapper and projection/pose-merge logic.
- Evaluation wiring (`evaluation_args.py`, `evaluate.py`).

### Out of scope
- Vendor DA3 code behavior beyond its documented I/O.
- Dataset loading logic changes.


## Requirements
- [ ] DA3 weights + config + SALAD are downloaded into torch cache and never into the repo or cwd.
- [ ] DA3 config copy exists at `src/configs/depth_anything_3.yaml` with null weight paths.
- [ ] Forward path writes images to a temp dir (RAM-backed when available) and cleans output dirs after parsing.
- [ ] Pose merging pre-rotates faces to canonical orientation before averaging; output is w2c and matches evaluation format.
- [ ] Face size aligns with DA3 `process_res=504`.
- [ ] Evaluation can select the new model without breaking existing models.
- [ ] Evaluation uses `SequenceChunker` for `da3_perspective_transform`.
- [ ] Metrics or performance gates: no excessive disk usage in the repo; temp outputs removed.
- [ ] Scope assertions (must remain unchanged): existing VGGT/ViPE model code paths and dataset loaders.


## Verification
- [ ] Manual check: confirm camera_poses parsing (c2w -> w2c) and face pre-rotation math with a synthetic identity-case test.
- [ ] Test: run `src/splat_init/evaluate.py` with an existing model to verify no regressions.
- [ ] Edge case or risk area: ensure `temporary_storage_in_ram=True` asserts when `/dev/shm` is unavailable.
- [ ] Artifact or output inspection: verify temp output dirs are deleted after run (no new files in workspace).


# Questions
-- Important questions that require clarification.

1. None.

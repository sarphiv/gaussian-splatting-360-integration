# Context
Review the pycolmap-cuda12 wrapper implementation and its evaluation integration to
ensure it matches the interface and behavior of existing pose initialization models.


# Review
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- This file is an initial checklist only. Report outcomes (including skipped tests) in chat.


## Scope
### In scope
- `src/splat_init/models/pycolmap_perspective_transform.py` implementation details.
- Wiring in `configs/evaluation_args.py` and `src/splat_init/evaluate.py`.
- Pose output shapes, coordinate conventions, and merge logic.

### Out of scope
- Installing pycolmap-cuda12 or validating CUDA runtime setup.
- Large-scale accuracy benchmarking beyond a sanity run.


## Requirements
- [ ] New wrapper returns `poses_w2c` shaped `[1, S, 4, 4]` and `depth=None`.
- [ ] Per-face pose merges align with existing cube-face conventions.
- [ ] Only 4 faces (indices 0,1,4,5) are used with `OTCProjector(alpha=1e-9)`.
- [ ] Missing registrations are skipped; identity pose is emitted if none register.
- [ ] COLMAP artifacts are not persisted beyond temporary directories.
- [ ] When GPU extraction is enabled, `num_threads=1` is enforced and `gpu_index` is a string.
- [ ] Metrics or performance gates: a short sequence completes without GPU OOM.
- [ ] Scope assertions (must remain unchanged): existing model behavior and CLI flags.


## Verification
- [ ] Manual check: inspect logs for feature extraction, matching, and mapping stages.
- [ ] Test: run `pose-evaluate` on a tiny scene subset with the new model selected.
- [ ] Edge case or risk area: partial COLMAP registration or missing faces.
- [ ] Artifact or output inspection: confirm no COLMAP artifacts are persisted outside temp.


# Questions
-- Important questions that require clarification.

1. None.

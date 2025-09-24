# VGGT Perspective Transform Cleanup

## General Plan
- Capture the essential functionality in the existing perspective-transform VGGT module.
- Redesign the module with clear structure mirroring the naive equirectangular variant while keeping the perspective projection pathway.
- Reintroduce the cleaned implementation with concise preprocessing, loss computation, and logging helpers.

## Detailed Plan
### Capture the essential functionality in the existing perspective-transform VGGT module
- Inventory how the current module constructs cubemap faces, forwards them through VGGT, merges per-face predictions, and computes losses.
- Identify which utilities (projector, pose decoding, Markley averaging, geodesic distance, logging) must persist to maintain behaviour.

### Redesign the module with clear structure mirroring the naive equirectangular variant while keeping the perspective projection pathway
- Define typed helper dataclasses/functions to organize projector usage, pose aggregation, and relative pose conversion.
- Plan the Lightning hooks so training/validation steps share a single tidy helper and logging matches the naive module conventions.

### Reintroduce the cleaned implementation with concise preprocessing, loss computation, and logging helpers
- Implement the refactored projector invocation, VGGT forward pass, loss stack, and logging with clear docstrings and type annotations.
- Validate shapes and control flow to ensure the perspective module remains drop-in compatible with existing training scripts.

## TODO / Questions
- [x] Confirm the Markley averaging weights for faces remain appropriate after refactor.
- [x] Verify that logging and file outputs stay consistent with the previous behaviour for downstream tools.
- Assumptions: Batch size stays 1 and stages remain `train`/`val`.

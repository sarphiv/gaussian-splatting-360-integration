# Remove VGGT chunking for perspective transform

## General Plan
- Adjust the VGGT perspective transform forward path so that all faces are processed in one pass without manual chunking.
- Update constructor parameters, documentation, and internal helpers to reflect the removal of chunking behaviour.
- Sanity check tensor shapes and dtype handling so the single forward call still returns the expected outputs.

## Detailed Plan
- Modify `_forward_vggt` in `src/splat_init/models/vggt_perspective_transform.py` to call the VGGT model once on the full face tensor, relying on autocast only around that single call, and keep the float32 conversion of outputs.
- Remove the `faces_per_forward` hyperparameter and any associated state or comments, making sure `save_hyperparameters()` and docstrings stay accurate after the change.
- Inspect the surrounding code paths (e.g., `_prepare_vggt_input` and `_shared_step`) to confirm downstream reshaping still matches the new forward output and add/adjust assertions if needed for clarity.

### Revision 1
- Since evaluation now OOMs, wrap the VGGT forward pass in inference/no-grad context to drop gradient bookkeeping while keeping the single-shot call structure.

### Revision 2
- Disable unused VGGT heads and aggressively discard extraneous prediction tensors so the forward pass keeps only pose and depth required for evaluation.
- Convert the VGGT weights to half precision on CUDA and execute the depth head with tight frame chunking to further cap activation usage.

### Revision 3
- Drop defensive checks tied to optional heads or devices now that evaluation always runs on CUDA with the standard camera/depth heads enabled.
- Assume RTX 3090 execution, casting VGGT modules directly to `bfloat16` without runtime guards.

## TODO
- [x] Update `_forward_vggt` to eliminate manual chunking.
- [x] Prune constructor args and docstrings tied to chunked inference.
- [x] Re-run through `_shared_step` logic to ensure shapes/dtypes remain correct post-change.
- [x] Guard the VGGT forward in inference mode to minimise memory use during evaluation.
- [x] Disable unused VGGT heads and drop surplus prediction tensors to lower activation memory.
- [x] Implement CUDA graph capture per input shape to reuse allocations and stabilise peak memory.

## Questions / Assumptions
- Assuming GPU memory is sufficient to handle all perspective faces in one VGGT forward pass for current datasets.
- Assuming no external configs depend on the `faces_per_forward` hyperparameter being present.

# Context
Update the depth-capable models to return keypoints as the second output item, using the new keypoints helper and (for perspective models) the new cube projector inverse. Models: `vggt_naive_equirectangular.py`, `vggt_perspective_transform.py`, `vipe_panorama.py`, `da3_perspective_transform.py`.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- For each listed model, update `forward` return signature to:
  `tuple[poses, list[tuple[Tensor, Tensor]] | None, dict[str, Tensor]]`.
- Use depth predictions + confidence where available to compute keypoints via the new helper.
- Keep GPU memory usage low (avoid stacking huge grids on GPU; consider CPU keypoint extraction).
- Use a shared random depth sampling utility for ViPE (reusable with GroundTruth).
- Target files:
  - `src/splat_init/models/vggt_naive_equirectangular.py`
  - `src/splat_init/models/vggt_perspective_transform.py`
  - `src/splat_init/models/vipe_panorama.py`
  - `src/splat_init/models/da3_perspective_transform.py`
 - Replace `OTCProjector(...)` with new `CubeProjector(...)` and remove any `alpha=` argument.

### Out of scope
- `pycolmap_perspective_transform.py`, `sequence_chunker.py`, `ground_truth.py` (handled elsewhere).
- `evaluate_poses.py` changes.


## Steps
- [x] Collect necessary information.
    - [x] Note VGGT depth/conf shapes (`[B,S,1,H,W]`), and current depth discard in `vggt_perspective_transform.py`.
    - [x] Note ViPE optional depth output (`return_depth` flag).
    - [x] Inspect DA3 streaming outputs: `DA3_Streaming.save_depth_conf_result` writes per-frame `results_output/frame_XXXX.npz` with `depth` and `conf` when `save_depth_conf_result=True`.
- [ ] Formulate overall approach to solve the task.
    - [ ] **VggtNaiveEquirectangular**:
        - Capture `depth_conf` from VGGT predictions.
        - Update `_gather_predictions` (or a new helper) to return `depth_conf` alongside `depth`.
        - Use helper on the original input images; let helper resize depth/conf to input resolution.
        - Use confidence threshold `0.9`.
        - Return `poses, keypoints, {}` on `.to(images)` device.
    - [ ] **VggtPerspectiveTransform**:
        - Keep `depth_conf` from `depth_head`.
        - Use new `CubeProjector` with enabled faces (right, left, forward, back) to inverse-project `depth_pred_faces` and `depth_conf_faces` to equirectangular size (`output_size=(H,W)`).
        - Use helper to compute keypoints in equirectangular pixel coords.
        - Use confidence threshold `0.9`.
        - Face indices are `[0, 1, 4, 5]` (same as existing `FACE_INDICES` constants) and must match `CubeProjector` enabled faces order.
    - [ ] **VipePanorama**:
        - Always enable depth (`return_depth=True`).
        - Use a shared random depth sampling helper (same as GroundTruth) instead of full-grid keypoints.
        - Sample 10% of valid depth pixels (finite and > 0); ignore confidence (use ones).
        - If depth is unavailable (e.g. SLAM map missing), return empty keypoints for each frame.
    - [ ] **Da3PerspectiveTransform**:
        - Ensure DA3 produces depth/conf for each frame (config already updated; optionally enforce `Model.save_depth_conf_result=True` as override).
        - Load per-frame `results_output/frame_XXXX.npz` (depth + conf) inside the DA3 temp output directory before it is cleaned up; map to face order indices.
        - Likely change `_run_da3` to return `(w2c, depth, conf)` by reading `results_output` before the temp dir is deleted.
        - Use `CubeProjector.inverse(..., output_size=(H,W))` to merge face depth/conf into equirectangular.
        - Use helper to compute keypoints.
        - Use confidence threshold `0.9`.
        - DA3 `results_output/frame_XXXX.npz` contains `depth` and `conf` shaped `[H, W]` (float32); stack into `[S, 1, H, W]`.
- [ ] Append to the plan.
    - [ ] Update the plan if new information or issues arise.
    - [ ] Update assumptions and questions if necessary.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. A shared random sampling utility can be used for both ViPE and GroundTruth (default sample ratio 10%).


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

# Context
Add a utility function under `src/utilities/` that converts per-image depth + confidence into a list of keypoints tied to input pixel coordinates and 3D world points. This function will be reused by multiple models.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Add a new utility module (suggested: `src/utilities/keypoints.py` or `src/utilities/depth_keypoints.py`) with a function like:
  `keypoints_from_depth(poses, rgb, depth, depth_confidence, confidence_threshold) -> list[tuple[Tensor, Tensor]]`.
- Add a second helper for random sampling (shared by ViPE + GroundTruth), e.g.:
  `sample_keypoints_from_depth(poses, rgb, depth, sample_ratio=0.1) -> list[tuple[Tensor, Tensor]]`.
- Required behavior:
  - `poses` are world-to-camera matrices; use them to transform camera-frame 3D points into world coordinates.
  - Input `rgb` is used to determine the target pixel grid shape.
  - If `depth` spatial dims do not match `rgb`, resize `depth` (and `depth_confidence` for alignment) bilinearly to match `rgb`.
  - Filter pixels by `depth_confidence >= confidence_threshold` and valid depth values (`depth > 0` and finite).
  - Return `list[(xy, xyz)]` per image in batch/sequence order, where:
    - `xy` is shape `[2, N]` (x first row, y second row), pixel coordinates in input image space.
    - `xyz` is shape `[3, N]`, world-space 3D point for each pixel.
  - Accept both `[B, S, ...]` and `[S, ...]` inputs by flattening leading dims to a list of images; preserve order.
  - `poses` may be `[B, S, 4, 4]` or `[S, 4, 4]`; flatten to the same leading order as images.
  - Expected channel shapes: `rgb` is `[... , 3 or 4, H, W]`, `depth` is `[... , 1, H, W]` or `[... , H, W]`, `depth_confidence` is `[... , 1, H, W]` or `[... , H, W]`.
- Keep GPU memory usage low:
  - Prefer per-image processing.
  - Avoid building full-resolution grids on GPU; compute lon/lat and rays only for selected pixel indices.
  - Return tensors on the same device as the input images (use `.to(images)`).

### Out of scope
- Modifying any models or pipelines that will call this helper.
- Adding global config for thresholds.


## Steps
- [x] Collect necessary information.
    - [x] Confirm pose conventions are world-to-camera in datasets/models.
    - [x] Note OTCProjector equirectangular mapping (lon/lat) to derive rays.
- [x] Formulate overall approach to solve the task.
    - [x] Define equirectangular pixel-to-ray mapping:
          x_norm = 2*x/(W-1) - 1, y_norm = 2*y/(H-1) - 1,
          lon = pi * x_norm, lat = -0.5*pi * y_norm,
          dir = [sin(lon)*cos(lat), sin(lat), cos(lon)*cos(lat)].
    - [x] For each image:
          - Build mask from confidence and valid depth.
          - Gather (x,y) indices for valid pixels.
          - Compute per-pixel rays only for selected indices and multiply by depth to get camera-frame points.
          - Transform to world coordinates using pose w2c: if `x_cam = R * x_world + t`, then `x_world = R^T @ (x_cam - t)`.
          - Store `xy` and `xyz` tensors.
    - [x] Implement random sampling helper:
          - Build valid-depth mask (`depth > 0`, finite).
          - Sample `ceil(sample_ratio * num_valid)` indices without replacement.
          - Use the same ray + world transform logic as `keypoints_from_depth`.
    - [x] Handle shape variants (e.g., depth/conf with/without channel dims) by squeezing as needed.
- [x] Implement keypoint extraction utilities.
    - [x] Add tensor-shape normalization and per-image flattening helpers.
    - [x] Compute rays only for selected pixel indices to keep GPU memory low.
    - [x] Implement confidence-filtered and random-sampling variants.
- [x] Append to the plan.
    - [x] Update the plan if new information or issues arise.
    - [x] Update assumptions and questions if necessary.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. The confidence threshold is provided by the caller and can be used directly.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

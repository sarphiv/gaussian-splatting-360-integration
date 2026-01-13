# Context
Create a new cubemap projector utility as a drop-in replacement for OTCProjector. It must live in `src/utilities/cube_projector.py`, support per-face enable flags in `__init__`, preserve the existing face order, and add an inverse projection back to equirectangular with an explicit output size argument.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Add `src/utilities/cube_projector.py` implementing a cubemap projector with:
  - Same forward call signature as `OTCProjector.__call__` (RGBA + optional depth).
  - Face order matching `_FACE_ORDER = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")`.
  - `__init__(face_size, face_forward=True, face_left=True, face_right=True, face_up=True, face_down=True, face_back=True)`.
  - Forward outputs only the enabled faces in face-index order; e.g. disabling two faces yields shape `[B, 4, C, F, F]`.
  - An `inverse(*face_tensors, output_size=(H, W))` method that accepts the enabled faces in the same order as forward output and returns equirectangular tensors.
  - Assert that all enabled faces are provided to `inverse` (missing faces should fail, not fill); pixels mapping to disabled faces should remain zero-filled.
  - Use standard cubemap mapping (no OTC tangent warp).
  - Forward outputs should mirror OTCProjector types and shapes:
      - `rgb_faces`: `[B, num_faces, 3, F, F]`
      - `alpha_faces`: `[B, num_faces, 1, F, F]`
      - `depth_faces`: `[B, num_faces, 1, F, F]` (zeros if `depth is None`)
- Keep face order mapping consistent with existing usage in:
  - `train_splats_otf-nvs.py` (`_FORWARD_FACE = 4`, `_LEFT_FACE = 1`, `_RIGHT_FACE = 0`, `_BACK_FACE = 5`).
- Cache sampling grids by device/dtype to avoid recomputation, similar to `OTCProjector`.

### Out of scope
- Updating model imports or behavior.
- Updating `src/utilities/otc_projector.py`.
- Writing tests or benchmarks.


## Steps
- [x] Collect necessary information.
    - [x] Read `src/utilities/otc_projector.py` for face order and mapping.
    - [x] Identify face index expectations in `train_splats_otf-nvs.py` and models.
    - [x] Note inverse usage requirement from the task description.
- [x] Formulate overall approach to solve the task.
    - [x] Define face index mapping for booleans:
          +X = right, -X = left, +Y = up, -Y = down, +Z = forward, -Z = back.
    - [x] Implement forward sampling that only computes enabled faces in sorted face-index order:
          - Build `u_lin, v_lin` as a meshgrid over `[-1, 1]` for the face size.
          - For each face, build direction vectors using the same formulas as `OTCProjector._dir_for_face`:
                +X: (x, y, z) = (1, -v, -u)
                -X: (x, y, z) = (-1, -v, u)
                +Y: (x, y, z) = (u, -1, -v)
                -Y: (x, y, z) = (u, 1, v)
                +Z: (x, y, z) = (u, -v, 1)
                -Z: (x, y, z) = (-u, -v, -1)
          - Normalize directions, convert to lon/lat, then to grid coords `(x_norm, y_norm)` as in `OTCProjector` (no tan warp).
          - Use `grid_sample` with `align_corners=True` and `padding_mode="border"` to match existing behavior.
          - Keep grid cached per `(device, dtype)` for speed.
    - [x] Implement inverse that samples from enabled faces into an equirectangular grid sized by `output_size=(H, W)` using the same direction conventions as `OTCProjector` (standard cube mapping, no tan warp):
          - For each equirect pixel, compute direction:
                lon = pi * (2*x/(W-1) - 1)
                lat = -0.5*pi * (2*y/(H-1) - 1)
                dir = [sin(lon)*cos(lat), sin(lat), cos(lon)*cos(lat)]
          - Face selection via `argmax(abs(x), abs(y), abs(z))`, ties resolved by `_FACE_ORDER` preference:
                "+X" > "-X" > "+Y" > "-Y" > "+Z" > "-Z".
          - Convert direction to per-face (u, v) in [-1, 1] by inverting `_dir_for_face`:
                +X: u = -z/x, v = -y/x
                -X: u = -z/x, v = y/x
                +Y: u = -x/y, v = z/y
                -Y: u = x/y,  v = z/y
                +Z: u = x/z,  v = -y/z
                -Z: u = x/z,  v = y/z
          - Convert u/v to grid coords directly (no tangent warp): u_lin = u, v_lin = v.
          - For each enabled face, build a grid and sample using `grid_sample` (`align_corners=True`, `padding_mode="border"`), then mask+accumulate into the equirect output.
          - Support variable channel counts: each `face_tensor` can be `[B, num_faces, C, F, F]`; return an equirect tensor `[B, C, H, W]` per input argument.
    - [x] Accept variable numbers of face tensors (`[B, num_faces, C, F, F]`) and return one equirectangular tensor per input, preserving channel count.
    - [x] Validate that `inverse` receives the exact number of enabled faces (assert if mismatch).
- [x] Implement the cubemap projector utility.
    - [x] Add `CubeProjector` with face-enable flags and face-order filtering.
    - [x] Cache forward and inverse sampling grids by device/dtype.
    - [x] Mirror OTCProjector forward outputs and depth fallback behavior.
    - [x] Add inverse mapping with explicit output size and missing-face asserts.
- [x] Append to the plan.
    - [x] Update the plan if new information or issues arise.
    - [x] Update assumptions and questions if necessary.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. The `output_size` argument to `inverse` is provided as `(height, width)` matching input panorama tensor shapes.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

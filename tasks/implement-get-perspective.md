# General Plan
- Extend existing room/file indexing logic to support perspective frame naming and directory layout.
- Build per-room mappings for perspective RGB/depth/pose assets so we can load them deterministically.
- Implement `get_perspective` to reuse the shared loaders/stacks and return a `SceneSample` mirroring `__getitem__` but backed by perspective data.
- Optimize `get_perspective` data loading so repeated access is faster.

# Detailed Plan
## Extend indexing helpers
- Update the room extraction regex (or helper) to accept both equirectangular and numbered perspective frame suffixes without breaking existing cases.
- Verify `_prefix_up_to_domain` remains valid for perspective filenames, adjusting if necessary.

## Build perspective asset lookup
- Discover `area_dir / "data" / {rgb, depth, pose}` and enumerate all RGB assets, matching depth/pose via the shared prefix.
- Use the room list already kept after `max_sequence_length` filtering so perspective views stay aligned with panorama rooms.
- Group perspective views by room, sort them for determinism, and store only the file paths for lazy loading, mirroring existing behavior.

## Implement `get_perspective`
- For the requested dataset index, resolve the room id and associated perspective paths.
- Load RGB/depth/pose tensors following the same normalization/stacking as `__getitem__`.
- Return a `SceneSample` with the canonical origin and stacked tensors; ensure assertions guard against missing data.

## Improve perspective loading speed
- Introduce multithreaded I/O for per-view loading to overlap PNG/JSON reads across frames.
- Make the worker count configurable (with a sensible default) and reuse the batching utilities for stacking.

# TODO / Questions
- [x] Confirm perspective directories contain assets for every kept room; add asserts if mismatches appear.
- [x] Ensure room ids between panoramic and perspective assets align (add asserts during indexing).
- [x] Decide on a good default for threaded perspective loading (empirically balance parallelism vs overhead).

## Assumptions
- Perspective files live under `area_dir / "data" / {...}` and follow the provided naming patterns.
- Depth filenames match prefixes even if frame ids differ from RGB (will map strictly by shared prefix).

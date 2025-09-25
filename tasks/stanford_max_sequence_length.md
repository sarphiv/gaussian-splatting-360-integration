# Add max sequence length filter to Stanford dataset

## General Plan
- Expose an optional maximum sequence length parameter on the Stanford 2D-3D dataset.
- Ensure room indexing respects the limit by excluding rooms exceeding the bound.
- Wire the new parameter through existing dataset factory helpers while keeping current behaviour default.

## Detailed Plan
- Update `Stanford2D3DDataset` to accept `max_sequence_length: int | None = None`, store it, and document the new argument.
- During room indexing, skip any room whose view count exceeds the provided maximum, keeping per-room tensors contiguous for the remaining rooms.
- Adjust dataset construction helpers (e.g., `_stanford_callables`) so callers can supply the optional max length without affecting default usage.

## TODO
- [x] Add the optional parameter to `Stanford2D3DDataset` and record it on the instance.
- [x] Filter out rooms whose sequence length exceeds the configured maximum.
- [x] Update dataset factory helpers or call sites to pass through the new parameter.

## Questions / Assumptions
- Assuming evaluation setups that need the cap will explicitly configure it; otherwise default `None` should preserve existing behaviour.
- Excluding long rooms entirely is acceptable; we do not need to truncate sequences.

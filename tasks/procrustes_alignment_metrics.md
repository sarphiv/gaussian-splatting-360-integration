# Procrustes Alignment Metrics

## General Plan
- Derive per-scene position errors after alignment and organize them by sequence length.
- Report aggregate statistics that highlight mean errors for sequences longer than three views.
- Restrict the interactive visualization to only show scenes whose sequence length exceeds three views.
- Emit per-scene error logs whenever the visualization switches between scenes.

## Detailed Plan
### Derive per-scene errors
- After aligning each scene, compute the per-point Euclidean distances between aligned predictions and targets.
- Aggregate the mean distance per scene into a mapping keyed by the scene's sequence length for later reporting.

### Report aggregate statistics
- Iterate over the grouped error data to log counts and representative error values for every observed sequence length.
- Compute and log the mean error for each sequence length above three to satisfy the additional reporting requirement.

### Restrict visualization
- Build a filtered list of scenes whose sequence length is greater than three while keeping all scenes for statistics.
- Update the visualization logic to operate on the filtered lists, handling the edge case where no qualifying scenes exist.

### Emit per-scene logs on navigation
- Track the mean error associated with each filtered scene alongside the visualization data.
- Log the mean error for the current scene whenever the visualization advances or reverses.

## TODO
- [x] Implement error aggregation keyed by sequence length.
- [x] Add logging for grouped errors and mean errors for sequence lengths above three.
- [x] Filter visualization inputs to scenes with sequence length greater than three and guard against empty results.
- [x] Log the mean error whenever the visualization switches scenes.

## Questions / Assumptions
- Assuming "position error" refers to the mean Euclidean distance between aligned prediction and target points per scene.

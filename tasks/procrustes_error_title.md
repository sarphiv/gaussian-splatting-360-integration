# Task: Show per-scene mean error in Procrustes plot title

## General Plan
- Inspect the Procrustes visualization update logic to confirm where scene-level errors are logged and how titles are set.
- Adjust the scene switching code to display the mean error in the plot title instead of logging it repeatedly.
- Sanity-check the updated visualization flow to ensure no regressions in aggregation metrics or interaction behavior.

## Detailed Plan
### 1. Inspect the Procrustes visualization update logic
- Review `investigations/procrustes_alignment.py` around the scene update functions to map out current logging and titling behavior.
- Identify the variables holding per-scene mean error and sequence metadata used during visualization.

### 2. Adjust the scene switching code to display the mean error in the plot title
- Update the helper/setter responsible for `ax.set_title` so it includes both the scene index/length and the mean error.
- Remove the per-scene logger invocation to avoid redundant console spam while keeping aggregate logging untouched.
- Ensure the initial plot title (before any button interaction) mirrors the same format.

### 3. Sanity-check the visualization flow
- Re-read the modified code to confirm that button callbacks and data structures remain valid after the change.
- Verify the code style aligns with project conventions (docstrings, typing, logging) and that no unused helpers remain.

## TODO
- [x] Confirm scene update logic and data structures used for error/titles.
- [x] Refactor title-setting code to include mean error and drop per-scene logging.
- [x] Review code after edits for style adherence and lingering references.

## Questions / Assumptions
- Assuming runtime tests/plots are not required for this change due to interactive nature.
- Assuming aggregate error logging earlier in `main` should remain unchanged.

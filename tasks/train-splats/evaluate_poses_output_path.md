# Context
Update `src/splat_init/evaluate_poses.py` to write outputs under `args.output_dir/<scene-id>/poses/` and update `investigations/threesixty_explorer.py` to read from the new path. Do not modify or touch any existing output data on disk.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Adjust output path in `evaluate_poses.py` to include a `poses` subdirectory.
- Update `investigations/threesixty_explorer.py` to load `model_output.pt` and `metrics.pt` from the `poses` subdirectory.

### Out of scope
- Modifying existing output data on disk.
- Changing the evaluation logic or metrics content.
- Adding any compatibility fallback to the old output path.


## Steps
- [x] Collect necessary information.
    - [x] Locate `evaluate_poses.py` output write calls and search for any readers.
    - [x] Identify `investigations/threesixty_explorer.py` path usage.
- [ ] Formulate overall approach to solve the task.
    - [ ] Change the output directory from `args.output_dir/<scene-id>` to `args.output_dir/<scene-id>/poses`.
    - [ ] Update `threesixty_explorer.py` to read from `pred_scene_path / "poses"` while keeping the rest of the logic intact.
    - [ ] Verify the new directory structure is created before writing `model_output.pt` and `metrics.pt`.
- [ ] Append to the plan.
    - [ ] Update the plan if new information or issues arise.
    - [ ] Update assumptions and questions if necessary.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. No other internal script depends on the old `<output>/<scene-id>/model_output.pt` path.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

# Context
Update dependency wiring so the project installs VGGT from the new git submodule instead of the remote git URL. This subtask touches `pyproject.toml` and `uv.lock` only.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Update `[tool.uv.sources]` in `pyproject.toml` to point `vggt` at `vendor/vggt`.
- Decide whether `vggt` should be editable (and reflect that in `pyproject.toml`).
- Refresh `uv.lock` so the VGGT source is local-path-based.

### Out of scope
- Adding the VGGT submodule itself (handled by another subtask).
- Code changes in `src/` or `tests/`.
- Documentation updates unless necessary for dependency setup.


## Steps
- [x] Collect necessary information.
    - [x] Confirm `pyproject.toml` currently pins `vggt` via git URL + commit.
    - [x] Note `uv.lock` currently references the git URL for VGGT.
    - [x] Verify `vipe` uses a local editable source for comparison.
- [x] Formulate overall approach to solve the task.
    - [x] Replace `vggt` source in `[tool.uv.sources]` with a local path (`vendor/vggt`).
    - [x] Mark the `vggt` source as `editable = true`.
    - [x] Update `uv.lock` to reflect the new source.
    - [x] Spot-check the lock entry for `vggt` to confirm it is path-based.
    - [x] Sanity check devcontainer config for VGGT (no changes expected unless path wiring breaks).
- [x] Append to the plan.
    - [x] Editable setting unchanged from the planned choice (`editable = true`).
    - [x] No `uv lock` run needed; updated lock entries directly.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. VGGT can be installed from its repository root without extra build configuration beyond what `uv` can infer.
2. Updating `uv.lock` is expected and acceptable for this change.


# Questions
-- Important questions about the task that cannot be answered without help.

1. Should any devcontainer file be adjusted if `vendor/vggt` is added, or is the current setup sufficient?

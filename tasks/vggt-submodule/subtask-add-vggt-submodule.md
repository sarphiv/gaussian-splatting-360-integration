# Context
Add VGGT as a git submodule under `vendor/vggt` so it mirrors the existing ViPE submodule setup. This subtask only handles submodule wiring (.gitmodules + vendor tree), not Python dependency configuration.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Add `vendor/vggt` as a git submodule using the VGGT repo URL.
- Update `.gitmodules` with the new VGGT entry (path, URL, optional branch).
- Ensure the submodule is checked out at the intended revision.
- Confirm the `vendor/` ignore rule does not block the submodule add.

### Out of scope
- `pyproject.toml` or `uv.lock` changes.
- Any code changes in `src/` or `tests/`.
- Running project tests or installs.


## Steps
- [x] Collect necessary information.
    - [x] Locate current submodule config in `.gitmodules` (only `vendor/vipe`).
    - [x] Identify current VGGT source in `pyproject.toml` (git URL + pinned commit).
    - [x] Confirm `vendor/` is currently used for submodules and is gitignored.
    - [x] Note existing VGGT usage in code for context only.
- [x] Formulate overall approach to solve the task.
    - [x] Add `vendor/vggt` as a submodule pointing to `https://github.com/facebookresearch/vggt.git`.
    - [x] Check out commit `e56963328b7476e615ce8dda9164d381f8dc07a3`.
    - [x] Update `.gitmodules` with path/url (no branch tracking).
    - [x] Verify the submodule add is not blocked by the `/vendor/` ignore rule (use `git submodule add -f` if needed).
- [x] Append to the plan.
    - [x] Confirm the `pyproject.toml` pin matches `e56963328b7476e615ce8dda9164d381f8dc07a3`.
    - [x] Add the VGGT submodule with `git submodule add -f` if the ignore rule blocks it.
    - [x] Checkout the pinned commit inside `vendor/vggt`.
    - [x] Document that the `/vendor/` gitignore required using `-f`.
    - [x] Stage the submodule gitlink with `git add -f vendor/vggt`.
    - [x] Verify the gitlink points at `e56963328b7476e615ce8dda9164d381f8dc07a3`.
    - [x] Record any submodule add issues (e.g., ignore rules, nested git config).


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. The VGGT submodule should be pinned to commit `e56963328b7476e615ce8dda9164d381f8dc07a3`.
2. The VGGT repository is intended to live under `vendor/` alongside `vipe`.


# Questions
-- Important questions about the task that cannot be answered without help.

1. Do you want the submodule path to be `vendor/vggt`, or a different location?

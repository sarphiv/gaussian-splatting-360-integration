# Context
Review the change that converts VGGT from a git URL dependency into a git submodule under `vendor/` and wires it into `pyproject.toml`/`uv.lock`.


# Review
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- This file is an initial checklist only. Report outcomes (including skipped tests) in chat.


## Scope
### In scope
- `.gitmodules` entry for `vendor/vggt`.
- `vendor/vggt` submodule commit/branch selection.
- `pyproject.toml` `[tool.uv.sources]` update for `vggt`.
- `uv.lock` update to reflect the local VGGT source.

### Out of scope
- Runtime or training behavior changes.
- Any edits under `src/` or `tests/`.


## Requirements
- [ ] `vendor/vggt` is a git submodule pointing at the intended VGGT repo URL.
- [ ] Submodule is pinned to the agreed revision (or branch if specified).
- [ ] `pyproject.toml` uses a local path source for `vggt` (editable or not per decision).
- [ ] `uv.lock` reflects the local path source for `vggt`.
- [ ] Devcontainer sanity check: `.devcontainer/Containerfile` and `.devcontainer/docker-compose.yml` remain compatible with the local `vendor/vggt` source.
- [ ] Scope assertions (must remain unchanged): no changes outside `.gitmodules`, `vendor/vggt`, `pyproject.toml`, `uv.lock`, and devcontainer files if absolutely required.


## Verification
- [ ] Manual check: inspect `.gitmodules` for the new `vggt` entry and correct URL/path.
- [ ] Test: `git submodule status` shows `vendor/vggt` at the expected revision.
- [ ] Edge case or risk area: ensure `/vendor/` ignore rule did not block submodule addition.
- [ ] Artifact or output inspection: confirm `uv.lock` has a path-based `vggt` source entry.
- [ ] Manual check: review `.devcontainer/Containerfile` and `.devcontainer/docker-compose.yml` for path assumptions about `vendor/`.


# Questions
-- Important questions that require clarification.

1. Confirm the exact VGGT commit or branch that should be pinned in the submodule.

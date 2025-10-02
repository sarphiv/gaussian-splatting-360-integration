# Rename RoomSample360

## General Plan
- Identify every code and documentation reference to `RoomSample360` to determine the full rename surface, including class definitions, imports, and narrative docs.
- Plan the class rename and dependent updates so all type hints, exports, and docstrings consistently use `SceneSample`.
- Plan the documentation refresh and follow-up verification to ensure no stale references remain.

## Detailed Plan
### Identify every reference to `RoomSample360`
- Confirm the following files contain the `RoomSample360` identifier and note the context to be updated:
  - `src/splat_init/data/datamodule_360.py`: class definition, docstrings, `__all__` export, type hints within `DataModule360`, helper functions, and dataset stubs.
  - `src/splat_init/models/vggt_perspective_transform.py`: import from the datamodule and all LightningModule signatures using the type.
  - `src/splat_init/models/vggt_naive_equirectangular.py`: import, docstring prose, preprocessing helper signatures, and batch typing.
  - `src/splat_init/data/stanford_2d_3d.py`: dataset return type, iterator helper, and constructor docstring.
  - `tasks/vggt_naive_equirectangular.md`: narrative references to the data type in the plan notes.
- Double-check for any additional hits (e.g., notebooks under `investigations/`) and decide whether they require changes or can be left as historical scratch work.

### Rename the class and dependent code paths
- In `src/splat_init/data/datamodule_360.py`, rename the `RoomSample360` class to `SceneSample`, update its docstring, adjust the `__all__` tuple, and replace every local type hint or inline comment referencing the old name.
- In `src/splat_init/models/vggt_perspective_transform.py`, update the import and every method signature or type alias that relies on `RoomSample360` to use `SceneSample`.
- In `src/splat_init/models/vggt_naive_equirectangular.py`, rename the imported symbol, all annotations, and docstrings that mention the old type, ensuring consistency in helper return types and batch structures.
- In `src/splat_init/data/stanford_2d_3d.py`, swap the import and constructor/return annotations so the dataset emits `SceneSample` instances; adjust iterator helper annotations accordingly.
- Update any remaining code locations discovered in the identification phase (e.g., notebook snippets) to maintain a repo-wide consistent type name.

### Refresh documentation and verify the rename
- Edit `tasks/vggt_naive_equirectangular.md` (and any other markdown hits) so the prose uses `SceneSample`.
- If notebook text/code cells reference the old name, replace them while keeping execution counts untouched.
- Rerun `rg "RoomSample360"` to confirm the identifier has been fully removed, and spot-check diffs for accidental regressions.

## TODO / Questions
- [x] Run `rg "RoomSample360"` after the changes to ensure no residual references remain (only references left in this task plan).
- [x] Review the updated docs for clarity once the rename is complete.
- Assumption: `SceneSample` retains the exact structure/semantics of `RoomSample360` and no API beyond the name change is required.

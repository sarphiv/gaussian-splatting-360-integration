# Context
Implement refactoring of the `OTCProjector` class out of `src/splat_init/models/vggt_perspective_transform.py` into the shared `src/utilities` package (proposed path: `src/utilities/otc_projector.py`). Ensure functionality remains unchanged while centralizing the projector for reuse, updating all imports, and adding any necessary tests or adjustments.

# Plan
- Only use the existing headings of the template.
- Lines beginning with `-` or `#` may not be modified.
- Plans are written as nested to-do lists of actions and changes.
- The plan is a living document so it must be updated to be accurate.
- The plan also serves as a log, so all actions and changes must be present.
- All items must be broken down into general and detailed steps in that order.
- Update the general plan first, then the detailed plan, then execute, then review. Repeat this line until completion.

## General
- [] Collect necessary information.
- [] Formulate overall approach to solve the task.
- [] Append to the plan with the above approach.
- [] ...
- [] Audit current `OTCProjector` implementation and dependencies.
- [] Extract `OTCProjector` into `src/utilities/otc_projector.py` with required helpers.
- [] Update all code to import the refactored projector and remove duplicated definitions.
 - [] Avoid creating or running tests per instruction; keep existing coverage untouched.
 - [] Document module-level API and ensure style guidelines (type hints, docstrings, loguru usage) are met.

## Detailed
- [] Collect necessary information.
    - [x] Explore the codebase.
    - [x] Explore the data for e.g. relevant metadata.
    - [] Search online for relevant information.
    - [] Describe a small experiment to verify or deepen understanding.
    - [] ...
- [] Formulate overall approach to solve the task.
    - [] Which code areas to change and with what.
    - [] Perhaps more information must be collected, which necessitates updating the plan.
    - [] ...
- [] Append to the plan with the above approach.
    - [] Only update the detailed section after the current update to the general plan is completed.
    - [] If issues or new information arises, the plan must be updated accordingly.
    - [] Assumptions and questions should be updated whenever necessary.
    - [] ...
- [] Inventory all usages of `OTCProjector` (search across `src/` and `tests/`) and note dependencies (imports, constants, tensor shapes).
- [] Create `src/utilities/otc_projector.py` with the extracted class, docstrings, and exports; ensure any constants remain accessible.
- [] Update `src/splat_init/models/vggt_perspective_transform.py` to import from utilities and remove the inline class definition.
- [] Adjust any `__init__.py` or utility aggregator files if needed for cleaner imports.
- [] Update references elsewhere to use the new utilities path; keep runtime behavior intact.
 - [] Do not add or run tests; ensure code is consistent without modifying test suites.
- [] Perform final self-review for style, dead imports, and adherence to project conventions.


# Assumptions
- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. `OTCProjector` relocation does not require API changes beyond its import path.
2. No external consumers outside the repo depend on the current module path.
3. Creating `src/utilities/otc_projector.py` aligns with existing utilities naming conventions.


# Questions
- Important questions about the task that cannot be answered without help.

1. Should the projector expose any helper functions separately from the class when moved?
2. Is there a preferred test suite or quick check to validate this refactor (e.g., specific `pytest` markers)?
3. Should utilities modules be surfaced via an `__init__.py` for simpler imports?


# Output
OTCProjector and `cube_face_relative_rotations` relocated to `src/utilities/otc_projector.py` and imported into `src/splat_init/models/vggt_perspective_transform.py`; functionality unchanged and tests were not run per instructions.

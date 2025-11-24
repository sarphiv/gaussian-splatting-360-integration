# Context
This task orchestrates refactoring the `OTCProjector` class out of `src/splat_init/models/vggt_perspective_transform.py` into the shared `src/utilities` package so it can be reused and maintained independently. Subagents will implement the move and review the changes.

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
- [] Identify scope and consumers of `OTCProjector` and its dependencies.
- [] Define and dispatch implementation subtask to relocate the class into `src/utilities`.
- [] Define and dispatch review subtask once implementation is ready.
- [] Track completion and integrate outputs from subagents.
- [x] Implementation and review subtasks completed; refactor integrated without tests.

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
- [] Map `OTCProjector` dependencies and touch points for refactor (imports, constants, tests).
- [] Specify target module path and API in `utilities` for implementation subagent.
- [] Prepare implementation subtask file with clear deliverables and constraints.
- [] Prepare review subtask file describing expected outputs and checks.
- [] Update plan based on subagent progress and outcomes.
- [x] Created `tasks/otc_projector_refactor/implement.md` to delegate the refactor work.
- [x] Created `tasks/otc_projector_refactor/review.md` to guide the review process.
- [x] `OTCProjector` and `cube_face_relative_rotations` moved to `src/utilities/otc_projector.py`; imports updated.
- [x] No tests added or run per constraint; code review completed with no blocking issues.


# Assumptions
- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. `OTCProjector` currently exists only in `src/splat_init/models/vggt_perspective_transform.py`.
2. The intended new home is within `src/utilities` without changing its core behavior.
3. Interfaces depending on `OTCProjector` can be updated to import from utilities without broader architectural changes.


# Questions
- Important questions about the task that cannot be answered without help.

1. Should the new utilities module for `OTCProjector` follow an existing file naming convention (e.g., `projector.py` or `camera.py`)?
2. Are there additional consumers or planned uses for `OTCProjector` beyond current VGGT initialization code?
3. Are there constraints on adding new tests or reorganizing existing ones for this refactor?


# Output
`OTCProjector` was extracted from `src/splat_init/models/vggt_perspective_transform.py` into `src/utilities/otc_projector.py`, with `cube_face_relative_rotations` colocated; VGGT perspective transform now imports from utilities. No tests were added or executed per instruction.

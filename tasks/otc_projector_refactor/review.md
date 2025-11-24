# Context
Review the implementation that moves `OTCProjector` from `src/splat_init/models/vggt_perspective_transform.py` into the `src/utilities` package. Verify correctness, style, and that all usages are updated without behavioral regressions.

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
- [] Review code changes for correctness and consistency with style guidelines.
- [] Confirm all imports and references to `OTCProjector` point to the utilities module.
 - [] Confirm no new tests were added and no tests were executed per constraints; focus on code review only.
- [] Provide actionable feedback or approval.

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
- [] Examine new utilities module for completeness (docstrings, type hints, loguru usage, exports).
- [] Verify `vggt_perspective_transform` no longer defines `OTCProjector` and correctly imports it.
- [] Search for stale references to the old path across `src/` and `tests/`.
 - [] Verify no tests were added/modified; do not run any tests.
 - [] Validate correctness through inspection only, respecting the no-test constraint.
- [] Summarize findings, blocking issues, and recommendations.


# Assumptions
- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. The implementation subagent provides clear test results or guidance on what was executed.
2. The refactor is confined to the projector code and related imports, not broader architectural changes.
3. Available compute/time permits running at least a small targeted `pytest` subset.


# Questions
- Important questions about the task that cannot be answered without help.

1. Were any functional changes intended beyond relocating the class?
2. Are there preferred tests or datasets to prioritize when validating the projector behavior?
3. Should any documentation or README updates accompany the refactor?


# Output
Verified `OTCProjector` relocation to `src/utilities/otc_projector.py` with `cube_face_relative_rotations`; `vggt_perspective_transform.py` now imports from utilities and no residual definitions remain. No tests were added or executed per instruction. Code review found no blocking issues.

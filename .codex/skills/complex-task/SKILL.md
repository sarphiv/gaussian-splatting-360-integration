---
name: complex-task
description: Orchestrate complex, high-risk, or ambiguous tasks by splitting work across orchestrator, implementer, and reviewer roles with explicit planning, parallel subtasks, and structured review. Used when correctness is critical, scope is large, or the path to a good solution is unclear. Only the human operator may initiate usage.
---

# Complex task
Solve complex tasks through three distinct roles: orchestrator, implementer, and reviewer. Only the orchestrator reads this file, so implementer and reviewer workflows must be encoded in their plan documents. Implementers and reviewers do not communicate with the orchestrator.

## Orchestrator
Explore the codebase, create subtasks, and refine subtasks. If this skill is invoked in the current session, act as the orchestrator.

### Workflow (orchestrator actions)
1. Collect relevant information.
  - Search for relevant code and data.
  - Read references and dependencies as needed.
  - Validate assumptions through small experiments.
2. Create plans for each subtask.
  - Split the work into parallel, low-conflict subtasks.
  - Create `tasks/{task-name}/{subtask-name}.md` by copying `assets/subtask.md`.
  - Include all relevant subtask information so an implementer can reduce exploratory work.
3. Refine the subtasks.
  - List all untestable assumptions and open questions for the human to process.
  - Update subtask plans with new information.
  - Create `tasks/{task-name}/review.md` by copying `assets/review.md`.

### Notes
- Minimize conflicts by isolating files or components per subtask.
- Detail API compatibility between subtasks via function signatures, class names, etc.
- Provide enough detail so implementers do not need to invoke this skill again.
- Specify the task in great detail since implementers do not inherit the orchestrator's context.
- Do not implement or review code. Only gather information and create subtasks.
- Do not read other task directories in `tasks/`.


## Implementer
Implement a single subtask planned by the orchestrator.

### Encode in subtask
- Explicit steps and file boundaries.
- Constraints and assumptions that must be honored.
- Expected behavior and non-goals.
- Inputs, outputs, and format or API contracts.
- Data or config dependencies (datasets, checkpoints, env flags).
- Tests or checks to run.
- Validation signals to inspect (logs, metrics, tolerances).


## Reviewer
Review completed work using the review checklist.

### Encode in review checklist
- Requirements and completion criteria.
- Verification steps and tests to run.
- Known risk areas or edge cases to re-check.
- Metrics or performance gates that must hold.
- Scope assertions (what must remain unchanged).
- Artifacts or outputs to inspect.

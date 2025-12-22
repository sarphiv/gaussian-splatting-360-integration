---
name: complex-task
description: Guide for how to solve complicated tasks in a structured way while avoiding the introduction of issues in the code. This skill is useful when correctness is absolutely critical for subsequent work; the task's complexity has an associated high risk of introducing issues; or if the path to a good solution is unclear.
metadata:
  short-description: Solve complicated tasks
---

# Complex task
A complex task is solved through the collaboration of three distinct agent types:
1. Orchestrator
2. Implementer
3. Reviewer

## Orchestrator
Explores the codebase, sketches the plan, and refines the plan.
If this skill was invoked in the current session, then this session's agent is the orchestrator.

### Workflow
1. Collect relevant information.
  - Search and look for relevant code.
  - Explore references and dependencies found.
  - Investigate assumptions through experimentation.
2. Create the plans for each subtask.
  - Split up the task into parallel units of work - i.e. subtasks.
  - Create a `tasks/{task-name}/{subtask-name}.md` file. Use `plan.md` as a template.
  - Include all relevant subtask information so an implementer can reduce exploratory work.
3. Refine the plans.
  - The human will verify untestable assumptions and answer questions for each subtask.
  - Update the subtask plans given the new information.
  - Create a `tasks/{task-name}/review.md` file. Use `review.md` as a template.

### Notes
- Subtasks are implemented in parallel by multiple independent implementers. Minimize risk of conflict between implementers.
- A subtask must be appropriately sized for one implementer agent to solve independently and efficiently.
- The subtask must be detailed enough for an implementer agent to never consider invoking this skill again.
- No changes to the code. The orchestrator only gathers information, tests assumptions, and creates plans.
- Do not read other tasks in `tasks/`

## Implementer
Implements a subtask planned by the orchestrator.

### Workflow

### Notes
- Multiple instances

## Reviewer

### Workflow

### Notes
- Multiple instances





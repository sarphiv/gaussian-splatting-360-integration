# Context
Investigate `SequenceChunker` behavioral changes between 40a3f6dd and current that could degrade pose quality (even if execution is error-free), and propose minimal corrective steps.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Differences in device/dtype handling and chunking behavior in `src/splat_init/models/sequence_chunker.py`.
- Interaction between `SequenceChunker` defaults and caller-provided chunking settings.

### Out of scope
- Changes in other files except to reference expected inputs/outputs.


## Steps
- [x] Collect necessary information.
    - [x] Diff `sequence_chunker.py` against 40a3f6dd and list behavioral changes.
    - [x] Identify callsites and defaults for chunking (`args.model.chunker`).
    - [x] Note device/dtype changes that could alter numerical behavior.
- [x] Reconcile chunking behavior with DA3 settings.
    - [x] Note that DA3 has its own internal chunk_size/overlap unrelated to SequenceChunker.
- [ ] Formulate overall approach to solve the task.
    - [ ] Decide whether to restore device placement of `images` and `pose_pred`.
    - [ ] Decide whether chunking should be opt-in or default for DA3 in evaluation scripts.
    - [ ] Draft a minimal patch that preserves new API (`chunking: tuple[int, int] | None`) but restores numerical equivalence.
    - [ ] Define a quick validation (same input, compare pose drift or per-frame error).
- [ ] Append to the plan.
    - [ ] Update assumptions/questions if new callsites are discovered.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. Chunking behavior (enabled vs disabled) materially affects pose quality for long sequences.
2. Device placement differences do not change results for models that internally move tensors (validate).


# Questions
-- Important questions about the task that cannot be answered without help.

1. Was DA3 previously evaluated with chunking enabled, and what values were used?
2. Is the regression observed only when using `chunking=None`?

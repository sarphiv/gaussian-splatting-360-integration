# Context
Plan the integration steps needed to expose the pycolmap-cuda12 wrapper through
the evaluation CLI and model registry so it can be selected like existing models.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Extend `configs/evaluation_args.py` to include the new model name and options.
- Update `src/splat_init/evaluate.py` to construct the new model.
- Preserve the existing CLI usage pattern (no new flags required).
- Keep changes isolated to config/evaluation plumbing.

### Out of scope
- Implementing the pycolmap wrapper itself (handled in the wrapper subtask).
- Modifying datasets or model outputs.
- Installing pycolmap or CUDA dependencies.


## Steps
- [x] Collect necessary information.
    - [x] Review `configs/evaluation_args.py` and `src/splat_init/evaluate.py`.
    - [x] Note how models are mapped and how SequenceChunker is used.
    - [x] Confirm dataset outputs supply RGBA tensors expected by model wrappers.
    - [x] Describe a small CLI run to validate new model wiring.
- [x] Formulate overall approach to solve the task.
    - [x] Add `"pycolmap_perspective_transform"` to `ModelArgs.model` Literal.
    - [x] Do not add new CLI args; rely on model initializer defaults.
    - [x] If we later need overrides, add them as optional init params without changing CLI.
    - [x] Update `_build_model` in `src/splat_init/evaluate.py` to import and instantiate
          `PycolmapPerspectiveTransform` with the selected args.
    - [x] Ensure the `SequenceChunker` call path is unchanged for other models.
    - [x] Document the expected class name and constructor signature in this file.
        - [x] Expect `PycolmapPerspectiveTransform()` in
              `splat_init.models.pycolmap_perspective_transform`.
    - [x] Decide chunking default: use `chunker=None` for pycolmap in CLI runs unless
          users explicitly override, because COLMAP expects the full sequence.
    - [x] Note behavior if chunking is enabled: SequenceChunker will call the model
          on shorter sequences; ensure this is acceptable in docs/comments.
        - [x] Note: chunking behavior stays unchanged; the wrapper sees chunked sequences.
- [ ] Append to the plan.
    - [x] Confirm argument naming/tyro ergonomics require no changes.
    - [x] Record no backward-compat concerns beyond the new model label.
    - [ ] Add notes after running a minimal evaluation invocation.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. None.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

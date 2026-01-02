# Context
Wire the new DA3-Streaming panorama model into the evaluation CLI, including args and model construction logic.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Update `src/configs/evaluation_args.py` to include the new model name and defaults.
- Update `src/splat_init/evaluate.py` to construct the new model and wrap it in `SequenceChunker`.

### Out of scope
- Implementing the DA3 model itself.
- Changing dataset loaders or metrics.


## Steps
- [x] Collect necessary information.
    - [x] Inspect `evaluate.py` to see how models are instantiated and used.
    - [x] Inspect `evaluation_args.py` for model choice and chunker defaults.
    - [x] Describe a small experiment: run evaluation with an existing model to ensure the new wiring does not break current paths.
- [x] Formulate overall approach to solve the task.
    - [x] Add the DA3 model name `da3_perspective_transform` to `ModelArgs.model` Literal and update defaults/comments if needed.
    - [x] Import `Da3PerspectiveTransform` in `evaluate.py` and extend `_build_model` to handle it.
        - [x] Always wrap with `SequenceChunker(model=da3_model, ...)` for memory control and long sequences.
    - [x] Keep changes minimal and avoid altering evaluation metrics.
- [x] Append to the plan.
    - [x] Confirm no additional DA3 CLI arguments are needed for evaluation wiring.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. SequenceChunker chunk size/overlap will be tuned to fit the DA3 memory budget in practice.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

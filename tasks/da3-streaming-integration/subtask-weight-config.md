# Context
Build the DA3-Streaming asset management layer: a clean config file in `src/configs` plus a Python helper that downloads DA3 + SALAD weights into the torch cache and loads a YAML config with weight paths injected. The helper now lives under `src/utilities/da3_assets.py`.


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- Add `src/configs/depth_anything_3.yaml` copied from `vendor/depth-anything-3/da3_streaming/configs/base_config.yaml` with `Weights.*` set to null.
- Implement a helper module (new file) that resolves torch cache, downloads weights/config if missing, and returns absolute paths.
- Implement a helper function to load the DA3 config YAML and override `Weights` entries (and optional runtime overrides).

### Out of scope
- Implementing the DA3 streaming model wrapper or evaluation wiring.
- Modifying vendor code.


## Steps
- [x] Collect necessary information.
    - [x] Read `vendor/depth-anything-3/da3_streaming/configs/base_config.yaml` and `scripts/download_weights.sh`.
    - [x] Locate evaluation and model code to see how configs are typically handled.
    - [x] Identify torch cache and download utilities available (torch, huggingface_hub).
    - [x] Describe a small experiment: load the YAML config via the helper and assert weights paths are absolute (no download).
- [x] Formulate overall approach to solve the task.
    - [x] Create `src/configs/depth_anything_3.yaml` with the same structure as `base_config.yaml` but `Weights: {DA3: null, DA3_CONFIG: null, SALAD: null}`.
    - [x] Add a helper module `src/utilities/da3_assets.py` with:
        - [x] A `DA3StreamingAssets` dataclass containing `da3`, `da3_config`, `salad` (all `Path`).
        - [x] `resolve_torch_cache_root()` using `torch.hub.get_dir()` (parent of `hub/`) or `_get_torch_home()` if present.
        - [x] `ensure_da3_streaming_assets()` that downloads:
            - [x] SALAD ckpt from GitHub to `torch_cache/da3_streaming/dino_salad.ckpt` (skip if present).
            - [x] DA3 config.json + model.safetensors from HuggingFace `depth-anything/DA3NESTED-GIANT-LARGE-1.1` into the same cache dir (skip if present).
        - [x] Logging via `loguru` with clear paths and sizes.
    - [x] Add `load_da3_streaming_config(base_config_path, assets, overrides=None)` that loads YAML and injects weight paths into `Weights` keys using `assets`; apply any overrides.
    - [x] Document expected return types with docstrings and type hints.
    - [x] Ensure only ASCII text in the YAML file.
- [x] Append to the plan.
    - [x] Confirm weight-host URLs and repo IDs match `vendor/depth-anything-3/da3_streaming/scripts/download_weights.sh`.
    - [x] Confirm the model variant remains `DA3NESTED-GIANT-LARGE-1.1`.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. None.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

# Context
Build the DA3-Streaming asset management layer: a clean config file in `src/configs` plus a Python helper that downloads DA3 + SALAD weights into the torch cache and loads a YAML config with weight paths injected.


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
- [ ] Formulate overall approach to solve the task.
    - [ ] Create `src/configs/depth_anything_3.yaml` with the same structure as `base_config.yaml` but `Weights: {DA3: null, DA3_CONFIG: null, SALAD: null}`.
    - [ ] Add a helper module (e.g. `src/splat_init/models/da3_streaming_assets.py`) with:
        - [ ] A `DA3StreamingAssets` dataclass containing `da3`, `da3_config`, `salad` (all `Path`).
        - [ ] `resolve_torch_cache_root()` using `torch.hub.get_dir()` (parent of `hub/`) or `_get_torch_home()` if present.
        - [ ] `ensure_da3_streaming_assets()` that downloads:
            - [ ] SALAD ckpt from GitHub to `torch_cache/da3_streaming/dino_salad.ckpt` (skip if present).
            - [ ] DA3 config.json + model.safetensors from HuggingFace `depth-anything/DA3NESTED-GIANT-LARGE-1.1` into the same cache dir (skip if present).
        - [ ] Logging via `loguru` with clear paths and sizes.
    - [ ] Add `load_da3_streaming_config(base_config_path, assets, overrides=None)` that loads YAML and injects weight paths into `Weights` keys using `assets`; apply any overrides.
    - [ ] Document expected return types with docstrings and type hints.
    - [ ] Ensure only ASCII text in the YAML file.
- [ ] Append to the plan.
    - [ ] Update this plan if weight-host URLs or repo IDs change.
    - [ ] Record any adjustments if the model variant differs from `DA3NESTED-GIANT-LARGE-1.1`.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. None.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

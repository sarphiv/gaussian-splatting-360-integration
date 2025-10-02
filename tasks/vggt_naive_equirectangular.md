# VGGT Naive Equirectangular

## General Plan
- Clarify data shapes and preprocessing needed to reuse VGGT's loader logic directly on in-memory tensors.
- Design a clean LightningModule that feeds the preprocessed ERP views directly into VGGT and mirrors the existing loss stack.
- Define auxiliary utilities for pose decoding, loss computation, and position logging to keep the training/validation steps tidy.

## Detailed Plan
### Clarify data shapes and preprocessing needed to reuse VGGT's loader logic directly on in-memory tensors
- Deconstruct `load_and_preprocess_images` to capture its resize, crop, and padding behavior (including multiples-of-14 handling and whitening).
- Implement an in-memory preprocessing function that ingests `SceneSample` tensors, applies the same normalization, and returns RGB batches plus per-view masks mirroring the loader output.
- Establish the resizing strategy so RGB, alpha, and depth tensors all land on the same `VGGT_TARGET_SIZE` grid expected by the model outputs without ever serializing to disk.

### Design a clean LightningModule that feeds the preprocessed ERP views directly into VGGT and mirrors the existing loss stack
- Instantiate the pretrained VGGT model once, set it to eval, and implement a forward wrapper with autocast for inference.
- Structure helper methods to convert a `SceneSample` sample into batched VGGT inputs (RGB, depth, alpha masks) using the new in-memory preprocessing.
- Implement training/validation hooks that reshape predictions back to `[B, S, ...]`, merge per-view poses, enforce camera-rel pose normalization, and compute the depth/rotation/translation losses with clear weighting.

### Define auxiliary utilities for pose decoding, loss computation, and position logging to keep the training/validation steps tidy
- Reimplement quaternion→rotation matrix conversion and geodesic distance helpers with focused docstrings and type hints.
- Centralize loss computation and logging dictionary creation to avoid duplicated logic between stages.
- Write compact helpers to append predicted/target translations to stage-specific text files for downstream inspection.

## TODO / Questions
- [x] Confirm that serializing ERP tensors to temporary PNGs is acceptable and document any performance trade-offs. (Outcome: not acceptable.)
- [ ] Verify the resizing pipeline keeps RGB/depth/mask tensors aligned at `VGGT_TARGET_SIZE` before losses.
- [ ] Ensure the merged pose outputs and logged files match the expectations from the perspective baseline.
- Assumptions: Batch size remains 1; stage names stay limited to `train` and `val`.

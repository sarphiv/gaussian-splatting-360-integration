# Context
Plan the implementation of a pycolmap-cuda12 model wrapper that mirrors the existing
perspective-transform models and returns per-panorama w2c poses for sequences of 360
images, using the documented pycolmap API (no internet needed for implementers).


# Plan
-- Only use the existing headings of the template.
-- Lines beginning with `--` or `#` may not be modified.
-- The steps are also a log so all changes must be present.
-- Steps are written as nested verb-first to-do lists of actions.
-- Update the steps, execute, review. Repeat this until completion.


## Scope
### In scope
- New LightningModule wrapper in `src/splat_init/models/pycolmap_perspective_transform.py`.
- Projection of panoramas into perspective faces via `OTCProjector`.
- pycolmap feature extraction, matching, and incremental mapping on projected faces.
- Merging per-face poses into per-panorama pose outputs consistent with existing models.
- Minimal logging and debug outputs for inspection.
- Four cube faces (indices 0,1,4,5) and `OTCProjector(alpha=1e-9)` for cube map geometry.
- Avoid equirectangular camera models (COLMAP/pycolmap camera list does not include one).

### Out of scope
- Installing pycolmap-cuda12 or managing system-level CUDA dependencies.
- Changes outside `src/splat_init/models` (handled in integration subtask).
- Dataset format changes or new dataloaders.
- Persisting COLMAP artifacts (database/sparse) beyond temp directories.


## Steps
- [x] Collect necessary information.
    - [x] Review existing wrappers in `src/splat_init/models`.
    - [x] Inspect `OTCProjector` and face rotation utilities.
    - [x] Note evaluation expectations for pose outputs.
    - [x] Describe a small colmap run to validate pose extraction and merging:
        - [x] Project 2-3 panoramas to faces, run `extract_features`, `match_sequential`, `incremental_mapping`.
        - [x] Check `reconstruction.images` for registered images per face group.
        - [x] Confirm `image.cam_from_world.matrix()` -> 3x4 w2c and expand to 4x4.
        - [x] Validate partial merges: skip missing faces; identity fallback if all missing.
- [ ] Formulate overall approach to solve the task.
    - [ ] Define the `PycolmapPerspectiveTransform` API and constructor arguments.
    - [ ] Specify face projection parameters:
        - [ ] Use 4 faces with indices `[0, 1, 4, 5]` and `OTCProjector(alpha=1e-9)`.
        - [ ] Group projected images by face during COLMAP processing.
    - [ ] Outline the pycolmap pipeline using documented API signatures:
        - [ ] `pycolmap.extract_features(database_path, image_path, image_names, camera_mode, camera_model, reader_options, extraction_options, device)`.
        - [ ] `pycolmap.match_sequential(database_path, matching_options, pairing_options, verification_options, device)` for face-grouped sequences.
        - [ ] Optional fallback to `pycolmap.match_exhaustive(...)` if sequential fails.
        - [ ] `pycolmap.match_exhaustive(database_path, matching_options, pairing_options, verification_options, device)` (same option types).
        - [ ] `pycolmap.incremental_mapping(database_path, image_path, output_path, options, input_path, initial_image_pair_callback, next_image_callback)` returning recon dict.
        - [ ] Pass `image_names` in face-grouped order, e.g. `face_{face_idx}/frame_{frame_idx:06d}.png`.
    - [ ] Define pose extraction from reconstructions and merge to panorama space:
        - [ ] Use `reconstruction.images` / `reconstruction.image(image_id)` to access `Image`.
        - [ ] Use `image.cam_from_world` (Rigid3d) and `Rigid3d.matrix()` -> 3x4 w2c; expand to 4x4.
        - [ ] Convert numpy to torch and map back via `image.name` to `(frame_idx, face_idx)`.
        - [ ] Pick the reconstruction with max `num_images` when multiple models exist.
    - [ ] Describe logging, temporary storage, and cleanup strategy.
    - [ ] Use `loguru` for stage timing and reconstruction stats logging.
    - [ ] Plan the helper methods and file layout:
        - [ ] `_temporary_directory` for RAM-backed temp storage.
        - [ ] `_project_faces` to accept `[B, S, C, H, W]` and return `[S, F, 3, Hf, Wf]`.
        - [ ] `_write_face_images` to emit uint8 PNGs with stable naming and face-grouped ordering
              (e.g. `face_0/frame_000000.png`), to aid sequential matching.
        - [ ] `_run_colmap` to build database, extract features, match, and map.
        - [ ] `_poses_from_reconstruction` to parse per-image w2c transforms.
        - [ ] `_merge_face_poses` reusing DA3-style merge for panorama pose.
    - [ ] Specify return contract: `(poses_w2c, None, extras)` with extras including per-face poses.
    - [ ] Document camera intrinsics derivation for faces:
        - [ ] Use `face_size` and `alpha` from `OTCProjector` to set `fx=fy=0.5*face_size/tan(alpha)`.
        - [ ] Use `cx=cy=0.5*(face_size-1)` to align with pixel centers.
    - [ ] Ensure all new methods include docstrings and type annotations.
    - [ ] Set `ImageReaderOptions`/camera configuration:
        - [ ] `camera_mode=pycolmap.CameraMode.SINGLE` for shared intrinsics.
        - [ ] `camera_model="PINHOLE"` with `camera_params=[fx, fy, cx, cy]`.
        - [ ] Use `ImageReaderOptions.camera_model` / `camera_params` to fix intrinsics.
    - [ ] Set GPU/CPU options explicitly:
        - [ ] `FeatureExtractionOptions.use_gpu` and `FeatureExtractionOptions.gpu_index`.
        - [ ] `FeatureMatchingOptions.use_gpu` and `FeatureMatchingOptions.gpu_index`.
        - [ ] `device=pycolmap.Device.cuda` if GPU requested, else `Device.cpu` (enum also has `Device.auto`).
        - [ ] `SequentialPairingOptions.overlap` to control temporal matching within face groups.
    - [ ] Define merge behavior for missing registrations:
        - [ ] Only merge faces that registered for each frame.
        - [ ] If no faces registered for a frame, emit identity pose at origin.
        - [ ] If no reconstructions, emit identity poses for all frames.
- [ ] Append to the plan.
    - [ ] Update the plan if pycolmap API details require adjustments.
    - [ ] Update assumptions and questions after confirming camera model support.
    - [ ] Record any constraints discovered during a small test run.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. None.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

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
- [x] Formulate overall approach to solve the task.
    - [x] Define the `PycolmapPerspectiveTransform` API and constructor arguments.
    - [x] Specify face projection parameters:
        - [x] Use 4 faces with indices `[0, 1, 4, 5]` and `OTCProjector(alpha=1e-9)`.
        - [x] Group projected images by face during COLMAP processing.
    - [x] Outline the pycolmap pipeline using documented API signatures:
        - [x] `pycolmap.extract_features(database_path, image_path, image_names, camera_mode, camera_model, reader_options, extraction_options, device)`.
        - [x] Use `pycolmap.match_exhaustive(database_path, matching_options, pairing_options, verification_options, device)` to match like the default CLI pipeline.
        - [x] `pycolmap.incremental_mapping(database_path, image_path, output_path, options, input_path, initial_image_pair_callback, next_image_callback)` returning recon dict.
        - [x] Pass `image_names` in face-grouped order, e.g. `face_{face_idx}/frame_{frame_idx:06d}.png`.
    - [x] Define pose extraction from reconstructions and merge to panorama space:
        - [x] Use `reconstruction.images` / `reconstruction.image(image_id)` to access `Image`.
        - [x] Use `image.cam_from_world()` (Rigid3d) and `Rigid3d.matrix()` -> 3x4 w2c; expand to 4x4.
        - [x] Convert numpy to torch and map back via `image.name` to `(frame_idx, face_idx)`.
        - [x] Pick the reconstruction with max `num_points3D` to mirror COLMAP's mapper output ordering.
    - [x] Describe logging, temporary storage, and cleanup strategy.
    - [x] Use `loguru` for stage timing and reconstruction stats logging.
    - [x] Plan the helper methods and file layout:
        - [x] `_temporary_directory` for RAM-backed temp storage.
        - [x] `_project_faces` to accept `[B, S, C, H, W]` and return `[S, F, 3, Hf, Wf]`.
        - [x] `_write_face_images` to emit uint8 PNGs with stable naming and face-grouped ordering
              (e.g. `face_0/frame_000000.png`), to aid sequential matching.
        - [x] `_run_colmap` to build database, extract features, match, and map.
        - [x] `_poses_from_reconstruction` to parse per-image w2c transforms.
        - [x] `_merge_face_poses` reusing DA3-style merge for panorama pose.
    - [x] Specify return contract: `(poses_w2c, None, extras)` with extras including per-face poses.
    - [x] Document camera intrinsics derivation for faces:
        - [x] Use cube-map 90° FOV pinhole intrinsics: `fx=fy=0.5*(face_size-1)`.
        - [x] Use `cx=cy=0.5*(face_size-1)` to align with pixel centers.
    - [x] Ensure all new methods include docstrings and type annotations.
    - [x] Set `ImageReaderOptions`/camera configuration:
        - [x] `camera_mode=pycolmap.CameraMode.SINGLE` for shared intrinsics.
        - [x] `camera_model="PINHOLE"` with `camera_params="fx,fy,cx,cy"`.
        - [x] Use `ImageReaderOptions.camera_model` / `camera_params` to fix intrinsics.
    - [x] Set GPU/CPU options explicitly:
        - [x] `FeatureExtractionOptions.use_gpu` and `FeatureExtractionOptions.gpu_index`.
        - [x] `FeatureMatchingOptions.use_gpu` and `FeatureMatchingOptions.gpu_index`.
        - [x] When `use_gpu=True` for extraction, set `FeatureExtractionOptions.num_threads=1`
              to avoid multi-threaded GPU SIFT OpenGL context failures.
        - [x] `gpu_index` should be a string (e.g. `"0"`), not an int.
        - [x] `device=pycolmap.Device.cuda` if GPU requested, else `Device.cpu` (enum also has `Device.auto`).
    - [x] Define merge behavior for missing registrations:
        - [x] Only merge faces that registered for each frame.
        - [x] If no faces registered for a frame, emit identity pose at origin.
        - [x] If no reconstructions, emit identity poses for all frames.
- [x] Append to the plan.
    - [x] Update the plan if pycolmap API details require adjustments.
        - [x] Note `ImageReaderOptions.camera_params` expects a comma-separated string.
        - [x] Note `image.cam_from_world()` is a method on `pycolmap.Image`.
        - [x] Note that `alpha` is unrelated to FOV for cube-map faces; use 90° FOV intrinsics.
        - [x] Note the matching default is `match_exhaustive` to mirror COLMAP CLI.
    - [x] Update assumptions and questions after confirming camera model support.
    - [x] Record no constraints because no small test run was performed.


# Assumptions
-- Only for untestable assumptions. Testable assumptions should be verified through experiments.

1. None.


# Questions
-- Important questions about the task that cannot be answered without help.

1. None.

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from matplotlib.widgets import Button

NDArrayFloat = NDArray[np.float64]


def read_position_file(path: Path) -> List[NDArrayFloat]:
    """Return a list of scenes loaded from a comma-delimited position file."""
    raw_lines = path.read_text().splitlines()
    scenes: List[List[List[float]]] = []
    current_scene: List[List[float]] = []

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        if line == "---":
            if current_scene:
                scenes.append(np.asarray(current_scene, dtype=np.float64))
                current_scene = []
            continue
        parts = [part.strip() for part in line.split(",")]
        assert len(parts) == 3, "Each position must have exactly three coordinates."
        current_scene.append([float(part) for part in parts])

    if current_scene:
        scenes.append(np.asarray(current_scene, dtype=np.float64))

    return scenes


def procrustes_align(pred: NDArrayFloat, target: NDArrayFloat) -> NDArrayFloat:
    """Align predicted positions to targets using similarity Procrustes analysis."""
    assert pred.shape == target.shape, "Predicted and target arrays must match in shape."
    pred_centroid = pred.mean(axis=0)
    target_centroid = target.mean(axis=0)

    pred_centered = pred - pred_centroid
    target_centered = target - target_centroid

    covariance = pred_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)

    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt

    scale_numerator = singular_values.sum()
    scale_denominator = np.sum(pred_centered ** 2)
    assert scale_denominator > 0.0, "Predicted scene must span more than a single point."
    scale = scale_numerator / scale_denominator

    aligned = scale * (pred_centered @ rotation) + target_centroid
    return aligned


def main() -> None:
    """Align predicted positions to targets, report MSE, and render a 3D visualization."""
    repo_root = Path(__file__).resolve().parent.parent
    pred_path = repo_root / "outputs" / "positions_perspective_pred_val.txt"
    target_path = repo_root / "outputs" / "positions_perspective_target_val.txt"
    # pred_path = repo_root / "positions_pred_val.txt"
    # target_path = repo_root / "positions_target_val.txt"

    pred_scenes = read_position_file(pred_path)
    target_scenes = read_position_file(target_path)

    assert len(pred_scenes) == len(target_scenes), "Predicted and target files must have matching scene counts."

    aligned_scenes: List[NDArrayFloat] = []
    retained_targets: List[NDArrayFloat] = []

    for index, (pred_scene, target_scene) in enumerate(zip(pred_scenes, target_scenes), start=1):
        assert pred_scene.shape == target_scene.shape, f"Scene {index} has mismatched point counts."
        if pred_scene.shape[0] < 2:
            logger.warning(
                "Skipping scene {} because it only has {} point(s).",
                index,
                pred_scene.shape[0],
            )
            continue
        aligned_scene = procrustes_align(pred_scene, target_scene)
        aligned_scenes.append(aligned_scene)
        retained_targets.append(target_scene)

    assert aligned_scenes, "No scenes with at least two points were found for alignment."

    aligned_points = np.vstack(aligned_scenes)
    target_points = np.vstack(retained_targets)

    mse = np.mean(np.sum((aligned_points - target_points) ** 2, axis=1))
    logger.info("Mean squared error after Procrustes alignment: {:.6f}", mse)

    scene_count = len(aligned_scenes)
    logger.info("Rendering {} aligned scene(s).", scene_count)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.2)

    current_index = 0

    target_scene = retained_targets[current_index]
    aligned_scene = aligned_scenes[current_index]

    target_scatter = ax.scatter(
        target_scene[:, 0],
        target_scene[:, 1],
        target_scene[:, 2],
        label="Target",
        c="tab:blue",
        s=15,
    )
    aligned_scatter = ax.scatter(
        aligned_scene[:, 0],
        aligned_scene[:, 1],
        aligned_scene[:, 2],
        label="Aligned prediction",
        c="tab:orange",
        s=15,
    )

    line_artists = []

    def draw_connectors(scene_index: int) -> None:
        nonlocal line_artists
        for artist in line_artists:
            artist.remove()
        line_artists = [
            ax.plot(
                (pred_point[0], target_point[0]),
                (pred_point[1], target_point[1]),
                (pred_point[2], target_point[2]),
                c="0.6",
                linewidth=0.6,
            )[0]
            for pred_point, target_point in zip(
                aligned_scenes[scene_index], retained_targets[scene_index]
            )
        ]

    def update_scene(scene_index: int) -> None:
        nonlocal current_index
        current_index = scene_index % scene_count
        aligned = aligned_scenes[current_index]
        target = retained_targets[current_index]
        target_scatter._offsets3d = (target[:, 0], target[:, 1], target[:, 2])
        aligned_scatter._offsets3d = (aligned[:, 0], aligned[:, 1], aligned[:, 2])
        draw_connectors(current_index)
        ax.set_title(
            f"Procrustes Alignment of Predictions to Targets (Scene {current_index + 1}/{scene_count})"
        )
        fig.canvas.draw_idle()

    draw_connectors(current_index)
    ax.set_title(
        f"Procrustes Alignment of Predictions to Targets (Scene {current_index + 1}/{scene_count})"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.view_init(elev=20, azim=35)

    ax_prev = fig.add_axes([0.3, 0.05, 0.15, 0.05])
    ax_next = fig.add_axes([0.55, 0.05, 0.15, 0.05])
    button_prev = Button(ax_prev, "Previous")
    button_next = Button(ax_next, "Next")

    def on_prev_click(event) -> None:  # type: ignore[override]
        _ = event
        update_scene(current_index - 1)

    def on_next_click(event) -> None:  # type: ignore[override]
        _ = event
        update_scene(current_index + 1)

    button_prev.on_clicked(on_prev_click)
    button_next.on_clicked(on_next_click)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

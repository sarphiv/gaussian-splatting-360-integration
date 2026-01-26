from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ExperimentArgs:
    model: Literal[
        "vggt_naive_equirectangular",
        "vggt_perspective_transform",
        "vipe_panorama",
        "da3_perspective_transform",
        "pycolmap_perspective_transform",
        "ground_truth",
    ] = "vipe_panorama"
    chunker: tuple[int, int] = (48, 8)
    dataset_stride: int = 4


@dataclass(frozen=True)
class Args:
    output_dir: Path = Path("outputs")

    evaluate_poses: bool = False
    train_splats: bool = True
    evaluate_splats: bool = True

    experiments: list[ExperimentArgs] = field(default_factory=lambda: [
        ExperimentArgs("ground_truth", (0, 0), 16),
        ExperimentArgs("ground_truth", (0, 0), 8),
        ExperimentArgs("ground_truth", (0, 0), 4),
        ExperimentArgs("ground_truth", (0, 0), 2),

        ExperimentArgs("pycolmap_perspective_transform", (0, 0), 16),
        ExperimentArgs("pycolmap_perspective_transform", (0, 0), 8),
        ExperimentArgs("pycolmap_perspective_transform", (0, 0), 4),
        ExperimentArgs("pycolmap_perspective_transform", (0, 0), 2),

        ExperimentArgs("vipe_panorama", (48, 8), 16),
        ExperimentArgs("vipe_panorama", (48, 8), 8),
        ExperimentArgs("vipe_panorama", (48, 8), 4),
        ExperimentArgs("vipe_panorama", (48, 8), 2),

        ExperimentArgs("da3_perspective_transform", (0, 0), 16),
        ExperimentArgs("da3_perspective_transform", (0, 0), 8),
        ExperimentArgs("da3_perspective_transform", (0, 0), 4),
        ExperimentArgs("da3_perspective_transform", (0, 0), 2),

        ExperimentArgs("vggt_perspective_transform", (14, 6), 16),
        ExperimentArgs("vggt_perspective_transform", (14, 6), 8),
        ExperimentArgs("vggt_perspective_transform", (14, 6), 4),
        ExperimentArgs("vggt_perspective_transform", (14, 6), 2),

        ExperimentArgs("vggt_naive_equirectangular", (96, 16), 16),
        ExperimentArgs("vggt_naive_equirectangular", (96, 16), 8),
        ExperimentArgs("vggt_naive_equirectangular", (96, 16), 4),
        ExperimentArgs("vggt_naive_equirectangular", (96, 16), 2),
    ])


__all__ = [
    "Args",
    "ExperimentArgs"
]

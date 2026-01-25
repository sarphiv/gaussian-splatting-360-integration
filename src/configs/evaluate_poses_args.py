from __future__ import annotations

from dataclasses import dataclass
from os import environ
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelArgs:
    model: Literal[
        "vggt_naive_equirectangular",
        "vggt_perspective_transform",
        "vipe_panorama",
        "da3_perspective_transform",
        "pycolmap_perspective_transform",
        "ground_truth",
    ] = "pycolmap_perspective_transform"
    # Naive: (96, 16), Persp: (14, 6), ViPE: (48, 8), DA3: (0, 0), Pycolmap: (0, 0), GT: (0, 0)
    chunker: tuple[int, int] = (0, 0)  # (size, overlap)
    dtype: Literal["float32", "bfloat16"] = "float32"


@dataclass(frozen=True)
class DataArgs:
    dataset_dir: Path = Path(environ.get("DATASET_360_LOC_ROOT", ""))
    dataset_stride: int = 8 # NOTE: Must be even and above 1 for strided training
    dataset_offset: int = dataset_stride // 2 # NOTE: If using CLI, remember to also set this
    dataset_fps: float = 2 / dataset_stride # NOTE: If using CLI, remember to also set this
    dataset_image_size: tuple[int, int] = (1538, 768) # Width x Height to resize to

    dataloader_workers: int = 8


@dataclass(frozen=True)
class Args:
    seed: int = 1337

    results_dir: Path = Path(datetime.now(timezone.utc).strftime("outputs/%Y-%m-%dT%H:%M:%S"))
    data: DataArgs = DataArgs()
    model: ModelArgs = ModelArgs()


__all__ = [
    "Args",
    "DataArgs",
    "ModelArgs",
]

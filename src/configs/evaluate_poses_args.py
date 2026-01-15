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
    ] = "vipe_panorama"
    dtype: Literal["float32", "bfloat16"] = "float32"

    # Naive: (40, 15), Persp: (7, 4), ViPE: (32, 8), DA3: None, Pycolmap: None, GT: None
    chunker: tuple[int, int] | None = (32, 8)  # (size, overlap)


@dataclass(frozen=True)
class DataArgs:
    dataset_name: Literal["stanford_2d_3d", "360_loc"] = "360_loc"
    dataset_dir: Path = Path(environ.get("DATASET_360_LOC_ROOT", ""))
    dataset_stride: int = 4
    dataset_fps: float = 2 / dataset_stride
    dataset_image_size: tuple[int, int] = (1538, 768) # Width x Height

    dataloader_workers: int = 8


@dataclass(frozen=True)
class Args:
    seed: int = 1337

    output_dir: Path = Path(datetime.now(timezone.utc).strftime("outputs/%Y-%m-%dT%H:%M:%S"))
    data: DataArgs = DataArgs()
    model: ModelArgs = ModelArgs()


__all__ = [
    "Args",
    "DataArgs",
    "ModelArgs",
]

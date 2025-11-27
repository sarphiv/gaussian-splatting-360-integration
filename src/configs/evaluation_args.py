from __future__ import annotations

from dataclasses import dataclass
from os import environ
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelArgs:
    model: Literal["vggt_naive_equirectangular", "vggt_perspective_transform"] = "vggt_naive_equirectangular"
    dtype: Literal["float32", "bfloat16"] = "float32"

    chunker_chunk_size: int = 16
    chunker_chunk_overlap: int = 6


@dataclass(frozen=True)
class DataArgs:
    dataloader_workers: int = 4

    dataset_name: Literal["stanford_2d_3d", "360_loc"] = "360_loc"
    dataset_dir: Path = Path(environ.get("DATASET_360_LOC_ROOT", ""))
    dataset_stride: int = 3
    dataset_image_size: tuple[int, int] = (1538, 768) # Width x Height


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

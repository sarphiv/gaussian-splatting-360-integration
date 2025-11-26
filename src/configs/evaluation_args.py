from __future__ import annotations

from dataclasses import dataclass
from os import environ
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from lightning.fabric.plugins.precision.precision import (
    _PRECISION_INPUT_STR as PRECISION_INPUT_STR,
)


@dataclass(frozen=True)
class ModelArgs:
    precision: PRECISION_INPUT_STR = "16-mixed"
    model: Literal["vggt_naive_equirectangular", "vggt_perspective_transform"] = "vggt_perspective_transform"

    chunker_chunk_size: int = 16
    chunker_chunk_overlap: int = 6


@dataclass(frozen=True)
class DataArgs:
    dataloader_workers: int = 8

    dataset_name: Literal["stanford_2d_3d", "360_loc"] = "360_loc"
    dataset_dir: Path = Path(environ.get("DATASET_360_LOC_ROOT", ""))
    dataset_stride: int = 8


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

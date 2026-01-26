from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path


@dataclass(frozen=True)
class DataArgs:
    dataset_dir: Path = Path(environ.get("DATASET_360_LOC_ROOT", ""))
    dataloader_workers: int = 8


@dataclass(frozen=True)
class Args:
    results_dir: Path = Path("outputs/vggt_perspective_transform-8")
    data: DataArgs = DataArgs()


__all__ = [
    "Args",
    "DataArgs",
]

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from os import environ
from pathlib import Path
from typing import Literal
import os


@dataclass(frozen=True)
class DataArgs:
    """Dataset selection and loading settings for splat training."""

    dataset_name: Literal["stanford_2d_3d", "360_loc"] = "360_loc"
    dataset_dir: Path = Path(environ.get("DATASET_360_LOC_ROOT", ""))
    dataset_image_size: tuple[int, int] = (6144, 3072)  # Width x Height
    dataloader_workers: int = 8


@dataclass(frozen=True)
class ProjectionArgs:
    """Projection settings for converting panoramas to cube faces."""

    face_size: int = 1024
    projection_batch_size: int = 16
    image_workers: int = min(4, os.cpu_count() or 1)
    images_per_worker: int = 4


@dataclass(frozen=True)
class Args:
    """Arguments for the splat training wrapper entrypoint."""

    output_dir: Path = Path("outputs/2026-01-11T20:34:06")
    data: DataArgs = DataArgs()
    projection: ProjectionArgs = ProjectionArgs()


__all__ = [
    "Args",
    "DataArgs",
    "ProjectionArgs",
]

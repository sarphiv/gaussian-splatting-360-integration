from __future__ import annotations

from dataclasses import dataclass
from os import cpu_count
from pathlib import Path


@dataclass(frozen=True)
class ParallelArgs:
    """Parallelism settings for batched image evaluation."""
    jobs: int = max(1, min(4, cpu_count() or 1))
    images_per_job: int = 8
    image_workers: int = 4


@dataclass(frozen=True)
class Args:
    """Top-level CLI arguments for splat evaluation."""
    results_dir: Path = Path("outputs/2026-01-24T01:00:13")
    parallel: ParallelArgs = ParallelArgs()


__all__ = [
    "Args",
    "ParallelArgs",
]

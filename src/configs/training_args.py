"""Typed configuration dataclasses for training and data loading.

These are used by the 360° training entrypoint. Keep focused and minimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ
from pathlib import Path
from typing import Sequence
from lightning.fabric.plugins.precision.precision import (
    _PRECISION_INPUT_STR as PRECISION_INPUT_STR,
)


@dataclass(frozen=True)
class ModelArgs:
    """Minimal model hyperparameters.

    Extend as the model matures. Defaults target a stable baseline.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-6


@dataclass(frozen=True)
class DataArgs:
    """Dataset configuration for 360° training.

    Provide one or more Stanford area directories for each stage.
    """

    train_areas: Sequence[Path] = tuple(
        Path(environ.get("DATASET_STANFORD_2D_3D_ROOT", "")) / area
        for area in ("area_1", "area_2", "area_5a", "area_5b")
    )
    val_areas: Sequence[Path] = tuple(
        Path(environ.get("DATASET_STANFORD_2D_3D_ROOT", "")) / area
        for area in ("area_3", "area_4", "area_6")
    )
    test_areas: Sequence[Path] = field(default_factory=list)


@dataclass(frozen=True)
class Args:
    # Run + logging
    entity_name: str = environ.get("WANDB_ENTITY", "")
    project_name: str = environ.get("WANDB_PROJECT", "")
    run_name: str | None = None
    logging_step_period: int = 20
    checkpoint_save_n_best: int = 3
    checkpoint_save_every_n_steps: int = 2000

    # Data loading
    batch_size: int = 1  # rooms per batch
    dataloader_workers: int = 4
    seed: int = 1337

    # Trainer
    max_epochs: int = 1
    precision: PRECISION_INPUT_STR = "16-mixed"

    # Sub-configs
    data: DataArgs = DataArgs()
    model: ModelArgs = ModelArgs()


__all__ = [
    "Args",
    "DataArgs",
    "ModelArgs",
]

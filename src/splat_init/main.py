from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import tyro
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch

from splat_init.data.datamodule_360 import DataModule360
from splat_init.data.stanford_2d_3d import Stanford2D3DDataset
from splat_init.models.vggt_perspective_transform import VggtPerspectiveTransform
from splat_init.models.vggt_naive_equirectangular import VggtNaiveEquirectangular
from configs.training_args import Args



# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _stanford_callables(
    paths: Sequence[Path],
    max_sequence_length: int | None = None,
) -> list[Callable[[], Stanford2D3DDataset]]:
    """Create dataset constructors for provided Stanford area directories."""

    return [lambda p=p: Stanford2D3DDataset(p, max_sequence_length) for p in paths]


def _build_datamodule(args: Args) -> DataModule360:
    train_fns = _stanford_callables(args.data.train_areas, args.data.max_sequence_length)
    val_fns = _stanford_callables(args.data.val_areas, args.data.max_sequence_length) if args.data.val_areas else []

    dm = DataModule360(
        train_datasets=train_fns,
        val_datasets=val_fns,
        batch_size=args.batch_size,
        num_workers=args.dataloader_workers,
        seed=args.seed,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False,
    )
    return dm


def _build_trainer(args: Args, logger: WandbLogger) -> L.Trainer:
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"{logger.experiment.id}" + ":top:{epoch:02d}:{step}:{val_loss:.3f}",
            every_n_train_steps=args.checkpoint_save_every_n_steps,
            save_top_k=args.checkpoint_save_n_best,
            mode="min",
            monitor="val_loss",
        ),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"{logger.experiment.id}" + ":all:{epoch:02d}:{step}:{val_loss:.3f}",
            every_n_train_steps=args.checkpoint_save_every_n_steps,
            save_top_k=-1,
        ),
    ]

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        logger=logger,
        log_every_n_steps=args.logging_step_period,
        callbacks=callbacks,
    )
    return trainer



# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    # Parse args and seed deterministically.
    args = tyro.cli(Args)
    L.seed_everything(args.seed)

    # Logger
    logger = WandbLogger(name=args.run_name, project=args.project_name, entity=args.entity_name)
    logger.experiment.config.update(asdict(args))

    # Data
    dm = _build_datamodule(args)

    # Model
    model = VggtPerspectiveTransform(output_dir=args.model.output_dir)
    # model = VggtNaiveEquirectangular()

    # Trainer
    torch.set_float32_matmul_precision("medium")
    trainer = _build_trainer(args, logger)

    # Fit + Evaluate
    trainer.fit(model, dm)
    if len(args.data.val_areas) > 0:
        trainer.validate(model, dm)


if __name__ == "__main__":
    main()

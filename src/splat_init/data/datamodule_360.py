"""360° Panorama Lightning DataModule and shared sample type.

This module provides a DataModule for 360° panorama datasets that yield one
sample per room (multiple views per room). Batches are lists of such room
samples. The DataModule is data-oriented, stage-agnostic beyond PyTorch
Lightning's standard setup stages, and aims for deterministic shuffling.

Conventions
- A dataset item is a ``SceneSample`` with fields:
  - id:    e.g. ``stanford-2d-3d/area-1/conference-room-1``
  - rgba:  uint8 tensor ``[S, 4, H, W]`` (RGBA; A masks cutouts)
  - depth: float32 tensor ``[S, 1, H, W]``
  - pose:  float32 tensor ``[S, 4, 4]``
- A DataModule batch is a Python list[SceneSample]. The collate function can
  optionally apply a transform to each element.

Determinism
- Shuffling uses a seeded ``torch.Generator``. Worker seeding initializes
  Python's ``random``, ``numpy``, and ``torch`` RNGs with a deterministic
  per-worker seed derived from the base seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# -----------------------------------------------------------------------------
# Shared sample type
# -----------------------------------------------------------------------------


@dataclass
class SceneSample:
    """One scene-worth of aligned panoramic views.

    Attributes
    - id:           Canonical identifier string.
    - rgba:         Tensor of shape [S, 4, H, W], dtype float32 (RGBA; A is cutout mask).
    - depth:        Tensor of shape [S, 1, H, W], dtype float32 (meters).
    - pose:         Tensor of shape [S, 4, 4], dtype float32.
    - focal_length: Optional tensor of shape [S], dtype float32 (pixels).
    """

    id: str
    rgba: Tensor
    depth: Tensor
    pose: Tensor
    focal_length: Tensor | None


# -----------------------------------------------------------------------------
# DataModule
# -----------------------------------------------------------------------------


class DataModule360(L.LightningDataModule):
    """Lightning DataModule for 360° panorama datasets producing room samples.

    Parameters
    - train_datasets: List of callables constructing training datasets.
    - val_datasets:   Optional list of callables for validation datasets.
    - test_datasets:  Optional list of callables for test datasets.
    - batch_size:     Number of rooms per batch.
    - num_workers:    DataLoader workers per loader.
    - seed:           Base seed for deterministic shuffling and worker RNGs.
    - pin_memory:     Pin memory for faster host→device transfer.
    - persistent_workers: Keep workers alive between iterations.
    - prefetch_factor: Batches prefetched per worker (requires num_workers>0).
    - shuffle_train/val/test: Shuffle flags per stage.
    - transform:      Optional callable applied to each SceneSample in collate.
    """

    def __init__(
        self,
        *,
        train_datasets: Sequence[Callable[[], Dataset[SceneSample]]],
        val_datasets: Sequence[Callable[[], Dataset[SceneSample]]] | None = None,
        test_datasets: Sequence[Callable[[], Dataset[SceneSample]]] | None = None,
        batch_size: int = 1,
        num_workers: int = 4,
        seed: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = 2,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
        transform: Callable[[SceneSample], SceneSample] | None = None,
    ) -> None:
        super().__init__()
        self._train_fns = list(train_datasets)
        self._val_fns = list(val_datasets) if val_datasets else []
        self._test_fns = list(test_datasets) if test_datasets else []

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self._transform = transform

        # Lazily created datasets
        self._train_ds: Dataset[SceneSample] | None = None
        self._val_ds: Dataset[SceneSample] | None = None
        self._test_ds: Dataset[SceneSample] | None = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None) and self._train_ds is None:
            train_list = [fn() for fn in self._train_fns]
            self._train_ds = _concat(train_list)
        if stage in ("validate", "fit", None) and self._val_ds is None:
            val_list = [fn() for fn in self._val_fns] if self._val_fns else []
            self._val_ds = _concat(val_list) if val_list else _EMPTY_DATASET
        if stage in ("test", None) and self._test_ds is None:
            test_list = [fn() for fn in self._test_fns] if self._test_fns else []
            self._test_ds = _concat(test_list) if test_list else _EMPTY_DATASET

    def train_dataloader(self) -> DataLoader[List[SceneSample]]:
        assert self._train_ds is not None, "call setup('fit') before train_dataloader()"
        return make_dataloader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            seed=self.seed + 0x9E3779B1 * 0,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader[List[SceneSample]]:
        assert self._val_ds is not None, "call setup('validate') before val_dataloader()"
        return make_dataloader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            seed=self.seed + 0x9E3779B1 * 1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> DataLoader[List[SceneSample]]:
        assert self._test_ds is not None, "call setup('test') before test_dataloader()"
        return make_dataloader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test,
            num_workers=self.num_workers,
            seed=self.seed + 0x9E3779B1 * 2,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self._collate,
        )

    def _collate(self, batch: List[SceneSample]) -> List[SceneSample]:
        if self._transform is None:
            return batch
        return [self._transform(x) for x in batch]


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def _concat(datasets: Sequence[Dataset[SceneSample]]) -> Dataset[SceneSample]:
    """Concatenate datasets if there are more than one, else return the single.

    This keeps indexing fast while allowing multi-area/multi-dataset training.
    """

    assert len(datasets) > 0, "expected at least one dataset"
    return datasets[0] if len(datasets) == 1 else ConcatDataset(list(datasets))


# Minimal empty dataset to satisfy loader types when a stage has no data.
class _EmptyDataset(Dataset[SceneSample]):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> SceneSample:  # pragma: no cover - never called
        raise IndexError


_EMPTY_DATASET: Dataset[SceneSample] = _EmptyDataset()


def _seed_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and Torch using the worker's initial seed.

    Relies on torch to set each worker's base seed deterministically when a
    ``generator`` is passed to DataLoader.
    """
    import random
    import numpy as np

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataloader(
    ds: Dataset[SceneSample],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    collate_fn: Callable[[List[SceneSample]], List[SceneSample]],
) -> DataLoader[List[SceneSample]]:
    """Factory for DataLoader with deterministic shuffling and worker seeding."""

    gen = torch.Generator()
    gen.manual_seed(seed)

    kwargs: dict = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        collate_fn=collate_fn,
        generator=gen,
        worker_init_fn=_seed_worker,
    )
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "SceneSample",
    "DataModule360",
]

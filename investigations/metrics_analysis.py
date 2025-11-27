"""Utilities for inspecting evaluation metrics across model runs.

This script expects one or more output directories from ``src/splat_init/evaluate.py``.
Populate ``PRED_PATHS`` (alias ``PRED_PATH``) with the run folders you want to
inspect and execute the sections below to visualize runtime and pose quality
metrics.
"""
# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import torch as th
from loguru import logger


## %% Paths to prediction runs
# Add the output folders you want to analyze. Each should contain per-scene
# ``metrics.pt`` files produced by ``src/splat_init/evaluate.py``.
PRED_PATHS: list[Path] = [Path(p) for p in [
    "../outputs/2025-11-27T05:34:31",  # ViPE, dataset stride 4
    "../outputs/2025-11-27T08:02:11",  # ViPE, dataset stride 8
    "../outputs/2025-11-27T07:47:02",  # ViPE, dataset stride 16
    "../outputs/2025-11-27T06:39:15",  # VGGT Naive, dataset stride 4
    "../outputs/2025-11-27T06:50:17",  # VGGT Naive, dataset stride 8
    "../outputs/2025-11-27T06:54:15",  # VGGT Naive, dataset stride 16
    "../outputs/2025-11-27T06:59:04",  # VGGT Perspective, dataset stride 4
    "../outputs/2025-11-27T07:26:26",  # VGGT Perspective, dataset stride 8
    "../outputs/2025-11-27T07:36:05",  # VGGT Perspective, dataset stride 16
]]


## %% Data structures and loading helpers
@dataclass
class SceneMetrics:
    """Pose and runtime metrics for a single evaluated scene."""

    run_path: Path
    scene_id: str
    model_name: str
    dataset_stride: int
    chunker_chunk_size: int
    chunker_chunk_overlap: int
    sequence_length: int
    elapsed_seconds: float
    translation_mean: float
    translation_std: float
    rotation_geodesic_mean: float
    rotation_geodesic_std: float
    rotation_pointing_mean: float
    rotation_pointing_std: float
    rotation_roll_mean: float
    rotation_roll_std: float
    gpu_memory_peak: float
    cpu_memory_rss: float

    @property
    def fps(self) -> float:
        """Frames processed per second for the scene."""

        return self.sequence_length / self.elapsed_seconds


def load_all_metrics(run_paths: Iterable[Path]) -> list[SceneMetrics]:
    """Load metrics for all scenes across the provided run directories."""

    records: list[SceneMetrics] = []
    for run_path in run_paths:
        for metrics_file in sorted(run_path.rglob("metrics.pt")):
            raw = th.load(metrics_file, map_location="cpu")
            scene_id = metrics_file.parent.name
            records.append(
                SceneMetrics(
                    run_path=run_path,
                    scene_id=scene_id,
                    model_name=str(raw["model_name"]),
                    dataset_stride=int(raw["dataset_stride"]),
                    chunker_chunk_size=int(raw["chunker_chunk_size"]),
                    chunker_chunk_overlap=int(raw["chunker_chunk_overlap"]),
                    sequence_length=int(raw["sequence_length"]),
                    elapsed_seconds=float(raw["elapsed_seconds"]),
                    translation_mean=float(raw["translation_error_mean"]),
                    translation_std=float(raw["translation_error_std"]),
                    rotation_geodesic_mean=float(raw["rotation_geodesic_mean"]),
                    rotation_geodesic_std=float(raw["rotation_geodesic_std"]),
                    rotation_pointing_mean=float(raw["rotation_pointing_mean"]),
                    rotation_pointing_std=float(raw["rotation_pointing_std"]),
                    rotation_roll_mean=float(raw["rotation_roll_mean"]),
                    rotation_roll_std=float(raw["rotation_roll_std"]),
                    gpu_memory_peak=float(raw["gpu_memory_peak"]),
                    cpu_memory_rss=float(raw["cpu_memory_rss"]),
                )
            )

    logger.info("Loaded {} scenes of metrics", len(records))
    return records


## %% Load metrics into memory
metrics = load_all_metrics(PRED_PATHS)


## %% Runtime and memory plots
def _bar_plot(values: list[float], labels: list[str], ylabel: str, title: str) -> None:
    """Render a simple bar plot with value annotations."""

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.bar_label(bars, fmt="{:.2f}")
    fig.tight_layout()
    plt.show()


def plot_fps_per_run(records: list[SceneMetrics]) -> None:
    """Plot mean frames-per-second per run directory (average over scenes)."""

    per_run: dict[Path, list[float]] = {}
    for record in records:
        per_run.setdefault(record.run_path, []).append(record.fps)

    labels: list[str] = []
    fps_values: list[float] = []
    for run_path, fps_list in per_run.items():
        fps = float(np.mean(fps_list)) if fps_list else float("nan")
        model_names = {rec.model_name for rec in records if rec.run_path == run_path}
        model_label = ",".join(sorted(model_names)) if model_names else "unknown"
        strides = {rec.dataset_stride for rec in records if rec.run_path == run_path}
        assert len(strides) == 1, "Each run is expected to use a single dataset stride"
        stride = strides.pop()
        labels.append(f"{model_label}/{stride}")
        fps_values.append(fps)

    _bar_plot(fps_values, labels, ylabel="Frames per second", title="Processing speed per run")


def plot_memory_bars(records: list[SceneMetrics]) -> None:
    """Plot peak GPU and CPU memory per run."""

    per_run_gpu: dict[Path, float] = {}
    per_run_cpu: dict[Path, float] = {}
    run_paths_in_order: list[Path] = []
    seen: set[Path] = set()
    for record in records:
        per_run_gpu[record.run_path] = max(per_run_gpu.get(record.run_path, 0.0), record.gpu_memory_peak)
        per_run_cpu[record.run_path] = max(per_run_cpu.get(record.run_path, 0.0), record.cpu_memory_rss)
        if record.run_path not in seen:
            seen.add(record.run_path)
            run_paths_in_order.append(record.run_path)

    labels: list[str] = []
    gpu_gb: list[float] = []
    cpu_gb: list[float] = []
    for run_path in run_paths_in_order:
        model_names = {rec.model_name for rec in records if rec.run_path == run_path}
        model_label = ",".join(sorted(model_names)) if model_names else "unknown"
        strides = {rec.dataset_stride for rec in records if rec.run_path == run_path}
        assert len(strides) == 1, "Each run is expected to use a single dataset stride"
        stride = strides.pop()
        labels.append(f"{model_label}/{stride}")
        gpu_gb.append(per_run_gpu.get(run_path, float("nan")) / 1024**3)
        cpu_gb.append(per_run_cpu.get(run_path, float("nan")) / 1024**3)

    _bar_plot(gpu_gb, labels, ylabel="GPU peak (GB)", title="Peak GPU memory per run")
    _bar_plot(cpu_gb, labels, ylabel="CPU RSS (GB)", title="Peak CPU memory per run")


def plot_chunker_bars(records: list[SceneMetrics]) -> None:
    """Plot chunker chunk size and overlap per run as overlaid bars."""

    per_run: dict[Path, SceneMetrics] = {}
    run_paths_in_order: list[Path] = []
    seen: set[Path] = set()
    for record in records:
        if record.run_path not in seen:
            seen.add(record.run_path)
            run_paths_in_order.append(record.run_path)
        per_run.setdefault(record.run_path, record)

    labels: list[str] = []
    sizes: list[float] = []
    overlaps: list[float] = []
    for run_path in run_paths_in_order:
        record = per_run[run_path]
        model_names = {rec.model_name for rec in records if rec.run_path == run_path}
        model_label = ",".join(sorted(model_names)) if model_names else "unknown"
        strides = {rec.dataset_stride for rec in records if rec.run_path == run_path}
        assert len(strides) == 1, "Each run is expected to use a single dataset stride"
        stride = strides.pop()
        labels.append(f"{model_label}/{stride}")
        sizes.append(float(record.chunker_chunk_size))
        overlaps.append(float(record.chunker_chunk_overlap))

    indices = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(indices, sizes, width=width, label="Chunk size", alpha=0.8)
    ax.bar(indices, overlaps, width=width, label="Chunk overlap", alpha=0.6)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Frames")
    ax.set_title("Chunker chunk size and overlap")
    ax.legend()
    fig.tight_layout()
    plt.show()


## %% Aggregation helpers for error plots
@dataclass
class AggregatedMetric:
    """Aggregated pose error for a model at a given dataset stride."""

    stride: int
    mean: float
    std: float


def aggregate_by_stride(records: Iterable[SceneMetrics], metric_mean: str, metric_std: str) -> dict[str, list[AggregatedMetric]]:
    """Aggregate pose metrics per model and dataset stride.

    Uses sequence length as the weight so that longer sequences contribute proportionally
    to the combined mean and variance.
    """

    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for record in records:
        key = (record.model_name, record.dataset_stride)
        stats = grouped.setdefault(key, {"frames": 0.0, "sum_mean": 0.0, "sum_second": 0.0})
        mean_value = getattr(record, metric_mean)
        std_value = getattr(record, metric_std)
        n = float(record.sequence_length)
        stats["frames"] += n
        stats["sum_mean"] += mean_value * n
        stats["sum_second"] += (std_value**2 + mean_value**2) * n

    aggregated: dict[str, list[AggregatedMetric]] = {}
    for (model_name, stride), stats in grouped.items():
        frames = stats["frames"]
        mean = stats["sum_mean"] / frames
        variance = max(stats["sum_second"] / frames - mean**2, 0.0)
        std = float(np.sqrt(variance))
        aggregated.setdefault(model_name, []).append(AggregatedMetric(stride, mean, std))

    for metrics in aggregated.values():
        metrics.sort(key=lambda item: item.stride)

    return aggregated


def plot_metric_vs_stride(
    records: Iterable[SceneMetrics],
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    to_degrees: bool = False,
) -> None:
    """Plot per-scene errors with error bars and highlight the per-stride mean.

    Small circles (with std error bars) show individual scenes; large circles show
    the weighted mean at each stride. Colors are keyed by model.
    """

    records = list(records)
    aggregated = aggregate_by_stride(records, metric_mean, metric_std)

    model_order: list[str] = []
    seen_models: set[str] = set()
    for record in records:
        if record.model_name not in seen_models:
            seen_models.add(record.model_name)
            model_order.append(record.model_name)

    model_count = len(model_order)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, model_name in enumerate(model_order):
        color = cmap(idx % cmap.N)
        offset = (idx - (model_count - 1) / 2) * 0.2
        per_model_records = [rec for rec in records if rec.model_name == model_name]

        strides = np.array([rec.dataset_stride for rec in per_model_records], dtype=float) + offset
        means = np.array([getattr(rec, metric_mean) for rec in per_model_records], dtype=float)
        stds = np.array([getattr(rec, metric_std) for rec in per_model_records], dtype=float)
        if to_degrees:
            means = np.degrees(means)
            stds = np.degrees(stds)

        ax.errorbar(
            strides + np.random.uniform(-0.1, 0.1, size=strides.shape),
            means,
            yerr=stds,
            fmt="o",
            markersize=4,
            capsize=3,
            linestyle="",
            color=color,
            alpha=0.6,
            label=model_name,
        )

        aggregated_metrics = aggregated.get(model_name, [])
        if aggregated_metrics:
            agg_strides = np.array([m.stride for m in aggregated_metrics], dtype=float) + offset
            agg_means = np.array([m.mean for m in aggregated_metrics], dtype=float)
            if to_degrees:
                agg_means = np.degrees(agg_means)

            ax.plot(
                agg_strides,
                agg_means,
                color=color,
                alpha=0.7,
                linewidth=1.5,
            )
            ax.scatter(
                agg_strides,
                agg_means,
                color=color,
                s=90,
                edgecolor="k",
                linewidth=1.2,
                alpha=0.9,
            )

    ax.set_xlabel("Dataset stride")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(ncol=max(1, model_count))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1.0, 2.0, 4.0, 6.0, 8.0]))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    plt.show()


## %% FPS and memory visualizations
plot_fps_per_run(metrics)
plot_memory_bars(metrics)
plot_chunker_bars(metrics)


## %% Pose error plots
plot_metric_vs_stride(metrics, "translation_mean", "translation_std", ylabel="Translation error (m)")


plot_metric_vs_stride(
    metrics,
    "rotation_geodesic_mean",
    "rotation_geodesic_std",
    ylabel="Rotation geodesic (deg)",
    to_degrees=True,
)


plot_metric_vs_stride(
    metrics,
    "rotation_pointing_mean",
    "rotation_pointing_std",
    ylabel="Rotation pointing (deg)",
    to_degrees=True,
)


plot_metric_vs_stride(
    metrics,
    "rotation_roll_mean",
    "rotation_roll_std",
    ylabel="Rotation roll (deg)",
    to_degrees=True,
)

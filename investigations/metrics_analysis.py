"""Utilities for inspecting evaluation metrics across model runs.

This script expects one or more output directories from ``src/splat_init/evaluate_poses.py``.
Populate ``PRED_PATHS`` (alias ``PRED_PATH``) with the run folders you want to
inspect and execute the sections below to visualize runtime and pose quality
metrics.
"""
# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import torch as th
from loguru import logger

PRED_PATHS: list[Path] = [
    Path("../outputs") / f"{model}-{stride}"
    for model in [
        "pycolmap_perspective_transform",
        "vggt_naive_equirectangular",
        "vggt_perspective_transform",
        "da3_perspective_transform",
        "vipe_panorama"
    ]
    for stride in [16, 8, 4, 2]
]

PRED_PATHS_ALL: list[Path] = [
    Path("../outputs") / f"{model}-{stride}"
    for model in [
        "ground_truth",
        "pycolmap_perspective_transform",
        "vggt_naive_equirectangular",
        "vggt_perspective_transform",
        "da3_perspective_transform",
        "vipe_panorama"
    ]
    for stride in [16, 8, 4, 2]
]


## %% Data structures and loading helpers
MODEL_DISPLAY_NAMES = {
    "pycolmap_perspective_transform": "COLMAP",
    "vggt_naive_equirectangular": "VGGT naive",
    "vggt_perspective_transform": "VGGT cube map",
    "da3_perspective_transform": "DA3 streaming",
    "vipe_panorama": "ViPE",
    "ground_truth": "Ground truth",
}


class PoseClass(StrEnum):
    """Outcome class for a pose evaluation run."""

    FAIL = "fail"
    WONK = "wonk"
    OKAY = "okay"
    TRAIN = "train"


POSE_CLASS_COLORS = {
    PoseClass.TRAIN: "#2ca02c",
    PoseClass.OKAY: "#f1c40f",
    PoseClass.WONK: "#d62728",
    PoseClass.FAIL: "#111111",
}


@dataclass
class SplatStepMetrics:
    """Splat evaluation metrics for a single optimization step."""

    psnr: list[float]
    ssim: list[float]
    lpips: list[float]


@dataclass
class SceneMetrics:
    """Pose and runtime metrics for a single evaluated scene."""

    run_path: Path
    scene_id: str
    scene_idx: int
    pose_class: PoseClass
    model_name: str
    dataset_stride: int
    dataset_offset: int
    dataset_fps: float
    dataset_image_width: int
    dataset_image_height: int
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
    gpu_memory_allocated: float
    gpu_memory_peak: float
    cpu_memory_rss: float
    splat_results: dict[str, SplatStepMetrics] | None

    @property
    def fps(self) -> float:
        """Frames processed per second for the scene."""

        return self.sequence_length / self.elapsed_seconds


def load_all_metrics(run_paths: Iterable[Path]) -> list[SceneMetrics]:
    """Load metrics for all scenes across the provided run directories."""

    class_names = ["fail", "wonk", "okay", "train"]
    records: list[SceneMetrics] = []
    for run_path in run_paths:
        for metrics_file in sorted(run_path.rglob("poses/metrics.pt")):
            raw = th.load(metrics_file, map_location="cpu")
            scene_id = metrics_file.parent.parent.name
            poses_dir = metrics_file.parent
            class_files = [name for name in class_names if (poses_dir / f"{name}.txt").exists()]
            assert len(class_files) == 1, f"Expected one pose class file in {poses_dir}"
            results_file = poses_dir.parent / "results" / "metrics.pt"
            splat_results: dict[str, SplatStepMetrics] | None = None
            if results_file.exists():
                raw_results = th.load(results_file, map_location="cpu")
                assert isinstance(raw_results, dict), f"Unexpected results format: {type(raw_results)}"
                metrics_by_step: dict[str, SplatStepMetrics] = {}
                for step, metrics in raw_results.items():
                    assert isinstance(metrics, dict), f"Unexpected metrics format for {step}: {type(metrics)}"
                    step_metrics = SplatStepMetrics(
                        psnr=[float(value) for value in metrics["psnr"]],
                        ssim=[float(value) for value in metrics["ssim"]],
                        lpips=[float(value) for value in metrics["lpips"]],
                    )
                    metrics_by_step[str(step)] = step_metrics
                splat_results = metrics_by_step
            records.append(
                SceneMetrics(
                    run_path=run_path,
                    scene_id=scene_id,
                    scene_idx=int(raw["scene_idx"]),
                    pose_class=PoseClass(class_files[0]),
                    model_name=str(raw["model_name"]),
                    dataset_stride=int(raw["dataset_stride"]),
                    dataset_offset=int(raw["dataset_offset"]),
                    dataset_fps=float(raw["dataset_fps"]),
                    dataset_image_width=int(raw["dataset_image_width"]),
                    dataset_image_height=int(raw["dataset_image_height"]),
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
                    gpu_memory_allocated=float(raw["gpu_memory_allocated"]),
                    gpu_memory_peak=float(raw["gpu_memory_peak"]),
                    cpu_memory_rss=float(raw["cpu_memory_rss"]),
                    splat_results=splat_results,
                )
            )

    logger.info("Loaded {} scenes of metrics", len(records))
    return records


## %% Load metrics into memory
metrics = load_all_metrics(PRED_PATHS)
metrics_all = load_all_metrics(PRED_PATHS_ALL)


## %% Runtime and memory plots
def _bar_plot(
    values: list[float],
    labels: list[str],
    ylabel: str,
    title: str = "",
    *,
    yerr: list[float] | None = None,
    bar_label_fmt: str = "{:.2f}",
) -> None:
    """Render a simple bar plot with value annotations."""

    assert len(values) == len(labels), "Each bar value must have a label."
    if yerr is not None:
        assert len(yerr) == len(values), "Each bar value must have a matching error bar."

    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(len(labels))
    bars = ax.bar(indices, values)
    if yerr is not None:
        ax.errorbar(
            indices,
            values,
            yerr=yerr,
            fmt="none",
            ecolor="orange",
            alpha=0.8,
            capsize=3,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if values:
        max_value = max(values)
        if yerr:
            max_value = max(max_value, max(value + err for value, err in zip(values, yerr)))
        if max_value > 0:
            ax.set_ylim(0, max_value * 1.12)

    parsed_labels: list[tuple[str, str]] = []
    for label in labels:
        group, stride = label.rsplit("/", 1) if "/" in label else ("", label)
        parsed_labels.append((group, stride))

    stride_labels = [stride for _, stride in parsed_labels]
    ax.set_xticks(indices)
    ax.set_xticklabels(stride_labels)
    ax.tick_params(axis="x", pad=6)

    groups: list[tuple[str, int, int]] = []
    if parsed_labels:
        group_start = 0
        current_group = parsed_labels[0][0]
        for idx, (group, _) in enumerate(parsed_labels + [("", "")]):
            group_change = idx == len(parsed_labels) or group != current_group
            if not group_change:
                continue
            if current_group:
                groups.append((current_group, group_start, idx - 1))
            group_start = idx
            current_group = group

    for group, start, end in groups:
        center = (start + end) / 2
        ax.text(
            center,
            -0.12,
            group,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
        )

    for _, _, end in groups[:-1]:
        boundary = end + 0.5
        ax.vlines(
            boundary,
            -0.12,
            0.00,
            transform=ax.get_xaxis_transform(),
            color="0.4",
            linewidth=1.4,
            clip_on=False,
        )
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            bar_label_fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    if any(group for group, _ in parsed_labels):
        fig.subplots_adjust(bottom=0.25)
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
        model_label = (
            ",".join(sorted(MODEL_DISPLAY_NAMES.get(name, name) for name in model_names)) if model_names else "unknown"
        )
        strides = {rec.dataset_stride for rec in records if rec.run_path == run_path}
        assert len(strides) == 1, "Each run is expected to use a single dataset stride"
        stride = strides.pop()
        labels.append(f"{model_label}/{stride}")
        fps_values.append(fps)

    _bar_plot(fps_values, labels, ylabel="Frames per second")


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
        model_label = (
            ",".join(sorted(MODEL_DISPLAY_NAMES.get(name, name) for name in model_names)) if model_names else "unknown"
        )
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
        model_label = (
            ",".join(sorted(MODEL_DISPLAY_NAMES.get(name, name) for name in model_names)) if model_names else "unknown"
        )
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
    # ax.legend()
    fig.tight_layout()
    plt.show()


def plot_pose_stat_bars(
    records: Iterable[SceneMetrics],
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    *,
    pose_classes: Iterable[PoseClass] | None = None,
    to_degrees: bool = False,
) -> None:
    """Plot mean pose metric per model/stride with scene-level variance error bars."""

    records = list(records)
    pose_classes = set(pose_classes) if pose_classes is not None else set(PoseClass)

    per_run: dict[Path, list[SceneMetrics]] = {}
    run_paths_in_order: list[Path] = []
    seen: set[Path] = set()
    for record in records:
        if record.run_path not in seen:
            seen.add(record.run_path)
            run_paths_in_order.append(record.run_path)
        per_run.setdefault(record.run_path, []).append(record)

    labels: list[str] = []
    means: list[float] = []
    errors: list[float] = []
    for run_path in run_paths_in_order:
        run_records = [rec for rec in per_run[run_path] if rec.pose_class in pose_classes]
        if not run_records:
            continue
        model_name = run_records[0].model_name
        stride = run_records[0].dataset_stride
        labels.append(f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}/{stride}")

        values = np.array([getattr(rec, metric_mean) for rec in run_records], dtype=float)
        variances = np.array([getattr(rec, metric_std) ** 2 for rec in run_records], dtype=float)
        if to_degrees:
            factor = 180.0 / np.pi
            values = values * factor
            variances = variances * (factor**2)
        means.append(float(np.mean(values)))
        errors.append(float(np.sqrt(np.sum(variances)) / len(variances)))

    _bar_plot(means, labels, ylabel=ylabel, yerr=errors)


def _select_splat_step(metrics_by_step: dict[str, SplatStepMetrics], step: str | None) -> str:
    """Select a splat evaluation step key, defaulting to the latest."""

    if step is not None:
        assert step in metrics_by_step, f"Requested step {step} missing from splat metrics."
        return step

    def step_key(name: str) -> tuple[int, str]:
        matches = re.findall(r"\d+", name)
        step_index = int(matches[-1]) if matches else -1
        return (step_index, name)

    return max(metrics_by_step, key=step_key)


def plot_splat_metric_bars(
    records: Iterable[SceneMetrics],
    metric: str,
    ylabel: str,
    *,
    pose_classes: Iterable[PoseClass] | None = None,
    step: str | None = None,
    bar_label_fmt: str = "{:.3f}",
) -> None:
    """Plot mean splat image metric per model/stride with per-scene variance error bars."""

    assert metric in {"psnr", "ssim", "lpips"}, f"Unexpected splat metric: {metric}"

    records = list(records)
    pose_classes = set(pose_classes) if pose_classes is not None else set(PoseClass)

    per_run: dict[Path, list[SceneMetrics]] = {}
    run_paths_in_order: list[Path] = []
    seen: set[Path] = set()
    for record in records:
        if record.run_path not in seen:
            seen.add(record.run_path)
            run_paths_in_order.append(record.run_path)
        per_run.setdefault(record.run_path, []).append(record)

    labels: list[str] = []
    means: list[float] = []
    errors: list[float] = []
    for run_path in run_paths_in_order:
        run_records = [
            rec
            for rec in per_run[run_path]
            if rec.pose_class in pose_classes and rec.splat_results is not None
        ]
        if not run_records:
            continue
        model_name = run_records[0].model_name
        stride = run_records[0].dataset_stride
        labels.append(f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}/{stride}")

        scene_means: list[float] = []
        scene_variances: list[float] = []
        for record in run_records:
            assert record.splat_results is not None
            step_key = _select_splat_step(record.splat_results, step)
            step_metrics = record.splat_results[step_key]
            values = np.array(getattr(step_metrics, metric), dtype=float)
            scene_means.append(float(np.mean(values)))
            scene_variances.append(float(np.var(values)))

        means.append(float(np.mean(scene_means)))
        errors.append(float(np.sqrt(np.sum(scene_variances)) / len(scene_variances)))

    _bar_plot(means, labels, ylabel=ylabel, yerr=errors, bar_label_fmt=bar_label_fmt)


def plot_pose_class_distribution(records: Iterable[SceneMetrics]) -> None:
    """Plot stacked pose-class distribution per model across scenes and strides."""

    model_order: list[str] = []
    seen_models: set[str] = set()
    for record in records:
        if record.model_name not in seen_models:
            seen_models.add(record.model_name)
            model_order.append(record.model_name)

    class_order = [PoseClass.TRAIN, PoseClass.OKAY, PoseClass.WONK, PoseClass.FAIL]
    counts: dict[str, dict[PoseClass, int]] = {model: {cls: 0 for cls in class_order} for model in model_order}
    totals: dict[str, int] = {model: 0 for model in model_order}
    for record in records:
        counts[record.model_name][record.pose_class] += 1
        totals[record.model_name] += 1

    indices = np.arange(len(model_order))
    bottoms = np.zeros(len(model_order), dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))

    for pose_class in class_order:
        values = np.array(
            [
                (counts[model][pose_class] / totals[model] * 100.0) if totals[model] > 0 else 0.0
                for model in model_order
            ],
            dtype=float,
        )
        bars = ax.bar(
            indices,
            values,
            bottom=bottoms,
            color=POSE_CLASS_COLORS[pose_class],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            if value <= 0.0:
                continue
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_y() + bar.get_height() / 2
            text_color = "black" if pose_class == PoseClass.OKAY else "white"
            ax.text(x, y, f"{value:.0f}%", ha="center", va="center", fontsize=9, color=text_color)
        bottoms += values

    ax.set_xticks(indices)
    ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(name, name) for name in model_order], ha="center")
    ax.set_ylabel("Pose success classification")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    plt.show()


## %% FPS and memory visualizations
plot_fps_per_run(metrics)
plot_memory_bars(metrics)
plot_chunker_bars(metrics)
plot_pose_class_distribution(metrics)


## %% Pose error plots
plot_pose_stat_bars(
    metrics,
    "translation_mean",
    "translation_std", 
    ylabel="Translation error (m) - All"
)
plot_pose_stat_bars(
    metrics,
    "translation_mean",
    "translation_std",
    ylabel="Translation error (m) - Okay & Train",
    pose_classes=[PoseClass.OKAY, PoseClass.TRAIN],
)
plot_pose_stat_bars(
    metrics,
    "translation_mean",
    "translation_std",
    ylabel="Translation error (m) - Train only",
    pose_classes=[PoseClass.TRAIN],
)

plot_pose_stat_bars(
    metrics,
    "rotation_geodesic_mean",
    "rotation_geodesic_std", 
    ylabel="Rotation geodesic error ($^\\circ$) - All",
    to_degrees=True
)
plot_pose_stat_bars(
    metrics,
    "rotation_geodesic_mean",
    "rotation_geodesic_std",
    ylabel="Rotation geodesic error ($^\\circ$) - Okay & Train",
    pose_classes=[PoseClass.OKAY, PoseClass.TRAIN],
    to_degrees=True
)
plot_pose_stat_bars(
    metrics,
    "rotation_geodesic_mean",
    "rotation_geodesic_std",
    ylabel="Rotation geodesic error ($^\\circ$) - Train only",
    pose_classes=[PoseClass.TRAIN],
    to_degrees=True
)

plot_pose_stat_bars(
    metrics,
    "rotation_pointing_mean",
    "rotation_pointing_std", 
    ylabel="Rotation pointing error ($^\\circ$) - All",
    to_degrees=True
)
plot_pose_stat_bars(
    metrics,
    "rotation_pointing_mean",
    "rotation_pointing_std",
    ylabel="Rotation pointing error ($^\\circ$) - Okay & Train",
    pose_classes=[PoseClass.OKAY, PoseClass.TRAIN],
    to_degrees=True
)
plot_pose_stat_bars(
    metrics,
    "rotation_pointing_mean",
    "rotation_pointing_std",
    ylabel="Rotation pointing error ($^\\circ$) - Train only",
    pose_classes=[PoseClass.TRAIN],
    to_degrees=True
)

plot_pose_stat_bars(
    metrics,
    "rotation_roll_mean",
    "rotation_roll_std", 
    ylabel="Rotation roll error ($^\\circ$) - All",
    to_degrees=True
)
plot_pose_stat_bars(
    metrics,
    "rotation_roll_mean",
    "rotation_roll_std",
    ylabel="Rotation roll error ($^\\circ$) - Okay & Train",
    pose_classes=[PoseClass.OKAY, PoseClass.TRAIN],
    to_degrees=True
)
plot_pose_stat_bars(
    metrics,
    "rotation_roll_mean",
    "rotation_roll_std",
    ylabel="Rotation roll error ($^\\circ$) - Train only",
    pose_classes=[PoseClass.TRAIN],
    to_degrees=True
)


## %% Splat image metric plots
plot_splat_metric_bars(
    metrics_all,
    "psnr",
    ylabel="Splat PSNR - Okay & Train",
    pose_classes=[PoseClass.OKAY, PoseClass.TRAIN],
)
plot_splat_metric_bars(
    metrics_all,
    "psnr",
    ylabel="Splat PSNR - Train only",
    pose_classes=[PoseClass.TRAIN],
)

plot_splat_metric_bars(
    metrics_all,
    "ssim",
    ylabel="Splat SSIM - Okay & Train",
    pose_classes=[PoseClass.OKAY, PoseClass.TRAIN],
)
plot_splat_metric_bars(
    metrics_all,
    "ssim",
    ylabel="Splat SSIM - Train only",
    pose_classes=[PoseClass.TRAIN],
)

plot_splat_metric_bars(
    metrics_all,
    "lpips",
    ylabel="Splat LPIPS - Okay & Train",
    pose_classes=[PoseClass.OKAY, PoseClass.TRAIN],
)
plot_splat_metric_bars(
    metrics_all,
    "lpips",
    ylabel="Splat LPIPS - Train only",
    pose_classes=[PoseClass.TRAIN],
)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any
import shutil
import urllib.request

from loguru import logger
import torch
import yaml

SALAD_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
DA3_REPO_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_BASE_URL = f"https://huggingface.co/{DA3_REPO_ID}/resolve/main"
DA3_CONFIG_NAME = "config.json"
DA3_MODEL_NAME = "model.safetensors"


@dataclass(frozen=True)
class DA3StreamingAssets:
    """Absolute paths to DA3 streaming weight artifacts."""

    da3: Path
    da3_config: Path
    salad: Path


def resolve_torch_cache_root() -> Path:
    """Return the root of the torch cache directory."""
    hub_dir = Path(torch.hub.get_dir()).expanduser()
    get_torch_home = getattr(torch.hub, "_get_torch_home", None)
    return (
        hub_dir.parent
        if hub_dir.name == "hub"
        else Path(get_torch_home()).expanduser()
        if get_torch_home is not None
        else hub_dir
    )


def ensure_da3_streaming_assets() -> DA3StreamingAssets:
    """Ensure DA3 streaming weights/config exist in the torch cache and return their paths."""
    cache_root = resolve_torch_cache_root()
    cache_dir = cache_root / "da3_streaming"
    cache_dir.mkdir(parents=True, exist_ok=True)

    salad_path = cache_dir / "dino_salad.ckpt"
    da3_config_path = cache_dir / DA3_CONFIG_NAME
    da3_path = cache_dir / DA3_MODEL_NAME

    _ensure_file("SALAD", SALAD_URL, salad_path)
    _ensure_file("DA3 config", f"{DA3_BASE_URL}/{DA3_CONFIG_NAME}", da3_config_path)
    _ensure_file("DA3 weights", f"{DA3_BASE_URL}/{DA3_MODEL_NAME}", da3_path)

    return DA3StreamingAssets(
        da3=da3_path.resolve(),
        da3_config=da3_config_path.resolve(),
        salad=salad_path.resolve(),
    )


def load_da3_streaming_config(
    base_config_path: Path | str,
    assets: DA3StreamingAssets,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the DA3 config YAML, inject weight paths, and apply optional overrides."""
    assert assets.da3.is_absolute()
    assert assets.da3_config.is_absolute()
    assert assets.salad.is_absolute()

    config_path = Path(base_config_path)
    config = yaml.safe_load(config_path.read_text())
    assert isinstance(config, dict)
    assert "Weights" in config
    assert isinstance(config["Weights"], dict)

    weights = config["Weights"]
    weights["DA3"] = str(assets.da3)
    weights["DA3_CONFIG"] = str(assets.da3_config)
    weights["SALAD"] = str(assets.salad)

    if overrides:
        _apply_overrides(config, overrides)

    return config


def _ensure_file(label: str, url: str, path: Path) -> None:
    if not path.exists():
        _download_file(url, path)
        logger.info(f"Downloaded {label} {_human_size(path.stat().st_size)}")


def _download_file(url: str, dest: Path) -> None:
    logger.info(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(dest.suffix + ".partial")
    with urllib.request.urlopen(url) as response:
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    temp_path.replace(dest)


def _apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(config.get(key), dict):
            _apply_overrides(config[key], value)
        else:
            config[key] = value


def _human_size(byte_count: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(byte_count)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    unit = units[unit_index]
    return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"

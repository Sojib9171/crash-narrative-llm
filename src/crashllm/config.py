"""Configuration loading, environment variables and seeding."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve all paths relative to the project root."""
    path = Path(path or os.environ.get("CRASHLLM_CONFIG", DEFAULT_CONFIG))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg["paths"].items():
        p = Path(value).expanduser()
        cfg["paths"][key] = p if p.is_absolute() else PROJECT_ROOT / p

    for key in ("split_dir", "results_dir", "models_dir"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)

    cfg["_config_path"] = str(path)
    return cfg


def load_env() -> None:
    """Load API keys (OPENAI_API_KEY, HF_TOKEN) from a .env file if present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def get_secret(name: str) -> str | None:
    load_env()
    value = os.environ.get(name)
    return value or None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

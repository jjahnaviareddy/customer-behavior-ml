"""
src/utils.py
------------
Shared utilities: logging setup, config loading, reproducibility.
"""

import logging
import random
import numpy as np
import yaml
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across numpy and random."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(*paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def print_section(title: str, width: int = 60) -> None:
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")

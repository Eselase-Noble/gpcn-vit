"""Portable path configuration.

All data/result locations are resolved relative to a configurable data root so
the project runs unchanged on a laptop, a Colab VM (``/content/...``), or Google
Drive (``/content/drive/MyDrive/...``). No personal machine paths are hard-coded.

Resolution order for the data root:
    1. The ``GPCN_DATA_ROOT`` environment variable, if set.
    2. ``<repo>/data`` as a sensible local default.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root = three levels up from this file (src/utils/paths.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def get_data_root() -> Path:
    """Return the configurable data root directory.

    Set ``GPCN_DATA_ROOT`` to point at the dataset location, e.g. on Colab::

        os.environ["GPCN_DATA_ROOT"] = "/content/data"
        os.environ["GPCN_DATA_ROOT"] = "/content/drive/MyDrive/gpcn-research/data"
    """
    env = os.environ.get("GPCN_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else (REPO_ROOT / "data")


def get_results_root() -> Path:
    """Return the configurable results root (checkpoints, metrics, logs)."""
    env = os.environ.get("GPCN_RESULTS_ROOT")
    return Path(env).expanduser().resolve() if env else (REPO_ROOT / "results")


# Conventional sub-directories (created lazily on demand, never at import time).
def metadata_dir() -> Path:
    return get_data_root() / "metadata"


def splits_dir() -> Path:
    return get_data_root() / "splits"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it. Colab-safe."""
    path.mkdir(parents=True, exist_ok=True)
    return path

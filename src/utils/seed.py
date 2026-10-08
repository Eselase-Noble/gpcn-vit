"""Reproducible seeding utilities.

Seeds Python's ``random``, NumPy, and (if installed) PyTorch. Torch is imported
lazily so the metadata/EDA pipeline can run in a minimal environment without a
deep-learning stack.

Note: full GPU determinism can cost performance and is not guaranteed across
hardware/library versions. ``deterministic=True`` requests it on a best-effort
basis; we document rather than promise bit-exact reproducibility.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """Seed all relevant RNGs. Returns the seed for logging convenience."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch not installed (e.g. metadata-only environment) — that's fine.
        pass

    return seed

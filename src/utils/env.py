"""Environment capture for reproducibility (§17).

Records Python/PyTorch/CUDA/GPU details and the current git commit so every
experiment can be traced to the exact software state that produced it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Dict


def get_device():
    """Return a torch.device, preferring CUDA. Imports torch lazily."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "N/A"
    except Exception:
        return "N/A"


def collect_env() -> Dict[str, Any]:
    """Collect a reproducibility snapshot. Torch/CUDA fields are 'N/A' if absent."""
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "torch_version": "N/A",
        "cuda_version": "N/A",
        "cuda_available": False,
        "gpu_name": "N/A",
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None) or "N/A"
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info

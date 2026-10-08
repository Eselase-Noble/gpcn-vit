"""Torch Dataset for BreakHis images, driven by a metadata DataFrame subset.

A fold is just a slice of the metadata table (produced by the patient-level
split), so the dataset never has to know about splitting — it only loads the
rows it is given, which keeps patient-level separation impossible to violate
here. Each item carries its row index so predictions can be joined back to
``patient_id`` / ``filename`` for patient-level aggregation and persistence.
"""

from __future__ import annotations

from typing import Callable, Dict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BreakHisImageDataset(Dataset):
    """Loads (image, label, row_index) triples from a metadata subset.

    Parameters
    ----------
    df: metadata rows for THIS fold only (columns include ``filepath``,
        ``label_binary``). The index is reset internally so ``row_index`` maps
        cleanly back onto ``df.iloc``.
    transform: torchvision/timm transform applied to the PIL image.
    """

    def __init__(self, df: pd.DataFrame, transform: Callable) -> None:
        if "filepath" not in df.columns or "label_binary" not in df.columns:
            raise ValueError("df must contain 'filepath' and 'label_binary'")
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        x = self.transform(img)
        return {
            "image": x,
            "label": torch.tensor(int(row["label_binary"]), dtype=torch.long),
            "index": idx,
        }

    def labels(self):
        """Return the fold's integer labels (for class-weight computation)."""
        return self.df["label_binary"].to_numpy()

    def meta_for_indices(self, indices) -> pd.DataFrame:
        """Return patient_id/filename/label rows for a batch of dataset indices."""
        cols = [c for c in ("filename", "patient_id", "label_binary") if c in self.df.columns]
        return self.df.iloc[list(indices)][cols].reset_index(drop=True)

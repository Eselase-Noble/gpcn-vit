"""Baseline 2 — gated-attention MIL pooling over a patient's frozen embeddings.

Where Baseline 1 averages every instance equally, this LEARNS per-instance
attention weights and pools by them (Ilse et al., 2018, "Attention-based Deep
MIL"). It is still a non-graph aggregator — instances are a set, with no edges —
so B2 vs B1 isolates "does a learned weighting beat a flat mean?", and both vs a
graph model isolate "do relationships add anything beyond better pooling?".

Same frozen ViT-S/16 features and same folds as B1 (controlled comparison). The
torch model lives here; bag construction is pure-numpy so it is unit-testable
without torch.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def make_bags(embeddings: np.ndarray, patient_ids: np.ndarray, labels: np.ndarray) -> List[dict]:
    """Group per-instance embeddings into one bag per patient.

    Returns a list of ``{patient_id, X (n_instances, D) float32, y int, n}`` in
    sorted-patient order. Asserts a single label per patient (leakage/parse guard).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N, D), got {embeddings.shape}")
    pid = np.asarray(patient_ids)
    y = np.asarray(labels).astype(int)
    if not (len(pid) == len(y) == len(embeddings)):
        raise ValueError("embeddings, patient_ids, labels must align in length")

    nun = pd.DataFrame({"pid": pid, "label": y}).groupby("pid")["label"].nunique()
    mixed = nun[nun > 1]
    if len(mixed) > 0:
        raise ValueError(f"patients with mixed labels (bug): {list(mixed.index)}")

    bags = []
    for p in np.sort(pd.unique(pid)):
        m = pid == p
        bags.append({"patient_id": p, "X": embeddings[m], "y": int(y[m][0]), "n": int(m.sum())})
    return bags


def build_model(in_dim: int, hidden: int = 128, n_classes: int = 2, dropout: float = 0.25):
    """Construct the gated-attention MIL model (import torch lazily)."""
    import torch
    import torch.nn as nn

    class GatedAttentionMIL(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn_V = nn.Linear(in_dim, hidden)
            self.attn_U = nn.Linear(in_dim, hidden)
            self.attn_w = nn.Linear(hidden, 1)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(in_dim, n_classes)

        def forward(self, bag):
            """bag: (n_instances, in_dim) -> (logits (1, n_classes), attention (n, 1))."""
            h = self.dropout(bag)
            a = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h)))  # (n,1)
            a = torch.softmax(a, dim=0)                       # attention over instances
            z = (a * bag).sum(dim=0, keepdim=True)            # (1, in_dim) weighted pool
            return self.classifier(z), a

    return GatedAttentionMIL()

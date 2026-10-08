"""Fold-local class weights for imbalanced training (Baseline 0 spec).

Weights MUST be computed from the training images of a single fold only — never
from validation/test, and never globally across the dataset, because global
weights leak information from held-out folds. This module takes already-isolated
training labels and returns weights; it is the caller's responsibility to pass
*only* training-fold labels (enforced by the training pipeline, tested here).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def compute_class_weights(
    y_train: Sequence[int],
    scheme: str = "balanced",
    n_classes: int = 2,
) -> Dict[str, object]:
    """Compute class weights from TRAINING labels only.

    scheme="balanced" uses the sklearn convention: ``w_c = n / (K * count_c)``,
    which up-weights the minority (benign) class. Returns weights plus the raw
    class counts so they can be logged per fold for reproducibility.

    A class absent from the training fold gets weight 0.0 (and is flagged via its
    zero count) rather than producing an infinity.
    """
    y = np.asarray(y_train).astype(int)
    if y.size == 0:
        raise ValueError("empty y_train — cannot compute class weights")
    if scheme != "balanced":
        raise ValueError(f"unsupported scheme: {scheme!r}")

    n = y.size
    counts = {c: int((y == c).sum()) for c in range(n_classes)}
    weights = {
        c: (n / (n_classes * counts[c])) if counts[c] > 0 else 0.0
        for c in range(n_classes)
    }
    return {
        "scheme": scheme,
        "n_train": int(n),
        "counts": counts,
        "weights": weights,
        # convenience: weight vector ordered by class index, for torch loss
        "weight_vector": [weights[c] for c in range(n_classes)],
    }

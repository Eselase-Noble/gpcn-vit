"""Classification metrics for the GPCN research protocol (§7).

Primary metrics are ROC-AUC and PR-AUC. Because BreakHis is imbalanced
(~2.2:1 malignant:benign), we also report balanced accuracy, sensitivity
(recall of the positive/malignant class), specificity, precision, and F1 — raw
accuracy alone is misleading here.

Convention: positive class = malignant = 1 (``label_binary`` from the parser).
``y_prob`` is the predicted probability of the positive class.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "n",
    "n_pos",
    "n_neg",
    "threshold",
)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the full protocol metric set from labels and positive-class probs.

    Threshold-free metrics (ROC-AUC, PR-AUC) use ``y_prob`` directly;
    threshold-dependent metrics binarise at ``threshold``.

    Degenerate cases are handled gracefully: if only one class is present,
    ROC-AUC/PR-AUC are set to ``nan`` (undefined) rather than raising, so a bad
    fold does not crash an entire evaluation run.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("empty input to compute_metrics")

    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    both_classes = n_pos > 0 and n_neg > 0

    # Threshold-free ranking metrics.
    roc = float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan")
    pr = float(average_precision_score(y_true, y_prob)) if both_classes else float("nan")

    # Confusion matrix over fixed label set so missing classes don't shift it.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = _safe_div(tp, tp + fn)   # recall of malignant
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "n": int(y_true.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": float(threshold),
    }


def aggregate_metrics(
    per_fold: Sequence[Dict[str, float]],
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate a list of per-fold metric dicts into mean/std/n (for CV, §8).

    Returns ``{metric: {"mean": .., "std": .., "n_folds": ..}}``. NaNs (e.g. a
    degenerate fold's AUC) are ignored in the aggregation so one bad fold does
    not poison the summary; ``n_folds`` reflects how many folds contributed.
    """
    if not per_fold:
        raise ValueError("no folds to aggregate")
    keys = keys or [k for k in per_fold[0] if isinstance(per_fold[0][k], (int, float))]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([f[k] for f in per_fold if k in f], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n_folds": 0}
        else:
            out[k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "n_folds": int(vals.size),
            }
    return out

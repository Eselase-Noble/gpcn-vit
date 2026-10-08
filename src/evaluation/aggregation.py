"""Image→patient aggregation, threshold selection, and two-level evaluation.

Baseline 0 trains at the image level but the PRIMARY evaluation unit is the
patient (spec: ``docs/baseline0_spec.md``). This module:

  * aggregates per-image malignant probabilities to a patient-level probability
    (fixed rule: mean of image probs);
  * selects a decision threshold using validation data only (never test);
  * produces image-level AND patient-level metric dicts.

Prediction tables are plain DataFrames with columns:
    ``patient_id``, ``label_binary`` (0/1), ``prob`` (P(malignant)).
Per-image and per-patient predictions should be persisted by the caller so
alternative aggregation schemes can be tried later without retraining.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def aggregate_to_patient(
    preds: pd.DataFrame,
    method: str = "mean",
    patient_col: str = "patient_id",
    label_col: str = "label_binary",
    prob_col: str = "prob",
) -> pd.DataFrame:
    """Collapse image-level predictions to one row per patient.

    Asserts each patient has a single ground-truth label (a mixed label would
    indicate a parsing/leakage bug, mirroring the split module's guard).
    Returns columns: ``patient_id``, ``label_binary``, ``prob``, ``n_images``.
    """
    if method != "mean":
        raise ValueError(f"unsupported aggregation method: {method!r}")
    if preds.empty:
        raise ValueError("empty predictions")

    label_nunique = preds.groupby(patient_col)[label_col].nunique()
    mixed = label_nunique[label_nunique > 1]
    if len(mixed) > 0:
        raise ValueError(f"patients with mixed labels (bug): {list(mixed.index)}")

    grouped = preds.groupby(patient_col)
    out = pd.DataFrame({
        "patient_id": grouped.size().index,
        "label_binary": grouped[label_col].first().to_numpy(),
        "prob": grouped[prob_col].mean().to_numpy(),
        "n_images": grouped.size().to_numpy(),
    }).reset_index(drop=True)
    return out


def select_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    objective: str = "balanced_accuracy",
) -> float:
    """Pick a decision threshold on VALIDATION data only (never test).

    Scans candidate thresholds (midpoints between sorted unique probabilities)
    and returns the one maximising ``objective``. Falls back to 0.5 if the
    validation set has a single class (threshold undefined from data).
    """
    y_val = np.asarray(y_val).astype(int)
    p_val = np.asarray(p_val, dtype=float)
    if y_val.size == 0:
        raise ValueError("empty validation set for threshold selection")
    if len(np.unique(y_val)) < 2:
        return 0.5

    uniq = np.unique(p_val)
    if uniq.size == 1:
        candidates = np.array([uniq[0] - 1e-6, uniq[0] + 1e-6])
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        candidates = np.concatenate([[0.0], mids, [1.0]])

    best_t, best_score = 0.5, -np.inf
    for t in candidates:
        m = compute_metrics(y_val, p_val, threshold=float(t))
        score = m[objective]
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def evaluate_predictions(
    preds: pd.DataFrame,
    threshold: float,
    aggregation: str = "mean",
) -> Dict[str, Dict[str, float]]:
    """Return both image-level and patient-level metrics at a fixed threshold.

    ``threshold`` must have been chosen on train/val, not on these predictions
    if they are the test fold.
    """
    image_metrics = compute_metrics(
        preds["label_binary"].to_numpy(),
        preds["prob"].to_numpy(),
        threshold=threshold,
    )
    patient = aggregate_to_patient(preds, method=aggregation)
    patient_metrics = compute_metrics(
        patient["label_binary"].to_numpy(),
        patient["prob"].to_numpy(),
        threshold=threshold,
    )
    return {"image": image_metrics, "patient": patient_metrics}

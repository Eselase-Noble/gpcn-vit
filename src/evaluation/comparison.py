"""Paired, cross-validated model comparison on held-out patients (protocol §Step 5).

Because the 5 test folds PARTITION the patients, concatenating every fold's test
predictions yields exactly ONE held-out prediction per patient per model. That
gives a fully cross-validated per-patient vector on which models can be compared
*paired* (same patients) — the correct unit, not per-fold means.

``paired_bootstrap_auc`` resamples patients (with replacement) to get a CI and a
two-sided p-value on the ROC-AUC difference between two models. With 81 patients
this is weak power — report it as supportive, not confirmatory (§Step 5: state
the limits of inference with so few patients/folds).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def load_pooled_predictions(run_dir: Path) -> pd.DataFrame:
    """Concatenate ``fold_*/preds_test_patient.csv`` into one row per patient.

    Raises if a patient appears in more than one test fold (would violate the
    partition / indicate a leak).
    """
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("fold_*/preds_test_patient.csv"))
    if not files:
        raise FileNotFoundError(f"no fold_*/preds_test_patient.csv under {run_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dups = int(df["patient_id"].duplicated().sum())
    if dups:
        raise ValueError(f"{dups} patient(s) appear in >1 test fold under {run_dir}")
    return (df[["patient_id", "label_binary", "prob"]]
            .sort_values("patient_id").reset_index(drop=True))


def pooled_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y, p = df["label_binary"].to_numpy().astype(int), df["prob"].to_numpy(dtype=float)
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "n_patients": int(len(y)), "n_pos": int((y == 1).sum())}


def paired_bootstrap_auc(y: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                         n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap of ROC-AUC(A) - ROC-AUC(B) over patients (same resample)."""
    y = np.asarray(y).astype(int)
    prob_a, prob_b = np.asarray(prob_a, float), np.asarray(prob_b, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], prob_a[idx]) - roc_auc_score(y[idx], prob_b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"auc_a": float(roc_auc_score(y, prob_a)),
            "auc_b": float(roc_auc_score(y, prob_b)),
            "diff": float(roc_auc_score(y, prob_a) - roc_auc_score(y, prob_b)),
            "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(min(p, 1.0)), "n_boot": int(len(diffs))}

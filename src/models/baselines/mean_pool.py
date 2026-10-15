"""Baseline 1 — patient-level MEAN POOLING of frozen patch embeddings.

Contrast with Baseline 0, whose patient score is the mean of per-image malignant
*posteriors*. Here we mean-pool the frozen ViT *embeddings* to one vector per
patient, then fit a light linear classifier. This is the simplest non-graph
aggregation: it discards all intra-patient structure (order, count, relations),
so it is the floor any graph model must beat to claim that relationships matter.

Pure numpy/sklearn — no torch — so it is unit-testable off-GPU. The classifier
standardises features using TRAIN statistics only (no val/test leakage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def patient_mean_embeddings(
    embeddings: np.ndarray,
    patient_ids: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse per-image embeddings to one mean vector per patient.

    Returns ``(patients_sorted, X, y, n_images)`` where ``X`` has one row per
    patient (mean of that patient's image embeddings). Asserts a single
    ground-truth label per patient (a mixed label would indicate a leakage/parse
    bug, mirroring the split and aggregation guards).
    """
    embeddings = np.asarray(embeddings, dtype=float)
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

    patients = np.sort(pd.unique(pid))
    X = np.vstack([embeddings[pid == p].mean(axis=0) for p in patients])
    y_out = np.array([int(y[pid == p][0]) for p in patients])
    n_images = np.array([int((pid == p).sum()) for p in patients])
    return patients, X, y_out, n_images


@dataclass
class MeanPoolClassifier:
    """StandardScaler (train-only) + class-weighted logistic regression.

    ``C`` is the one real hyperparameter; tune it on validation, never on test.
    Positive class is malignant=1; ``predict_proba_pos`` returns P(malignant).
    """

    C: float = 1.0
    max_iter: int = 2000
    seed: int = 42

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanPoolClassifier":
        self.scaler_ = StandardScaler().fit(X)
        self.clf_ = LogisticRegression(
            C=self.C, class_weight="balanced", max_iter=self.max_iter,
            random_state=self.seed,
        ).fit(self.scaler_.transform(X), y)
        return self

    def predict_proba_pos(self, X: np.ndarray) -> np.ndarray:
        proba = self.clf_.predict_proba(self.scaler_.transform(X))
        pos = list(self.clf_.classes_).index(1)  # P(malignant), robust to class order
        return proba[:, pos]

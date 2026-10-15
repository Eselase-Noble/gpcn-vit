"""Baseline 1 mean-pooling: per-patient pooling + leakage-safe classifier."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.baselines.mean_pool import MeanPoolClassifier, patient_mean_embeddings


def _toy(n_per_class=20, images_per=5, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    emb, pid, lab = [], [], []
    for cls in (0, 1):
        center = np.zeros(dim)
        center[0] = 3.0 if cls == 1 else -3.0
        for k in range(n_per_class):
            p = f"c{cls}_p{k}"
            for _ in range(images_per):
                emb.append(center + rng.normal(scale=1.0, size=dim))
                pid.append(p)
                lab.append(cls)
    return np.array(emb), np.array(pid), np.array(lab)


def test_patient_mean_shapes_and_labels():
    emb, pid, lab = _toy()
    patients, X, y, n_images = patient_mean_embeddings(emb, pid, lab)
    assert X.shape == (len(patients), emb.shape[1])
    assert len(y) == len(patients) == len(n_images)
    assert set(np.unique(y)) == {0, 1}
    assert (n_images == 5).all()


def test_mixed_patient_label_raises():
    emb = np.zeros((2, 4))
    with pytest.raises(ValueError):
        patient_mean_embeddings(emb, np.array(["p", "p"]), np.array([0, 1]))


def test_classifier_separates_and_is_deterministic():
    emb, pid, lab = _toy(seed=1)
    _, X, y, _ = patient_mean_embeddings(emb, pid, lab)
    clf = MeanPoolClassifier(C=1.0, seed=42).fit(X, y)
    p1 = clf.predict_proba_pos(X)
    p2 = MeanPoolClassifier(C=1.0, seed=42).fit(X, y).predict_proba_pos(X)
    assert np.allclose(p1, p2)  # deterministic given seed

    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p1) > 0.95  # separable toy -> high AUC

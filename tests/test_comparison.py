"""Paired cross-model comparison: pooling across folds + paired bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.comparison import (
    load_pooled_predictions,
    paired_bootstrap_auc,
    pooled_metrics,
)


def _write_fold(run_dir, fold, patients, labels, probs):
    d = run_dir / f"fold_{fold}"
    d.mkdir(parents=True)
    pd.DataFrame({"patient_id": patients, "label_binary": labels,
                  "prob": probs, "n_images": [1] * len(patients)}).to_csv(
        d / "preds_test_patient.csv", index=False)


def test_pooling_one_row_per_patient(tmp_path):
    _write_fold(tmp_path, 0, ["p0", "p1"], [0, 1], [0.1, 0.9])
    _write_fold(tmp_path, 1, ["p2", "p3"], [0, 1], [0.2, 0.8])
    df = load_pooled_predictions(tmp_path)
    assert len(df) == 4 and df["patient_id"].is_unique
    assert pooled_metrics(df)["roc_auc"] == 1.0  # perfectly separated


def test_pooling_rejects_duplicate_patient(tmp_path):
    _write_fold(tmp_path, 0, ["p0", "p1"], [0, 1], [0.1, 0.9])
    _write_fold(tmp_path, 1, ["p1", "p2"], [1, 0], [0.5, 0.2])  # p1 repeats -> leak
    with pytest.raises(ValueError):
        load_pooled_predictions(tmp_path)


def test_paired_bootstrap_identical_models_centre_on_zero():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 20)
    p = rng.uniform(size=40)
    r = paired_bootstrap_auc(y, p, p, n_boot=500, seed=1)
    assert r["diff"] == 0.0
    assert r["ci_low"] <= 0.0 <= r["ci_high"]
    assert r["p_two_sided"] > 0.5  # cannot distinguish a model from itself


def test_paired_bootstrap_detects_better_model():
    y = np.array([0] * 20 + [1] * 20)
    good = y + np.random.default_rng(2).normal(0, 0.1, size=40)  # tracks label
    bad = np.random.default_rng(3).uniform(size=40)              # random
    r = paired_bootstrap_auc(y, good, bad, n_boot=800, seed=4)
    assert r["diff"] > 0.2
    assert r["ci_low"] > 0.0  # CI excludes zero -> good is better

"""Tests for the evaluation metrics (§7)."""

from __future__ import annotations

import math

import numpy as np

from src.evaluation.metrics import aggregate_metrics, compute_metrics


def test_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    m = compute_metrics(y_true, y_prob)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["balanced_accuracy"] == 1.0


def test_confusion_derived_values():
    # pred at 0.5: probs -> [1,0,1,0]; truth [1,1,0,0]
    # tp=1 (idx0), fn=1 (idx1), fp=1 (idx2), tn=1 (idx3)
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.4, 0.6, 0.1]
    m = compute_metrics(y_true, y_prob)
    assert m["sensitivity"] == 0.5     # tp/(tp+fn)=1/2
    assert m["specificity"] == 0.5     # tn/(tn+fp)=1/2
    assert m["precision"] == 0.5       # tp/(tp+fp)=1/2
    assert m["accuracy"] == 0.5
    assert m["n_pos"] == 2 and m["n_neg"] == 2


def test_single_class_auc_is_nan_not_crash():
    m = compute_metrics([1, 1, 1], [0.6, 0.7, 0.9])
    assert math.isnan(m["roc_auc"])
    assert math.isnan(m["pr_auc"])
    # threshold metrics still defined
    assert m["sensitivity"] == 1.0


def test_threshold_shifts_predictions():
    y_true = [0, 1]
    y_prob = [0.3, 0.6]
    strict = compute_metrics(y_true, y_prob, threshold=0.7)  # both predicted 0
    assert strict["sensitivity"] == 0.0
    loose = compute_metrics(y_true, y_prob, threshold=0.2)   # both predicted 1
    assert loose["specificity"] == 0.0


def test_aggregate_metrics_mean_std():
    folds = [
        {"roc_auc": 0.8, "pr_auc": 0.7},
        {"roc_auc": 1.0, "pr_auc": 0.9},
    ]
    agg = aggregate_metrics(folds)
    assert math.isclose(agg["roc_auc"]["mean"], 0.9)
    assert math.isclose(agg["roc_auc"]["std"], 0.1)
    assert agg["roc_auc"]["n_folds"] == 2


def test_aggregate_ignores_nan_folds():
    folds = [{"roc_auc": float("nan")}, {"roc_auc": 0.8}, {"roc_auc": 1.0}]
    agg = aggregate_metrics(folds)
    assert math.isclose(agg["roc_auc"]["mean"], 0.9)
    assert agg["roc_auc"]["n_folds"] == 2  # nan fold excluded

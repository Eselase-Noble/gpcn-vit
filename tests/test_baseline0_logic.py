"""Tests for the leakage-critical Baseline 0 logic: class weights, image->patient
aggregation, and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.aggregation import (
    aggregate_to_patient,
    evaluate_predictions,
    evaluate_two_level,
    select_threshold,
)
from src.training.class_weights import compute_class_weights


# ---- class weights ----------------------------------------------------------

def test_balanced_weights_upweight_minority():
    # 3 benign (0), 7 malignant (1) -> benign weight > malignant weight
    y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    cw = compute_class_weights(y)
    assert cw["counts"] == {0: 3, 1: 7}
    assert cw["weights"][0] > cw["weights"][1]
    # sklearn convention: n/(K*count) -> 10/(2*3) and 10/(2*7)
    assert np.isclose(cw["weights"][0], 10 / 6)
    assert np.isclose(cw["weights"][1], 10 / 14)
    assert cw["weight_vector"] == [cw["weights"][0], cw["weights"][1]]


def test_absent_class_gets_zero_not_inf():
    cw = compute_class_weights([1, 1, 1])
    assert cw["counts"][0] == 0
    assert cw["weights"][0] == 0.0
    assert np.isfinite(cw["weights"][1])


def test_empty_weights_raises():
    with pytest.raises(ValueError):
        compute_class_weights([])


# ---- image -> patient aggregation ------------------------------------------

def _preds(rows):
    return pd.DataFrame(rows, columns=["patient_id", "label_binary", "prob"])


def test_aggregate_mean_of_image_probs():
    preds = _preds([
        ("p1", 1, 0.9), ("p1", 1, 0.7),   # mean 0.8
        ("p2", 0, 0.2), ("p2", 0, 0.4), ("p2", 0, 0.0),  # mean 0.2
    ])
    agg = aggregate_to_patient(preds)
    agg = agg.set_index("patient_id")
    assert np.isclose(agg.loc["p1", "prob"], 0.8)
    assert np.isclose(agg.loc["p2", "prob"], 0.2)
    assert agg.loc["p1", "n_images"] == 2
    assert agg.loc["p2", "n_images"] == 3
    assert agg.loc["p1", "label_binary"] == 1


def test_aggregate_rejects_mixed_patient_labels():
    preds = _preds([("p1", 1, 0.9), ("p1", 0, 0.1)])
    with pytest.raises(ValueError):
        aggregate_to_patient(preds)


# ---- threshold selection (validation only) ---------------------------------

def test_threshold_separates_when_possible():
    y_val = np.array([0, 0, 1, 1])
    p_val = np.array([0.1, 0.2, 0.8, 0.9])
    t = select_threshold(y_val, p_val, objective="balanced_accuracy")
    # any threshold in (0.2, 0.8) gives perfect balanced accuracy
    pred = (p_val >= t).astype(int)
    assert (pred == y_val).all()


def test_threshold_single_class_falls_back_to_half():
    assert select_threshold(np.array([1, 1, 1]), np.array([0.3, 0.6, 0.9])) == 0.5


# ---- two-level evaluation ---------------------------------------------------

def test_evaluate_reports_both_levels():
    # patient p1 malignant (high probs), p2 benign (low probs)
    preds = _preds([
        ("p1", 1, 0.8), ("p1", 1, 0.9),
        ("p2", 0, 0.1), ("p2", 0, 0.2),
    ])
    out = evaluate_predictions(preds, threshold=0.5)
    assert set(out) == {"image", "patient"}
    assert out["patient"]["n"] == 2          # two patients
    assert out["image"]["n"] == 4            # four images
    assert out["patient"]["roc_auc"] == 1.0  # perfectly separable at patient level


def test_evaluate_two_level_applies_distinct_thresholds():
    # patient means: p1=0.55 (malignant), p2=0.45 (benign)
    preds = _preds([
        ("p1", 1, 0.50), ("p1", 1, 0.60),
        ("p2", 0, 0.40), ("p2", 0, 0.50),
    ])
    # image threshold 0.55 vs patient threshold 0.50 -> different decisions
    out = evaluate_two_level(preds, image_threshold=0.55, patient_threshold=0.50)
    # patient: 0.55>=0.50 -> malignant (correct); 0.45<0.50 -> benign (correct)
    assert out["patient"]["accuracy"] == 1.0
    # ranking is perfect at both levels regardless of threshold
    assert out["patient"]["roc_auc"] == 1.0

"""Tests for patient-level stratified k-fold CV (§8) and leakage safety (§3)."""

from __future__ import annotations

import pytest

from src.datasets.breakhis import build_metadata
from src.splits.cross_validation import make_patient_kfold, verify_kfold
from src.splits.patient_split import LeakageError


def test_kfold_partitions_patients_disjointly(breakhis_tree):
    df = build_metadata(breakhis_tree)  # 6 patients, 3 benign / 3 malignant
    folds = make_patient_kfold(df, n_splits=3, seed=42)
    assert len(folds) == 3
    # every patient in exactly one test fold; test folds disjoint
    verify_kfold(folds, df)  # raises on violation
    # within each fold, train/test are patient-disjoint
    for f in folds:
        assert set(f.train).isdisjoint(set(f.test))


def test_kfold_is_stratified(breakhis_tree):
    df = build_metadata(breakhis_tree)
    labels = df.drop_duplicates("patient_id").set_index("patient_id")["label_binary"]
    folds = make_patient_kfold(df, n_splits=3, seed=0)
    # with 3 benign + 3 malignant over 3 folds, each test fold should hold
    # exactly one of each class
    for f in folds:
        test_labels = labels.loc[f.test]
        assert (test_labels == 0).sum() == 1
        assert (test_labels == 1).sum() == 1


def test_kfold_val_carve_is_disjoint(breakhis_tree):
    df = build_metadata(breakhis_tree)
    folds = make_patient_kfold(df, n_splits=3, seed=42, val_fraction=0.34)
    for f in folds:
        s = set(f.train), set(f.val), set(f.test)
        assert s[0].isdisjoint(s[1])
        assert s[0].isdisjoint(s[2])
        assert s[1].isdisjoint(s[2])
        assert len(f.val) >= 1  # something was carved


def test_kfold_deterministic(breakhis_tree):
    df = build_metadata(breakhis_tree)
    a = make_patient_kfold(df, n_splits=3, seed=5)
    b = make_patient_kfold(df, n_splits=3, seed=5)
    assert [f.test for f in a] == [f.test for f in b]


def test_kfold_rejects_too_many_splits(breakhis_tree):
    df = build_metadata(breakhis_tree)
    with pytest.raises(ValueError):
        make_patient_kfold(df, n_splits=99)


def test_verify_kfold_catches_missing_patient(breakhis_tree):
    df = build_metadata(breakhis_tree)
    folds = make_patient_kfold(df, n_splits=3, seed=42)
    # Drop a patient from one test fold -> no longer a partition.
    folds[0].test = folds[0].test[:-1]
    with pytest.raises(LeakageError):
        verify_kfold(folds, df)

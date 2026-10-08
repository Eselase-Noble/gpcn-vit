"""Tests for patient-level splitting and leakage verification (§3)."""

from __future__ import annotations

import pytest

from src.datasets.breakhis import build_metadata
from src.splits.patient_split import (
    LeakageError,
    PatientSplit,
    make_patient_split,
    split_dataframe,
    verify_disjoint,
)


def test_split_is_patient_disjoint(breakhis_tree):
    df = build_metadata(breakhis_tree)
    split = make_patient_split(df, seed=42)
    train, val, test = (set(split.train), set(split.val), set(split.test))
    # The protocol's mandatory assertions:
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # Completeness: every patient assigned exactly once.
    assert train | val | test == set(df["patient_id"].unique())


def test_image_level_partition_has_no_patient_leakage(breakhis_tree):
    df = build_metadata(breakhis_tree)
    split = make_patient_split(df, seed=42)
    parts = split_dataframe(df, split)
    p_train = set(parts["train"]["patient_id"])
    p_val = set(parts["val"]["patient_id"])
    p_test = set(parts["test"]["patient_id"])
    assert p_train.isdisjoint(p_val)
    assert p_train.isdisjoint(p_test)
    assert p_val.isdisjoint(p_test)
    # No image lost or duplicated across folds.
    assert sum(len(p) for p in parts.values()) == len(df)


def test_split_is_deterministic(breakhis_tree):
    df = build_metadata(breakhis_tree)
    a = make_patient_split(df, seed=7)
    b = make_patient_split(df, seed=7)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_different_seed_can_differ(breakhis_tree):
    df = build_metadata(breakhis_tree)
    a = make_patient_split(df, seed=1)
    b = make_patient_split(df, seed=2)
    # Not a strict guarantee, but with these ratios the orderings should differ.
    assert (a.train, a.val, a.test) != (b.train, b.val, b.test)


def test_ratios_must_sum_to_one(breakhis_tree):
    df = build_metadata(breakhis_tree)
    with pytest.raises(ValueError):
        make_patient_split(df, ratios=(0.6, 0.3, 0.3))


def test_verify_disjoint_catches_injected_leak(breakhis_tree):
    df = build_metadata(breakhis_tree)
    split = make_patient_split(df, seed=42)
    # Inject a leak: put a train patient also into test.
    leaked = PatientSplit(
        train=split.train,
        val=split.val,
        test=split.test + [split.train[0]],
        seed=split.seed,
        ratios=split.ratios,
    )
    with pytest.raises(LeakageError):
        verify_disjoint(leaked, df)


def test_split_roundtrip_json(breakhis_tree, tmp_path):
    df = build_metadata(breakhis_tree)
    split = make_patient_split(df, seed=42)
    p = split.to_json(tmp_path / "split.json")
    loaded = PatientSplit.from_json(p)
    assert set(loaded.train) == set(split.train)
    assert set(loaded.test) == set(split.test)
    verify_disjoint(loaded, df)

"""Canonical CV-fold persistence: save -> load round-trips and re-verifies."""
from __future__ import annotations

import pandas as pd
import pytest

from src.splits.cross_validation import load_folds, make_patient_kfold, save_folds


def _toy_df(n_patients=15, images_per=4):
    rows = []
    for i in range(n_patients):
        pid = f"P{i:02d}"
        label = i % 2  # alternate benign/malignant -> both classes present
        for j in range(images_per):
            rows.append({"filename": f"{pid}_{j}.png", "patient_id": pid, "label_binary": label})
    return pd.DataFrame(rows)


def test_save_load_roundtrip(tmp_path):
    df = _toy_df()
    folds = make_patient_kfold(df, n_splits=3, seed=42, val_fraction=0.25)
    path = save_folds(folds, tmp_path / "cv.json", dataset="breakhis", magnification="40X")
    reloaded = load_folds(path, df=df)  # df -> re-verify partition on load

    assert len(reloaded) == len(folds)
    for a, b in zip(folds, reloaded):
        assert sorted(a.train) == sorted(b.train)
        assert sorted(a.val) == sorted(b.val)
        assert sorted(a.test) == sorted(b.test)
        assert a.seed == b.seed


def test_test_folds_partition_all_patients(tmp_path):
    df = _toy_df()
    folds = make_patient_kfold(df, n_splits=3, seed=42, val_fraction=0.0)
    reloaded = load_folds(save_folds(folds, tmp_path / "cv.json"))
    seen = set()
    for f in reloaded:
        assert not (seen & set(f.test))  # test folds are disjoint
        seen |= set(f.test)
    assert seen == set(df["patient_id"].unique())


def test_rejects_unknown_schema(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"schema": "nope", "folds": []}')
    with pytest.raises(ValueError):
        load_folds(p)

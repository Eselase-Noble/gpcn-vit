"""Patient-level stratified k-fold cross-validation (§8).

Produces ``n_splits`` folds where:
  * folds are built over *patients*, never images (leakage rule, §3);
  * folds are stratified by the patient's class (benign/malignant), so each
    test fold preserves the dataset's imbalance;
  * every patient appears in exactly one test fold;
  * an optional inner validation set is carved (stratified) from each fold's
    training patients, for early stopping / model selection without touching test.

Each fold is returned as a :class:`~src.splits.patient_split.PatientSplit` so the
same leakage-verification and JSON (de)serialisation machinery applies.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .patient_split import LeakageError, PatientSplit, _patient_labels, verify_disjoint

logger = logging.getLogger(__name__)


def make_patient_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    val_fraction: float = 0.0,
) -> List[PatientSplit]:
    """Create patient-level stratified k-fold CV splits.

    Parameters
    ----------
    df: image-level metadata (must contain ``patient_id`` and ``label_binary``).
    n_splits: number of folds (default 5, per the agreed protocol).
    seed: RNG seed for fold assignment (and the inner val carve).
    val_fraction: fraction of each fold's *training patients* to hold out as an
        inner validation set (stratified). ``0.0`` => no val (train/test only).

    Returns
    -------
    list of PatientSplit, one per fold, each already leakage-verified.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    labels = _patient_labels(df)  # index: patient_id -> 0/1 (asserts no mixed labels)
    patients = np.sort(labels.index.to_numpy())
    y = labels.loc[patients].to_numpy()

    if n_splits > len(patients):
        raise ValueError(f"n_splits={n_splits} > n_patients={len(patients)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[PatientSplit] = []

    for i, (trainval_idx, test_idx) in enumerate(skf.split(patients, y)):
        test_p = patients[test_idx].tolist()
        trainval_p = patients[trainval_idx]
        trainval_y = y[trainval_idx]

        if val_fraction > 0.0:
            train_p, val_p = train_test_split(
                trainval_p,
                test_size=val_fraction,
                stratify=trainval_y,
                random_state=seed + i,
            )
            train_p, val_p = sorted(train_p.tolist()), sorted(val_p.tolist())
        else:
            train_p, val_p = sorted(trainval_p.tolist()), []

        n = len(patients)
        split = PatientSplit(
            train=train_p,
            val=val_p,
            test=sorted(test_p),
            seed=seed,
            ratios=(len(train_p) / n, len(val_p) / n, len(test_p) / n),
            meta={
                "cv": True,
                "fold": i,
                "n_splits": n_splits,
                "val_fraction": val_fraction,
                "n_patients": n,
                "n_train": len(train_p),
                "n_val": len(val_p),
                "n_test": len(test_p),
            },
        )
        verify_disjoint(split, df)
        folds.append(split)

    verify_kfold(folds, df)
    logger.info(
        "Built %d-fold patient-level CV: %d patients, val_fraction=%.2f.",
        n_splits,
        len(patients),
        val_fraction,
    )
    return folds


def verify_kfold(folds: List[PatientSplit], df: pd.DataFrame) -> None:
    """Assert CV folds are valid: test folds partition all patients, disjointly."""
    all_patients = set(df["patient_id"].unique())
    seen_test: set = set()
    for f in folds:
        test = set(f.test)
        overlap = seen_test & test
        if overlap:
            raise LeakageError(f"patient(s) in multiple test folds: {sorted(overlap)}")
        seen_test |= test
    if seen_test != all_patients:
        missing = all_patients - seen_test
        extra = seen_test - all_patients
        raise LeakageError(
            f"test folds do not partition patients; missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    logger.info("k-fold verification PASSED: %d folds partition %d patients.",
                len(folds), len(all_patients))

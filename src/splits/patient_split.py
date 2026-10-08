"""Patient-level train/validation/test splitting with leakage verification.

Implements the critical data-leakage rule (§3): a patient must NEVER appear in
more than one split. Splitting is performed on *patients*, not images, and is
stratified by the patient's class so each split preserves the benign/malignant
balance.

The split is deterministic given ``seed``. ``verify_disjoint`` provides the
automated assertions required by the protocol; callers should treat a raised
``LeakageError`` as a hard stop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LeakageError(AssertionError):
    """Raised when a patient-level split violates disjointness."""


@dataclass
class PatientSplit:
    """A patient-disjoint split, stored as sets of patient IDs per fold."""

    train: List[str]
    val: List[str]
    test: List[str]
    seed: int
    ratios: Tuple[float, float, float]
    meta: Dict = field(default_factory=dict)

    def as_sets(self) -> Dict[str, set]:
        return {"train": set(self.train), "val": set(self.val), "test": set(self.test)}

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train": sorted(self.train),
            "val": sorted(self.val),
            "test": sorted(self.test),
            "seed": self.seed,
            "ratios": list(self.ratios),
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def from_json(cls, path: Path) -> "PatientSplit":
        d = json.loads(Path(path).read_text())
        return cls(
            train=d["train"],
            val=d["val"],
            test=d["test"],
            seed=d["seed"],
            ratios=tuple(d["ratios"]),
            meta=d.get("meta", {}),
        )


def _patient_labels(df: pd.DataFrame) -> pd.Series:
    """Map each patient to a single class label, asserting label consistency.

    In BreakHis a patient's slide is entirely benign or entirely malignant. We
    assert this holds; a mixed-label patient would indicate a parsing bug or a
    dataset assumption violation and must surface, not be silently averaged.
    """
    per_patient = df.groupby("patient_id")["label_binary"].nunique()
    mixed = per_patient[per_patient > 1]
    if len(mixed) > 0:
        raise LeakageError(
            "Patients have mixed class labels (parsing/assumption error): "
            f"{list(mixed.index)}"
        )
    return df.groupby("patient_id")["label_binary"].first()


def make_patient_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> PatientSplit:
    """Create a stratified, patient-disjoint train/val/test split.

    Stratification is per class: each class's patients are shuffled with a
    seeded RNG and partitioned by ``ratios``, then concatenated. This keeps the
    class balance similar across folds while guaranteeing patient disjointness.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    labels = _patient_labels(df)  # index: patient_id, value: 0/1
    rng = np.random.default_rng(seed)

    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    for cls in sorted(labels.unique()):
        patients = labels.index[labels == cls].to_numpy()
        patients = np.sort(patients)  # deterministic base order before shuffle
        rng.shuffle(patients)
        n = len(patients)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        # Remainder to test so the three folds always cover all patients.
        train.extend(patients[:n_train].tolist())
        val.extend(patients[n_train : n_train + n_val].tolist())
        test.extend(patients[n_train + n_val :].tolist())

    split = PatientSplit(
        train=sorted(train),
        val=sorted(val),
        test=sorted(test),
        seed=seed,
        ratios=ratios,
        meta={
            "n_patients": int(len(labels)),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
    )
    verify_disjoint(split, df)

    # Guard: a requested (ratio > 0) fold that ends up empty is almost always a
    # silent evaluation failure (too few patients per class for the rounding).
    # We warn loudly rather than raise, so tiny/unit cases still run.
    for fold, ratio, ids in (
        ("train", ratios[0], train),
        ("val", ratios[1], val),
        ("test", ratios[2], test),
    ):
        if ratio > 0 and len(ids) == 0:
            logger.warning(
                "Split fold '%s' is EMPTY despite ratio=%.2f (only %d patients). "
                "Too few patients per class — consider patient-level "
                "cross-validation (protocol §8) instead of a single split.",
                fold,
                ratio,
                len(labels),
            )
    return split


def verify_disjoint(split: PatientSplit, df: pd.DataFrame | None = None) -> None:
    """Assert the split is patient-disjoint and complete. Raises LeakageError.

    Checks (the protocol's required assertions, §3):
      * train/val/test patient sets are pairwise disjoint
      * if ``df`` is given, every patient in the data lands in exactly one fold
    """
    s = split.as_sets()
    train, val, test = s["train"], s["val"], s["test"]

    if not train.isdisjoint(val):
        raise LeakageError(f"train∩val leakage: {sorted(train & val)}")
    if not train.isdisjoint(test):
        raise LeakageError(f"train∩test leakage: {sorted(train & test)}")
    if not val.isdisjoint(test):
        raise LeakageError(f"val∩test leakage: {sorted(val & test)}")

    if df is not None:
        all_patients = set(df["patient_id"].unique())
        assigned = train | val | test
        missing = all_patients - assigned
        extra = assigned - all_patients
        if missing:
            raise LeakageError(f"patients missing from all splits: {sorted(missing)}")
        if extra:
            raise LeakageError(f"split references unknown patients: {sorted(extra)}")

    logger.info(
        "Leakage verification PASSED: train=%d val=%d test=%d patients (disjoint).",
        len(train),
        len(val),
        len(test),
    )


def split_dataframe(
    df: pd.DataFrame, split: PatientSplit
) -> Dict[str, pd.DataFrame]:
    """Partition the image-level DataFrame by the patient split, image-leakage-safe."""
    sets = split.as_sets()
    return {
        fold: df[df["patient_id"].isin(ids)].reset_index(drop=True)
        for fold, ids in sets.items()
    }

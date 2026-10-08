"""Shared test fixtures.

Builds a synthetic BreakHis directory tree that mimics the official naming
convention and folder layout, so the metadata/EDA/split pipeline can be tested
without the real (large) dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``import src.*`` resolve when pytest is run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# (tumor_class, tumor_type_code, tumor_type_folder)
_BENIGN = [("B", "A", "adenosis"), ("B", "F", "fibroadenoma")]
_MALIGNANT = [("M", "DC", "ductal_carcinoma"), ("M", "LC", "lobular_carcinoma")]
_MAGS = ["40", "100", "200", "400"]


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # empty placeholder; parsing is filename-only


@pytest.fixture
def breakhis_tree(tmp_path: Path) -> Path:
    """Create a synthetic BreakHis tree.

    Layout: 6 patients (3 benign, 3 malignant), each with images at all four
    magnifications and a variable number of images per magnification, so
    per-patient/per-magnification counts are non-trivial.

    Returns the root directory to scan.
    """
    root = tmp_path / "BreaKHis_v1"
    year = "14"
    patient_counter = 22000

    def add_patient(group, idx: int, class_folder: str, n_per_mag: int):
        nonlocal patient_counter
        tumor_class, type_code, type_folder = group[idx % len(group)]
        slide_id = f"{patient_counter}{chr(65 + idx)}"  # e.g. 22000A
        patient_counter += 1
        for mag in _MAGS:
            for seq in range(1, n_per_mag + 1):
                fname = f"SOB_{tumor_class}_{type_code}-{year}-{slide_id}-{mag}-{seq:03d}.png"
                fp = (root / "breast" / class_folder / "SOB" / type_folder
                      / f"SOB_{tumor_class}_{type_code}_{year}-{slide_id}" / f"{mag}X" / fname)
                _touch(fp)

    # 3 benign patients, 3 malignant patients; vary image counts per patient
    for i in range(3):
        add_patient(_BENIGN, i, "benign", n_per_mag=2 + i)       # 2,3,4 per mag
    for i in range(3):
        add_patient(_MALIGNANT, i, "malignant", n_per_mag=3 + i)  # 3,4,5 per mag

    return root

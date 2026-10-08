"""Tests for BreakHis filename parsing and metadata ingestion."""

from __future__ import annotations

from src.datasets.breakhis import build_metadata, parse_filename


def test_parse_canonical_filename():
    img = parse_filename("SOB_B_A-14-22549AB-40-001.png")
    assert img is not None
    assert img.patient_id == "14-22549AB"      # year-slide => leakage unit
    assert img.tumor_class == "B"
    assert img.label == "benign"
    assert img.label_binary == 0
    assert img.tumor_type == "adenosis"
    assert img.magnification == "40X"
    assert img.mag_value == 40
    assert img.seq == "001"


def test_parse_malignant_multichar_type():
    img = parse_filename("SOB_M_DC-14-10926-100-015.png")
    assert img is not None
    assert img.label == "malignant"
    assert img.label_binary == 1
    assert img.tumor_type == "ductal_carcinoma"
    assert img.magnification == "100X"
    assert img.patient_id == "14-10926"


def test_parse_rejects_bad_filename():
    assert parse_filename("not_a_breakhis_file.png") is None
    assert parse_filename("random.txt") is None


def test_build_metadata_counts(breakhis_tree):
    df = build_metadata(breakhis_tree)
    # 6 patients total
    assert df["patient_id"].nunique() == 6
    # benign totals: (2+3+4) per mag * 4 mags = 9*4 = 36
    # malignant totals: (3+4+5) per mag * 4 mags = 12*4 = 48
    assert (df["label"] == "benign").sum() == 36
    assert (df["label"] == "malignant").sum() == 48
    assert len(df) == 84
    # one slide per patient (BreakHis assumption)
    assert (df.groupby("patient_id")["slide_id"].nunique() == 1).all()


def test_build_metadata_magnification_filter(breakhis_tree):
    df40 = build_metadata(breakhis_tree, magnification="40X")
    assert (df40["magnification"] == "40X").all()
    assert len(df40) == 84 // 4  # equal images across the 4 magnifications

"""Tests for EDA summary and reproducibility utilities."""

from __future__ import annotations

from src.datasets.breakhis import build_metadata
from src.eda.summary import format_summary, summarize
from src.utils.env import collect_env
from src.utils.seed import set_seed


def test_eda_summary_answers_milestone_questions(breakhis_tree):
    df = build_metadata(breakhis_tree)
    s = summarize(df)
    assert s["n_patients"] == 6
    assert s["n_images"] == 84
    assert s["class_counts"] == {"benign": 36, "malignant": 48}
    # all four magnifications present
    assert set(s["images_per_magnification"]) == {"40X", "100X", "200X", "400X"}
    # one slide per patient
    assert s["slides_per_patient"]["max"] == 1
    # report renders without error and mentions patient count
    text = format_summary(s)
    assert "Patients available" in text


def test_set_seed_repeatable():
    set_seed(123)
    import numpy as np

    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert (a == b).all()


def test_collect_env_keys():
    env = collect_env()
    for key in ("python_version", "git_commit", "torch_version", "cuda_available"):
        assert key in env

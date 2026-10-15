"""Baseline 2 attention-MIL: bag construction (pure) + torch forward (if available)."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.baselines.attention_mil import make_bags


def test_make_bags_groups_by_patient():
    emb = np.arange(24, dtype=float).reshape(6, 4)
    pid = np.array(["a", "a", "a", "b", "b", "c"])
    lab = np.array([1, 1, 1, 0, 0, 1])
    bags = make_bags(emb, pid, lab)
    assert [b["patient_id"] for b in bags] == ["a", "b", "c"]
    assert [b["n"] for b in bags] == [3, 2, 1]
    assert [b["y"] for b in bags] == [1, 0, 1]
    assert bags[0]["X"].shape == (3, 4)


def test_make_bags_rejects_mixed_labels():
    emb = np.zeros((2, 4))
    with pytest.raises(ValueError):
        make_bags(emb, np.array(["p", "p"]), np.array([0, 1]))


def test_attention_forward_shapes_and_weights():
    torch = pytest.importorskip("torch")
    from src.models.baselines.attention_mil import build_model

    torch.manual_seed(0)
    model = build_model(in_dim=8, hidden=16).eval()
    bag = torch.randn(5, 8)
    logits, attn = model(bag)
    assert logits.shape == (1, 2)
    assert attn.shape == (5, 1)
    assert abs(float(attn.sum()) - 1.0) < 1e-5  # attention is a distribution over instances

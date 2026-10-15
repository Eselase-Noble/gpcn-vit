"""Feature-similarity k-NN graph construction + GNN forward (if torch available)."""
from __future__ import annotations

import numpy as np
import pytest

from src.graphs.feature import knn_feature_edges


def test_knn_links_within_clusters():
    # two tight clusters far apart; k=1 must link each point to its cluster-mate
    X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    e = knn_feature_edges(X, k=1)
    und = {(min(s, d), max(s, d)) for s, d in zip(e[0], e[1])}
    assert (0, 1) in und and (2, 3) in und
    assert (0, 2) not in und and (1, 3) not in und  # no cross-cluster edge at k=1


def test_knn_singleton_and_symmetry():
    assert knn_feature_edges(np.zeros((1, 4)), k=8).shape == (2, 0)  # singleton -> no edges
    X = np.random.default_rng(0).normal(size=(6, 5))
    e = knn_feature_edges(X, k=3)
    pairs = set(zip(e[0].tolist(), e[1].tolist()))
    assert all((d, s) in pairs for s, d in pairs)  # every edge is symmetric


def test_knn_caps_k_at_n_minus_1():
    X = np.random.default_rng(1).normal(size=(3, 4))
    e = knn_feature_edges(X, k=10)  # k capped to 2 -> fully connected triangle
    und = {(min(s, d), max(s, d)) for s, d in zip(e[0], e[1])}
    assert und == {(0, 1), (0, 2), (1, 2)}


def test_gnn_forward_runs_with_and_without_edges():
    torch = pytest.importorskip("torch")
    from src.models.gnn.sage import build_model

    torch.manual_seed(0)
    model = build_model(in_dim=8, hidden=16, layers=2).eval()
    x = torch.randn(5, 8)
    e = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
    assert model(x, e).shape == (1, 2)
    # isolated-node / no-edge case (the 'none' control) must not crash
    assert model(x, torch.empty((2, 0), dtype=torch.long)).shape == (1, 2)

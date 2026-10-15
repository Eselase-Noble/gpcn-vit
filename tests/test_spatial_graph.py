"""Within-slide spatial graph construction + degree-preserving rewire control."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.graphs.spatial import (
    build_slide_graphs,
    grid_edges,
    radius_edges,
    rewire_degree_preserving,
)


def _degrees(edge_index, n):
    deg = np.zeros(n, dtype=int)
    for s in edge_index[0]:
        deg[int(s)] += 1
    return deg


def test_grid_edges_2x2_connectivity():
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    e4 = grid_edges(rows, cols, connectivity=4)
    e8 = grid_edges(rows, cols, connectivity=8)
    assert e4.shape == (2, 8)    # 4 undirected orthogonal edges, both directions
    assert e8.shape == (2, 12)   # K4: 6 undirected edges, both directions
    # symmetric (every i->j has j->i)
    und = {(min(s, d), max(s, d)) for s, d in zip(e8[0], e8[1])}
    assert len(und) == 6


def test_grid_edges_respects_gaps():
    # two tiles diagonal with a missing cell between -> only 8-conn links them
    rows = np.array([0, 1])
    cols = np.array([0, 1])
    assert grid_edges(rows, cols, 4).shape[1] == 0   # not orthogonally adjacent
    assert grid_edges(rows, cols, 8).shape[1] == 2   # diagonally adjacent


def test_radius_edges():
    x = np.array([0, 100, 1000])
    y = np.array([0, 0, 0])
    e = radius_edges(x, y, radius=150)  # links 0-1 only
    und = {(min(s, d), max(s, d)) for s, d in zip(e[0], e[1])}
    assert und == {(0, 1)}


def test_rewire_preserves_degree():
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(5), 5)
    cols = np.tile(np.arange(5), 5)
    e = grid_edges(rows, cols, connectivity=8)
    n = 25
    rw = rewire_degree_preserving(e, seed=1)
    assert np.array_equal(np.sort(_degrees(e, n)), np.sort(_degrees(rw, n)))  # same degree seq
    # connectivity actually changed (not a no-op on a 5x5 grid)
    assert {(min(s, d), max(s, d)) for s, d in zip(e[0], e[1])} != \
           {(min(s, d), max(s, d)) for s, d in zip(rw[0], rw[1])}


def test_build_slide_graphs_no_cross_slide_edges():
    # two slides with IDENTICAL grid positions must not be linked together
    rows = []
    for sid in ("s1", "s2"):
        for r in range(2):
            for c in range(2):
                rows.append({"slide_id": sid, "label": 0 if sid == "s1" else 1,
                             "row": r, "col": c, "x": c * 256, "y": r * 256,
                             "filepath": f"{sid}_{r}_{c}.png"})
    graphs = build_slide_graphs(pd.DataFrame(rows), connectivity=8)
    assert len(graphs) == 2
    for gph in graphs:
        assert gph["n_nodes"] == 4
        assert gph["edge_index"].shape == (2, 12)  # each slide is its own K4
        assert gph["edge_index"].max() < 4          # node indices local to the slide
    assert {g["label"] for g in graphs} == {0, 1}

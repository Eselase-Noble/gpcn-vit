"""Within-slide spatial graphs for WSI tiles (B3 substrate).

Nodes are tissue tiles from ONE slide; edges connect tiles that are spatial
neighbours WITHIN THAT SLIDE (grid adjacency on (row, col), or a pixel radius on
(x, y)). Edges are never created across slides — there is no meaningful adjacency
between separate slides (see docs/wsi_protocol.md).

Also provides a DEGREE-PRESERVING REWIRE control: same node set and same degree
sequence, randomized connectivity. Comparing the real spatial graph against this
control is how we test whether *meaningful* structure matters, not merely node
features plus connectivity density.

Framework-agnostic numpy (edge_index is a (2, E) int array of directed edges,
both directions per undirected edge — the convention GNN libraries expect).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

_OFFSETS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_OFFSETS_8 = _OFFSETS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def grid_edges(rows: np.ndarray, cols: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Edges between tiles whose grid cells are adjacent (4- or 8-connectivity).

    Returns a (2, E) int array of directed edges (both directions per undirected
    edge). Tiles need not form a full grid — only present cells are linked.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    offsets = _OFFSETS_4 if connectivity == 4 else _OFFSETS_8
    index = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    src, dst = [], []
    for (r, c), i in index.items():
        for dr, dc in offsets:
            j = index.get((r + dr, c + dc))
            if j is not None:
                src.append(i)
                dst.append(j)
    return np.array([src, dst], dtype=int) if src else np.empty((2, 0), dtype=int)


def radius_edges(x: np.ndarray, y: np.ndarray, radius: float) -> np.ndarray:
    """Edges between tiles within Euclidean ``radius`` in level-0 pixels.

    Brute force (fine for prototype slide sizes). Returns (2, E) directed edges.
    """
    xy = np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=1)
    src, dst = [], []
    r2 = float(radius) ** 2
    for i in range(len(xy)):
        d2 = ((xy - xy[i]) ** 2).sum(axis=1)
        for j in np.where((d2 <= r2) & (np.arange(len(xy)) != i))[0]:
            src.append(i)
            dst.append(int(j))
    return np.array([src, dst], dtype=int) if src else np.empty((2, 0), dtype=int)


def _undirected_set(edge_index: np.ndarray):
    return {(min(int(s), int(d)), max(int(s), int(d)))
            for s, d in zip(edge_index[0], edge_index[1]) if s != d}


def rewire_degree_preserving(edge_index: np.ndarray, seed: int = 0,
                             n_swaps: Optional[int] = None) -> np.ndarray:
    """Randomize connectivity while PRESERVING every node's degree (double-edge swap).

    The relationship-necessity control: if the real spatial graph does not beat
    this, the measured gain came from degree/features, not meaningful structure.
    """
    rng = np.random.default_rng(seed)
    eset = _undirected_set(edge_index)
    edges = list(eset)
    m = len(edges)
    if m < 2:
        return edge_index
    n_swaps = n_swaps if n_swaps is not None else 10 * m
    for _ in range(n_swaps):
        i, j = int(rng.integers(0, m)), int(rng.integers(0, m))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue
        new1 = (min(a, d), max(a, d))
        new2 = (min(c, b), max(c, b))
        if new1 in eset or new2 in eset:
            continue
        eset.discard((a, b))
        eset.discard((c, d))
        eset.add(new1)
        eset.add(new2)
        edges[i], edges[j] = new1, new2
    src, dst = [], []
    for s, d in eset:
        src += [s, d]
        dst += [d, s]
    return np.array([src, dst], dtype=int)


def build_slide_graphs(tiles: pd.DataFrame, connectivity: int = 8) -> List[dict]:
    """Build one spatial graph per slide from a tiles table (groups by slide_id).

    Grouping by ``slide_id`` STRUCTURALLY guarantees edges stay within a slide.
    Each graph: ``{slide_id, label, n_nodes, filepath, x, y, edge_index}`` where
    node order follows the slide's row order in ``tiles``.
    """
    required = {"slide_id", "row", "col", "x", "y", "filepath"}
    missing = required - set(tiles.columns)
    if missing:
        raise ValueError(f"tiles table missing columns: {sorted(missing)}")

    graphs: List[dict] = []
    for sid, g in tiles.groupby("slide_id", sort=True):
        g = g.reset_index(drop=True)
        edge_index = grid_edges(g["row"].to_numpy(), g["col"].to_numpy(), connectivity)
        label = int(g["label"].iloc[0]) if "label" in g else -1
        graphs.append({
            "slide_id": sid,
            "label": label,
            "n_nodes": len(g),
            "filepath": g["filepath"].to_numpy(),
            "x": g["x"].to_numpy(),
            "y": g["y"].to_numpy(),
            "edge_index": edge_index,
        })
    return graphs

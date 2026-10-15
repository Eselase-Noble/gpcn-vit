"""Per-slide WSI graph assembly: feature alignment + edge modes + no cross-slide edges."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.graphs.wsi_graph import build_wsi_graphs


def _tiles_two_slides():
    rows = []
    for sid, lab in (("s1", 0), ("s2", 1)):
        for r in range(2):
            for c in range(2):
                rows.append({"slide_id": sid, "label": lab, "row": r, "col": c,
                             "x": c * 256, "y": r * 256, "filepath": f"/t/{sid}_{r}_{c}.png"})
    return pd.DataFrame(rows)


def _emb_lookup(tiles, dim=4):
    # distinct feature per tile so misalignment would be detectable
    return {f.split("/")[-1]: np.full(dim, i, dtype=float)
            for i, f in enumerate(tiles["filepath"])}


def test_spatial_graphs_align_features_and_stay_within_slide():
    tiles = _tiles_two_slides()
    graphs = build_wsi_graphs(tiles, _emb_lookup(tiles), connectivity=8, mode="spatial")
    assert len(graphs) == 2
    for g in graphs:
        assert g["features"].shape == (4, 4)
        assert g["edge_index"].shape == (2, 12)   # per-slide K4 (8-conn 2x2)
        assert g["edge_index"].max() < 4          # indices local to the slide
    assert [g["label"] for g in graphs] == [0, 1]
    # features aligned in tile order: slide s1 took embedding ids 0..3
    assert np.array_equal(graphs[0]["features"][:, 0], [0, 1, 2, 3])


def test_mode_none_has_no_edges():
    tiles = _tiles_two_slides()
    graphs = build_wsi_graphs(tiles, _emb_lookup(tiles), mode="none")
    assert all(g["edge_index"].shape == (2, 0) for g in graphs)


def test_missing_embedding_raises():
    tiles = _tiles_two_slides()
    bad = _emb_lookup(tiles)
    bad.pop(next(iter(bad)))  # drop one tile's embedding
    with pytest.raises(KeyError):
        build_wsi_graphs(tiles, bad, mode="spatial")

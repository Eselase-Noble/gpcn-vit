"""Assemble per-slide WSI graphs: spatial edges + aligned tile node features.

Ties together the tiling table (coordinates), the cached tile embeddings (node
features), and the edge builders. Grouping by ``slide_id`` guarantees edges stay
within one slide (the WSI protocol's hard rule). Supports the same three
conditions as the BreakHis rung so the spatial hypothesis gets the same controls:
  * spatial  — real grid adjacency (4/8-connectivity)
  * rewired  — degree-preserving randomized edges
  * none     — no edges

Pure numpy (edge_index + float32 features); the runner converts to torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .spatial import grid_edges, rewire_degree_preserving


def build_wsi_graphs(tiles: pd.DataFrame, emb_by_filename: Dict[str, np.ndarray],
                     connectivity: int = 8, mode: str = "spatial", seed: int = 0) -> List[dict]:
    """One graph per slide. ``emb_by_filename`` maps a tile PNG name -> embedding.

    Node order follows each slide's row order in ``tiles``; features are gathered
    in that same order so nodes and embeddings stay aligned.
    """
    if mode not in ("spatial", "rewired", "none"):
        raise ValueError(f"mode must be spatial|rewired|none, got {mode!r}")
    required = {"slide_id", "row", "col", "x", "y", "filepath", "label"}
    missing = required - set(tiles.columns)
    if missing:
        raise ValueError(f"tiles table missing columns: {sorted(missing)}")

    graphs: List[dict] = []
    for i, (sid, g) in enumerate(tiles.groupby("slide_id", sort=True)):
        g = g.reset_index(drop=True)
        names = [Path(fp).name for fp in g["filepath"]]
        missing_emb = [n for n in names if n not in emb_by_filename]
        if missing_emb:
            raise KeyError(f"{sid}: {len(missing_emb)} tiles have no embedding "
                           f"(e.g. {missing_emb[0]})")
        feats = np.stack([emb_by_filename[n] for n in names]).astype(np.float32)
        if mode == "none":
            edges = np.empty((2, 0), dtype=int)
        else:
            edges = grid_edges(g["row"].to_numpy(), g["col"].to_numpy(), connectivity)
            if mode == "rewired":
                edges = rewire_degree_preserving(edges, seed=seed + i)
        graphs.append({
            "slide_id": sid, "label": int(g["label"].iloc[0]), "n": len(g),
            "features": feats, "edge_index": edges,
            "x": g["x"].to_numpy(), "y": g["y"].to_numpy(),
        })
    return graphs

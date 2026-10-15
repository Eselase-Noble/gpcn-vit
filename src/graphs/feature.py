"""Feature/morphology-similarity graphs over a patient's image embeddings.

BreakHis has no spatial coordinates, so the only *relational* structure we can
build on it is similarity in feature space: connect a patient's images to their
k nearest neighbours in the frozen ViT embedding space. This tests whether
modelling relationships among a patient's images adds anything beyond pooling
them (B1/B2). The degree-preserving rewire control lives in ``spatial`` and is
reused here — same k-NN degree sequence, randomized connectivity.

Edges are built PER PATIENT (one bag), so they never cross patients — mirroring
the within-slide rule for the WSI spatial graph.
"""
from __future__ import annotations

import numpy as np


def knn_feature_edges(X: np.ndarray, k: int = 8, metric: str = "euclidean") -> np.ndarray:
    """Symmetric k-NN graph in embedding space for one patient's instances.

    Returns a (2, E) int array of directed edges (both directions per undirected
    edge). ``k`` is capped at n-1; a singleton bag yields no edges.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    n = len(X)
    if n <= 1:
        return np.empty((2, 0), dtype=int)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)  # +1 = self
    _, idx = nn.kneighbors(X)
    und = set()
    for i in range(n):
        for j in idx[i][1:]:  # drop self (first neighbour)
            a, b = int(i), int(j)
            und.add((min(a, b), max(a, b)))
    src, dst = [], []
    for a, b in und:
        src += [a, b]
        dst += [b, a]
    return np.array([src, dst], dtype=int) if src else np.empty((2, 0), dtype=int)

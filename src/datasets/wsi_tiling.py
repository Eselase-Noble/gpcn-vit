"""Coordinate-preserving WSI tiling primitives (CAMELYON16 spatial-graph substrate).

BreakHis has no spatial coordinates, so the spatial-neighbour hypothesis needs a
true WSI (user decision, 2026-10-15: CAMELYON16). These are the pure-numpy
primitives — tissue detection and a tile grid that RECORDS each tile's (x, y)
position — so downstream we can build genuine *within-slide* spatial edges.
openslide-dependent slide I/O lives in ``scripts/tile_wsi.py``; keeping it out of
here lets the geometry be unit-tested without the heavy dependency.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu's threshold on a 0..255 single-channel array (between-class variance)."""
    v = np.asarray(values, dtype=float).ravel()
    hist, edges = np.histogram(v, bins=nbins, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2[~np.isfinite(sigma_b2)] = -1.0
    return float(centers[int(np.argmax(sigma_b2))])


def tissue_mask(thumb_rgb: np.ndarray, sat_threshold: Optional[float] = None) -> np.ndarray:
    """Boolean tissue mask from an RGB thumbnail via HSV-saturation Otsu.

    Background glass is bright and low-saturation; stained tissue is more
    saturated. Returns an HxW bool array (True = tissue). Mirrors the standard
    CLAM/WSI heuristic without requiring OpenCV.
    """
    rgb = np.asarray(thumb_rgb, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB thumbnail, got {rgb.shape}")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(cmax > 0, (cmax - cmin) / cmax, 0.0) * 255.0
    t = sat_threshold if sat_threshold is not None else otsu_threshold(sat)
    return sat > t


def tile_grid(width: int, height: int, tile_size: int, stride: Optional[int] = None) -> List[dict]:
    """Enumerate fully-contained tiles with grid index AND level-0 pixel (x, y).

    Each entry: ``{row, col, x, y}`` in LEVEL-0 pixel coordinates. The (x, y) is
    what makes spatial edges possible later — two tiles are spatial neighbours iff
    their grid cells are adjacent WITHIN THE SAME slide (never across slides).
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    stride = stride or tile_size
    ys = list(range(0, max(0, height - tile_size + 1), stride))
    xs = list(range(0, max(0, width - tile_size + 1), stride))
    tiles: List[dict] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            tiles.append({"row": row, "col": col, "x": int(x), "y": int(y)})
    return tiles


def tile_tissue_fraction(tile: dict, mask: np.ndarray, downsample: float, tile_size: int) -> float:
    """Fraction of a level-0 tile covered by tissue, read from a thumbnail mask.

    ``downsample`` = level-0 pixels per mask pixel (slide_dim / thumb_dim).
    """
    x0 = int(tile["x"] / downsample)
    y0 = int(tile["y"] / downsample)
    x1 = max(int((tile["x"] + tile_size) / downsample), x0 + 1)
    y1 = max(int((tile["y"] + tile_size) / downsample), y0 + 1)
    region = mask[y0:y1, x0:x1]
    return float(region.mean()) if region.size else 0.0


def filter_tissue_tiles(tiles, mask, downsample, tile_size, min_tissue: float = 0.25):
    """Keep only tiles whose tissue fraction >= ``min_tissue``."""
    return [t for t in tiles if tile_tissue_fraction(t, mask, downsample, tile_size) >= min_tissue]

"""WSI tiling geometry + tissue detection (coordinate-preserving)."""
from __future__ import annotations

import numpy as np

from src.datasets.wsi_tiling import (
    filter_tissue_tiles,
    otsu_threshold,
    tile_grid,
    tile_tissue_fraction,
    tissue_mask,
)


def test_tile_grid_coordinates_non_overlapping():
    tiles = tile_grid(width=1000, height=600, tile_size=256, stride=256)
    xs = sorted({t["x"] for t in tiles})
    ys = sorted({t["y"] for t in tiles})
    assert xs == [0, 256, 512]  # fully-contained only (768+256=1024>1000 excluded)
    assert ys == [0, 256]       # 512+256=768>600 excluded
    assert len(tiles) == 6
    # (row, col) grid index maps back to pixel coords
    t = next(t for t in tiles if t["row"] == 1 and t["col"] == 2)
    assert (t["x"], t["y"]) == (512, 256)


def test_tile_grid_overlap_stride():
    tiles = tile_grid(600, 300, tile_size=256, stride=128)
    assert sorted({t["x"] for t in tiles}) == [0, 128, 256]  # 384+256=640>600 excluded


def test_tissue_mask_picks_saturated_region():
    thumb = np.full((20, 20, 3), 240, dtype=np.uint8)  # bright low-saturation background
    thumb[5:15, 5:15] = [150, 30, 120]                  # saturated "tissue" block
    mask = tissue_mask(thumb)
    assert mask[10, 10]            # tissue detected
    assert not mask[0, 0]          # background rejected
    assert 0 < mask.sum() < mask.size


def test_tile_tissue_fraction_and_filter():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:5, 0:5] = True  # top-left quadrant is tissue (thumbnail space)
    # downsample=10 -> level-0 100x100 maps to 10x10 thumb; tile 50px -> 5x5 thumb cells
    tiles = tile_grid(100, 100, tile_size=50, stride=50)
    assert tile_tissue_fraction({"x": 0, "y": 0}, mask, downsample=10, tile_size=50) == 1.0
    kept = filter_tissue_tiles(tiles, mask, downsample=10, tile_size=50, min_tissue=0.5)
    assert {(t["x"], t["y"]) for t in kept} == {(0, 0)}


def test_otsu_threshold_bimodal():
    vals = np.concatenate([np.full(100, 10.0), np.full(100, 200.0)])
    assert 10 < otsu_threshold(vals) < 200

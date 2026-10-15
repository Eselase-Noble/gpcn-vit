#!/usr/bin/env python3
"""Tile CAMELYON16 slides into patches, PRESERVING each patch's (x, y) position.

For every .tif slide: read a thumbnail, detect tissue, enumerate a tile grid at a
chosen level, keep tissue tiles, save each patch PNG and append a row to
``tiles.csv`` with (slide_id, label, level, row, col, x, y, filepath). Those
coordinates are what later lets us build *within-slide* spatial-neighbour graphs
(never across slides — see docs/wsi_protocol.md).

Requires openslide. In Colab:
    apt-get install -y openslide-tools && pip install openslide-python pillow
Run from the repo root.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.datasets.wsi_tiling import filter_tissue_tiles, tile_grid, tissue_mask  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tile_wsi")


def slide_label(name: str) -> int:
    """Slide-level label from the CAMELYON16 filename convention."""
    n = name.lower()
    if "tumor" in n:
        return 1
    if "normal" in n:
        return 0
    return -1  # unknown (e.g. test_* slides -> resolve from reference.csv separately)


def _robust_thumbnail(slide, thumb_max):
    """Build a low-res overview block-by-block, tolerating malformed TIFF tiles.

    openslide's get_thumbnail() decodes a whole pyramid level in one RGBA pass, so
    a single bad tile (common in a few CAMELYON16 slides -> 'TIFFRGBAImageGet
    failed') kills the slide. Reading in blocks isolates the damage: unreadable
    blocks are left white (treated as background by the tissue mask).
    """
    from PIL import Image

    W0, H0 = slide.level_dimensions[0]
    ds = max(max(W0, H0) / float(thumb_max), 1.0)
    level = slide.get_best_level_for_downsample(ds)
    lw, lh = slide.level_dimensions[level]
    lvl_ds = slide.level_downsamples[level]
    img = Image.new("RGB", (lw, lh), (255, 255, 255))
    block, bad = 2048, 0
    for by in range(0, lh, block):
        for bx in range(0, lw, block):
            w, h = min(block, lw - bx), min(block, lh - by)
            loc = (int(bx * lvl_ds), int(by * lvl_ds))
            try:
                img.paste(slide.read_region(loc, level, (w, h)).convert("RGB"), (bx, by))
            except Exception:  # noqa: BLE001  malformed tile -> leave as background
                bad += 1
    if bad:
        log.warning("  %d unreadable thumbnail block(s) filled as background", bad)
    img.thumbnail((thumb_max, thumb_max))
    return img


def tile_slide(path, out_root, tile_size, stride, level, min_tissue, thumb_max):
    import openslide  # noqa: F401  (ensures the dependency is present)

    slide = openslide.OpenSlide(str(path))
    sid = Path(path).stem
    label = slide_label(sid)
    W0, H0 = slide.level_dimensions[0]

    # Thumbnail for tissue detection; downsample = level-0 px per thumbnail px.
    thumb = _robust_thumbnail(slide, thumb_max)
    tw, th = thumb.size
    downsample = max(W0 / tw, H0 / th)
    mask = tissue_mask(np.asarray(thumb))

    lvl = min(level, slide.level_count - 1)
    lvl_ds = slide.level_downsamples[lvl]
    read_span = int(round(tile_size * lvl_ds))               # level-0 px per tile
    stride_span = int(round((stride or tile_size) * lvl_ds))  # level-0 px per step

    grid = tile_grid(W0, H0, read_span, stride=stride_span)
    kept = filter_tissue_tiles(grid, mask, downsample, read_span, min_tissue=min_tissue)
    log.info("%s: %dx%d, %d grid tiles, %d tissue tiles (level %d, label %d)",
             sid, W0, H0, len(grid), len(kept), lvl, label)

    out_dir = Path(out_root) / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for t in kept:
        try:
            patch = slide.read_region((t["x"], t["y"]), lvl, (tile_size, tile_size)).convert("RGB")
        except Exception:  # noqa: BLE001  malformed tile -> skip this patch only
            skipped += 1
            continue
        fn = out_dir / f"{sid}_r{t['row']}_c{t['col']}_x{t['x']}_y{t['y']}.png"
        patch.save(fn)
        rows.append({"slide_id": sid, "label": label, "level": lvl,
                     "row": t["row"], "col": t["col"], "x": t["x"], "y": t["y"],
                     "filepath": str(fn)})
    if skipped:
        log.warning("  %s: skipped %d unreadable tile(s)", sid, skipped)
    slide.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slides", default="data/raw/camelyon16", help="dir of .tif slides")
    ap.add_argument("--out", default="data/processed/camelyon16_tiles")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=None, help="default = tile-size (non-overlapping)")
    ap.add_argument("--level", type=int, default=1, help="openslide level (1 ~ 20x on a 40x slide)")
    ap.add_argument("--min-tissue", type=float, default=0.25)
    ap.add_argument("--thumb-max", type=int, default=2048)
    args = ap.parse_args()

    slides = sorted(Path(args.slides).glob("*.tif"))
    if not slides:
        log.error("no .tif slides under %s", args.slides)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows, failed = [], []
    for s in slides:
        try:
            all_rows.extend(tile_slide(s, out, args.tile_size, args.stride,
                                       args.level, args.min_tissue, args.thumb_max))
        except Exception as e:  # noqa: BLE001  one bad slide must not kill the batch
            log.error("SKIPPING slide %s: %s", s.name, e)
            failed.append(s.name)
    if failed:
        log.warning("%d slide(s) failed and were skipped: %s", len(failed), failed)

    csv_path = out / "tiles.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["slide_id", "label", "level",
                                           "row", "col", "x", "y", "filepath"])
        w.writeheader()
        w.writerows(all_rows)
    log.info("wrote %d tiles from %d slides -> %s", len(all_rows), len(slides), csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

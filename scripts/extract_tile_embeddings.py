#!/usr/bin/env python3
"""Cache FROZEN ViT-S/16 embeddings for CAMELYON16 tiles (spatial-graph node features).

Reuses the same frozen ImageNet encoder as the BreakHis comparison, so node
features are produced by an identical representation — only the aggregation
(graph vs none) varies. Reads a ``tiles.csv`` (from scripts/tile_wsi.py), embeds
each tile PNG, and caches to ``results/embeddings/<tag>.npz`` with slide_id as the
"patient" grouping so the existing extractor machinery applies unchanged.

Requires torch/timm (Colab). Run from the repo root.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.embeddings.extractor import extract_embeddings  # noqa: E402
from src.utils.env import get_device  # noqa: E402
from src.utils.paths import get_results_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("extract_tile_embeddings")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", default="data/processed/camelyon16_tiles/tiles.csv")
    ap.add_argument("--backbone", default="vit_small_patch16_224")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tag", default="vits_imagenet_cam16")
    args = ap.parse_args()

    from src.models.baselines.patch_vit import build_model  # torch import deferred

    set_seed(42)
    tiles_path = Path(args.tiles)
    if not tiles_path.exists():
        log.error("missing tiles.csv %s — run scripts/tile_wsi.py first", tiles_path)
        return 1
    tiles = pd.read_csv(tiles_path)
    if tiles.empty:
        log.error("tiles.csv is empty — tiling produced no tiles")
        return 1

    # Map the WSI tiles table onto the columns extract_embeddings expects.
    df = pd.DataFrame({
        "filename": tiles["filepath"].map(lambda p: Path(p).name),
        "filepath": tiles["filepath"],
        "patient_id": tiles["slide_id"],      # slide is the grouping unit for WSIs
        "label_binary": tiles["label"].astype(int),
    })

    model, info = build_model({"backbone": args.backbone, "pretrained": True,
                               "freeze_backbone": True})
    model.eval()
    eval_tf = model.build_transforms(is_training=False)

    out_dir = get_results_root() / "embeddings"
    npz = extract_embeddings(model, df, eval_tf, get_device(), out_dir, args.tag,
                             batch_size=args.batch_size, resume=True)
    log.info("Tile embeddings cached: %s (encoder=%s, dim=%s, N=%d tiles, %d slides)",
             npz, info["backbone"], info["embed_dim"], len(df), tiles["slide_id"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Cache FROZEN patch embeddings once — the shared substrate for B1/B2/graphs.

Controlled-comparison decision (user, 2026-10-15): all aggregation and graph
models operate on a FROZEN ImageNet-pretrained ViT-S/16. The encoder never sees
BreakHis labels, so these embeddings are leakage-free by construction and can be
shared across every fold and model. Baseline 0 (end-to-end fine-tuned) remains a
separate image-level reference, not part of this frozen-encoder comparison.

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

import yaml  # noqa: E402

from src.datasets.breakhis import build_metadata  # noqa: E402
from src.embeddings.extractor import extract_embeddings  # noqa: E402
from src.utils.env import get_device  # noqa: E402
from src.utils.paths import get_data_root, get_results_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("extract_embeddings")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/breakhis_40x.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tag", default=None, help="cache tag (default: vits_imagenet_<mag>)")
    args = ap.parse_args()

    from src.models.baselines.patch_vit import build_model  # torch import deferred to runtime

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)
    mag = cfg["dataset"].get("magnification", "40X")
    data_root = cfg["dataset"].get("data_root") or get_data_root()
    device = get_device()

    df = build_metadata(Path(data_root), magnification=None if mag == "all" else mag)
    if df.empty:
        log.error("no images under %s", data_root)
        return 1

    # FROZEN ImageNet encoder: pretrained weights, no fine-tuning, eval transform.
    model_cfg = dict(cfg.get("model", {}) or {})
    model_cfg.update(backbone=model_cfg.get("backbone", "vit_small_patch16_224"),
                     pretrained=True, freeze_backbone=True)
    model, info = build_model(model_cfg)
    model.eval()
    eval_tf = model.build_transforms(is_training=False)

    tag = args.tag or f"vits_imagenet_{mag}"
    out_dir = get_results_root() / "embeddings"
    npz = extract_embeddings(model, df, eval_tf, device, out_dir, tag,
                             batch_size=args.batch_size, resume=True)
    log.info("Frozen embeddings cached: %s (encoder=%s, dim=%s, N=%d)",
             npz, info["backbone"], info["embed_dim"], len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

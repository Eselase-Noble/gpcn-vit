#!/usr/bin/env python3
"""CAMELYON16 spatial-graph rung (the real test of the spatial hypothesis).

Builds a per-slide spatial graph (tiles = nodes, within-slide grid adjacency =
edges) with frozen ViT-S/16 tile features, and trains a GraphSAGE GNN to a
slide-level label under slide-level stratified k-fold CV. Runs the same control
triad as BreakHis — ``--graph {spatial,rewired,none}`` — so any gain is tested
against random-structure and no-message-passing controls.

IMPORTANT: this needs ENOUGH slides for CV. The 6-slide prototype only proves the
pipeline runs end-to-end; download a larger subset before trusting the numbers.

Inputs (Colab):
  * tiles:      <tiles_dir>/tiles.csv                     (scripts/tile_wsi.py)
  * embeddings: results/embeddings/vits_imagenet_cam16.npz (scripts/extract_tile_embeddings.py)

Requires torch. Run from the repo root.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.evaluation.aggregation import select_threshold  # noqa: E402
from src.evaluation.metrics import aggregate_metrics, compute_metrics  # noqa: E402
from src.graphs.wsi_graph import build_wsi_graphs  # noqa: E402
from src.models.gnn.sage import build_model  # noqa: E402
from src.splits.cross_validation import make_patient_kfold  # noqa: E402
from src.training.class_weights import compute_class_weights  # noqa: E402
from src.utils.env import collect_env, get_device  # noqa: E402
from src.utils.paths import ensure_dir, get_results_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("graph_camelyon")


def _to_torch(g, device):
    return (torch.tensor(g["features"], device=device),
            torch.tensor(g["edge_index"], dtype=torch.long, device=device))


def _bag_loss(model, graphs, crit, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for g in graphs:
            x, e = _to_torch(g, device)
            total += float(crit(model(x, e), torch.tensor([g["label"]], device=device)).item())
    return total / max(len(graphs), 1)


def _probs(model, graphs, device):
    model.eval()
    out = []
    with torch.no_grad():
        for g in graphs:
            x, e = _to_torch(g, device)
            out.append(float(torch.softmax(model(x, e), dim=1)[0, 1].item()))
    return np.array(out)


def run_fold(fold_graphs, split, hp, objective, out_dir, seed, device):
    tr = [g for g in fold_graphs if g["slide_id"] in set(split.train)]
    va = [g for g in fold_graphs if g["slide_id"] in set(split.val)]
    te = [g for g in fold_graphs if g["slide_id"] in set(split.test)]

    cw = compute_class_weights(np.array([g["label"] for g in tr]))
    crit = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw["weight_vector"], dtype=torch.float32, device=device))

    set_seed(seed)
    torch.manual_seed(seed)
    in_dim = tr[0]["features"].shape[1]
    model = build_model(in_dim, hidden=hp["hidden"], layers=hp["layers"], dropout=hp["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    rng = np.random.default_rng(seed)
    best_state, best_val = copy.deepcopy(model.state_dict()), np.inf
    for _ in range(hp["epochs"]):
        model.train()
        for i in rng.permutation(len(tr)):
            g = tr[i]
            x, e = _to_torch(g, device)
            opt.zero_grad()
            crit(model(x, e), torch.tensor([g["label"]], device=device)).backward()
            opt.step()
        vl = _bag_loss(model, va, crit, device) if va else _bag_loss(model, tr, crit, device)
        if vl < best_val:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    yte = np.array([g["label"] for g in te])
    test_prob = _probs(model, te, device)
    thr_src = va if va else tr
    threshold = select_threshold(np.array([g["label"] for g in thr_src]),
                                 _probs(model, thr_src, device), objective=objective)
    patient_m = compute_metrics(yte, test_prob, threshold=threshold)

    pd.DataFrame({"patient_id": [g["slide_id"] for g in te], "label_binary": yte,
                  "prob": test_prob, "n_images": [g["n"] for g in te]}).to_csv(
        out_dir / "preds_test_patient.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(
        {"patient": patient_m, "patient_threshold": threshold, "best_val_loss": best_val,
         "split_meta": split.meta, "hyperparams": hp, "seed": seed}, indent=2, default=str))
    log.info("fold %s: test slide ROC-AUC=%.4f PR-AUC=%.4f (n_test=%d, val_loss=%.4f)",
             split.meta.get("fold"), patient_m["roc_auc"], patient_m["pr_auc"], len(te), best_val)
    return patient_m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", default="data/processed/camelyon16_tiles/tiles.csv")
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--graph", choices=["spatial", "rewired", "none"], default="spatial")
    ap.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    hp = {"epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
          "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout}

    tiles_path = Path(args.tiles)
    emb_path = Path(args.embeddings) if args.embeddings else get_results_root() / "embeddings" / "vits_imagenet_cam16.npz"
    if not tiles_path.exists():
        log.error("missing %s — run scripts/tile_wsi.py", tiles_path); return 1
    if not emb_path.exists():
        log.error("missing %s — run scripts/extract_tile_embeddings.py", emb_path); return 1

    tiles = pd.read_csv(tiles_path)
    data = np.load(emb_path, allow_pickle=True)
    emb_by_filename = {str(fn): e for fn, e in zip(data["filenames"], data["embeddings"])}
    n_slides = tiles["slide_id"].nunique()
    log.info("graph=%s | %d tiles, %d slides, device=%s", args.graph, len(tiles), n_slides, device)
    if n_slides < args.n_splits + 1:
        log.error("only %d slides — need >= %d for %d-fold CV. Download more slides "
                  "(scripts/download_camelyon16_subset.py --n-tumor/--n-normal).",
                  n_slides, args.n_splits + 1, args.n_splits)
        return 1

    graphs = build_wsi_graphs(tiles, emb_by_filename, connectivity=args.connectivity,
                              mode=args.graph, seed=args.seed)

    # Slide-level stratified folds (reuse the patient-level CV machinery on slide_id).
    slide_df = pd.DataFrame({"patient_id": tiles["slide_id"], "label_binary": tiles["label"]})
    folds = make_patient_kfold(slide_df, n_splits=args.n_splits, seed=args.seed, val_fraction=0.15)

    env = collect_env()
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(env.get("git_commit", "nogit"))[:8]
    run_root = ensure_dir(get_results_root() / "graph_camelyon" / "spatial_cv" / args.graph / run_id)
    (run_root / "run_config.json").write_text(json.dumps(
        {"env": env, "graph": args.graph, "connectivity": args.connectivity,
         "hyperparams": hp, "n_slides": int(n_slides), "model": "graphsage"}, indent=2, default=str))

    results = [run_fold(graphs, split, hp, "balanced_accuracy",
                        ensure_dir(run_root / f"fold_{split.meta.get('fold')}"), args.seed, device)
               for split in folds]

    summary = {"patient": aggregate_metrics(results), "n_folds": len(folds),
               "graph": args.graph, "n_slides": int(n_slides), "model": "graphsage"}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("CAMELYON graph[%s] CV slide ROC-AUC: %.4f ± %.4f | PR-AUC: %.4f ± %.4f",
             args.graph, summary["patient"]["roc_auc"]["mean"], summary["patient"]["roc_auc"]["std"],
             summary["patient"]["pr_auc"]["mean"], summary["patient"]["pr_auc"]["std"])
    log.info("Done. Results under %s", run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

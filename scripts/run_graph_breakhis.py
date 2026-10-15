#!/usr/bin/env python3
"""BreakHis feature-similarity GRAPH rung + relationship-necessity controls.

Builds a per-patient k-NN graph over the patient's frozen image embeddings and
trains a small GraphSAGE GNN to a patient-level prediction. Runs the SAME folds /
metrics / val-only-threshold protocol as B0/B1/B2 so results are comparable.

The point is NOT "does a GNN beat mean-pool" — it is whether MEANINGFUL
connectivity helps. So ``--graph`` selects one of three conditions on identical
nodes/features:
  * feature  — real k-NN similarity edges
  * rewired  — degree-preserving randomized edges (control: same degrees, random)
  * none     — no edges (control: pure per-node transform + readout; no messages)

A real-graph gain is only credible if 'feature' beats BOTH 'rewired' and 'none'.

Requires torch (Colab). Run from the repo root.
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.evaluation.aggregation import select_threshold  # noqa: E402
from src.evaluation.metrics import aggregate_metrics, compute_metrics  # noqa: E402
from src.graphs.feature import knn_feature_edges  # noqa: E402
from src.graphs.spatial import rewire_degree_preserving  # noqa: E402
from src.models.gnn.sage import build_model  # noqa: E402
from src.splits.cross_validation import load_folds  # noqa: E402
from src.training.class_weights import compute_class_weights  # noqa: E402
from src.utils.env import collect_env, get_device  # noqa: E402
from src.utils.paths import ensure_dir, get_data_root, get_results_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("graph_breakhis")


def _mask(ids, keep):
    return np.array([i in keep for i in ids])


def _edges(X, mode, k, seed):
    if mode == "none":
        return np.empty((2, 0), dtype=int)
    e = knn_feature_edges(X, k=k)
    return rewire_degree_preserving(e, seed=seed) if mode == "rewired" else e


def _build_graphs(emb, pid, lab, mode, k, seed):
    graphs = []
    for i, p in enumerate(np.sort(pd.unique(pid))):
        m = pid == p
        X = emb[m]
        graphs.append({
            "patient_id": p, "y": int(lab[m][0]), "n": int(m.sum()),
            "x": torch.tensor(X, dtype=torch.float32),
            "edge_index": torch.tensor(_edges(X, mode, k, seed + i), dtype=torch.long),
        })
    return graphs


def _bag_loss(model, graphs, crit, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for g in graphs:
            logits = model(g["x"].to(device), g["edge_index"].to(device))
            total += float(crit(logits, torch.tensor([g["y"]], device=device)).item())
    return total / max(len(graphs), 1)


def _probs(model, graphs, device):
    model.eval()
    out = []
    with torch.no_grad():
        for g in graphs:
            logits = model(g["x"].to(device), g["edge_index"].to(device))
            out.append(float(torch.softmax(logits, dim=1)[0, 1].item()))
    return np.array(out)


def run_fold(split, emb, pid, lab, mode, hp, objective, out_dir, seed, device):
    in_tr, in_va, in_te = (_mask(pid, set(split.train)),
                           _mask(pid, set(split.val)),
                           _mask(pid, set(split.test)))
    train_g = _build_graphs(emb[in_tr], pid[in_tr], lab[in_tr], mode, hp["k"], seed)
    val_g = _build_graphs(emb[in_va], pid[in_va], lab[in_va], mode, hp["k"], seed + 1000)
    test_g = _build_graphs(emb[in_te], pid[in_te], lab[in_te], mode, hp["k"], seed + 2000)

    cw = compute_class_weights(np.array([g["y"] for g in train_g]))
    crit = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw["weight_vector"], dtype=torch.float32, device=device))

    set_seed(seed)
    torch.manual_seed(seed)
    model = build_model(emb.shape[1], hidden=hp["hidden"], layers=hp["layers"],
                        dropout=hp["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    rng = np.random.default_rng(seed)
    best_state, best_val = copy.deepcopy(model.state_dict()), np.inf
    for _ in range(hp["epochs"]):
        model.train()
        for i in rng.permutation(len(train_g)):
            g = train_g[i]
            opt.zero_grad()
            logits = model(g["x"].to(device), g["edge_index"].to(device))
            crit(logits, torch.tensor([g["y"]], device=device)).backward()
            opt.step()
        vl = _bag_loss(model, val_g, crit, device)
        if vl < best_val:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    val_prob, test_prob = _probs(model, val_g, device), _probs(model, test_g, device)
    yva = np.array([g["y"] for g in val_g])
    yte = np.array([g["y"] for g in test_g])
    threshold = select_threshold(yva, val_prob, objective=objective)
    patient_m = compute_metrics(yte, test_prob, threshold=threshold)

    pd.DataFrame({"patient_id": [g["patient_id"] for g in test_g],
                  "label_binary": yte, "prob": test_prob,
                  "n_images": [g["n"] for g in test_g]}).to_csv(
        out_dir / "preds_test_patient.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "patient": patient_m, "patient_threshold": threshold, "graph": mode,
        "selection": "min_val_bag_loss", "best_val_loss": best_val,
        "threshold_objective": objective, "hyperparams": hp,
        "split_meta": split.meta, "seed": seed,
    }, indent=2, default=str))
    log.info("[%s] fold %s: test patient ROC-AUC=%.4f PR-AUC=%.4f (val_loss=%.4f, thr=%.3f)",
             mode, split.meta.get("fold"), patient_m["roc_auc"], patient_m["pr_auc"],
             best_val, threshold)
    return patient_m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/breakhis_40x.yaml")
    ap.add_argument("--graph", choices=["feature", "rewired", "none"], default="feature")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.25)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)
    mag = cfg["dataset"].get("magnification", "40X")
    objective = (cfg.get("training", {}) or {}).get("threshold_objective", "balanced_accuracy")
    device = get_device()
    hp = {"k": args.k, "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
          "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout}

    folds_path = Path(args.folds) if args.folds else get_data_root() / "splits" / f"cv_{mag}_seed{seed}.json"
    emb_path = Path(args.embeddings) if args.embeddings else get_results_root() / "embeddings" / f"vits_imagenet_{mag}.npz"
    for pth, hint in [(folds_path, "scripts/make_cv_folds.py"), (emb_path, "scripts/extract_embeddings.py")]:
        if not pth.exists():
            log.error("missing %s — run %s first", pth, hint)
            return 1

    folds = load_folds(folds_path)
    data = np.load(emb_path, allow_pickle=True)
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    pid = data["patient_ids"].astype(str)
    lab = data["labels"].astype(int)
    log.info("graph=%s | loaded %d embeddings (dim=%d), %d folds, device=%s",
             args.graph, len(emb), emb.shape[1], len(folds), device)

    env = collect_env()
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(env.get("git_commit", "nogit"))[:8]
    run_root = ensure_dir(get_results_root() / "graph_breakhis" / f"{mag}_cv" / args.graph / run_id)
    (run_root / "run_config.json").write_text(json.dumps(
        {"config": cfg, "env": env, "graph": args.graph, "hyperparams": hp,
         "encoder": "frozen_vits_imagenet", "model": "graphsage"}, indent=2, default=str))

    results = [run_fold(split, emb, pid, lab, args.graph, hp, objective,
                        ensure_dir(run_root / f"fold_{split.meta.get('fold')}"), seed, device)
               for split in folds]

    summary = {"patient": aggregate_metrics(results), "n_folds": len(folds),
               "graph": args.graph, "encoder": "frozen_vits_imagenet", "model": "graphsage"}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("GRAPH[%s] CV patient ROC-AUC: %.4f ± %.4f | PR-AUC: %.4f ± %.4f",
             args.graph, summary["patient"]["roc_auc"]["mean"], summary["patient"]["roc_auc"]["std"],
             summary["patient"]["pr_auc"]["mean"], summary["patient"]["pr_auc"]["std"])
    log.info("Done. Results under %s", run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

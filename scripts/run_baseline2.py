#!/usr/bin/env python3
"""Baseline 2 runner — gated-attention MIL over frozen embeddings, patient-level CV.

Same canonical folds, frozen ViT-S/16 embeddings, metrics, val-only threshold, and
per-patient export layout as B0/B1 — so B2 is directly comparable. Model selection
uses LOWEST validation bag-loss (consistent with the project's selection rule;
patient ROC-AUC saturates on ~10 val patients and cannot select epochs).

Inputs (produced earlier, in Colab):
  * folds:       data/splits/cv_<mag>_seed<seed>.json       (scripts/make_cv_folds.py)
  * embeddings:  results/embeddings/vits_imagenet_<mag>.npz (scripts/extract_embeddings.py)

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
from src.models.baselines.attention_mil import build_model, make_bags  # noqa: E402
from src.splits.cross_validation import load_folds  # noqa: E402
from src.training.class_weights import compute_class_weights  # noqa: E402
from src.utils.env import collect_env, get_device  # noqa: E402
from src.utils.paths import ensure_dir, get_data_root, get_results_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("baseline2")


def _mask(ids, keep):
    return np.array([i in keep for i in ids])


def _bag_loss(model, bags, crit, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for b in bags:
            logits, _ = model(torch.tensor(b["X"], device=device))
            total += float(crit(logits, torch.tensor([b["y"]], device=device)).item())
    return total / max(len(bags), 1)


def _probs(model, bags, device):
    model.eval()
    out = []
    with torch.no_grad():
        for b in bags:
            logits, _ = model(torch.tensor(b["X"], device=device))
            out.append(float(torch.softmax(logits, dim=1)[0, 1].item()))
    return np.array(out)


def run_fold(split, emb, pid, lab, cfg, objective, out_dir, seed, device):
    in_tr, in_va, in_te = (_mask(pid, set(split.train)),
                           _mask(pid, set(split.val)),
                           _mask(pid, set(split.test)))
    train_bags = make_bags(emb[in_tr], pid[in_tr], lab[in_tr])
    val_bags = make_bags(emb[in_va], pid[in_va], lab[in_va])
    test_bags = make_bags(emb[in_te], pid[in_te], lab[in_te])

    cw = compute_class_weights(lab[in_tr])
    weight = torch.tensor(cw["weight_vector"], dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=weight)

    set_seed(seed)
    torch.manual_seed(seed)
    model = build_model(emb.shape[1], hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    rng = np.random.default_rng(seed)
    best_state, best_val = copy.deepcopy(model.state_dict()), np.inf
    for epoch in range(cfg["epochs"]):
        model.train()
        for i in rng.permutation(len(train_bags)):
            b = train_bags[i]
            opt.zero_grad()
            logits, _ = model(torch.tensor(b["X"], device=device))
            crit(logits, torch.tensor([b["y"]], device=device)).backward()
            opt.step()
        vl = _bag_loss(model, val_bags, crit, device)
        if vl < best_val:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    val_prob = _probs(model, val_bags, device)
    test_prob = _probs(model, test_bags, device)
    yva = np.array([b["y"] for b in val_bags])
    yte = np.array([b["y"] for b in test_bags])
    threshold = select_threshold(yva, val_prob, objective=objective)
    patient_m = compute_metrics(yte, test_prob, threshold=threshold)

    pd.DataFrame({"patient_id": [b["patient_id"] for b in test_bags],
                  "label_binary": yte, "prob": test_prob,
                  "n_images": [b["n"] for b in test_bags]}).to_csv(
        out_dir / "preds_test_patient.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "patient": patient_m, "patient_threshold": threshold,
        "selection": "min_val_bag_loss", "best_val_loss": best_val,
        "threshold_objective": objective, "hyperparams": cfg,
        "n_train_patients": len(train_bags), "n_val_patients": len(val_bags),
        "n_test_patients": len(test_bags), "split_meta": split.meta, "seed": seed,
    }, indent=2, default=str))
    log.info("fold %s: test patient ROC-AUC=%.4f PR-AUC=%.4f (val_loss=%.4f, thr=%.3f)",
             split.meta.get("fold"), patient_m["roc_auc"], patient_m["pr_auc"],
             best_val, threshold)
    return patient_m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/breakhis_40x.yaml")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.25)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)
    mag = cfg["dataset"].get("magnification", "40X")
    objective = (cfg.get("training", {}) or {}).get("threshold_objective", "balanced_accuracy")
    device = get_device()
    hp = {"epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
          "hidden": args.hidden, "dropout": args.dropout}

    folds_path = Path(args.folds) if args.folds else get_data_root() / "splits" / f"cv_{mag}_seed{seed}.json"
    emb_path = Path(args.embeddings) if args.embeddings else get_results_root() / "embeddings" / f"vits_imagenet_{mag}.npz"
    if not folds_path.exists():
        log.error("missing folds %s — run scripts/make_cv_folds.py first", folds_path)
        return 1
    if not emb_path.exists():
        log.error("missing embeddings %s — run scripts/extract_embeddings.py first", emb_path)
        return 1

    folds = load_folds(folds_path)
    data = np.load(emb_path, allow_pickle=True)
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    pid = data["patient_ids"].astype(str)
    lab = data["labels"].astype(int)
    log.info("loaded %d embeddings (dim=%d), %d folds, device=%s", len(emb), emb.shape[1], len(folds), device)

    env = collect_env()
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(env.get("git_commit", "nogit"))[:8]
    run_root = ensure_dir(get_results_root() / "baseline2" / f"{mag}_cv" / run_id)
    (run_root / "run_config.json").write_text(json.dumps(
        {"config": cfg, "env": env, "folds": str(folds_path), "embeddings": str(emb_path),
         "hyperparams": hp, "encoder": "frozen_vits_imagenet", "model": "attention_mil"},
        indent=2, default=str))

    results = [run_fold(split, emb, pid, lab, hp, objective,
                        ensure_dir(run_root / f"fold_{split.meta.get('fold')}"), seed, device)
               for split in folds]

    summary = {"patient": aggregate_metrics(results), "n_folds": len(folds),
               "encoder": "frozen_vits_imagenet", "model": "attention_mil", "hyperparams": hp}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("B2 CV patient ROC-AUC: %.4f ± %.4f | PR-AUC: %.4f ± %.4f",
             summary["patient"]["roc_auc"]["mean"], summary["patient"]["roc_auc"]["std"],
             summary["patient"]["pr_auc"]["mean"], summary["patient"]["pr_auc"]["std"])
    log.info("Done. Results under %s", run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Baseline 1 runner — mean-pool frozen embeddings -> patient-level classifier.

Reuses the SAME canonical folds, metrics, threshold selection, and per-patient
export layout as Baseline 0, so the two are directly comparable patient-by-patient
(paired tests). Model selection (C) and the decision threshold use train/val only;
each outer test fold is scored exactly once.

Inputs (both produced earlier, in Colab):
  * folds:       data/splits/cv_<mag>_seed<seed>.json       (scripts/make_cv_folds.py)
  * embeddings:  results/embeddings/vits_imagenet_<mag>.npz (scripts/extract_embeddings.py)

Pure numpy/sklearn — no torch. Run from the repo root.
"""
from __future__ import annotations

import argparse
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

from src.evaluation.aggregation import select_threshold  # noqa: E402
from src.evaluation.metrics import aggregate_metrics, compute_metrics  # noqa: E402
from src.models.baselines.mean_pool import MeanPoolClassifier, patient_mean_embeddings  # noqa: E402
from src.splits.cross_validation import load_folds  # noqa: E402
from src.utils.env import collect_env  # noqa: E402
from src.utils.paths import ensure_dir, get_data_root, get_results_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("baseline1")


def _mask(ids, keep):
    return np.array([i in keep for i in ids])


def _patient_frame(patients, probs, labels, n_images):
    return pd.DataFrame({
        "patient_id": patients, "label_binary": labels,
        "prob": probs, "n_images": n_images,
    })


def run_fold(split, emb, pid, lab, cand_C, objective, out_dir, seed):
    in_tr, in_va, in_te = (_mask(pid, set(split.train)),
                           _mask(pid, set(split.val)),
                           _mask(pid, set(split.test)))

    _, Xtr, ytr, _ = patient_mean_embeddings(emb[in_tr], pid[in_tr], lab[in_tr])
    p_va, Xva, yva, nva = patient_mean_embeddings(emb[in_va], pid[in_va], lab[in_va])
    p_te, Xte, yte, nte = patient_mean_embeddings(emb[in_te], pid[in_te], lab[in_te])

    # Select C on validation patient ROC-AUC (threshold-free), never on test.
    best_C, best_auc = cand_C[0], -np.inf
    for C in cand_C:
        clf = MeanPoolClassifier(C=C, seed=seed).fit(Xtr, ytr)
        auc = compute_metrics(yva, clf.predict_proba_pos(Xva))["roc_auc"]
        if np.isfinite(auc) and auc > best_auc:
            best_auc, best_C = auc, C

    clf = MeanPoolClassifier(C=best_C, seed=seed).fit(Xtr, ytr)
    val_prob = clf.predict_proba_pos(Xva)
    test_prob = clf.predict_proba_pos(Xte)

    threshold = select_threshold(yva, val_prob, objective=objective)
    patient_m = compute_metrics(yte, test_prob, threshold=threshold)

    _patient_frame(p_va, val_prob, yva, nva).to_csv(out_dir / "preds_val_patient.csv", index=False)
    _patient_frame(p_te, test_prob, yte, nte).to_csv(out_dir / "preds_test_patient.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "patient": patient_m, "patient_threshold": threshold,
        "selection": "val_patient_roc_auc", "selected_C": best_C,
        "val_roc_auc": best_auc, "threshold_objective": objective,
        "n_train_patients": int(len(ytr)), "n_val_patients": int(len(p_va)),
        "n_test_patients": int(len(p_te)), "split_meta": split.meta, "seed": seed,
    }, indent=2, default=str))
    log.info("fold %s: test patient ROC-AUC=%.4f PR-AUC=%.4f (C=%g, thr=%.3f)",
             split.meta.get("fold"), patient_m["roc_auc"], patient_m["pr_auc"],
             best_C, threshold)
    return patient_m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/breakhis_40x.yaml")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)
    mag = cfg["dataset"].get("magnification", "40X")
    objective = (cfg.get("training", {}) or {}).get("threshold_objective", "balanced_accuracy")

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
    emb = np.asarray(data["embeddings"], dtype=float)
    pid = data["patient_ids"].astype(str)
    lab = data["labels"].astype(int)
    log.info("loaded %d embeddings (dim=%d), %d folds", len(emb), emb.shape[1], len(folds))

    env = collect_env()
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(env.get("git_commit", "nogit"))[:8]
    run_root = ensure_dir(get_results_root() / "baseline1" / f"{mag}_cv" / run_id)
    (run_root / "run_config.json").write_text(json.dumps(
        {"config": cfg, "env": env, "folds": str(folds_path), "embeddings": str(emb_path),
         "C_grid": args.C, "encoder": "frozen_vits_imagenet", "model": "mean_pool"},
        indent=2, default=str))

    results = [run_fold(split, emb, pid, lab, args.C, objective,
                        ensure_dir(run_root / f"fold_{split.meta.get('fold')}"), seed)
               for split in folds]

    summary = {"patient": aggregate_metrics(results), "n_folds": len(folds),
               "encoder": "frozen_vits_imagenet", "model": "mean_pool"}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("B1 CV patient ROC-AUC: %.4f ± %.4f | PR-AUC: %.4f ± %.4f",
             summary["patient"]["roc_auc"]["mean"], summary["patient"]["roc_auc"]["std"],
             summary["patient"]["pr_auc"]["mean"], summary["patient"]["pr_auc"]["std"])
    log.info("Done. Results under %s", run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

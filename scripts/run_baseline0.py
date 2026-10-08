#!/usr/bin/env python3
"""Baseline 0 runner — fine-tune ViT-S/16, evaluate image- AND patient-level.

Config-driven (configs/breakhis_40x.yaml). Runs the single prototype split by
default, or 5-fold patient-level CV with --cv. Persists per-image & per-patient
predictions, metrics, thresholds, class weights, config, env, and git commit.

Examples
--------
    # fast plumbing check in Colab (tiny subset, 1 epoch, CPU/GPU)
    python scripts/run_baseline0.py --config configs/breakhis_40x.yaml --smoke

    # prototype on single split
    python scripts/run_baseline0.py --config configs/breakhis_40x.yaml

    # reported results: 5-fold patient-level CV
    python scripts/run_baseline0.py --config configs/breakhis_40x.yaml --cv

Run from the repo root. Requires torch/timm (Colab).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.breakhis import build_metadata  # noqa: E402
from src.datasets.image_dataset import BreakHisImageDataset  # noqa: E402
from src.evaluation.aggregation import (  # noqa: E402
    aggregate_to_patient,
    evaluate_predictions,
    select_threshold,
)
from src.evaluation.metrics import aggregate_metrics  # noqa: E402
from src.splits.cross_validation import make_patient_kfold  # noqa: E402
from src.splits.patient_split import make_patient_split, split_dataframe  # noqa: E402
from src.training.class_weights import compute_class_weights  # noqa: E402
from src.utils.env import collect_env, get_device  # noqa: E402
from src.utils.paths import ensure_dir, get_data_root, get_results_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("baseline0")


def _loader(ds, batch_size, shuffle, seed, num_workers):
    import torch

    g = torch.Generator()
    g.manual_seed(seed)

    def _wi(worker_id):
        np.random.seed(seed + worker_id)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, generator=g, worker_init_fn=_wi,
    )


def run_fold(fold_df, model_cfg, train_cfg, split, device, out_dir, seed, num_workers):
    """Train+evaluate one fold. Returns metrics dict (image+patient on test)."""
    from src.models.baselines.patch_vit import build_model
    from src.training.trainer import Trainer, TrainConfig

    parts = split_dataframe(fold_df, split)  # train/val/test image-level dfs
    train_df, val_df, test_df = parts["train"], parts["val"], parts["test"]
    log.info("fold images: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    model, model_info = build_model(model_cfg)
    train_tf = model.build_transforms(is_training=True)
    eval_tf = model.build_transforms(is_training=False)

    cw = compute_class_weights(train_df["label_binary"].to_numpy())
    log.info("fold class weights (train-only): %s", cw["weights"])

    bs = train_cfg["batch_size"]
    train_loader = _loader(BreakHisImageDataset(train_df, train_tf), bs, True, seed, num_workers)
    val_loader = _loader(BreakHisImageDataset(val_df, eval_tf), bs, False, seed, num_workers)
    test_loader = _loader(BreakHisImageDataset(test_df, eval_tf), bs, False, seed, num_workers)

    tcfg = TrainConfig(
        epochs=train_cfg["epochs"], lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        monitor=train_cfg.get("monitor", "roc_auc"),
        monitor_level=train_cfg.get("monitor_level", "patient"),
    )
    trainer = Trainer(model, device, cw["weight_vector"], tcfg, out_dir / "checkpoints")
    trainer.fit(train_loader, val_loader)
    trainer.load_best()

    # threshold from VALIDATION only (image-level: more samples than patients)
    val_preds = trainer.predict_df(val_loader)
    threshold = select_threshold(
        val_preds["label_binary"].to_numpy(), val_preds["prob"].to_numpy(),
        objective=train_cfg.get("threshold_objective", "balanced_accuracy"),
    )
    log.info("selected threshold (val): %.4f", threshold)

    test_preds = trainer.predict_df(test_loader)
    results = evaluate_predictions(test_preds, threshold=threshold)

    # persist everything
    val_preds.to_csv(out_dir / "preds_val.csv", index=False)
    test_preds.to_csv(out_dir / "preds_test_image.csv", index=False)
    aggregate_to_patient(test_preds).to_csv(out_dir / "preds_test_patient.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "image": results["image"], "patient": results["patient"],
        "threshold": threshold, "threshold_source": "val_image",
        "class_weights": cw, "model_info": model_info,
        "split_meta": split.meta,
    }, indent=2, default=str))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default="configs/breakhis_40x.yaml")
    ap.add_argument("--cv", action="store_true", help="5-fold patient-level CV (reported results)")
    ap.add_argument("--smoke", action="store_true", help="tiny subset + 1 epoch to test plumbing")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()
    env = collect_env()
    log.info("device=%s env=%s", device, json.dumps(env))

    mag = cfg["dataset"].get("magnification", "40X")
    data_root = cfg["dataset"].get("data_root") or get_data_root()
    df = build_metadata(Path(data_root), magnification=None if mag == "all" else mag)
    if df.empty:
        log.error("no images parsed under %s", data_root)
        return 1

    model_cfg = cfg.get("model", {}) or {}
    model_cfg.setdefault("backbone", "vit_small_patch16_224")
    train_cfg = dict(cfg.get("training", {}) or {})
    train_cfg.setdefault("epochs", 15)
    train_cfg.setdefault("batch_size", 32)
    train_cfg.setdefault("learning_rate", 3e-5)

    if args.smoke:
        df = df.groupby("label_binary", group_keys=False).apply(lambda g: g.head(40))
        train_cfg["epochs"] = 1
        log.warning("SMOKE MODE: subset=%d images, epochs=1 (plumbing test only)", len(df))

    run_root = ensure_dir(get_results_root() / "baseline0" /
                          (f"{mag}_cv" if args.cv else f"{mag}_single"))
    (run_root / "run_config.json").write_text(json.dumps(
        {"config": cfg, "env": env, "cv": args.cv, "smoke": args.smoke}, indent=2, default=str))

    ratios = tuple(cfg.get("split", {}).get("ratios", (0.70, 0.15, 0.15)))

    if args.cv:
        folds = make_patient_kfold(df, n_splits=cfg.get("split", {}).get("n_splits", 5),
                                   seed=seed, val_fraction=0.15)
        fold_results = []
        for split in folds:
            fout = ensure_dir(run_root / f"fold_{split.meta['fold']}")
            res = run_fold(df, model_cfg, train_cfg, split, device, fout, seed, args.num_workers)
            fold_results.append(res)
        summary = {
            "image": aggregate_metrics([r["image"] for r in fold_results]),
            "patient": aggregate_metrics([r["patient"] for r in fold_results]),
            "n_folds": len(folds),
        }
        (run_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log.info("CV patient-level ROC-AUC: %.4f ± %.4f",
                 summary["patient"]["roc_auc"]["mean"], summary["patient"]["roc_auc"]["std"])
        log.info("CV patient-level PR-AUC:  %.4f ± %.4f",
                 summary["patient"]["pr_auc"]["mean"], summary["patient"]["pr_auc"]["std"])
    else:
        split = make_patient_split(df, ratios=ratios, seed=seed)
        res = run_fold(df, model_cfg, train_cfg, split, device,
                       ensure_dir(run_root / "single"), seed, args.num_workers)
        log.info("TEST patient-level: ROC-AUC=%.4f PR-AUC=%.4f balAcc=%.4f",
                 res["patient"]["roc_auc"], res["patient"]["pr_auc"],
                 res["patient"]["balanced_accuracy"])
        log.info("TEST image-level:   ROC-AUC=%.4f PR-AUC=%.4f",
                 res["image"]["roc_auc"], res["image"]["pr_auc"])
    log.info("Done. Results under %s", run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

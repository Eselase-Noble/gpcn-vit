#!/usr/bin/env python3
"""Freeze the CANONICAL patient-level CV folds once, for all models to reuse.

Baseline 0 re-derives folds in-process; from Baseline 1 onward every model loads
the SAME folds from this artifact so paired comparisons share identical outer
test patients (audit point 1). Folds are deterministic given (patients, seed);
this script just writes them to ``data/splits/cv_<mag>_seed<seed>.json`` and
verifies the artifact round-trips and re-passes leakage checks.

Run from the repo root.
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
from src.splits.cross_validation import load_folds, make_patient_kfold, save_folds  # noqa: E402
from src.utils.paths import get_data_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("make_cv_folds")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/breakhis_40x.yaml")
    ap.add_argument("--val-fraction", type=float, default=0.15,
                    help="inner val carve per fold (match Baseline 0 = 0.15)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: data/splits/cv_<mag>_seed<seed>.json)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    split_cfg = cfg.get("split", {}) or {}
    seed = split_cfg.get("seed", cfg.get("reproducibility", {}).get("seed", 42))
    n_splits = split_cfg.get("n_splits", 5)
    mag = cfg["dataset"].get("magnification", "40X")
    data_root = cfg["dataset"].get("data_root") or get_data_root()

    set_seed(seed)
    df = build_metadata(Path(data_root), magnification=None if mag == "all" else mag)
    if df.empty:
        log.error("no images parsed under %s", data_root)
        return 1

    folds = make_patient_kfold(df, n_splits=n_splits, seed=seed, val_fraction=args.val_fraction)
    out = Path(args.out) if args.out else (get_data_root() / "splits" / f"cv_{mag}_seed{seed}.json")
    save_folds(folds, out, dataset="breakhis", magnification=mag,
               extra={"val_fraction": args.val_fraction, "config": args.config})

    # Fail loudly if the artifact does not round-trip and re-verify.
    reloaded = load_folds(out, df=df)
    assert len(reloaded) == len(folds)
    for a, b in zip(folds, reloaded):
        assert sorted(a.test) == sorted(b.test) and sorted(a.val) == sorted(b.val)
    log.info("Canonical folds OK: %d folds, %d patients, seed=%d -> %s",
             len(folds), df["patient_id"].nunique(), seed, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

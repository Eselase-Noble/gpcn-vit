#!/usr/bin/env python3
"""Milestone 1 pipeline: metadata -> EDA -> patient-level split -> leakage check.

Usage
-----
    # point at your BreakHis images (or set GPCN_DATA_ROOT)
    python scripts/run_milestone1.py --data-root /content/data/BreaKHis_v1 \
        --magnification 40X --seed 42

Outputs (under the results/data roots):
    data/metadata/breakhis_metadata.csv
    results/metrics/eda_summary.json
    data/splits/breakhis_<mag>_seed<seed>.json

Run from the repository root so ``import src.*`` resolves.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make repo root importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.breakhis import build_metadata  # noqa: E402
from src.eda.summary import format_summary, summarize  # noqa: E402
from src.splits.patient_split import make_patient_split  # noqa: E402
from src.utils.env import collect_env  # noqa: E402
from src.utils.paths import ensure_dir, get_data_root, get_results_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=str, default=None,
                    help="BreakHis image root (default: GPCN_DATA_ROOT or <repo>/data).")
    ap.add_argument("--magnification", type=str, default="40X",
                    help="Magnification filter, e.g. 40X. Use 'all' for all.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()

    set_seed(args.seed)
    env = collect_env()
    logging.info("Environment: %s", json.dumps(env))

    data_root = Path(args.data_root) if args.data_root else get_data_root()
    mag = None if args.magnification.lower() == "all" else args.magnification

    df = build_metadata(data_root, magnification=mag)
    if df.empty:
        logging.error("No BreakHis images parsed under %s. Check the path/layout.", data_root)
        return 1

    # 1) persist metadata
    meta_dir = ensure_dir(get_data_root() / "metadata")
    meta_csv = meta_dir / "breakhis_metadata.csv"
    df.to_csv(meta_csv, index=False)
    logging.info("Wrote metadata: %s", meta_csv)

    # 2) EDA
    summary = summarize(df)
    summary["_env"] = env
    summary["_magnification"] = args.magnification
    print(format_summary(summary))
    metrics_dir = ensure_dir(get_results_root() / "metrics")
    (metrics_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # 3) patient-level split + leakage verification (raises on leakage)
    split = make_patient_split(df, ratios=tuple(args.ratios), seed=args.seed)
    tag = args.magnification.lower()
    split_path = ensure_dir(get_data_root() / "splits") / f"breakhis_{tag}_seed{args.seed}.json"
    split.to_json(split_path)
    logging.info("Wrote patient-disjoint split: %s", split_path)
    logging.info("Split sizes (patients): %s", split.meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

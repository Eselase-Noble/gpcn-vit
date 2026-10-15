#!/usr/bin/env python3
"""Cross-model comparison report — paired, per-patient, cross-validated.

Auto-discovers the latest run of each model under a results root, pools each
model's held-out per-patient predictions (one row per patient), and compares every
model against a reference with a paired bootstrap on ROC-AUC. Writes a markdown
table + JSON under results/comparison/.

    python scripts/compare_models.py \
        --results-root /content/drive/MyDrive/MYRESEARCH/gpcn/results --ref baseline1

Pure numpy/pandas/sklearn — runs wherever the result CSVs live (Colab/Drive).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.comparison import (  # noqa: E402
    load_pooled_predictions,
    paired_bootstrap_auc,
    pooled_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("compare_models")

# name -> glob of candidate run dirs (each containing fold_*/preds_test_patient.csv)
MODEL_GLOBS = {
    "baseline0_vit_finetuned": "baseline0/*_cv/*",
    "baseline1_meanpool": "baseline1/*_cv/*",
    "baseline2_attention": "baseline2/*_cv/*",
    "graph_feature": "graph_breakhis/*_cv/feature/*",
    "graph_rewired": "graph_breakhis/*_cv/rewired/*",
    "graph_none": "graph_breakhis/*_cv/none/*",
}


def _latest_valid(root: Path, pattern: str):
    for cand in sorted(root.glob(pattern), reverse=True):  # latest timestamp first
        if cand.is_dir() and list(cand.glob("fold_*/preds_test_patient.csv")):
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--ref", default="baseline1_meanpool", help="reference model name for paired tests")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results_root)
    preds, meta = {}, {}
    for name, glob in MODEL_GLOBS.items():
        run = _latest_valid(root, glob)
        if run is None:
            log.warning("no run found for %s (%s) — skipping", name, glob)
            continue
        df = load_pooled_predictions(run)
        preds[name] = df
        meta[name] = {"run_dir": str(run), **pooled_metrics(df)}
        log.info("%-26s n=%d  pooled ROC-AUC=%.4f  PR-AUC=%.4f  (%s)",
                 name, meta[name]["n_patients"], meta[name]["roc_auc"],
                 meta[name]["pr_auc"], run.name)

    if args.ref not in preds:
        log.error("reference '%s' not found among %s", args.ref, list(preds)); return 1

    # Align every model to the reference's patient order (same held-out patients).
    ref = preds[args.ref].set_index("patient_id")
    y = ref["label_binary"].to_numpy().astype(int)
    comparisons = {}
    for name, df in preds.items():
        if name == args.ref:
            continue
        d = df.set_index("patient_id").reindex(ref.index)
        if d["prob"].isna().any():
            log.warning("%s patient set differs from ref — skipping paired test", name); continue
        comparisons[name] = paired_bootstrap_auc(y, d["prob"].to_numpy(),
                                                  ref["prob"].to_numpy(),
                                                  n_boot=args.n_boot, seed=42)

    # ---- report ----
    lines = ["# Cross-model comparison (paired, per-patient, pooled over CV folds)", "",
             f"Reference = **{args.ref}**. Paired bootstrap on ROC-AUC (n_boot={args.n_boot}).",
             "81 patients is low power — treat p-values as supportive, not confirmatory.", "",
             "| model | pooled ROC-AUC | pooled PR-AUC | ΔAUC vs ref | 95% CI | p |",
             "|---|---|---|---|---|---|"]
    for name in MODEL_GLOBS:
        if name not in meta:
            continue
        m = meta[name]
        if name == args.ref:
            lines.append(f"| {name} (ref) | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | — | — | — |")
        elif name in comparisons:
            c = comparisons[name]
            lines.append(f"| {name} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
                         f"{c['diff']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['p_two_sided']:.3f} |")
    report = "\n".join(lines) + "\n"

    out_dir = Path(args.out) if args.out else (root / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"comparison_{stamp}.md").write_text(report)
    (out_dir / f"comparison_{stamp}.json").write_text(json.dumps(
        {"reference": args.ref, "models": meta, "paired_vs_ref": comparisons}, indent=2))
    print("\n" + report)
    log.info("wrote report -> %s", out_dir / f"comparison_{stamp}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# PROGRESS — where we are right now

> Living status file. Update at the end of every working session. The scientific
> protocol lives in `docs/MASTER_INSTRUCTIONS.md` (original prompt, verbatim) and
> `docs/experiment_protocol.md`. The running history lives in `docs/research_log.md`.

**Last updated:** 2026-10-15
**Current milestone:** B0 complete (clean CV run: patient ROC-AUC 0.9567 ± 0.0498, PR-AUC 0.9797 ± 0.0252,
commit `ae8206b`). B1 (mean-pool) + shared substrate + CAMELYON16 WSI pipeline added as CODE — awaiting Colab run (47 tests).
**Next action (Colab, in order):** (1) `scripts/make_cv_folds.py` → freeze canonical folds;
(2) `scripts/extract_embeddings.py` → cache frozen ViT-S/16 ImageNet embeddings; (3) `scripts/run_baseline1.py`
→ B1 CV. In parallel: `scripts/download_camelyon16_subset.py` (pip install boto3) then `scripts/tile_wsi.py`.

**Eval protocol DECIDED (2026-10-08):** prototype/debug on the single 57/13/11 split; report final
results under **patient-level stratified 5-fold CV** (mean ± std). Applies to every model on the ladder.

**Decisions (2026-10-15):**
- Controlled aggregation/graph comparison runs on a **FROZEN ImageNet ViT-S/16** (leakage-free; B0's
  fine-tuned encoder is a separate image-level reference). Encoder held fixed; we vary aggregation/structure.
- **Canonical folds persisted** to `data/splits/cv_<mag>_seed<seed>.json` — every model loads the SAME folds.
- **CAMELYON16** added (small subset) ONLY for the spatial-graph rung — BreakHis has no coordinates.
  See `docs/wsi_protocol.md` for what the spatial graph represents (within-slide edges only).
- B0 audit: no correctness bug forcing a rerun. Fold-0 epoch-0 selection is a legitimate hard-split outcome;
  label-smoothing/augmentation is a DEFERRED separate fine-tuning experiment, not the frozen-encoder comparison.
  Caveat to note when reporting: CV std is population (`ddof=0`), 10-patient val sets make selection/thresholds noisy.

---

## Workflow (agreed)
Claude Code (edits here) → push to GitHub `Eselase-Noble/gpcn-vit` → you `git pull`
in Colab → run → bring results/errors back → decide next experiment together.
Claude does **not** jump ahead or design the whole project unilaterally.

## Milestone ladder & status

| Stage | Description | Status |
|-------|-------------|--------|
| **M1** | Repo skeleton, Colab setup, BreakHis metadata, EDA, patient-level split + leakage check | ✅ **done, verified on real data** |
| **M2** | Evaluation protocol: patient-level stratified k-fold CV + metrics module (ROC-AUC, PR-AUC, balanced acc, sens/spec, F1) | ✅ **done (27 tests)** |
| **B0** | Baseline 0 — independent patch ViT classifier (ViT-S/16, weighted loss, patient-level eval, resumable trainer, embedding cache) | ✅ **CV done (0.9567±0.0498)** |
| **S** | Shared substrate — canonical folds artifact + frozen ViT-S/16 embedding cache (`make_cv_folds.py`, `extract_embeddings.py`) | 🔸 **code done — awaiting Colab run** |
| B1 | Baseline 1 — mean pooling of frozen embeddings → logreg (`mean_pool.py`, `run_baseline1.py`) | 🔸 **code done — awaiting Colab run** |
| B2 | Baseline 2 — attention pooling / MIL | ⬜ next increment |
| B3 | Baseline 3 — spatial KNN graph + GNN (CAMELYON16; `wsi_tiling.py`, `tile_wsi.py`, `download_camelyon16_subset.py` ready) | 🔸 **WSI tiling pipeline code done** |
| B4 | Baseline 4 — feature KNN graph + GNN | ⬜ not started |
| B5 | Baseline 5 — hybrid graph + GNN | ⬜ not started |
| GN | Graph-necessity controls (real / random / edge-shuffled) | ⬜ not started |
| M6 | GPCN-ViT | ⬜ not started |
| — | Strong SOTA baselines → independent datasets → true WSI (CAMELYON) → generalization | ⬜ future |

## Milestone 1 — verified facts (real BreakHis 40X, seed 42)
- 81 patients (40X; full set 82 — one malignant patient has no 40X image; confirmed not a collision)
- 1,995 images; 625 benign / 1,370 malignant (matches official 40X exactly)
- 1 slide/patient; 1–64 images/patient (mean 24.6)
- Class imbalance 2.2:1 malignant:benign (68.7% malignant)
- Patient-disjoint split verified: train 57 / val 13 / test 11 patients

## Open decisions (need the user)
1. ~~Evaluation protocol~~ — DECIDED: single split to prototype, 5-fold patient-level CV for reported results.
2. **Repo visibility:** private (needs token in Colab) vs public (code only; data stays in Drive).

## Known items to handle later (not now)
- Singleton bags: min 1 image/patient → degenerate graphs; need a bag-size policy at B1+.
- Output location: metadata/splits currently written under `GPCN_DATA_ROOT`; consider a
  dedicated processed/ dir to keep artifacts out of the raw dataset tree.
- `--magnification all` run (expect 82 patients, 7,909 images) when we broaden scope.

## Repo
- GitHub: `Eselase-Noble/gpcn-vit` (private) · default branch `master`
- Commits authored solely by the user (no Claude co-author).

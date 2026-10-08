# Research Log

## 2026-10-08 — Baseline 0: patch ViT training stack (code complete)

Spec locked in `docs/baseline0_spec.md` (fixed ViT-S/16 backbone across ladder,
patient-level primary eval, fold-local weighted loss, val-only thresholding).
- `src/models/baselines/patch_vit.py`: timm vit_small_patch16_224 wrapper
  (forward/embed/transforms/param-counts).
- `src/datasets/image_dataset.py`: metadata-slice image Dataset (leakage-safe).
- `src/training/trainer.py`: resumable fine-tuning (checkpoint+RNG restore,
  per-epoch metrics JSON, best-by-val-monitor); §13 Colab recovery.
- `src/embeddings/extractor.py`: resumable ViT embedding cache for later graph rungs.
- `scripts/run_baseline0.py`: config-driven; --smoke / single split / --cv.
- configs/breakhis_40x.yaml extended with model+training sections.
- Verified locally: all modules compile (py_compile), config parses, 35 tests pass.
  Torch/timm not installable locally (py3.14/CPU) → full run happens in Colab.
- AWAITING: Colab run (smoke → single split → 5-fold CV). Results table stays N/A
  until real numbers return.

## 2026-10-08 — Milestone 2: evaluation module

- `src/evaluation/metrics.py`: `compute_metrics` (ROC-AUC, PR-AUC, accuracy,
  balanced accuracy, sensitivity, specificity, precision, F1) + `aggregate_metrics`
  (mean ± std over folds, NaN-safe). Positive class = malignant = 1.
- `src/splits/cross_validation.py`: `make_patient_kfold` — patient-level
  stratified k-fold, optional inner val carve, per-fold + partition leakage checks.
- Protocol decision recorded: prototype on single split, report under 5-fold CV.
- Tests: +12 (27 total passing).
- Next: Baseline 0 — independent patch ViT classifier (Colab GPU).

## 2026-10-08 — Milestone 1: pipeline skeleton

- Created modular repo structure (`src/`, `configs/`, `experiments/`, `docs/`, …).
- Implemented BreakHis **metadata ingestion** (`src/datasets/breakhis.py`):
  filename parser → patient_id = `year-slide`, class, tumor type, magnification.
- Implemented **EDA** (`src/eda/summary.py`) answering the 8 Milestone-1 questions.
- Implemented **patient-level split + leakage verification**
  (`src/splits/patient_split.py`): stratified by class, deterministic by seed,
  raises `LeakageError` on any train/val/test overlap or missing patient.
- Reproducibility utils: `set_seed`, environment capture (`src/utils/`).
- Portable paths via `GPCN_DATA_ROOT` (Colab/Drive-friendly); no hard-coded paths.
- Tests: 15 passing (parsing, metadata counts, EDA, split determinism,
  disjointness, injected-leak detection, JSON round-trip).
- Verified end-to-end CLI on a synthetic BreakHis tree.

### Methodological note
With few patients per class, rounding can yield an empty val/test fold. Added a
loud warning; for the real 82-patient BreakHis the single split is non-degenerate,
but per §8 patient-level cross-validation is the preferred evaluation and should
be added before baseline comparisons.

### Milestone-1 exit questions — ANSWERED on real BreakHis 40X (seed 42)
Run in Colab; env: python 3.13, torch 2.11.0+cpu, commit 43512be.
1. patients — **81** (at 40X; full dataset is 82 — one malignant patient has no
   40X image. Confirmed NOT a collision: slides_per_patient max = 1.)
2. images — **1,995** (matches official BreakHis 40X exactly)
3. benign/malignant — **625 / 1,370** (matches official 40X exactly)
4. images per magnification — 40X: 1,995
5. images per patient — min 1, max 64, mean 24.6, median 23
6. slides per patient — **1** (confirms clean patient IDs)
7. class balance — **imbalanced**, 68.7% malignant; patients 57 malignant / 24 benign
8. splits patient-disjoint — **verified** (train 57 / val 13 / test 11 patients)

### Decisions arising from real data
- **Imbalance (2.2:1):** results tables must lead with PR-AUC, balanced accuracy,
  sensitivity, specificity — not raw accuracy. (Baseline "always malignant" = 68.7% acc.)
- **Small test fold (11 patients):** single-split estimates are high-variance.
  RECOMMEND adopting patient-level stratified k-fold CV (§8) before Baseline 0.
- **Singleton bags:** min 1 image/patient → degenerate graphs for later bag/graph
  rungs; needs a documented bag-size policy when we reach Baseline 1+.
- Milestone 1 COMPLETE. Next rung (pending CV decision): Baseline 0 — patch ViT.

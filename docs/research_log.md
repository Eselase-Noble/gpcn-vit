# Research Log

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

### Milestone-1 exit questions — answered once run on real data
Run `python scripts/run_milestone1.py --data-root <BreakHis> --magnification 40X`.
(Counts below are populated from `results/metrics/eda_summary.json`.)
1. patients — N/A (pending real data)
2. images — N/A
3. benign/malignant — N/A
4. images per magnification — N/A
5. images per patient — N/A
6. slides per patient — N/A (expect 1)
7. class balance — N/A
8. splits patient-disjoint — verified by code (raises on violation)

# GPCN-ViT: Graph Patch Correlation Network with Vision Transformer

A **reproducible computational-pathology research framework**. The scientific
goal is *not* "make GPCN win". It is to determine, step by step, whether explicit
relationships between histopathology patches carry useful diagnostic information,
which relationships matter, and only then whether GPCN exploits them better than
strong baselines.

> **Scope note:** BreakHis is a controlled histopathology-**image** benchmark, not
> a WSI dataset. True WSI experiments come later (CAMELYON16/17).

## Status — Milestone 1 ✅

Dataset pipeline + patient-level evaluation protocol. **No models yet** (that is
intentional — see `docs/experiment_protocol.md`).

Implemented:
- BreakHis metadata ingestion (`src/datasets/breakhis.py`)
- EDA answering the 8 Milestone-1 questions (`src/eda/summary.py`)
- Patient-level train/val/test split with leakage verification (`src/splits/patient_split.py`)
- Reproducibility + portable-path utilities (`src/utils/`)
- Test suite (`tests/`, 15 tests)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the tests
pytest

# run the Milestone-1 pipeline on your BreakHis data
export GPCN_DATA_ROOT=/path/to/BreaKHis_v1      # or pass --data-root
python scripts/run_milestone1.py --magnification 40X --seed 42
```

Outputs:
- `data/metadata/breakhis_metadata.csv` — one row per image
- `results/metrics/eda_summary.json` — EDA + environment snapshot
- `data/splits/breakhis_40x_seed42.json` — patient-disjoint split

## Google Colab

```python
import os
os.environ["GPCN_DATA_ROOT"] = "/content/data/BreaKHis_v1"
# or Google Drive:
# os.environ["GPCN_DATA_ROOT"] = "/content/drive/MyDrive/gpcn-research/data/BreaKHis_v1"
!python scripts/run_milestone1.py --magnification 40X
```

Paths are resolved from `GPCN_DATA_ROOT` / `GPCN_RESULTS_ROOT` — no machine paths
are hard-coded, and the device is auto-detected (`cuda` if available).

## The one rule that cannot be broken

**Patient-level separation.** A patient never appears in more than one split.
`src/splits/patient_split.py` enforces this and raises `LeakageError` on any
violation. See `docs/dataset_protocol.md` and `docs/experiment_protocol.md`.

## Repository layout

```
configs/      YAML experiment configs (breakhis_40x.yaml)
src/          datasets, preprocessing, embeddings, graphs, models, training,
              evaluation, visualization, utils, eda, splits
scripts/      runnable entry points (run_milestone1.py)
experiments/  per-rung experiment dirs (00_patch_baseline … 06_gpcn)
tests/        unit tests
docs/         dataset_protocol, experiment_protocol, research_log
results/      metrics, figures, checkpoints, logs, embeddings
```

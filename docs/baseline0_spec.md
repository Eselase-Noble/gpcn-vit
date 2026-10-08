# Baseline 0 — Independent Patch/Image ViT Classifier (locked spec)

Decided 2026-10-08. This spec is binding; changing it changes the scientific
interpretation of the whole ladder (§9, §24) — if a change is needed, STOP and
reconcile with `docs/MASTER_INSTRUCTIONS.md`.

## Research question it answers
"Does explicit relational modelling improve diagnosis **when the underlying
visual representation is held constant**?" Baseline 0 is that fixed visual
representation — the floor the pooling/graph/GPCN rungs must beat.

## Backbone (FIXED across the entire ladder)
- timm `vit_small_patch16_224`, **ImageNet pretrained**, **fine-tuned** on BreakHis.
- Input resolution **224×224**; preprocessing from timm's pretrained cfg (mean/std).
- **Do not freeze** the backbone for the primary experiment (unless documented).
- Same backbone init/protocol reused for Patch ViT, Mean Pool, Attention Pool,
  Spatial/Feature/Hybrid GNN, and GPCN where architecturally applicable.
- **No pathology-pretrained foundation models** in the primary graph-necessity
  experiment (confounds the question). They are a separate later SOTA/robustness study.
- Record: exact timm model name, pretrained weight cfg, torch/timm versions,
  input preprocessing, total + trainable parameter counts.

## Embedding caching
- Cache per-image ViT features (resumable, §13) so graph rungs don't recompute.
- Store under `results/embeddings/` keyed by (backbone, fold, image).

## Training
- Loss: **class-weighted** cross-entropy / weighted BCE.
- Class weights computed **per CV fold from TRAINING images only** — never val/test,
  never global (that would leak held-out folds). Record class counts + weights per fold.
- **No oversampling / duplication** for the primary experiment.
- Validation/test kept at **natural class distribution**.
- Checkpoint regularly; resumable from checkpoint (Colab disconnection).

## Evaluation (PATIENT-LEVEL is primary)
- Train at image level; report **both** image-level and patient-level metrics.
- Patient-level is the primary evidence for comparing the ladder.
- Aggregation rule (fixed): `patient_prob = mean(image_probs)` over that patient's images.
- Metrics (both levels): ROC-AUC, PR-AUC, sensitivity, specificity, accuracy,
  balanced accuracy, precision, F1.
- **ROC-AUC and PR-AUC are the threshold-independent primary metrics.**
- Threshold for threshold-dependent metrics is chosen on **train/val**, NEVER the test fold.
- **Persist per-image AND per-patient predictions** so alternative aggregation
  strategies can be evaluated later without retraining.

## Protocol
- Prototype/debug on the single 57/13/11 split; report under patient-level 5-fold CV
  (mean ± std). Primary metrics PR-AUC + balanced metrics (imbalance 2.2:1).

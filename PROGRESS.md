# PROGRESS — where we are right now

> Living status file. Update at the end of every working session. The scientific
> protocol lives in `docs/MASTER_INSTRUCTIONS.md` (original prompt, verbatim) and
> `docs/experiment_protocol.md`. The running history lives in `docs/research_log.md`.

**Last updated:** 2026-10-08
**Current milestone:** Baseline 0 — patch ViT. AUDITED; selection/threshold/provenance fixes applied ✅ (36 tests).
**Next action:** YOU re-run CLEAN in Colab — prototype single split (fresh run dir), then `--cv`. Bring metrics back.
  First run numbers were INVALID (epoch-0 selection + non-traceable dir); discard them.

**Eval protocol DECIDED (2026-10-08):** prototype/debug on the single 57/13/11 split; report final
results under **patient-level stratified 5-fold CV** (mean ± std). Applies to every model on the ladder.

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
| **B0** | Baseline 0 — independent patch ViT classifier (ViT-S/16, weighted loss, patient-level eval, resumable trainer, embedding cache) | 🔸 **code done — awaiting Colab run** |
| B1 | Baseline 1 — mean pooling over patient bag | ⬜ not started |
| B2 | Baseline 2 — attention pooling | ⬜ not started |
| B3 | Baseline 3 — spatial KNN graph + GNN | ⬜ not started |
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

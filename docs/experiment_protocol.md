# Experiment Protocol

## Research progression (do not skip ahead)

```
Are patches independent?  →  Do patch relationships carry useful info?
→  Which relationships (spatial / feature / hybrid)?  →  Do graph models beat
non-graph?  →  Does GPCN exploit relationships better?  →  SOTA  →  generalize
→  only then adaptive/swarm extensions.
```

The first milestone is **not GPCN**. It is a correct, reproducible dataset
pipeline and patient-level evaluation protocol.

## Experimental ladder (models)

| # | Model          | Graph             | Status |
|---|----------------|-------------------|--------|
| 0 | Patch ViT      | No                | N/A    |
| 1 | Mean Pool      | No                | N/A    |
| 2 | Attention Pool | Implicit          | N/A    |
| 3 | Spatial GNN    | Spatial           | N/A    |
| 4 | Feature GNN    | Feature           | N/A    |
| 5 | Hybrid GNN     | Spatial + Feature | N/A    |
| 6 | GPCN-ViT       | Spatial + Feature | N/A    |

## Graph-necessity controls

Real vs Random vs Edge-shuffled graph (matched nodes/edges/degree), plus
Spatial / Feature / Hybrid. Goal: isolate whether performance depends on
*meaningful* relationships rather than merely adding a GNN.

## Evaluation metrics

Primary: ROC-AUC, PR-AUC. Also: accuracy, balanced accuracy, sensitivity/recall,
specificity, precision, F1. Later: ECE, Brier, reliability diagrams. Record
cost: params, trainable params, GPU memory, train/inference time, #nodes, #edges.

## Fair-comparison rule

Hold constant across models: dataset, patient split, preprocessing, resolution,
augmentation, backbone, training budget, optimizer, LR schedule, epochs,
evaluation protocol. Document any unavoidable difference. Never tune a baseline
down to flatter GPCN.

## Statistical rigor

Patient-level cross-validation and/or multiple seeds; report mean ± std and
confidence intervals. No superiority claims from a single lucky run.

## Splitting rule (enforced in code)

Patient-level only. `src/splits/patient_split.py` asserts train/val/test are
pairwise disjoint and cover all patients; a violation raises `LeakageError`.

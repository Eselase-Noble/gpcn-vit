# GPCN Research Project — Master Instructions (original prompt, verbatim)

> This is the first-ever prompt given for this project. Preserved unchanged as the
> authoritative statement of scope, scientific protocol, and working rules. If any
> later decision appears to conflict with this document, STOP and reconcile.

---

You are assisting with a research project called:

**GPCN-ViT: Graph Patch Correlation Network with Vision Transformer for Histopathology Image/WSI Diagnosis**

The project is being developed as a rigorous academic research project. Your role is to assist with software engineering, experiment implementation, reproducibility, debugging, data processing, evaluation, and research infrastructure.

## 1. IMPORTANT SCIENTIFIC CONTEXT

The central research hypothesis is NOT initially:

> "GPCN is better than existing models."

The first scientific question we need to establish is:

> **Do explicit relationships between histopathology patches provide diagnostically useful information beyond individual patch representations and conventional patch aggregation?**

The project must therefore first establish that modelling relationships between patches is scientifically justified.

Only after that has been demonstrated should we establish whether the proposed GPCN architecture is a better way of exploiting those relationships.

The research progression is:

```text
Histopathology problem
        ↓
Are patches sufficiently independent?
        ↓
Do relationships between patches contain useful information?
        ↓
What relationships are useful?
        ↓
Spatial relationships
Feature/morphological relationships
Hybrid spatial + feature relationships
        ↓
Do graph models outperform non-graph models?
        ↓
Can GPCN model those relationships better?
        ↓
Does GPCN outperform strong baselines/SOTA?
        ↓
Does the finding generalize to independent datasets?
        ↓
Only then investigate adaptive graph construction,
swarm intelligence, etc.
```

DO NOT introduce swarm intelligence, unnecessary modules, or additional complexity before the core graph hypothesis has been experimentally established.

---

# 2. DATASETS

The first dataset will be:

## BreakHis

BreakHis is the initial controlled benchmark.

Important characteristics:

* Breast histopathology dataset
* 7,909 microscopic images
* 82 patients
* 4 magnifications: 40X, 100X, 200X, 400X
* Binary classification: benign / malignant

The first major experiment should focus on: **BreakHis 40X**

The project may later expand to:

1. BreakHis all magnifications
2. BACH / ICIAR 2018
3. CAMELYON16
4. CAMELYON17
5. potentially other public datasets if scientifically justified

Important: **BreakHis is not a true WSI dataset.** Do not describe BreakHis
experiments as WSI experiments. It is being used initially to test the
fundamental relational hypothesis in controlled histopathology image data.
CAMELYON16/17 will later be used to investigate the hypothesis on actual
whole-slide images.

---

# 3. CRITICAL DATA LEAKAGE RULE

Patient-level separation is mandatory. Never randomly split individual images if
images from the same patient can occur in multiple splits.

```text
Patients
   ├── Training patients
   ├── Validation patients
   └── Test patients
```

A patient must NEVER appear in more than one split. The software must include
automated assertions/tests verifying this.

```python
assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)
```

If a proposed implementation violates patient-level separation, stop and report
the problem rather than proceeding.

---

# 4. FIRST SCIENTIFIC EXPERIMENT

The first major experiment is NOT GPCN. We need an incremental experimental ladder.

- **Baseline 0 — Independent patch/image classifier:** Image/Patch → ViT → Classifier → Prediction.
- **Baseline 1 — Mean pooling:** bag of patch embeddings → mean pooling → classifier.
- **Baseline 2 — Attention pooling:** patch embeddings → attention pooling → bag representation → classifier (important control; attention can model some relationships implicitly).
- **Baseline 3 — Spatial graph:** patch embeddings + spatial coordinates → spatial KNN graph → GNN → classifier.
- **Baseline 4 — Feature/morphological graph:** patch embeddings → feature similarity → feature KNN graph → GNN → classifier.
- **Baseline 5 — Hybrid graph:** spatial + feature relationships → hybrid graph → GNN → classifier.
- **Model 6 — GPCN-ViT:** only after the above baselines work. ViT → patch embeddings → spatial+feature graph → GPCN graph reasoning → multi-scale graph → message passing → slide/bag representation → classifier. GPCN augments ViT rather than simply replacing ViT attention.

---

# 5. GRAPH NECESSITY EXPERIMENTS

Experiments specifically designed to determine whether the graph itself is meaningful:

- **A. Real graph:** X + real adjacency A → GNN → prediction
- **B. Random graph:** X + random A (comparable graph characteristics) → GNN → prediction
- **C. Edge-shuffled graph:** preserve ~#nodes, ~#edges, ~degree while disrupting topology
- **D. Spatial graph:** A_spatial
- **E. Feature graph:** A_feature
- **F. Hybrid graph:** A_spatial + A_feature

Objective: determine whether performance depends on meaningful patch relationships
rather than merely adding a graph neural network.

---

# 6. REQUIRED EXPERIMENT TABLE

| Model          | Graph             | AUC | PR-AUC | Sensitivity | Specificity | Accuracy | F1 |
| -------------- | ----------------- | --: | -----: | ----------: | ----------: | -------: | -: |
| Patch ViT      | No                |     |        |             |             |          |    |
| Mean Pool      | No                |     |        |             |             |          |    |
| Attention Pool | Implicit          |     |        |             |             |          |    |
| Spatial GNN    | Spatial           |     |        |             |             |          |    |
| Feature GNN    | Feature           |     |        |             |             |          |    |
| Hybrid GNN     | Spatial + Feature |     |        |             |             |          |    |
| GPCN-ViT       | Spatial + Feature |     |        |             |             |          |    |

| Graph configuration | AUC | PR-AUC |
| ------------------- | --: | -----: |
| Real graph          |     |        |
| Random graph        |     |        |
| Edge-shuffled graph |     |        |
| Spatial graph       |     |        |
| Feature graph       |     |        |
| Hybrid graph        |     |        |

Do not fabricate results. Missing results must remain `N/A`.

---

# 7. EVALUATION METRICS

Primary: ROC-AUC, PR-AUC. Also report: accuracy, balanced accuracy,
sensitivity/recall, specificity, precision, F1. Later (uncertainty/calibration):
Expected Calibration Error (ECE), Brier score, reliability diagrams. Also record:
#parameters, trainable parameters, GPU memory, training time, inference time,
#nodes, #graph edges.

---

# 8. STATISTICAL RIGOR

Do not rely on one lucky random seed. Where feasible: patient-level
cross-validation, multiple random seeds, mean ± standard deviation, confidence
intervals. The evaluation framework should eventually support statistical
comparisons between models. Do not claim superiority from a single run with a
slightly higher metric.

---

# 9. FAIR COMPARISON RULE

Keep consistent whenever scientifically possible: dataset, patient split,
preprocessing, image resolution, augmentation, backbone, training budget,
optimizer, learning-rate schedule, number of epochs, evaluation protocol. If two
models require different settings, document the difference explicitly. Never
manipulate the baseline to make GPCN look better. The goal is a scientifically
defensible result, not merely a high score.

---

# 10. PROJECT STRUCTURE

Modular Python research repository (configs/, data/, src/{datasets,preprocessing,
embeddings,graphs,models/{baselines,gnn,gpcn},training,evaluation,visualization},
experiments/, notebooks/, results/, docs/). See repository README for the realized layout.

---

# 11. GOOGLE COLAB REQUIREMENT

Primarily runs on Google Colab with GPU. Do not assume a persistent local GPU
server, fixed filesystem, fixed CUDA, a permanently running process, unlimited
RAM/disk. Detect the device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Optimize experiments for Colab GPU execution.

---

# 12. COLAB FILESYSTEM

Configurable data root:

```python
DATA_ROOT = os.environ.get("GPCN_DATA_ROOT", "/content/data")
```

Optionally Google Drive: `/content/drive/MyDrive/gpcn-research/`. Do not hard-code
personal machine paths (e.g. `/home/noble/...`, `/kaggle/...`). The project must
remain portable.

---

# 13. COLAB SESSION DISCONNECTION

Sessions can terminate. Therefore: cache expensive embeddings, save checkpoints
regularly, save experiment configurations, save metrics after every epoch, make
experiments resumable, avoid recomputing expensive preprocessing, make training
restartable from checkpoints. If embedding extraction takes hours, it must be
resumable.

---

# 14. DO NOT PUT EVERYTHING IN NOTEBOOKS

Notebooks: exploration, visualization, EDA, result analysis. Core functionality
belongs in Python modules.

---

# 15. CONFIGURATION-DRIVEN EXPERIMENTS

Experiments configurable through YAML files. Changing an experiment should
preferably require changing configuration rather than source code.

---

# 16. EXPERIMENT TRACKING

Every experiment must save: experiment_id, dataset, split, seed, model, backbone,
hyperparameters, number_of_nodes, number_of_edges, training_time, best_epoch,
metrics, checkpoint_path, git_commit. Use TensorBoard and/or W&B if practical.
The project should remain usable without W&B if no credentials are provided.

---

# 17. REPRODUCIBILITY

Every experiment must record: random seed, Python version, PyTorch version, CUDA
version, GPU name, configuration, dataset version/path, git commit. Implement
reproducibility utilities (`set_seed(42)`). Recognize that exact GPU determinism
may cost performance and may not always be guaranteed.

---

# 18. CLAUDE CODE DEVELOPMENT RULE

Do NOT implement the entire project in one step. Work incrementally. Before a
major component: (1) explain what it does, (2) which research question it
supports, (3) inputs/outputs, (4) implement, (5) write tests, (6) run tests,
(7) report pass/fail, (8) only then move on. Never silently change the research
protocol. If a design decision could change the scientific interpretation, STOP
and ask for confirmation.

---

# 19. FIRST DEVELOPMENT TASK

Do NOT implement GPCN yet. First: project skeleton + BreakHis data pipeline.

```text
Milestone 1: Project structure → Colab setup → BreakHis metadata ingestion →
Patient/slide/magnification extraction → EDA → Patient-level train/val/test split
→ Leakage verification
```

At the end of Milestone 1, we should answer: (1) how many patients, (2) how many
images, (3) how many benign/malignant, (4) how many images per magnification,
(5) images per patient, (6) slides per patient, (7) classes balanced?,
(8) are splits patient-disjoint? Do not proceed to model training until these are
answered and the split is verified.

---

# 20. IMPORTANT RESEARCH PRINCIPLE

Never optimize for novelty before establishing validity. Priority: Validity →
Reproducibility → Strong baselines → Scientific evidence → GPCN → SOTA comparison
→ Generalization → Novel extensions. Do not add swarm intelligence, genetic
algorithms, unnecessary attention modules, arbitrary graph mechanisms, extra
losses, or complicated augmentation just to increase accuracy. Every component
must have a scientific justification and an ablation.

---

# 21. CURRENT RESEARCH POSITION

The existing GPCN concept includes: ViT representations, spatial relationships,
feature relationships, hybrid spatial+feature KNN, graph message passing,
multi-scale graph reasoning, uncertainty estimation. These are HYPOTHESES/
components to be experimentally validated, not established facts. Current reported
GPCN results are preliminary and must be reproduced under the new rigorous
protocol before being treated as final. Do not assume previous accuracy/AUC
values are automatically comparable to new experiments.

---

# 22. MOST IMPORTANT RULE

The goal is NOT "make GPCN achieve the highest possible accuracy." The goal is:

> **Determine scientifically whether explicit relational modelling between
> histopathology patches provides useful diagnostic information, determine what
> relationships are meaningful, and determine whether GPCN provides a superior
> mechanism for exploiting those relationships.**

If experiments show graphs do NOT provide meaningful additional information, that
is a valid research finding and we investigate why. Do not manipulate experiments
to force GPCN to win.

---

# 23. WHEN YOU WRITE CODE

Python 3, PyTorch, clean modular architecture, type hints where useful,
docstrings, meaningful names, unit tests, error handling, logging, configuration
files, no hard-coded personal paths, Colab-compatible, GPU-aware,
memory-efficient data loading, checkpointing, resumable preprocessing,
deterministic seeds where practical. Prefer simple, testable implementations over
unnecessary abstraction.

---

# 24. HOW TO RESPOND

When asked to implement something: (1) briefly explain the proposed
implementation, (2) identify scientific implications, (3) implement only the
requested scope, (4) run tests, (5) show important output, (6) mention
assumptions, (7) do not silently implement future phases. If you discover a
potential methodological flaw, tell me immediately. If a requested implementation
could introduce data leakage or invalidate the scientific comparison, stop and
explain before proceeding.

---

# FINAL OBJECTIVE

A reproducible computational pathology research framework progressing:
BreakHis → patient-level controlled experiments → non-graph baselines → graph
necessity experiments → spatial graph → feature graph → hybrid graph → GPCN-ViT →
strong contemporary baselines → independent datasets → true WSI experiments →
generalization → only then adaptive/swarm extensions.

The first milestone is **NOT GPCN**. The first milestone is proving that the
dataset pipeline, patient-level evaluation protocol, and baseline framework are
correct and reproducible.

**Additional working agreement (added after the original prompt):** The workflow is
You → Claude Code → implementation → run in Colab → bring results/errors back →
decide the next experiment together. Claude does not independently design the
whole project ahead or jump to GPCN.

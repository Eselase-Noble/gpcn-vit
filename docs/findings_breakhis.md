# Findings — BreakHis 40X relational hypothesis (as of 2026-10-15)

**Question.** Do explicit relationships among a patient's patches add diagnostic
value beyond strong non-graph aggregation?

**Protocol.** Patient-level stratified 5-fold CV, 81 patients. One FROZEN
ImageNet ViT-S/16 encoder shared by B1/B2/graphs (leakage-free; B0 is a separate
fine-tuned reference). Selection + thresholds on train/val only. Paired
comparison pools each model's held-out test predictions into one row per patient.

## Results (patient-level ROC-AUC)

| Model | Edges | Per-fold mean ± std | Pooled (81 pts) |
|---|---|---|---|
| B0 ViT-S/16 fine-tuned | — | 0.957 ± 0.050 | 0.929 |
| B1 mean-pool (frozen) | — | 0.931 ± 0.050 | 0.918 |
| B2 attention-MIL (frozen) | — | 0.938 ± 0.054 | 0.917 |
| Graph: feature kNN | similarity | 0.929 ± 0.063 | 0.877 |
| Graph: none | no edges | 0.933 ± 0.058 | 0.872 |
| Graph: rewired | degree-preserving random | 0.910 ± 0.086 | 0.856 |

Paired bootstrap vs the feature graph: **none** Δ−0.005, CI [−0.044,+0.031], p=0.79;
**rewired** Δ−0.021, CI [−0.065,+0.018], p=0.32; **B1** Δ+0.041, p=0.32.

## Conclusion

**No evidence that graph structure helps on BreakHis.**
- feature ≈ none (p=0.79, tight CI around 0): message passing is inert.
- feature vs rewired not significant: real edges do not reliably beat random.
- Non-graph methods are directionally higher; learned attention (B2) ≈ flat
  mean (B1). Graphs added cross-fold variance without signal.

**Why this is expected.** BreakHis images are individual microscopy fields —
redundant views of one tumor with NO spatial coordinates. There is no tissue
geometry for a graph to exploit; the only constructible structure is feature
similarity, which carries the same information the pooling already uses.

**Limits.** 81 patients / 5 folds is low power — we cannot claim graphs are
significantly *worse*, only that no benefit is detectable and point estimates
favour the simpler models.

## Implication

The spatial-neighbour hypothesis is **untestable on BreakHis** and must move to a
true WSI dataset (CAMELYON16), where tiles have genuine (x, y) coordinates and
within-slide adjacency encodes real tissue geometry. See `docs/wsi_protocol.md`.

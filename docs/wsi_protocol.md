# WSI protocol — CAMELYON16 (spatial-graph substrate)

**Decided 2026-10-15.** BreakHis consists of individual microscopy images with
**no spatial coordinates**, so it structurally *cannot* test the tissue-scale
spatial-neighbour hypothesis (the protocol forbids inventing adjacency between
separate BreakHis images). CAMELYON16 (whole-slide images) is added **only** to
provide genuine spatial structure for the spatial-graph rung (B3). It does not
replace BreakHis; the non-graph baselines and morphology/feature graphs still run
on BreakHis.

## Dataset

- **Source:** AWS Open Data mirror — `s3://camelyon-dataset` (region `us-west-2`,
  no credentials). Registry: https://registry.opendata.aws/camelyon/
- **Full size ~700 GB** → we deliberately pull only a **small prototype subset**
  (a few tumor + normal training slides + lesion annotations) via
  `scripts/download_camelyon16_subset.py`. Scale up only once the pipeline is
  proven and storage is arranged.
- **Task:** breast lymph-node metastasis (binary), slide-level label from the
  filename (`tumor_*`=1, `normal_*`=0); test slides resolve from `reference.csv`.

## Tiling (`scripts/tile_wsi.py`, geometry in `src/datasets/wsi_tiling.py`)

1. Thumbnail → HSV-saturation Otsu **tissue mask** (glass background is bright and
   low-saturation; stained tissue is saturated).
2. **Tile grid** at a chosen openslide level (default level 1 ≈ 20× on a 40× slide),
   256 px, non-overlapping. Only fully-contained tiles are kept.
3. Keep tiles with ≥ `min_tissue` (default 0.25) tissue fraction.
4. Each kept tile is saved with its **grid index (row, col)** and **level-0 pixel
   (x, y)** in `tiles.csv`. These coordinates are the whole point — they are what
   makes a spatial graph possible.

## What the spatial graph represents (binds at B3)

- **Nodes:** tissue tiles from a single slide.
- **Edges (spatial):** connect tiles whose grid cells are adjacent **within the
  same slide** (e.g. 4-/8-connectivity, or k-NN on (x, y)). **Never** connect
  tiles from different slides — there is no meaningful cross-slide adjacency.
- **Unit of analysis:** slide/patient, consistent with the BreakHis protocol
  (patients/slides are the evaluation unit, never tiles).
- **Encoder:** the same frozen ImageNet ViT-S/16 used for the BreakHis
  controlled comparison, so representation is held fixed while we vary structure.

## Controls (relationship-necessity, binds at GN)

Any claimed spatial-graph gain must be tested against structure-preserving
controls on the *same* tiles/features: randomized/rewired edges that preserve
degree/density, and a no-message-passing variant. A higher mean AUC alone is not
evidence that *meaningful connectivity* matters.

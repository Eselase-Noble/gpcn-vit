# Dataset Protocol — BreakHis

## Scope and framing (important)

BreakHis is used as an **initial controlled histopathology-image benchmark**, not
as a whole-slide-imaging (WSI) dataset. It lets us test the fundamental
relational hypothesis in controlled data before moving to true WSI datasets
(CAMELYON16/17). **Do not describe BreakHis experiments as WSI experiments.**

## Dataset facts (official)

- 7,909 microscopy images, 82 patients
- Binary label: benign / malignant
- 4 magnifications: 40X, 100X, 200X, 400X
- First major experiment: **BreakHis 40X**

## Filename convention

```
<PROCEDURE>_<CLASS>_<TYPE>-<YEAR>-<SLIDE>-<MAG>-<SEQ>.png
SOB_B_A-14-22549AB-40-001.png
```

| Field     | Example   | Meaning                                  |
|-----------|-----------|------------------------------------------|
| PROCEDURE | SOB       | Surgical Open Biopsy                     |
| CLASS     | B / M     | benign / malignant                       |
| TYPE      | A, DC, …  | tumor type (see `src/datasets/breakhis.py`) |
| YEAR      | 14        | slide acquisition year                   |
| SLIDE     | 22549AB   | slide identifier                         |
| MAG       | 40        | magnification                            |
| SEQ       | 001       | image sequence number                    |

## Patient identifier (leakage unit)

```
patient_id = f"{YEAR}-{SLIDE}"   # e.g. "14-22549AB"
```

In BreakHis each patient corresponds to exactly one slide (verified in EDA:
`slides_per_patient` max should be 1). **This identifier is the unit that must
never cross a split boundary.**

## Expected directory layout (parsing is filename-only, so layout is flexible)

```
BreaKHis_v1/histology_slides/breast/
  benign/SOB/<type>/SOB_B_<t>_<year>-<slide>/<mag>X/*.png
  malignant/SOB/<type>/SOB_M_<t>_<year>-<slide>/<mag>X/*.png
```

## How to obtain the data

Download BreakHis (BreaKHis_v1) from the official source and place/extract it so
that `GPCN_DATA_ROOT` (or `--data-root`) points at a directory containing the
`.png` images (searched recursively).

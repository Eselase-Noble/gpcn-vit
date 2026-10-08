"""BreakHis metadata ingestion.

Parses the official BreakHis image-naming convention into a structured table.
No pixels are read here — this module only builds metadata from file paths,
which is cheap, deterministic, and the foundation for leakage-free splitting.

Official filename format
------------------------
``<PROCEDURE>_<CLASS>_<TYPE>-<YEAR>-<SLIDE>-<MAG>-<SEQ>.png``

Example: ``SOB_B_A-14-22549AB-40-001.png``
    PROCEDURE = SOB   (Surgical Open Biopsy)
    CLASS     = B      -> benign  (M -> malignant)
    TYPE      = A      -> adenosis (see TUMOR_TYPES)
    YEAR      = 14
    SLIDE     = 22549AB
    MAG       = 40     -> 40X
    SEQ       = 001

Patient identifier
------------------
BreakHis contains 82 patients; each patient corresponds to exactly one slide.
We therefore define::

    patient_id = f"{YEAR}-{SLIDE}"      # e.g. "14-22549AB"

This identifier is the unit that MUST NOT cross train/val/test boundaries
(see ``src.splits.patient_split``). If this definition is wrong, every
downstream evaluation is compromised, so it is isolated and unit-tested here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

CLASS_MAP = {"B": "benign", "M": "malignant"}

TUMOR_TYPES = {
    # benign
    "A": "adenosis",
    "F": "fibroadenoma",
    "PT": "phyllodes_tumor",
    "TA": "tubular_adenoma",
    # malignant
    "DC": "ductal_carcinoma",
    "LC": "lobular_carcinoma",
    "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}

MAGNIFICATIONS = {"40": "40X", "100": "100X", "200": "200X", "400": "400X"}

# Groups: procedure, tumor_class, tumor_type, year, slide_id, mag, seq
_FILENAME_RE = re.compile(
    r"^(?P<procedure>[A-Z]+)_"
    r"(?P<tumor_class>[BM])_"
    r"(?P<tumor_type>[A-Z]+)-"
    r"(?P<year>\d+)-"
    r"(?P<slide_id>[A-Za-z0-9]+)-"
    r"(?P<mag>\d+)-"
    r"(?P<seq>\d+)\.png$"
)

METADATA_COLUMNS = [
    "filename",
    "filepath",
    "patient_id",
    "procedure",
    "tumor_class",      # B / M
    "label",            # benign / malignant
    "label_binary",     # 0 benign, 1 malignant
    "tumor_type_code",
    "tumor_type",
    "year",
    "slide_id",
    "magnification",    # 40X / 100X / ...
    "mag_value",        # 40 / 100 / ... (int)
    "seq",
]


@dataclass(frozen=True)
class BreakHisImage:
    """Parsed metadata for a single BreakHis image filename."""

    filename: str
    filepath: str
    patient_id: str
    procedure: str
    tumor_class: str
    label: str
    label_binary: int
    tumor_type_code: str
    tumor_type: str
    year: str
    slide_id: str
    magnification: str
    mag_value: int
    seq: str


def parse_filename(filename: str, filepath: Optional[str] = None) -> Optional[BreakHisImage]:
    """Parse one BreakHis filename into :class:`BreakHisImage`.

    Returns ``None`` (and logs a warning) if the name does not match the
    official convention, so a stray file never silently corrupts the table.
    """
    name = Path(filename).name
    m = _FILENAME_RE.match(name)
    if m is None:
        logger.warning("Filename does not match BreakHis convention: %s", name)
        return None

    g = m.groupdict()
    tumor_class = g["tumor_class"]
    mag = g["mag"]
    return BreakHisImage(
        filename=name,
        filepath=str(filepath) if filepath is not None else name,
        patient_id=f"{g['year']}-{g['slide_id']}",
        procedure=g["procedure"],
        tumor_class=tumor_class,
        label=CLASS_MAP[tumor_class],
        label_binary=0 if tumor_class == "B" else 1,
        tumor_type_code=g["tumor_type"],
        tumor_type=TUMOR_TYPES.get(g["tumor_type"], "unknown"),
        year=g["year"],
        slide_id=g["slide_id"],
        magnification=MAGNIFICATIONS.get(mag, f"{mag}X"),
        mag_value=int(mag),
        seq=g["seq"],
    )


def build_metadata(
    data_root: Path,
    magnification: Optional[str] = None,
    pattern: str = "*.png",
) -> pd.DataFrame:
    """Scan ``data_root`` recursively and build a metadata DataFrame.

    Parameters
    ----------
    data_root:
        Directory containing BreakHis images (searched recursively). The
        official layout nests images under ``benign/malignant`` and
        magnification sub-folders, but parsing relies only on filenames, so any
        layout works.
    magnification:
        Optional filter, e.g. ``"40X"``. ``None`` keeps all magnifications.
    pattern:
        Glob for candidate image files.

    Returns
    -------
    pandas.DataFrame with columns :data:`METADATA_COLUMNS`, sorted for
    determinism.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {data_root}. "
            "Set GPCN_DATA_ROOT or pass an explicit path (see README)."
        )

    records: List[dict] = []
    skipped = 0
    for fp in sorted(data_root.rglob(pattern)):
        parsed = parse_filename(fp.name, filepath=str(fp))
        if parsed is None:
            skipped += 1
            continue
        records.append(asdict(parsed))

    df = pd.DataFrame(records, columns=METADATA_COLUMNS)
    if magnification is not None and not df.empty:
        df = df[df["magnification"] == magnification].reset_index(drop=True)

    logger.info(
        "Built metadata: %d images, %d patients, %d skipped (non-matching).",
        len(df),
        df["patient_id"].nunique() if not df.empty else 0,
        skipped,
    )
    return df

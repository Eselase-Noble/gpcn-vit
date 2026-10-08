"""Exploratory data analysis for BreakHis metadata.

Answers the eight Milestone-1 exit questions (§19) from a metadata DataFrame:
    1. patients  2. images  3. benign/malignant counts  4. images per
    magnification  5. images per patient  6. slides per patient  7. class
    balance  8. (split disjointness is checked in src.splits.patient_split)
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute the Milestone-1 EDA summary as a plain dict (JSON-serialisable)."""
    if df.empty:
        return {"error": "empty metadata — no images parsed"}

    n_images = len(df)
    n_patients = df["patient_id"].nunique()
    class_counts = df["label"].value_counts().to_dict()

    images_per_patient = df.groupby("patient_id").size()
    # In BreakHis one patient == one slide; we verify rather than assume.
    slides_per_patient = df.groupby("patient_id")["slide_id"].nunique()

    benign = int(class_counts.get("benign", 0))
    malignant = int(class_counts.get("malignant", 0))

    summary: Dict[str, Any] = {
        "n_patients": int(n_patients),
        "n_images": int(n_images),
        "class_counts": {"benign": benign, "malignant": malignant},
        "class_balance_ratio": round(malignant / n_images, 4) if n_images else None,
        "images_per_magnification": (
            df.groupby("magnification").size().sort_index().to_dict()
        ),
        "images_per_patient": {
            "min": int(images_per_patient.min()),
            "max": int(images_per_patient.max()),
            "mean": round(float(images_per_patient.mean()), 2),
            "median": float(images_per_patient.median()),
        },
        "slides_per_patient": {
            "min": int(slides_per_patient.min()),
            "max": int(slides_per_patient.max()),
            "mean": round(float(slides_per_patient.mean()), 2),
        },
        "tumor_type_counts": df["tumor_type"].value_counts().to_dict(),
        "patients_per_class": (
            df.drop_duplicates("patient_id")["label"].value_counts().to_dict()
        ),
    }
    return summary


def format_summary(summary: Dict[str, Any]) -> str:
    """Render the summary as a human-readable report mapped to the 8 questions."""
    if "error" in summary:
        return f"EDA ERROR: {summary['error']}"

    cc = summary["class_counts"]
    ipp = summary["images_per_patient"]
    spp = summary["slides_per_patient"]
    lines = [
        "=" * 60,
        "BreakHis EDA Summary (Milestone 1)",
        "=" * 60,
        f"1. Patients available ............ {summary['n_patients']}",
        f"2. Images available ............. {summary['n_images']}",
        f"3. Benign / Malignant ........... {cc['benign']} / {cc['malignant']}",
        "4. Images per magnification:",
    ]
    for mag, n in summary["images_per_magnification"].items():
        lines.append(f"     {mag:>6} : {n}")
    lines += [
        f"5. Images per patient ........... min={ipp['min']} "
        f"max={ipp['max']} mean={ipp['mean']} median={ipp['median']}",
        f"6. Slides per patient ........... min={spp['min']} "
        f"max={spp['max']} mean={spp['mean']}  (expected 1 for BreakHis)",
        f"7. Class balance (malignant frac) {summary['class_balance_ratio']}",
        f"   Patients per class ........... {summary['patients_per_class']}",
        "   Tumor-type counts:",
    ]
    for t, n in summary["tumor_type_counts"].items():
        lines.append(f"     {t:>18} : {n}")
    lines.append("=" * 60)
    return "\n".join(lines)

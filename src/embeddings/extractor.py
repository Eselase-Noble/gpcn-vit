"""Resumable ViT embedding extraction + cache.

Later graph rungs (spatial/feature/hybrid, GPCN) consume per-image ViT features
instead of recomputing them each time (§13). Embeddings depend on the fine-tuned
weights, so a cache is keyed by a caller-supplied ``tag`` (e.g. backbone+fold).

The cache is resumable: a manifest records which filenames are already embedded,
so a dropped Colab session re-runs only the remainder.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_embeddings(
    model,
    df: pd.DataFrame,
    transform: Callable,
    device: torch.device,
    out_dir: Path,
    tag: str,
    batch_size: int = 64,
    resume: bool = True,
) -> Path:
    """Embed every image in ``df`` and cache to ``out_dir/<tag>.npz``.

    Saves arrays ``embeddings`` (N, D), ``filenames`` (N,), ``patient_ids`` (N,),
    ``labels`` (N,). Writes a ``<tag>.manifest.json`` for resumability.
    Returns the path to the .npz cache.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{tag}.npz"
    manifest_path = out_dir / f"{tag}.manifest.json"

    done: dict = {}
    if resume and manifest_path.exists() and npz_path.exists():
        done = json.loads(manifest_path.read_text()).get("embedded", {})
        logger.info("Resuming embedding cache '%s': %d already done.", tag, len(done))

    model.eval().to(device)
    rows = df.reset_index(drop=True)
    cache: dict = {}
    if done and npz_path.exists():
        prev = np.load(npz_path, allow_pickle=True)
        for i, fn in enumerate(prev["filenames"]):
            cache[str(fn)] = prev["embeddings"][i]

    pending = [r for _, r in rows.iterrows() if str(r["filename"]) not in cache]
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        imgs = torch.stack([
            transform(Image.open(r["filepath"]).convert("RGB")) for r in chunk
        ]).to(device)
        emb = model.embed(imgs).cpu().numpy()
        for r, e in zip(chunk, emb):
            cache[str(r["filename"])] = e
        logger.info("embedded %d/%d", min(start + batch_size, len(pending)), len(pending))

    # Assemble in df order.
    filenames = rows["filename"].astype(str).to_numpy()
    embeddings = np.stack([cache[fn] for fn in filenames])
    np.savez(
        npz_path,
        embeddings=embeddings,
        filenames=filenames,
        patient_ids=rows["patient_id"].to_numpy(),
        labels=rows["label_binary"].to_numpy(),
    )
    manifest_path.write_text(json.dumps(
        {"tag": tag, "n": len(filenames), "embedded": {fn: True for fn in filenames}},
        indent=2,
    ))
    logger.info("Wrote embedding cache: %s (%d x %d)", npz_path, *embeddings.shape)
    return npz_path

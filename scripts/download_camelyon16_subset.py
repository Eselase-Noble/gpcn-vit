#!/usr/bin/env python3
"""Download a SMALL prototype subset of CAMELYON16 from the AWS Open Data mirror.

CAMELYON16 is ~700 GB in full — do NOT pull it all. This grabs a handful of
tumor + normal training slides (and their lesion annotations) so we can build and
validate the tiling + spatial-graph pipeline, then scale up later once proven.

Bucket: s3://camelyon-dataset  (region us-west-2, AWS Open Data, no credentials).
The exact key layout is not pinned by the registry manifest and may change, so
this LISTS the bucket first and matches keys by pattern rather than hard-coding
paths. Run in Colab:

    pip install boto3
    python scripts/download_camelyon16_subset.py --n-tumor 3 --n-normal 3 --dest /content/cam16

Use --list-only to just print the CAMELYON16 key structure without downloading.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cam16_dl")

BUCKET = "camelyon-dataset"
REGION = "us-west-2"
PREFIX_CANDIDATES = ("CAMELYON16/", "camelyon16/", "CAMELYON/CAMELYON16/")


def _client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))


def list_keys(s3, prefix, suffix=None):
    keys, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            if suffix is None or o["Key"].lower().endswith(suffix):
                keys.append((o["Key"], o["Size"]))
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break
    return keys


def find_prefix(s3):
    """Locate the CAMELYON16 key prefix (tolerant of case/layout changes)."""
    for cand in PREFIX_CANDIDATES:
        if s3.list_objects_v2(Bucket=BUCKET, Prefix=cand, MaxKeys=1).get("KeyCount", 0) > 0:
            return cand
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="data/raw/camelyon16")
    ap.add_argument("--n-tumor", type=int, default=3)
    ap.add_argument("--n-normal", type=int, default=3)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    try:
        s3 = _client()
    except Exception as e:  # noqa: BLE001
        log.error("boto3 unavailable (%s). In Colab: pip install boto3", e)
        return 1

    prefix = find_prefix(s3)
    if not prefix:
        log.error("could not locate a CAMELYON16 prefix under s3://%s "
                  "(tried %s) — check the registry page", BUCKET, PREFIX_CANDIDATES)
        return 1
    log.info("CAMELYON16 prefix: s3://%s/%s", BUCKET, prefix)

    tif = list_keys(s3, prefix, suffix=".tif")
    xml = list_keys(s3, prefix, suffix=".xml")
    tumor = sorted(k for k, _ in tif if "tumor" in k.lower())
    normal = sorted(k for k, _ in tif if "normal" in k.lower())
    log.info("found %d .tif (%d tumor, %d normal), %d annotation .xml",
             len(tif), len(tumor), len(normal), len(xml))

    if args.list_only:
        for k, sz in tif[:20]:
            log.info("  %s  (%.1f GB)", k, sz / 1e9)
        return 0

    picks = tumor[: args.n_tumor] + normal[: args.n_normal]
    if not picks:
        log.error("no tumor/normal .tif matched; re-run with --list-only to inspect layout")
        return 1
    ann = {k for k, _ in xml}
    dest = Path(args.dest)
    for key in picks:
        out = dest / Path(key).name
        out.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s -> %s", key, out)
        s3.download_file(BUCKET, key, str(out))
        stem = Path(key).stem
        for a in (a for a in ann if Path(a).stem == stem):
            aout = dest / "annotations" / Path(a).name
            aout.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, a, str(aout))
            log.info("  + annotation %s", Path(a).name)
    log.info("done: %d slides under %s", len(picks), dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

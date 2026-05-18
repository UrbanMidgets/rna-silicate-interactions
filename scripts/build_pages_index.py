#!/usr/bin/env python3
"""Build a compact JSON index for the GitHub Pages molecular viewer."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "MANIFEST.tsv"
OUT = REPO_ROOT / "docs" / "data_index.json"
RAW_BASE = "https://raw.githubusercontent.com/UrbanMidgets/rna-silicate-interactions/main/"
BLOB_BASE = "https://github.com/UrbanMidgets/rna-silicate-interactions/blob/main/"

KEEP_SUFFIXES = (
    ".xyz",
    ".out",
    ".inp",
    ".opt",
    ".engrad",
    ".property.txt",
)


def classify_type(path: str) -> str:
    p = path.lower()
    if p.endswith("_trj.xyz"):
        return "trajectory"
    if p.endswith(".xyz"):
        return "structure"
    if p.endswith(".inp"):
        return "input"
    if p.endswith(".out"):
        return "output"
    if p.endswith(".engrad"):
        return "gradient"
    if p.endswith(".opt"):
        return "optimization"
    if p.endswith(".property.txt"):
        return "property"
    return "other"


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing manifest: {MANIFEST}")

    records = []
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            repo_path = row["repo_path"]
            path_low = repo_path.lower()
            if not any(path_low.endswith(s) for s in KEEP_SUFFIXES):
                continue

            rec = {
                "system": row["system"],
                "surface": row["surface"],
                "frame": row["frame"],
                "state": row["state"],
                "role": row["role"],
                "status": row["status"],
                "notes": row["notes"],
                "repo_path": repo_path,
                "web_path": RAW_BASE + repo_path,
                "blob_url": BLOB_BASE + repo_path,
                "size_bytes": int(row["size_bytes"] or "0"),
                "file_type": classify_type(repo_path),
                "file_name": Path(repo_path).name,
            }
            records.append(rec)

    meta = {
        "record_count": len(records),
        "source_manifest": "data/MANIFEST.tsv",
    }
    payload = {"meta": meta, "records": records}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()

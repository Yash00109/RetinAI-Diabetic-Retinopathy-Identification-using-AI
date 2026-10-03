"""Remove cross-split content duplicates from the manifest.

When two images in different splits have identical content (same MD5 hash),
keep the one in train (if any), otherwise keep the one in val, and remove
the duplicate from the other split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from dr_detection.manifest import normalize_image_path


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Remove cross-split content duplicates.")
    parser.add_argument("--manifest", default="data/processed/manifest_quality_accepted.csv")
    parser.add_argument("--output", default="data/manifests/aptos_fixed_manifest.csv")
    parser.add_argument("--report", default="experiments/EXP-000/dedup_report.json")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    print(f"Original manifest: {len(manifest)} rows")

    # Compute hashes for all images
    print("Computing file hashes (this may take a few minutes)...")
    hashes = []
    for _, row in manifest.iterrows():
        p = normalize_image_path(row["image_path"])
        if p.exists():
            hashes.append(file_hash(p))
        else:
            hashes.append(None)
    manifest["content_hash"] = hashes

    # Find duplicates
    hash_to_indices = defaultdict(list)
    for idx, h in enumerate(hashes):
        if h is not None:
            hash_to_indices[h].append(idx)

    duplicate_groups = {h: indices for h, indices in hash_to_indices.items() if len(indices) > 1}
    print(f"Found {len(duplicate_groups)} groups of content-identical images")

    # Priority: keep train > val > test
    split_priority = {"train": 0, "val": 1, "test": 2}
    indices_to_remove = set()
    dedup_log = []

    for h, indices in duplicate_groups.items():
        rows = manifest.iloc[indices]
        # Sort by split priority — keep the highest priority (lowest number)
        sorted_rows = rows.sort_values(by="split", key=lambda s: s.map(split_priority))
        keep_idx = sorted_rows.index[0]
        remove_indices = sorted_rows.index[1:]
        indices_to_remove.update(remove_indices)

        dedup_log.append({
            "hash": h,
            "kept": {
                "image_path": str(manifest.loc[keep_idx, "image_path"]),
                "split": str(manifest.loc[keep_idx, "split"]),
                "label": int(manifest.loc[keep_idx, "label"]),
            },
            "removed": [
                {
                    "image_path": str(manifest.loc[idx, "image_path"]),
                    "split": str(manifest.loc[idx, "split"]),
                    "label": int(manifest.loc[idx, "label"]),
                }
                for idx in remove_indices
            ],
        })

    print(f"Removing {len(indices_to_remove)} duplicate rows")

    # Remove duplicates and drop hash column
    cleaned = manifest.drop(index=list(indices_to_remove)).drop(columns=["content_hash"]).reset_index(drop=True)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    report = {
        "original_count": len(manifest),
        "duplicates_removed": len(indices_to_remove),
        "final_count": len(cleaned),
        "split_counts": cleaned["split"].value_counts().to_dict(),
        "dedup_log": dedup_log,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nCleaned manifest saved to: {output_path}")
    print(f"Dedup report saved to: {report_path}")
    print(f"Final split counts:")
    for split, count in sorted(cleaned["split"].value_counts().items()):
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()

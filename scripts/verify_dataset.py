"""Dataset verification and split-freeze script for RetinAI.

Phase 2 of the research pipeline rebuild:
- Verify APTOS dataset integrity (missing files, labels, duplicates)
- Verify train/val/test isolation (no image appears in multiple splits)
- Generate dataset_report.json
- Freeze the split as the canonical manifest for all experiments
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        # pyrefly: ignore [missing-attribute]
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from PIL import Image

from dr_detection.manifest import normalize_image_path


def file_hash(path: Path, algorithm: str = "md5") -> str:
    """Compute hash of file contents."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(manifest_path: str, output_dir: str, root_dir: str | Path | None = None) -> dict:
    """Verify dataset integrity and generate a report."""
    manifest = pd.read_csv(manifest_path)
    # pyrefly: ignore [bad-assignment]
    output_dir = Path(output_dir)
    # pyrefly: ignore [missing-attribute]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "manifest_path": manifest_path,
        "total_rows": len(manifest),
        "columns": list(manifest.columns),
    }
    
    # --- Basic checks ---
    required_cols = {"image_path", "label", "split"}
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        report["error"] = f"Missing required columns: {sorted(missing_cols)}"
        return report
    
    # --- Label verification ---
    labels = manifest["label"].values
    valid_labels = set(range(5))
    invalid_labels = set(labels) - valid_labels
    # pyrefly: ignore [bad-assignment]
    report["label_stats"] = {
        "valid_range": [0, 4],
        "unique_labels": sorted(set(int(x) for x in labels)),
        "invalid_labels": sorted(int(x) for x in invalid_labels) if invalid_labels else [],
        "class_distribution": {str(k): v for k, v in sorted(Counter(labels).items())},
    }
    
    # --- Split verification ---
    splits = manifest["split"].unique().tolist()
    split_counts = manifest["split"].value_counts().to_dict()
    # pyrefly: ignore [bad-assignment]
    report["split_stats"] = {
        "splits_found": sorted(splits),
        "split_counts": {str(k): v for k, v in split_counts.items()},
    }
    
    # Per-split class distribution
    split_class_dist = {}
    for split in sorted(splits):
        split_df = manifest[manifest["split"] == split]
        split_class_dist[split] = {str(k): v for k, v in sorted(Counter(split_df["label"]).items())}
    # pyrefly: ignore [bad-assignment]
    report["split_class_distribution"] = split_class_dist
    
    # --- File existence check ---
    missing_files = []
    existing_count = 0
    for _, row in manifest.iterrows():
        p = normalize_image_path(row["image_path"], root_dir=root_dir)
        if p.exists():
            existing_count += 1
        else:
            missing_files.append(str(p))
    
    # pyrefly: ignore [bad-assignment]
    report["file_stats"] = {
        "existing_images": existing_count,
        "missing_images": len(missing_files),
        "missing_files_sample": missing_files[:10] if missing_files else [],
    }
    
    # --- Duplicate check (by filename) ---
    filenames = manifest["image_path"].apply(lambda x: normalize_image_path(x).name)
    filename_counts = Counter(filenames)
    duplicates_by_name = {k: v for k, v in filename_counts.items() if v > 1}
    # pyrefly: ignore [bad-assignment]
    report["duplicate_stats"] = {
        "duplicate_filenames": len(duplicates_by_name),
        "duplicate_filenames_sample": dict(list(duplicates_by_name.items())[:10]),
    }
    
    # --- Train/Val/Test isolation check ---
    train_images = set(manifest[manifest["split"] == "train"]["image_path"].apply(lambda x: normalize_image_path(x).as_posix()).values)
    val_images = set(manifest[manifest["split"] == "val"]["image_path"].apply(lambda x: normalize_image_path(x).as_posix()).values)
    test_images = set(manifest[manifest["split"] == "test"]["image_path"].apply(lambda x: normalize_image_path(x).as_posix()).values)
    
    train_val_overlap = train_images & val_images
    train_test_overlap = train_images & test_images
    val_test_overlap = val_images & test_images
    
    # pyrefly: ignore [bad-assignment]
    report["isolation"] = {
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "is_isolated": len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0,
    }
    
    if train_val_overlap:
        # pyrefly: ignore [unsupported-operation]
        report["isolation"]["train_val_overlap_sample"] = list(train_val_overlap)[:5]
    if train_test_overlap:
        # pyrefly: ignore [unsupported-operation]
        report["isolation"]["train_test_overlap_sample"] = list(train_test_overlap)[:5]
    if val_test_overlap:
        # pyrefly: ignore [unsupported-operation]
        report["isolation"]["val_test_overlap_sample"] = list(val_test_overlap)[:5]
    
    # --- Content hash duplicate check (sample-based for speed) ---
    print("Computing file hashes for duplicate detection...")
    hash_to_paths: dict[str, list[str]] = {}
    sample_size = min(len(manifest), 500)  # Hash a sample for speed
    sample_df = manifest.sample(n=sample_size, random_state=42) if len(manifest) > sample_size else manifest
    
    for _, row in sample_df.iterrows():
        p = normalize_image_path(row["image_path"], root_dir=root_dir)
        if p.exists():
            try:
                h = file_hash(p)
                hash_to_paths.setdefault(h, []).append(str(p))
            except Exception:
                pass
    
    content_duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    # pyrefly: ignore [bad-assignment]
    report["content_duplicate_stats"] = {
        "sample_size": sample_size,
        "content_duplicates_found": len(content_duplicates),
        "content_duplicates_sample": dict(list(content_duplicates.items())[:5]),
    }
    
    # --- Cross-split content duplicates ---
    print("Checking for cross-split content duplicates...")
    split_hashes: dict[str, dict[str, str]] = {}
    for split in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split]
        split_sample = split_df.sample(n=min(len(split_df), 200), random_state=42) if len(split_df) > 200 else split_df
        for _, row in split_sample.iterrows():
            p = normalize_image_path(row["image_path"], root_dir=root_dir)
            if p.exists():
                try:
                    h = file_hash(p)
                    split_hashes.setdefault(h, {})[split] = str(p)
                except Exception:
                    pass
    
    cross_split_dupes = {h: splits_dict for h, splits_dict in split_hashes.items() if len(splits_dict) > 1}
    # pyrefly: ignore [bad-assignment]
    report["cross_split_content_duplicates"] = {
        "found": len(cross_split_dupes),
        "sample": dict(list(cross_split_dupes.items())[:5]),
    }
    
    # --- Image dimension statistics (sample) ---
    print("Sampling image dimensions...")
    dims = []
    dim_sample = manifest.sample(n=min(len(manifest), 100), random_state=42)
    for _, row in dim_sample.iterrows():
        p = normalize_image_path(row["image_path"], root_dir=root_dir)
        if p.exists():
            try:
                with Image.open(p) as img:
                    dims.append(img.size)
            except Exception:
                pass
    
    if dims:
        widths, heights = zip(*dims)
        # pyrefly: ignore [bad-assignment]
        report["image_dimensions"] = {
            "sample_size": len(dims),
            "width_range": [int(min(widths)), int(max(widths))],
            "height_range": [int(min(heights)), int(max(heights))],
            "mean_width": float(np.mean(widths)),
            "mean_height": float(np.mean(heights)),
        }
    
    # --- Summary ---
    # pyrefly: ignore [bad-assignment]
    report["summary"] = {
        "total_images": report["total_rows"],
        "existing_images": existing_count,
        "train_count": split_counts.get("train", 0),
        "val_count": split_counts.get("val", 0),
        "test_count": split_counts.get("test", 0),
        # pyrefly: ignore [bad-index]
        "splits_isolated": report["isolation"]["is_isolated"],
        "no_invalid_labels": len(invalid_labels) == 0,
        "no_cross_split_duplicates": len(cross_split_dupes) == 0,
        "dataset_verified": (
            # pyrefly: ignore [bad-index]
            report["isolation"]["is_isolated"]
            and len(invalid_labels) == 0
            and len(cross_split_dupes) == 0
            and len(missing_files) == 0
        ),
    }
    
    # Save report
    # pyrefly: ignore [unsupported-operation]
    report_path = output_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDataset report saved to: {report_path}")
    print(json.dumps(report["summary"], indent=2))
    
    return report


def freeze_split(manifest_path: str, output_path: str) -> None:
    """Copy the manifest as the canonical frozen split."""
    manifest = pd.read_csv(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(f"Frozen split saved to: {output}")
    print(f"  Train: {len(manifest[manifest['split'] == 'train'])}")
    print(f"  Val:   {len(manifest[manifest['split'] == 'val'])}")
    print(f"  Test:  {len(manifest[manifest['split'] == 'test'])}")


def main():
    parser = argparse.ArgumentParser(description="Verify dataset integrity and freeze splits.")
    parser.add_argument("--manifest", default="data/processed/manifest_quality_accepted.csv",
                       help="Path to the manifest CSV to verify")
    parser.add_argument("--data-dir", default=None,
                       help="Root directory for dataset images (default: auto-detected)")
    parser.add_argument("--output-dir", default="experiments/EXP-000",
                       help="Directory to save the dataset report")
    parser.add_argument("--freeze-output", default="data/manifests/aptos_fixed_manifest.csv",
                       help="Path to save the frozen canonical manifest")
    parser.add_argument("--skip-freeze", action="store_true",
                       help="Skip freezing the split (verification only)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("RetinAI Dataset Verification — Phase 2")
    print("=" * 60)
    
    report = verify_dataset(args.manifest, args.output_dir, root_dir=args.data_dir)
    
    if report["summary"]["dataset_verified"]:
        print("\n[OK] Dataset verification PASSED")
        if not args.skip_freeze:
            freeze_split(args.manifest, args.freeze_output)
    else:
        print("\n[WARNING] Dataset verification found issues — review dataset_report.json")
        if not report["summary"]["splits_isolated"]:
            print("  [FAIL] Split isolation failed — potential data leakage!")
        if not report["summary"]["no_invalid_labels"]:
            print("  [FAIL] Invalid labels found")
        if not report["summary"]["no_cross_split_duplicates"]:
            print("  [WARNING] Cross-split content duplicates detected")
        
        # Still freeze if splits are isolated (most critical requirement)
        if report["summary"]["splits_isolated"] and not args.skip_freeze:
            print("\n  Splits are isolated — freezing manifest despite other warnings.")
            freeze_split(args.manifest, args.freeze_output)


if __name__ == "__main__":
    main()

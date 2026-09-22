from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dr_detection.manifest import (
    load_dataset_sources,
    source_to_manifest,
    stratified_manifest_split,
    summarize_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified DR dataset manifest from configured sources.")
    parser.add_argument("--sources", default="configs/dataset_sources.json")
    parser.add_argument("--output", default="data/processed/manifest.csv")
    parser.add_argument("--summary", default="data/processed/manifest_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()

    sources = load_dataset_sources(args.sources)
    frames = []
    for source in sources:
        if not source.enabled and not args.include_disabled:
            print(f"Skipping disabled source: {source.name}")
            continue
        try:
            frame = source_to_manifest(source, require_images=args.require_images)
        except FileNotFoundError as exc:
            print(f"Skipping {source.name}: {exc}")
            continue
        frames.append(frame)
        print(f"Loaded {source.name}: {len(frame)} rows")

    if not frames:
        raise SystemExit("No dataset sources were loaded.")

    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[manifest["label"].between(0, 4)].reset_index(drop=True)
    manifest = stratified_manifest_split(manifest, args.seed, args.val_size, args.test_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)

    summary = summarize_manifest(manifest)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

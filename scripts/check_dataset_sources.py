from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dr_detection.manifest import load_dataset_sources, resolve_image_path


def check_source(source, sample: int) -> dict:
    csv_path = Path(source.csv_path)
    image_dir = Path(source.image_dir)
    result = {
        "name": source.name,
        "enabled": source.enabled,
        "type": source.type,
        "csv_exists": csv_path.exists(),
        "image_dir_exists": image_dir.exists(),
        "rows": 0,
        "label_counts": {},
        "sample_existing_images": 0,
        "sample_missing_images": 0,
        "notes": source.notes,
    }
    if not csv_path.exists():
        return result

    frame = pd.read_csv(csv_path)
    result["rows"] = int(len(frame))
    if source.label_col in frame.columns:
        result["label_counts"] = frame[source.label_col].value_counts().sort_index().to_dict()
    if source.image_col in frame.columns:
        checked = frame.head(sample)
        existing = 0
        missing = 0
        for image_id in checked[source.image_col].astype(str):
            if resolve_image_path(image_dir, image_id, source.image_exts).exists():
                existing += 1
            else:
                missing += 1
        result["sample_existing_images"] = existing
        result["sample_missing_images"] = missing
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check configured dataset source availability.")
    parser.add_argument("--sources", default="configs/dataset_sources.json")
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = [check_source(source, args.sample) for source in load_dataset_sources(args.sources)]
    text = json.dumps({"sources": results}, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

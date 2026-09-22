from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an APTOS 2019 dataset folder.")
    parser.add_argument("--root", default="data/raw/aptos")
    parser.add_argument("--csv", default="train.csv")
    parser.add_argument("--image-dir", default="train_images")
    args = parser.parse_args()

    root = Path(args.root)
    csv_path = root / args.csv
    image_dir = root / args.image_dir
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing labels CSV: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    frame = pd.read_csv(csv_path)
    missing_cols = {"id_code", "diagnosis"}.difference(frame.columns)
    if missing_cols:
        raise ValueError(f"CSV missing columns: {sorted(missing_cols)}")

    missing_images = []
    for image_id in frame["id_code"].astype(str):
        if not any((image_dir / f"{image_id}{ext}").exists() for ext in [".png", ".jpg", ".jpeg"]):
            missing_images.append(image_id)

    print(f"Rows: {len(frame)}")
    print("Class counts:")
    print(frame["diagnosis"].value_counts().sort_index())
    print(f"Missing images: {len(missing_images)}")
    if missing_images[:10]:
        print("First missing IDs:", ", ".join(missing_images[:10]))


if __name__ == "__main__":
    main()

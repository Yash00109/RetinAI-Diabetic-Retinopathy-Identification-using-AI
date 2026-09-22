from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image


def _image_bytes(value) -> bytes:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return value["bytes"]
        if value.get("path"):
            return Path(value["path"]).read_bytes()
    if isinstance(value, bytes):
        return value
    raise TypeError(f"Unsupported image cell type: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert bumbledeep/aptos Hugging Face parquet into APTOS-style files."
    )
    parser.add_argument("--parquet", default="data/raw/hf_aptos/train.parquet")
    parser.add_argument("--output-root", default="data/raw/aptos")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    output_root = Path(args.output_root)
    image_dir = output_root / "train_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(parquet_path)
    if args.limit:
        frame = frame.head(args.limit)
    if "image" not in frame.columns or "label_code" not in frame.columns:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")

    labels = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        image_id = f"hf_aptos_{idx:05d}"
        image = Image.open(io.BytesIO(_image_bytes(row["image"]))).convert("RGB")
        image.save(image_dir / f"{image_id}.png")
        labels.append({"id_code": image_id, "diagnosis": int(row["label_code"])})

    labels_frame = pd.DataFrame(labels)
    labels_frame.to_csv(output_root / "train.csv", index=False)
    print(f"Wrote {len(labels_frame)} images to {image_dir}")
    print(labels_frame["diagnosis"].value_counts().sort_index())


if __name__ == "__main__":
    main()

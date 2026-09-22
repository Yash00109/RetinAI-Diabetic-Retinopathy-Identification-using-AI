from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_label_file(path: Path, root: Path, image_root: Path) -> pd.DataFrame:
    rows = []
    subset = path.stem
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        image_value, label_value = parts[0], parts[-1]
        try:
            label = int(float(label_value))
        except ValueError:
            continue
        if label < 0 or label > 4:
            continue

        raw = Path(image_value)
        candidates = []
        if raw.suffix.lower() in IMAGE_EXTENSIONS:
            candidates.append(image_root / raw)
            candidates.append(image_root / subset / raw.name)
            candidates.append(root / raw)
        else:
            for ext in IMAGE_EXTENSIONS:
                candidates.append(image_root / f"{image_value}{ext}")
                candidates.append(image_root / subset / f"{image_value}{ext}")
                candidates.append(root / f"{image_value}{ext}")
        image_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        rows.append({"image": str(image_path.relative_to(root)), "diagnosis": label, "subset": subset})
    return pd.DataFrame(rows)


def normalize_csv(path: Path, root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    lower = {column.lower(): column for column in frame.columns}
    image_col = lower.get("image") or lower.get("image_path") or lower.get("filename") or frame.columns[0]
    label_col = lower.get("diagnosis") or lower.get("label") or lower.get("level") or frame.columns[1]
    rows = []
    for _, row in frame.iterrows():
        try:
            label = int(row[label_col])
        except (TypeError, ValueError):
            continue
        if label < 0 or label > 4:
            continue
        image_value = str(row[image_col]).strip()
        rows.append({"image": image_value, "diagnosis": label, "subset": path.stem})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize DDR grading labels for the unified manifest.")
    parser.add_argument("--root", default="data/raw/ddr", help="DDR extraction root.")
    parser.add_argument("--output", default="data/raw/ddr/labels.csv")
    args = parser.parse_args()

    root = Path(args.root)
    grading_root = root / "DR_grading" if (root / "DR_grading").exists() else root
    txt_files = [grading_root / name for name in ("train.txt", "valid.txt", "test.txt") if (grading_root / name).exists()]
    csv_files = sorted(path for path in root.glob("**/*.csv") if path.name != "labels.csv")

    if txt_files:
        frames = [parse_label_file(path, root, grading_root) for path in txt_files]
    elif csv_files:
        frames = [normalize_csv(path, root) for path in csv_files]
    else:
        raise SystemExit(f"No DDR train/valid/test txt files or CSV labels found under {root}.")

    labels = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["image"]).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output, index=False)
    print(f"Wrote {output} with {len(labels)} rows")
    print(labels["diagnosis"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

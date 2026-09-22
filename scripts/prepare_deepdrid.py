from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LABEL_CANDIDATES = (
    "diagnosis",
    "level",
    "dr_grade",
    "dr_level",
    "patient_dr_level",
    "retinopathy_grade",
    "Retinopathy grade",
    "DR_grade",
)
IMAGE_CANDIDATES = ("image", "image_id", "image_path", "filename", "file", "ID", "id")
QUALITY_CANDIDATES = ("quality", "image_quality", "overall_quality", "Overall quality")
LATERALITY_CANDIDATES = ("laterality", "eye", "Eye", "left_right")


def first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {column.lower(): column for column in frame.columns}
    for name in names:
        if name in frame.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def read_label_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name.lower(), path)
            index.setdefault(path.stem.lower(), path)
            try:
                index.setdefault(str(path.relative_to(root)).lower(), path)
            except ValueError:
                pass
    return index


def find_image(root: Path, image_value: str, image_index: dict[str, Path]) -> Path | None:
    raw = Path(str(image_value))
    keys = [str(image_value).strip().lower(), raw.name.lower(), raw.stem.lower()]
    if raw.suffix.lower() in IMAGE_EXTENSIONS:
        keys.append(str(raw).lower())
    else:
        for ext in IMAGE_EXTENSIONS:
            keys.append(f"{raw}{ext}".lower())
            keys.append(f"{raw.name}{ext}".lower())
    for key in keys:
        candidate = image_index.get(key)
        if candidate is not None:
            return candidate
    return None


def normalize_frame(frame: pd.DataFrame, csv_path: Path, root: Path, image_index: dict[str, Path]) -> pd.DataFrame:
    image_col = first_existing_column(frame, IMAGE_CANDIDATES)
    label_col = first_existing_column(frame, LABEL_CANDIDATES)
    if image_col is None or label_col is None:
        if len(frame.columns) >= 2:
            image_col = image_col or frame.columns[0]
            label_col = label_col or frame.columns[1]
        else:
            raise ValueError(f"{csv_path} does not contain image and diagnosis columns.")

    quality_col = first_existing_column(frame, QUALITY_CANDIDATES)
    laterality_col = first_existing_column(frame, LATERALITY_CANDIDATES)
    rows = []
    for _, row in frame.iterrows():
        try:
            label = int(row[label_col])
        except (TypeError, ValueError):
            continue
        if label < 0 or label > 4:
            continue

        image_value = str(row[image_col]).strip()
        image_path = find_image(root, image_value, image_index)
        if image_path is None:
            image_path = Path(csv_path.parent.name) / "Images" / image_value
        else:
            image_path = image_path.relative_to(root)

        item = {"image": str(image_path), "diagnosis": label, "subset": csv_path.parent.name}
        if quality_col is not None:
            item["quality"] = row[quality_col]
        if laterality_col is not None:
            item["laterality"] = row[laterality_col]
        rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize DeepDRiD regular fundus labels for the unified manifest.")
    parser.add_argument("--root", default="data/raw/deepdrid", help="DeepDRiD extraction root.")
    parser.add_argument("--output", default="data/raw/deepdrid/labels.csv")
    args = parser.parse_args()

    root = Path(args.root)
    label_files = sorted(
        path
        for path in root.glob("**/*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
        and "regular-fundus" in str(path).lower()
        and "upload" not in path.name.lower()
    )
    if not label_files:
        raise SystemExit(f"No DeepDRiD regular fundus label files found under {root}.")

    image_index = build_image_index(root)
    frames = [normalize_frame(read_label_table(path), path, root, image_index) for path in label_files]
    labels = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["image"]).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output, index=False)
    print(f"Wrote {output} with {len(labels)} rows")
    print(labels["diagnosis"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

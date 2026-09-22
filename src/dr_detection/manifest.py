from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetSource:
    name: str
    type: str
    root: str
    csv_path: str
    image_dir: str
    image_col: str
    label_col: str
    image_exts: tuple[str, ...]
    enabled: bool = True
    quality_col: str | None = None
    laterality_col: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetSource:
        return cls(
            name=str(raw["name"]),
            type=str(raw.get("type", "generic_csv")),
            root=str(raw.get("root", "")),
            csv_path=str(raw["csv_path"]),
            image_dir=str(raw["image_dir"]),
            image_col=str(raw.get("image_col", "image")),
            label_col=str(raw.get("label_col", "diagnosis")),
            image_exts=tuple(raw.get("image_exts", IMAGE_EXTENSIONS)),
            enabled=bool(raw.get("enabled", True)),
            quality_col=raw.get("quality_col"),
            laterality_col=raw.get("laterality_col"),
            notes=str(raw.get("notes", "")),
        )


def load_dataset_sources(path: str | Path) -> list[DatasetSource]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DatasetSource.from_dict(item) for item in raw.get("sources", [])]


def resolve_image_path(image_dir: str | Path, image_id: str, image_exts: tuple[str, ...]) -> Path:
    image_dir = Path(image_dir)
    raw = Path(str(image_id))
    candidates: list[Path] = []
    if raw.suffix.lower() in IMAGE_EXTENSIONS:
        candidates.append(image_dir / raw)
    else:
        candidates.extend(image_dir / f"{image_id}{ext}" for ext in image_exts)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def source_to_manifest(source: DatasetSource, require_images: bool = False) -> pd.DataFrame:
    csv_path = Path(source.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing labels CSV for {source.name}: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {source.image_col, source.label_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source.name} CSV missing columns: {sorted(missing)}")

    rows = []
    missing_images = []
    for _, row in frame.iterrows():
        image_id = str(row[source.image_col])
        image_path = resolve_image_path(source.image_dir, image_id, source.image_exts)
        if not image_path.exists():
            missing_images.append(image_id)
            if require_images:
                continue
        item = {
            "image_id": image_id,
            "image_path": str(image_path),
            "label": int(row[source.label_col]),
            "source": source.name,
            "exists": bool(image_path.exists()),
        }
        if source.quality_col and source.quality_col in frame.columns:
            item["quality"] = row[source.quality_col]
        if source.laterality_col and source.laterality_col in frame.columns:
            item["laterality"] = row[source.laterality_col]
        rows.append(item)

    manifest = pd.DataFrame(rows)
    if missing_images and require_images:
        print(f"{source.name}: skipped {len(missing_images)} rows with missing images")
    return manifest


def stratified_manifest_split(
    manifest: pd.DataFrame,
    seed: int,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    if "label" not in manifest.columns:
        raise ValueError("Manifest requires a 'label' column")
    frame = manifest.reset_index(drop=True).copy()
    labels = frame["label"].astype(int)
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val["label"].astype(int),
        random_state=seed,
    )
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return pd.concat([train, val, test], ignore_index=True)


def summarize_manifest(manifest: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(manifest)),
        "existing_images": int(manifest.get("exists", pd.Series(dtype=bool)).sum()) if "exists" in manifest else None,
        "sources": manifest["source"].value_counts().sort_index().to_dict() if "source" in manifest else {},
        "labels": manifest["label"].value_counts().sort_index().to_dict() if "label" in manifest else {},
    }
    if "split" in manifest:
        summary["splits"] = manifest["split"].value_counts().sort_index().to_dict()
        summary["split_labels"] = {
            split: group["label"].value_counts().sort_index().to_dict()
            for split, group in manifest.groupby("split")
        }
    return summary

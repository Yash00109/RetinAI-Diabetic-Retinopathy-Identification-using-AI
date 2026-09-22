from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from dr_detection.config import DataConfig
from dr_detection.quality import (
    QualityThresholds,
    analyze_fundus_quality,
    crop_to_retina,
    load_rgb_image,
)


class AptosDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str | Path,
        image_col: str = "id_code",
        label_col: str = "diagnosis",
        image_ext: str = ".png",
        transform: Callable | None = None,
        quality_filter: bool = False,
        quality_thresholds: QualityThresholds | None = None,
        crop_retina: bool = True,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_col = image_col
        self.label_col = label_col
        self.image_ext = image_ext
        self.transform = transform
        self.quality_filter = quality_filter
        self.quality_thresholds = quality_thresholds
        self.crop_retina = crop_retina

    def __len__(self) -> int:
        return len(self.frame)

    def _image_path(self, idx: int) -> Path:
        image_id = str(self.frame.loc[idx, self.image_col])
        suffix = "" if image_id.lower().endswith((".png", ".jpg", ".jpeg")) else self.image_ext
        return self.image_dir / f"{image_id}{suffix}"

    def __getitem__(self, idx: int):
        path = self._image_path(idx)
        rgb = load_rgb_image(path)

        if self.quality_filter:
            report = analyze_fundus_quality(rgb, self.quality_thresholds)
            if not report.accepted:
                raise ValueError(f"Rejected low-quality image {path}: {report.reasons}")

        if self.crop_retina:
            rgb = crop_to_retina(rgb)

        image = Image.fromarray(rgb)
        label = int(self.frame.loc[idx, self.label_col])

        if self.transform:
            image = self.transform(image)
        return image, label


class ManifestImageDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Callable | None = None,
        image_col: str = "image_path",
        label_col: str = "label",
    ) -> None:
        missing = {image_col, label_col}.difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        path = Path(str(self.frame.loc[idx, self.image_col]))
        image = Image.open(path).convert("RGB")
        label = int(self.frame.loc[idx, self.label_col])
        if self.transform:
            image = self.transform(image)
        return image, label


def read_labels(config: DataConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.csv_path)
    required = {config.image_col, config.label_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return frame


def filter_frame_by_quality(
    frame: pd.DataFrame,
    image_dir: str | Path,
    image_col: str,
    image_ext: str,
    thresholds: QualityThresholds,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_rows = []
    rejected_rows = []
    image_dir = Path(image_dir)

    for _, row in frame.iterrows():
        image_id = str(row[image_col])
        suffix = "" if image_id.lower().endswith((".png", ".jpg", ".jpeg")) else image_ext
        image_path = image_dir / f"{image_id}{suffix}"
        report = analyze_fundus_quality(image_path, thresholds)
        row_dict = row.to_dict()
        row_dict.update(report.as_dict())
        if report.accepted:
            accepted_rows.append(row_dict)
        else:
            rejected_rows.append(row_dict)

    return pd.DataFrame(accepted_rows), pd.DataFrame(rejected_rows)


def make_splits(config: DataConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = read_labels(config)
    labels = frame[config.label_col]
    train_val, test = train_test_split(
        frame,
        test_size=config.test_size,
        stratify=labels,
        random_state=seed,
    )
    relative_val_size = config.val_size / (1.0 - config.test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[config.label_col],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    csv_path: str
    image_dir: str
    image_col: str = "id_code"
    label_col: str = "diagnosis"
    image_ext: str = ".png"
    num_classes: int = 5
    val_size: float = 0.15
    test_size: float = 0.15
    quality_filter: bool = True


@dataclass(frozen=True)
class QualityConfig:
    min_brightness: float = 25.0
    max_brightness: float = 235.0
    min_contrast: float = 8.0
    min_blur_score: float = 18.0
    min_retina_area_ratio: float = 0.18
    max_retina_area_ratio: float = 0.97
    max_retina_center_offset: float = 0.30
    max_retina_aspect_ratio_delta: float = 0.55
    min_retina_circularity: float = 0.25
    min_orientation_landmark_confidence: float = 0.12
    min_orientation_landmark_distance: float = 0.16
    max_orientation_verticality: float = 0.80


@dataclass(frozen=True)
class ModelConfig:
    name: str = "efficientnet_b0"
    pretrained: bool = True
    drop_rate: float = 0.25


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    batch_size: int = 32
    image_size: int = 224
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    class_weighting: bool = True
    mixed_precision: bool = True
    early_stopping_patience: int = 5


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str = "artifacts/efficientnet_b0"
    use_wandb: bool = False
    wandb_project: str = "diabetic-retinopathy-efficientnet"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    quality: QualityConfig
    model: ModelConfig
    train: TrainConfig
    logging: LoggingConfig


def _build_dataclass(cls: type, values: dict[str, Any]) -> Any:
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        data=_build_dataclass(DataConfig, raw["data"]),
        quality=_build_dataclass(QualityConfig, raw.get("quality", {})),
        model=_build_dataclass(ModelConfig, raw["model"]),
        train=_build_dataclass(TrainConfig, raw["train"]),
        logging=_build_dataclass(LoggingConfig, raw.get("logging", {})),
    )

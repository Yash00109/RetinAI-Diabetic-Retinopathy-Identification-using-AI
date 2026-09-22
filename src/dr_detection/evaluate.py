from __future__ import annotations

import argparse
import json
from pathlib import Path

from dr_detection.config import load_config
from dr_detection.dataset import AptosDataset, filter_frame_by_quality, make_splits
from dr_detection.metrics import classification_metrics
from dr_detection.models import create_model
from dr_detection.quality import QualityThresholds
from dr_detection.transforms import build_transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficientnet_b0.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    cfg = load_config(args.config)
    _, val_frame, test_frame = make_splits(cfg.data, cfg.seed)
    frame = val_frame if args.split == "val" else test_frame

    thresholds = QualityThresholds(**cfg.quality.__dict__)
    rejected = None
    if cfg.data.quality_filter:
        frame, rejected = filter_frame_by_quality(
            frame,
            cfg.data.image_dir,
            cfg.data.image_col,
            cfg.data.image_ext,
            thresholds,
        )

    ds = AptosDataset(
        frame,
        cfg.data.image_dir,
        cfg.data.image_col,
        cfg.data.label_col,
        cfg.data.image_ext,
        transform=build_transforms(cfg.train.image_size, train=False),
        quality_filter=False,
        quality_thresholds=thresholds,
    )
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg.model.name, cfg.data.num_classes, pretrained=False, drop_rate=cfg.model.drop_rate)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    metrics = classification_metrics(y_true, y_pred, cfg.data.num_classes)
    output = Path(args.output) if args.output else Path(cfg.logging.output_dir) / f"{args.split}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if rejected is not None:
        rejected.to_csv(output.with_name(f"{args.split}_rejected_quality.csv"), index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

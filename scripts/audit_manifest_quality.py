from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dr_detection.quality import QualityThresholds, analyze_fundus_quality


def build_thresholds(args: argparse.Namespace) -> QualityThresholds:
    values = {
        field: getattr(args, field)
        for field in QualityThresholds.__dataclass_fields__
        if getattr(args, field, None) is not None
    }
    return QualityThresholds(**values)


def audit_manifest(manifest_path: Path, thresholds: QualityThresholds) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"].astype(bool)].reset_index(drop=True)
    if "image_path" not in manifest.columns:
        raise ValueError("Manifest must contain an image_path column.")

    rows = []
    for idx, row in manifest.iterrows():
        item = row.to_dict()
        try:
            report = analyze_fundus_quality(row["image_path"], thresholds)
            item.update(report.as_dict())
        except Exception as exc:
            item.update(
                {
                    "accepted": False,
                    "reasons": [f"audit_error:{type(exc).__name__}"],
                    "audit_error": str(exc),
                }
            )
        item["manifest_row"] = int(idx)
        rows.append(item)
    return pd.DataFrame(rows)


def write_filtered_manifest(frame: pd.DataFrame, path: Path, accepted: bool) -> None:
    output = frame[frame["accepted"].astype(bool) == accepted].copy()
    output = output.drop(
        columns=[
            "accepted",
            "reasons",
            "brightness",
            "contrast",
            "blur_score",
            "retina_area_ratio",
            "retina_center_offset",
            "retina_aspect_ratio",
            "retina_circularity",
            "orientation_landmark_confidence",
            "orientation_landmark_x",
            "orientation_landmark_y",
            "orientation_verticality",
            "audit_error",
            "manifest_row",
        ],
        errors="ignore",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def summarize_audit(frame: pd.DataFrame) -> dict:
    rejected = frame[~frame["accepted"].astype(bool)] if "accepted" in frame else pd.DataFrame()
    reason_counts: dict[str, int] = {}
    if not rejected.empty and "reasons" in rejected.columns:
        for reasons in rejected["reasons"]:
            if isinstance(reasons, str):
                values = [reasons]
            else:
                values = list(reasons)
            for reason in values:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1

    return {
        "rows": int(len(frame)),
        "accepted": int(frame["accepted"].astype(bool).sum()) if "accepted" in frame else 0,
        "rejected": int(len(rejected)),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "rejected_by_source": rejected["source"].value_counts().sort_index().to_dict()
        if "source" in rejected
        else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a unified DR manifest with fundus quality/orientation checks.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--output", default="artifacts/quality_audit/manifest_quality_audit.csv")
    parser.add_argument("--summary", default="artifacts/quality_audit/manifest_quality_summary.json")
    parser.add_argument("--accepted-output", default=None, help="Optional path for accepted manifest rows only.")
    parser.add_argument("--rejected-output", default=None, help="Optional path for rejected manifest rows only.")
    for field, dataclass_field in QualityThresholds.__dataclass_fields__.items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=type(dataclass_field.default), default=None)
    args = parser.parse_args()

    thresholds = build_thresholds(args)
    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    frame = audit_manifest(Path(args.manifest), thresholds)
    frame.to_csv(output, index=False)
    if args.accepted_output:
        write_filtered_manifest(frame, Path(args.accepted_output), accepted=True)
    if args.rejected_output:
        write_filtered_manifest(frame, Path(args.rejected_output), accepted=False)
    summary = summarize_audit(frame)
    summary["thresholds"] = thresholds.__dict__
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

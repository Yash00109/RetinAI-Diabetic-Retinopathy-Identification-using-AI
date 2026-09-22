from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from dr_detection.manifest import (
    load_dataset_sources,
    source_to_manifest,
    stratified_manifest_split,
    summarize_manifest,
)


class ManifestTests(unittest.TestCase):
    def test_builds_manifest_from_aptos_style_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            for idx in range(25):
                Image.new("RGB", (16, 16), (idx * 10, 20, 30)).save(image_dir / f"img_{idx}.png")
            csv_path = root / "labels.csv"
            pd.DataFrame(
                {
                    "id_code": [f"img_{idx}" for idx in range(25)],
                    "diagnosis": [idx % 5 for idx in range(25)],
                }
            ).to_csv(csv_path, index=False)
            config_path = root / "sources.json"
            config_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "name": "mini",
                                "type": "aptos_csv",
                                "root": str(root),
                                "csv_path": str(csv_path),
                                "image_dir": str(image_dir),
                                "image_col": "id_code",
                                "label_col": "diagnosis",
                                "image_exts": [".png"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            source = load_dataset_sources(config_path)[0]
            manifest = source_to_manifest(source, require_images=True)
            split_manifest = stratified_manifest_split(manifest, seed=7, val_size=0.2, test_size=0.2)
            summary = summarize_manifest(split_manifest)

            self.assertEqual(len(manifest), 25)
            self.assertEqual(set(split_manifest["split"]), {"train", "val", "test"})
            self.assertEqual(summary["rows"], 25)
            self.assertEqual(summary["existing_images"], 25)


if __name__ == "__main__":
    unittest.main()

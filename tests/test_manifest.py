from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image

from dr_detection.manifest import (
    load_dataset_sources,
    normalize_image_path,
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

    def test_windows_path_resolves_correctly_on_linux(self) -> None:
        """Regression test for cross-platform dataset path resolution:
        Proves that data\\raw\\aptos\\train_images\\example.png resolves
        correctly to data/raw/aptos/train_images/example.png on Linux/POSIX.
        """
        raw_win_path = r"data\raw\aptos\train_images\example.png"
        expected_posix = "data/raw/aptos/train_images/example.png"

        # 1. Normalize the path
        normalized = normalize_image_path(raw_win_path)

        # 2. Check POSIX representation matches expected
        self.assertEqual(normalized.as_posix(), expected_posix)

        # 3. Simulate Linux / POSIX path semantics:
        # On POSIX, a normalized path splits into components correctly:
        posix_path = PurePosixPath(normalized.as_posix())
        self.assertEqual(str(posix_path), expected_posix)
        self.assertEqual(posix_path.name, "example.png")
        self.assertEqual(posix_path.suffix, ".png")
        self.assertEqual(str(posix_path.parent), "data/raw/aptos/train_images")
        self.assertEqual(
            posix_path.parts,
            ("data", "raw", "aptos", "train_images", "example.png"),
        )

        # 4. Prove that unnormalized path fails on POSIX:
        unnormalized_posix = PurePosixPath(raw_win_path)
        # On POSIX without normalization, the entire Windows string is treated as a single filename:
        self.assertEqual(unnormalized_posix.parts, (raw_win_path,))
        self.assertNotEqual(unnormalized_posix.parts, posix_path.parts)

    def test_path_normalization_with_root_dir(self) -> None:
        """Test path normalization when root_dir is specified (e.g. Google Colab /content)."""
        raw_win_path = r"data\raw\aptos\train_images\example.png"
        root_dir = "/content/RetinAI-DR"
        normalized = normalize_image_path(raw_win_path, root_dir=root_dir)
        self.assertEqual(
            normalized.as_posix(),
            "/content/RetinAI-DR/data/raw/aptos/train_images/example.png",
        )

    def test_path_normalization_real_dataset_file(self) -> None:
        """Test path normalization resolves actual existing dataset file."""
        real_win_path = r"data\raw\aptos\train_images\hf_aptos_03571.png"
        resolved = normalize_image_path(real_win_path)
        self.assertTrue(resolved.exists(), f"Image {resolved} should exist on disk")

    def test_manifest_dataset_loads_with_windows_path(self) -> None:
        """Test ManifestImageDataset properly loads images from DataFrame containing Windows paths."""
        from dr_detection.dataset import ManifestImageDataset
        df = pd.DataFrame({
            "image_path": [r"data\raw\aptos\train_images\hf_aptos_03571.png"],
            "label": [2],
        })
        dataset = ManifestImageDataset(df)
        self.assertEqual(len(dataset), 1)
        img, label = dataset[0]
        self.assertEqual(label, 2)
        self.assertIsNotNone(img)


if __name__ == "__main__":
    unittest.main()

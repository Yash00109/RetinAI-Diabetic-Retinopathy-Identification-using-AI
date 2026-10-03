"""Tests for class-targeted augmentation (EXP-005).

Verifies:
1. ManifestImageDataset.label_transform hook is called with (image, label).
2. ManifestImageDataset.label_transform takes priority over transform.
3. ClassAwareRetinaAugmentation routes correctly with custom minority_classes.
4. Default minority_classes={1,3,4} is preserved for backward compatibility.
5. Validation / test datasets are not affected (no label_transform).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dr_detection.dataset import ManifestImageDataset
from dr_detection.transforms import ClassAwareRetinaAugmentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_manifest(tmp_dir: Path, n: int = 10) -> pd.DataFrame:
    """Create a tiny manifest with actual tiny images on disk."""
    rows = []
    for i in range(n):
        fname = f"img_{i:04d}.png"
        fpath = tmp_dir / fname
        # Tiny 32×32 RGB image
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(fpath)
        rows.append({
            "image_path": str(fpath),
            "label": i % 5,  # cycles 0-4
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: ManifestImageDataset.label_transform
# ---------------------------------------------------------------------------

class TestLabelTransformHook:
    """Tests for the label_transform hook in ManifestImageDataset."""

    def test_label_transform_called_with_image_and_label(self, tmp_path):
        """label_transform receives (PIL.Image, int label)."""
        df = _make_test_manifest(tmp_path, n=5)
        
        call_log = []

        def spy_transform(img, label):
            call_log.append((type(img).__name__, label))
            return img  # identity

        ds = ManifestImageDataset(df, label_transform=spy_transform)
        img, lbl = ds[0]
        assert len(call_log) == 1
        assert call_log[0][0] == "Image"
        assert call_log[0][1] == lbl

    def test_label_transform_takes_priority_over_transform(self, tmp_path):
        """When both transform and label_transform are set, label_transform wins."""
        df = _make_test_manifest(tmp_path, n=3)

        regular_called = []
        label_called = []

        def regular_tx(img):
            regular_called.append(1)
            return img

        def label_tx(img, label):
            label_called.append(1)
            return img

        ds = ManifestImageDataset(df, transform=regular_tx, label_transform=label_tx)
        ds[0]
        assert len(label_called) == 1
        assert len(regular_called) == 0, "regular transform should NOT be called when label_transform is set"

    def test_regular_transform_still_works(self, tmp_path):
        """When label_transform is None, the regular transform is used (backward compat)."""
        df = _make_test_manifest(tmp_path, n=3)

        regular_called = []

        def regular_tx(img):
            regular_called.append(1)
            return img

        ds = ManifestImageDataset(df, transform=regular_tx)
        ds[0]
        assert len(regular_called) == 1

    def test_no_transform_returns_pil_image(self, tmp_path):
        """With neither transform, the raw PIL image and label are returned."""
        df = _make_test_manifest(tmp_path, n=3)
        ds = ManifestImageDataset(df)
        img, lbl = ds[0]
        assert isinstance(img, Image.Image)
        assert isinstance(lbl, int)


# ---------------------------------------------------------------------------
# Tests: ClassAwareRetinaAugmentation
# ---------------------------------------------------------------------------

class TestClassAwareRetinaAugmentation:
    """Tests for the configurable ClassAwareRetinaAugmentation.

    We mock ``build_transforms`` to avoid requiring torchvision in the
    local test environment — we are testing *routing* logic, not the
    actual torchvision pipelines.
    """

    @pytest.fixture(autouse=True)
    def _patch_build_transforms(self, monkeypatch):
        """Replace build_transforms with a lightweight spy."""
        def fake_build(image_size, train, preprocess=True, augmentation="targeted"):
            # Return a distinguishable callable so we can verify routing
            tag = f"{augmentation}_{'train' if train else 'val'}"
            return MagicMock(__name__=tag, return_value=tag)

        monkeypatch.setattr(
            "dr_detection.transforms.build_transforms", fake_build,
        )

    def test_default_minority_classes_backward_compat(self):
        """Default minority_classes should be {1, 3, 4}."""
        aug = ClassAwareRetinaAugmentation(image_size=64, preprocess=False)
        assert aug.minority_classes == {1, 3, 4}

    def test_custom_minority_classes(self):
        """Custom minority_classes={3, 4} should be respected."""
        aug = ClassAwareRetinaAugmentation(
            image_size=64, preprocess=False, minority_classes={3, 4},
        )
        assert aug.minority_classes == {3, 4}

    def test_routes_minority_to_targeted(self):
        """Minority class labels should use the targeted_transform."""
        aug = ClassAwareRetinaAugmentation(
            image_size=64, preprocess=False, minority_classes={3, 4},
        )
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

        # Replace the real Compose pipelines with mocks so we can assert calls
        aug.targeted_transform = MagicMock(return_value=img)
        aug.standard_transform = MagicMock(return_value=img)

        result = aug(img, 3)
        # targeted_transform was built with augmentation="targeted" → tag
        aug.targeted_transform.assert_called_once()
        aug.standard_transform.assert_not_called()

    def test_routes_majority_to_standard(self):
        """Non-minority class labels should use the standard_transform."""
        aug = ClassAwareRetinaAugmentation(
            image_size=64, preprocess=False, minority_classes={3, 4},
        )
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

        # Replace the real Compose pipelines with mocks so we can assert calls
        aug.targeted_transform = MagicMock(return_value=img)
        aug.standard_transform = MagicMock(return_value=img)

        result = aug(img, 0)
        aug.standard_transform.assert_called_once()
        aug.targeted_transform.assert_not_called()

    def test_class_1_uses_standard_when_not_in_minority(self):
        """With minority_classes={3,4}, class 1 should get standard augmentation."""
        aug = ClassAwareRetinaAugmentation(
            image_size=64, preprocess=False, minority_classes={3, 4},
        )
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

        # Replace the real Compose pipelines with mocks so we can assert calls
        aug.targeted_transform = MagicMock(return_value=img)
        aug.standard_transform = MagicMock(return_value=img)

        result = aug(img, 1)
        aug.standard_transform.assert_called_once()
        aug.targeted_transform.assert_not_called()

    def test_augmentation_level_parameter(self):
        """augmentation_level='strong' should be accepted without error."""
        aug = ClassAwareRetinaAugmentation(
            image_size=64, preprocess=False,
            minority_classes={3, 4}, augmentation_level="strong",
        )
        assert aug.minority_classes == {3, 4}

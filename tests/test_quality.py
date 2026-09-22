from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image

from dr_detection.preprocessing import RetinaPreprocessConfig, RetinaPreprocessor, pad_to_square
from dr_detection.preprocessing import crop_to_retina as pil_crop_to_retina
from dr_detection.quality import QualityThresholds, analyze_fundus_quality, crop_to_retina


def synthetic_fundus(size: int = 256) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(image, center, int(size * 0.42), (110, 55, 35), -1)
    for offset in range(-70, 80, 35):
        cv2.line(image, center, (center[0] + offset, center[1] - 85), (180, 85, 65), 2)
    cv2.circle(image, (center[0] - 30, center[1] + 20), 12, (210, 160, 120), -1)
    return image


def synthetic_oriented_fundus(size: int = 256) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(image, center, int(size * 0.42), (110, 55, 35), -1)
    for offset in range(-70, 80, 35):
        cv2.line(image, center, (center[0] + offset, center[1] - 85), (180, 85, 65), 2)
    cv2.circle(image, (center[0] - 70, center[1]), 14, (230, 190, 130), -1)
    return image


class QualityTests(unittest.TestCase):
    def test_accepts_reasonable_synthetic_fundus(self) -> None:
        report = analyze_fundus_quality(
            synthetic_fundus(),
            QualityThresholds(min_blur_score=1.0, min_contrast=5.0),
        )
        self.assertTrue(report.accepted, report)
        self.assertGreater(report.retina_area_ratio, 0.18)

    def test_rejects_dark_blank_image(self) -> None:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        report = analyze_fundus_quality(image)
        self.assertFalse(report.accepted)
        self.assertIn("too_dark", report.reasons)

    def test_rejects_likely_rotated_fundus(self) -> None:
        image = np.rot90(synthetic_oriented_fundus())
        report = analyze_fundus_quality(
            image,
            QualityThresholds(min_blur_score=1.0, min_contrast=5.0),
        )
        self.assertFalse(report.accepted)
        self.assertIn("optic_disc_orientation_suspicious", report.reasons)

    def test_accepts_landmark_on_horizontal_axis(self) -> None:
        report = analyze_fundus_quality(
            synthetic_oriented_fundus(),
            QualityThresholds(min_blur_score=1.0, min_contrast=5.0),
        )
        self.assertTrue(report.accepted, report)
        self.assertIsNotNone(report.orientation_verticality)
        self.assertLess(report.orientation_verticality, 0.68)

    def test_crop_to_retina_reduces_background(self) -> None:
        image = synthetic_fundus()
        cropped = crop_to_retina(image)
        self.assertLess(cropped.shape[0], image.shape[0])
        self.assertLess(cropped.shape[1], image.shape[1])

    def test_training_preprocess_crops_and_pads(self) -> None:
        image = Image.fromarray(synthetic_fundus())
        cropped = pil_crop_to_retina(image)
        squared = pad_to_square(cropped)
        self.assertLess(cropped.size[0], image.size[0])
        self.assertEqual(squared.size[0], squared.size[1])

    def test_training_preprocess_returns_rgb_image(self) -> None:
        image = Image.fromarray(synthetic_fundus())
        preprocessor = RetinaPreprocessor(RetinaPreprocessConfig(enhance_contrast=False))
        processed = preprocessor(image)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(processed.size[0], processed.size[1])


if __name__ == "__main__":
    unittest.main()

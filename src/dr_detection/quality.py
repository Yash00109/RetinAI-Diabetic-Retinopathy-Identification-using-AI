from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityThresholds:
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
class QualityReport:
    accepted: bool
    reasons: tuple[str, ...]
    brightness: float
    contrast: float
    blur_score: float
    retina_area_ratio: float
    retina_center_offset: float
    retina_aspect_ratio: float
    retina_circularity: float
    orientation_landmark_confidence: float
    orientation_landmark_x: float | None
    orientation_landmark_y: float | None
    orientation_verticality: float | None

    def as_dict(self) -> dict[str, float | bool | list[str] | None]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "brightness": self.brightness,
            "contrast": self.contrast,
            "blur_score": self.blur_score,
            "retina_area_ratio": self.retina_area_ratio,
            "retina_center_offset": self.retina_center_offset,
            "retina_aspect_ratio": self.retina_aspect_ratio,
            "retina_circularity": self.retina_circularity,
            "orientation_landmark_confidence": self.orientation_landmark_confidence,
            "orientation_landmark_x": self.orientation_landmark_x,
            "orientation_landmark_y": self.orientation_landmark_y,
            "orientation_verticality": self.orientation_verticality,
        }


def load_rgb_image(image: str | Path | Image.Image | np.ndarray, max_side: int = 1024) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = image
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        array = np.asarray(img)
    else:
        with Image.open(image) as img:
            img = img.convert("RGB")
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            array = np.asarray(img)

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array.astype(np.uint8)


def estimate_retina_mask(rgb: np.ndarray) -> np.ndarray:
    """Estimate visible fundus area using intensity and saturation cues."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    mask = ((value > 12) & (saturation > 18)).astype(np.uint8)

    kernel_size = max(5, (min(rgb.shape[:2]) // 80) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def retina_geometry(mask: np.ndarray, image_shape: tuple[int, int]) -> dict[str, float]:
    height, width = image_shape
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return {
            "center_offset": 1.0,
            "aspect_ratio": 0.0,
            "circularity": 0.0,
        }

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    box_width = max(1, int(x1 - x0 + 1))
    box_height = max(1, int(y1 - y0 + 1))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    offset_x = abs(center_x - (width - 1) / 2.0) / max(1.0, width)
    offset_y = abs(center_y - (height - 1) / 2.0) / max(1.0, height)
    center_offset = float((offset_x**2 + offset_y**2) ** 0.5)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(mask.sum())
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    circularity = 0.0 if perimeter == 0 else float(4.0 * np.pi * area / (perimeter**2))

    return {
        "center_offset": center_offset,
        "aspect_ratio": float(box_width / box_height),
        "circularity": circularity,
    }


def estimate_orientation_landmark(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    """Estimate optic-disc-like bright landmark position relative to the retina center.

    This is a conservative heuristic for upload QA, not an anatomical detector. It only
    rejects orientation when a bright landmark is clear and lies on the top/bottom axis,
    which commonly indicates a 90-degree rotated fundus image.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return {"confidence": 0.0, "x": None, "y": None, "verticality": None}

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    box_width = max(1.0, float(x1 - x0 + 1))
    box_height = max(1.0, float(y1 - y0 + 1))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0].astype(np.float32)
    blurred = cv2.GaussianBlur(lightness, (0, 0), sigmaX=max(1.0, min(rgb.shape[:2]) / 120.0))
    foreground = blurred[mask]
    if foreground.size == 0:
        return {"confidence": 0.0, "x": None, "y": None, "verticality": None}

    cutoff = float(np.percentile(foreground, 99.2))
    candidates = mask & (blurred >= cutoff)
    cand_y, cand_x = np.where(candidates)
    if len(cand_x) == 0:
        return {"confidence": 0.0, "x": None, "y": None, "verticality": None}

    weights = np.maximum(blurred[candidates] - float(np.median(foreground)), 1.0)
    landmark_x = float(np.average(cand_x, weights=weights))
    landmark_y = float(np.average(cand_y, weights=weights))
    rel_x = float((landmark_x - center_x) / (box_width / 2.0))
    rel_y = float((landmark_y - center_y) / (box_height / 2.0))
    distance = abs(rel_x) + abs(rel_y)
    verticality = None if distance == 0 else float(abs(rel_y) / distance)
    contrast_gain = (float(np.mean(blurred[candidates])) - float(np.median(foreground))) / 255.0
    area_fraction = min(1.0, len(cand_x) / max(1.0, mask.sum() * 0.02))
    confidence = float(max(0.0, contrast_gain) * area_fraction)

    return {
        "confidence": confidence,
        "x": rel_x,
        "y": rel_y,
        "verticality": verticality,
    }


def crop_to_retina(rgb: np.ndarray, padding: float = 0.04) -> np.ndarray:
    mask = estimate_retina_mask(rgb)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return rgb

    height, width = rgb.shape[:2]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    pad_x = int((x1 - x0 + 1) * padding)
    pad_y = int((y1 - y0 + 1) * padding)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width - 1, x1 + pad_x)
    y1 = min(height - 1, y1 + pad_y)
    return rgb[y0 : y1 + 1, x0 : x1 + 1]


def analyze_fundus_quality(
    image: str | Path | Image.Image | np.ndarray,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    thresholds = thresholds or QualityThresholds()
    rgb = load_rgb_image(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = estimate_retina_mask(rgb)
    geometry = retina_geometry(mask, rgb.shape[:2])
    landmark = estimate_orientation_landmark(rgb, mask)

    if mask.any():
        foreground = gray[mask]
    else:
        foreground = gray.reshape(-1)

    brightness = float(np.mean(foreground))
    contrast = float(np.std(foreground))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    retina_area_ratio = float(mask.mean())
    retina_center_offset = geometry["center_offset"]
    retina_aspect_ratio = geometry["aspect_ratio"]
    retina_circularity = geometry["circularity"]
    orientation_confidence = float(landmark["confidence"] or 0.0)
    orientation_x = landmark["x"]
    orientation_y = landmark["y"]
    orientation_verticality = landmark["verticality"]

    reasons: list[str] = []

    def add_reason(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if brightness < thresholds.min_brightness:
        add_reason("too_dark")
    if brightness > thresholds.max_brightness:
        add_reason("too_bright")
    if contrast < thresholds.min_contrast:
        add_reason("low_contrast")
    if blur_score < thresholds.min_blur_score:
        add_reason("blurry")
    if retina_area_ratio < thresholds.min_retina_area_ratio:
        add_reason("retina_too_small_or_missing")
    if retina_area_ratio > thresholds.max_retina_area_ratio:
        add_reason("retina_crop_or_background_suspicious")
    if retina_center_offset > thresholds.max_retina_center_offset:
        add_reason("retina_off_center")
    if retina_aspect_ratio and abs(retina_aspect_ratio - 1.0) > thresholds.max_retina_aspect_ratio_delta:
        add_reason("retina_shape_suspicious")
    if retina_circularity < thresholds.min_retina_circularity:
        add_reason("retina_shape_suspicious")
    if (
        orientation_verticality is not None
        and orientation_x is not None
        and orientation_y is not None
        and orientation_confidence >= thresholds.min_orientation_landmark_confidence
        and (abs(orientation_x) + abs(orientation_y)) >= thresholds.min_orientation_landmark_distance
        and orientation_verticality > thresholds.max_orientation_verticality
    ):
        add_reason("optic_disc_orientation_suspicious")

    return QualityReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        retina_area_ratio=retina_area_ratio,
        retina_center_offset=retina_center_offset,
        retina_aspect_ratio=retina_aspect_ratio,
        retina_circularity=retina_circularity,
        orientation_landmark_confidence=orientation_confidence,
        orientation_landmark_x=float(orientation_x) if orientation_x is not None else None,
        orientation_landmark_y=float(orientation_y) if orientation_y is not None else None,
        orientation_verticality=float(orientation_verticality) if orientation_verticality is not None else None,
    )

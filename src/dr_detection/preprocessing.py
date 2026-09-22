from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class RetinaPreprocessConfig:
    crop_retina: bool = True
    square_pad: bool = True
    enhance_contrast: bool = True
    mask_threshold: int = 12
    crop_padding_ratio: float = 0.06
    blur_radius_ratio: float = 0.035


def crop_to_retina(image: Image.Image, threshold: int = 12, padding_ratio: float = 0.06) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    gray = arr.mean(axis=2)
    mask = gray > threshold
    if not mask.any():
        return rgb

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = int(max(x1 - x0, y1 - y0) * padding_ratio)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(rgb.width, x1 + pad)
    y1 = min(rgb.height, y1 + pad)
    return rgb.crop((x0, y0, x1, y1))


def pad_to_square(image: Image.Image, fill: int = 0) -> Image.Image:
    rgb = image.convert("RGB")
    size = max(rgb.size)
    left = (size - rgb.width) // 2
    top = (size - rgb.height) // 2
    right = size - rgb.width - left
    bottom = size - rgb.height - top
    return ImageOps.expand(rgb, border=(left, top, right, bottom), fill=(fill, fill, fill))


def ben_graham_enhance(image: Image.Image, blur_radius_ratio: float = 0.035) -> Image.Image:
    rgb = image.convert("RGB")
    radius = max(1.0, max(rgb.size) * blur_radius_ratio)
    blurred = rgb.filter(ImageFilter.GaussianBlur(radius=radius))
    img = np.asarray(rgb).astype(np.float32)
    blur = np.asarray(blurred).astype(np.float32)
    enhanced = np.clip((4.0 * img) - (4.0 * blur) + 128.0, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced, mode="RGB")


class RetinaPreprocessor:
    def __init__(self, config: RetinaPreprocessConfig | None = None) -> None:
        self.config = config or RetinaPreprocessConfig()

    def __call__(self, image: Image.Image) -> Image.Image:
        cfg = self.config
        out = image.convert("RGB")
        if cfg.crop_retina:
            out = crop_to_retina(out, threshold=cfg.mask_threshold, padding_ratio=cfg.crop_padding_ratio)
        if cfg.square_pad:
            out = pad_to_square(out)
        if cfg.enhance_contrast:
            out = ben_graham_enhance(out, blur_radius_ratio=cfg.blur_radius_ratio)
        return out

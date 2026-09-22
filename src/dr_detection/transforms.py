from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image

from dr_detection.preprocessing import RetinaPreprocessConfig, RetinaPreprocessor


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        import torch

        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ClipAndNormalize:
    def __init__(self, mean: list[float], std: list[float]):
        from torchvision import transforms

        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, tensor):
        import torch

        tensor = torch.clamp(tensor, 0, 1)
        return self.normalize(tensor)


class DihedralRotation:
    """Randomly applies one of the 8 transforms of the Dihedral group D4 (rotations + flips).
    Anatomically valid for circular retinal fundus images.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 7)
        # Rotations: 0, 90, 180, 270
        rot = (k % 4) * 90
        if rot > 0:
            img = img.rotate(rot, expand=False)
        # Flip if k >= 4
        if k >= 4:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class CLAHEGreenEnhance:
    """Adaptive histogram equalization on the green channel to enhance retinal microaneurysms and hemorrhages."""
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("RGB"))
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        # Enhance green channel which carries highest fundus pathology contrast
        arr[:, :, 1] = clahe.apply(arr[:, :, 1])
        return Image.fromarray(arr)


def build_transforms(
    image_size: int,
    train: bool,
    preprocess: bool = True,
    augmentation: str = "targeted",
):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("Install torchvision before building training transforms.") from exc

    steps = []
    if preprocess:
        steps.append(RetinaPreprocessor(RetinaPreprocessConfig()))

    if train:
        aug_steps = [
            DihedralRotation(),
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.04),
        ]
        
        if augmentation in ("targeted", "strong"):
            aug_steps.append(CLAHEGreenEnhance(clip_limit=2.5))
            if hasattr(transforms, "RandAugment"):
                aug_steps.append(transforms.RandAugment(num_ops=2, magnitude=7))
        elif hasattr(transforms, "RandAugment"):
            aug_steps.append(transforms.RandAugment(num_ops=2, magnitude=5))

        aug_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.20, scale=(0.015, 0.08), ratio=(0.4, 2.5)),
            ]
        )
        return transforms.Compose(steps + aug_steps)

    return transforms.Compose(
        steps
        + [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ClassAwareRetinaAugmentation:
    """Class-dependent dynamic augmentation multiplier and targeted transforms.
    Minority classes (Grade 1 Mild, Grade 3 Severe, Grade 4 Proliferative) receive
    targeted, higher-frequency and higher-diversity synthetic variations to equalize
    feature manifold representations.
    """
    def __init__(self, image_size: int, preprocess: bool = True):
        self.standard_transform = build_transforms(image_size, train=True, preprocess=preprocess, augmentation="standard")
        self.targeted_transform = build_transforms(image_size, train=True, preprocess=preprocess, augmentation="targeted")
        self.minority_classes = {1, 3, 4}

    def __call__(self, image: Image.Image, label: int):
        if label in self.minority_classes:
            return self.targeted_transform(image)
        return self.standard_transform(image)


def build_robustness_transforms(image_size: int, perturbation: str, preprocess: bool = True):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("Install torchvision for robustness transforms.") from exc

    steps = []
    if preprocess:
        steps.append(RetinaPreprocessor(RetinaPreprocessConfig()))

    steps.append(transforms.Resize((image_size, image_size)))

    if perturbation == "blur":
        steps.append(transforms.GaussianBlur(kernel_size=9, sigma=(1.5, 3.0)))
    elif perturbation == "brightness":
        steps.append(transforms.ColorJitter(brightness=(0.5, 2.0)))

    steps.extend([
        transforms.ToTensor(),
    ])

    if perturbation == "noise":
        steps.append(AddGaussianNoise(0.0, 0.15))

    steps.append(ClipAndNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(steps)

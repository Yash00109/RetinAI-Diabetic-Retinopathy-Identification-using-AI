from __future__ import annotations


def create_model(name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.25):
    import torch.nn as nn

    normalized = name.lower()
    if normalized == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, drop_rate=drop_rate)

    try:
        import timm

        return timm.create_model(
            normalized,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    except ImportError:
        pass

    torchvision_efficientnets = {
        "efficientnet_b0": ("EfficientNet_B0_Weights", "efficientnet_b0"),
        "efficientnet_b1": ("EfficientNet_B1_Weights", "efficientnet_b1"),
        "efficientnet_b2": ("EfficientNet_B2_Weights", "efficientnet_b2"),
        "efficientnet_b3": ("EfficientNet_B3_Weights", "efficientnet_b3"),
        "efficientnet_b4": ("EfficientNet_B4_Weights", "efficientnet_b4"),
    }
    if normalized in torchvision_efficientnets:
        import torchvision.models as tv_models

        weights_name, factory_name = torchvision_efficientnets[normalized]
        weights_cls = getattr(tv_models, weights_name)
        factory = getattr(tv_models, factory_name)
        weights = weights_cls.DEFAULT if pretrained else None
        model = factory(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_features, num_classes))
        return model

    raise ValueError(f"Unsupported model '{name}'. Install timm for more EfficientNet variants.")


class SimpleCNN:
    def __new__(cls, num_classes: int, drop_rate: float = 0.30):
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(128, num_classes),
        )

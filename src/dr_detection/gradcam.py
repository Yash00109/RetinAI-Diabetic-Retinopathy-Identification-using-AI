from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from dr_detection.config import load_config
from dr_detection.manifest import normalize_image_path
from dr_detection.models import create_model
from dr_detection.quality import crop_to_retina, load_rgb_image
from dr_detection.transforms import build_transforms


def _last_conv_layer(model):
    import torch.nn as nn

    last_layer = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_layer = module
    if last_layer is None:
        raise ValueError("No convolution layer found for Grad-CAM.")
    return last_layer


def make_gradcam(
    config_path: str | Path,
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    class_id: int | None = None,
) -> dict:
    import torch

    config_path = normalize_image_path(config_path)
    checkpoint_path = normalize_image_path(checkpoint_path)
    image_path = normalize_image_path(image_path)
    output_path = normalize_image_path(output_path)

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Any = create_model(cfg.model.name, cfg.data.num_classes, pretrained=False, drop_rate=cfg.model.drop_rate)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    rgb = crop_to_retina(load_rgb_image(image_path))
    image = Image.fromarray(rgb)
    transform: Any = build_transforms(cfg.train.image_size, train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    activations = []
    gradients = []
    target_layer = _last_conv_layer(model)

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    try:
        logits = model(tensor)
        predicted = int(logits.argmax(dim=1).item())
        target = predicted if class_id is None else class_id
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()

        acts = activations[-1][0]
        grads = gradients[-1][0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
    finally:
        handle_f.remove()
        handle_b.remove()

    heatmap = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heatmap = (255 * heatmap).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * rgb + 0.45 * heatmap).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    return {"predicted_class": predicted, "target_class": target, "output_path": str(output_path)}

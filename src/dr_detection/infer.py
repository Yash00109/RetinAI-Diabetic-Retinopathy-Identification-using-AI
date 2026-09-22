from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image

from dr_detection.config import load_config
from dr_detection.gradcam import make_gradcam
from dr_detection.models import create_model
from dr_detection.quality import (
    QualityThresholds,
    analyze_fundus_quality,
    crop_to_retina,
    load_rgb_image,
)
from dr_detection.transforms import build_transforms

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

DEFAULT_OPTIMAL_THRESHOLDS = [0.5842, 1.4821, 2.4783, 3.4610]

CLINICAL_RECOMMENDATIONS = {
    0: {
        "severity": "Normal",
        "color": "#10B981", # Green
        "referable": False,
        "urgency": "Routine",
        "action": "No apparent retinopathy detected. Annual routine screening recommended."
    },
    1: {
        "severity": "Mild NPDR",
        "color": "#F59E0B", # Amber
        "referable": False,
        "urgency": "Low-Moderate",
        "action": "Microaneurysms detected. Comprehensive dilated eye exam in 6–12 months."
    },
    2: {
        "severity": "Moderate NPDR",
        "color": "#F97316", # Orange
        "referable": True,
        "urgency": "Moderate",
        "action": "Referable DR. Referral to Ophthalmologist/Retinal Specialist within 3–6 months."
    },
    3: {
        "severity": "Severe NPDR",
        "color": "#EF4444", # Red
        "referable": True,
        "urgency": "High",
        "action": "High-risk pre-proliferative retinopathy. Urgent specialist referral within 2–4 weeks."
    },
    4: {
        "severity": "Proliferative DR",
        "color": "#991B1B", # Dark Red
        "referable": True,
        "urgency": "Critical",
        "action": "Neovascularization / severe lesions. Emergency retinal intervention within 24–48 hours."
    }
}

# In-memory model cache for fast serving
_LOADED_MODELS: dict[str, Any] = {}


def get_cached_model(config_path: str | Path, checkpoint_path: str | Path, device):
    import torch
    cache_key = f"{config_path}:{checkpoint_path}:{device.type}"
    if cache_key in _LOADED_MODELS:
        return _LOADED_MODELS[cache_key]

    cfg = load_config(config_path)
    model = create_model(cfg.model.name, cfg.data.num_classes, pretrained=False, drop_rate=cfg.model.drop_rate)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _LOADED_MODELS[cache_key] = (model, cfg)
    return model, cfg


def predict_image(
    config_path: str | Path = "configs/efficientnet_b0.json",
    checkpoint_path: str | Path = "artifacts/smoke_test/best_model.pt",
    image_path: str | Path = "",
    use_tta: bool = True,
    use_threshold_optimization: bool = True,
    thresholds: Optional[List[float]] = None,
    generate_gradcam: bool = False,
    gradcam_output_path: Optional[str | Path] = None,
) -> dict:
    """Enterprise inference pipeline with Quality Gate, TTA, Cohen's Kappa Threshold Optimization,
    Referable DR risk scoring, and optional Grad-CAM explainability.
    """
    import torch

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Checkpoint check
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = get_cached_model(config_path, checkpoint_path, device)

    # 1. Quality Gate Analysis
    quality_thresholds = QualityThresholds(**cfg.quality.__dict__)
    quality = analyze_fundus_quality(image_path, quality_thresholds)
    if not quality.accepted:
        return {
            "accepted": False,
            "quality": quality.as_dict(),
            "prediction": None,
            "clinical_recommendation": {
                "urgency": "Retake Required",
                "action": f"Image rejected by quality audit: {', '.join(quality.rejection_reasons)}. Please acquire a new clear, focused, centered fundus photograph."
            }
        }

    # 2. Preprocessing & Tensor Building
    rgb = crop_to_retina(load_rgb_image(image_path))
    image = Image.fromarray(rgb)
    transform = build_transforms(cfg.train.image_size, train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    # 3. Model Inference (with optional TTA)
    with torch.no_grad():
        logits = model(tensor)
        if use_tta:
            # 2-fold horizontal flip TTA
            logits_hflip = model(torch.flip(tensor, dims=[3]))
            logits = (logits + logits_hflip) / 2.0
            
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # 4. Continuous Expected Score & Decision Boundary
    expected_score = float(np.sum(probs * np.arange(len(probs))))
    
    if use_threshold_optimization:
        active_thresholds = thresholds or DEFAULT_OPTIMAL_THRESHOLDS
        pred = int(np.digitize(expected_score, sorted(active_thresholds)))
    else:
        pred = int(probs.argmax())

    # 5. Referable DR Risk Score: P(Grade >= 2)
    referable_probability = float(np.sum(probs[2:]))
    is_referable = bool(pred >= 2 or referable_probability >= 0.50)

    recommendation = CLINICAL_RECOMMENDATIONS.get(pred, CLINICAL_RECOMMENDATIONS[0])

    response = {
        "accepted": True,
        "quality": quality.as_dict(),
        "prediction": {
            "class_id": pred,
            "class_name": CLASS_NAMES.get(pred, str(pred)),
            "confidence": float(probs[pred]),
            "expected_continuous_score": round(expected_score, 4),
            "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "referable_dr": {
                "is_referable": is_referable,
                "probability": round(referable_probability, 4),
                "threshold": 0.50
            },
            "decision_rule": "Nelder-Mead Optimal Cutoffs" if use_threshold_optimization else "Argmax"
        },
        "clinical_recommendation": recommendation,
    }

    # 6. Optional Grad-CAM Heatmap Generation
    if generate_gradcam:
        if gradcam_output_path is None:
            gradcam_output_path = image_path.parent / f"{image_path.stem}_gradcam.png"
        cam_result = make_gradcam(config_path, checkpoint_path, image_path, gradcam_output_path, class_id=pred)
        response["gradcam"] = cam_result

    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinAI-DR Production Inference Engine")
    parser.add_argument("--config", default="configs/efficientnet_b0.json")
    parser.add_argument("--checkpoint", default="artifacts/smoke_test/best_model.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--gradcam-output", default=None)
    args = parser.parse_args()

    result = predict_image(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        generate_gradcam=args.gradcam,
        gradcam_output_path=args.gradcam_output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

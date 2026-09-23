import base64
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure 'src' is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dr_detection.infer import predict_image

DEFAULT_CONFIG = "configs/efficientnet_b0.json"
DEFAULT_CHECKPOINT = "artifacts/smoke_test/best_model.pt"


def get_prediction(
    image_path: str,
    config_path: str = DEFAULT_CONFIG,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    generate_gradcam: bool = False,
    gradcam_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Adapter function calling the enterprise dr_detection.infer.predict_image pipeline."""
    return predict_image(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        use_tta=True,
        use_threshold_optimization=True,
        generate_gradcam=generate_gradcam,
        gradcam_output_path=gradcam_output_path,
    )


def get_gradcam_base64(image_path: str, config_path: str = DEFAULT_CONFIG, checkpoint_path: str = DEFAULT_CHECKPOINT) -> Dict[str, Any]:
    """Runs inference and generates a base64-encoded Grad-CAM overlay for frontend rendering."""
    temp_cam = Path(image_path).parent / f"api_gradcam_{Path(image_path).stem}.png"
    result = predict_image(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        generate_gradcam=True,
        gradcam_output_path=temp_cam,
    )
    if not result["accepted"]:
        return {"accepted": False, "error": "Image rejected by quality gate", "quality": result["quality"]}

    cam_path = Path(result["gradcam"]["output_path"])
    b64_data = ""
    if cam_path.exists():
        b64_data = base64.b64encode(cam_path.read_bytes()).decode("utf-8")
        try:
            cam_path.unlink() # cleanup temp
        except Exception:
            pass

    return {
        "accepted": True,
        "predicted_class": result["prediction"]["class_id"],
        "class_name": result["prediction"]["class_name"],
        "gradcam_base64": b64_data,
    }

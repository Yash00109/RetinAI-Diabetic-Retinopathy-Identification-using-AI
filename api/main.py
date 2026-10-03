import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root and 'src' directory are in sys.path for cloud deployments (Render/Docker)
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from api.services.prediction import get_gradcam_base64, get_prediction, get_model_info

app = FastAPI(
    title="RetinAI-DR API",
    description="Research prototype API for Diabetic Retinopathy classification from retinal fundus images.",
    version="0.2.0-research"
)

# Allow CORS for client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "status": "online",
        "version": "0.2.0-research",
        "device": device_name,
        "disclaimer": "Research prototype — not for clinical use",
    }


@app.get("/model-info", response_model=ModelInfoResponse)
@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Returns metadata about the currently loaded model and its verified metrics."""
    return get_model_info()


@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    gradcam: bool = Query(False, description="Whether to compute and attach Grad-CAM heatmap"),
):
    ext = Path(file.filename or "image.png").suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format. Please upload PNG, JPG, or JPEG.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        result = get_prediction(temp_path, generate_gradcam=gradcam)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}") from e
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/explain", response_model=ExplainResponse)
@app.post("/api/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)):
    ext = Path(file.filename or "image.png").suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        explain_result = get_gradcam_base64(temp_path)
        if not explain_result["accepted"]:
            raise HTTPException(status_code=422, detail=explain_result.get("error", "Quality check failed"))
        return explain_result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM Generation Error: {str(e)}") from e
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

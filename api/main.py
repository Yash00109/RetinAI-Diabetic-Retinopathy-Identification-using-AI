import json
import logging
import os
import sys
import tempfile
import warnings
from io import BytesIO
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
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from api.schemas import (
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from api.services.prediction import get_gradcam_base64, get_prediction, get_model_info

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501,http://127.0.0.1:8501"


async def _read_validated_image(file: UploadFile) -> tuple[bytes, str]:
    """Read a bounded upload and verify it is a decodable supported image."""
    suffix = Path(file.filename or "image.png").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    chunks = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
        chunks.append(chunk)
    content = b"".join(chunks)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG", "TIFF", "BMP"}:
                    raise HTTPException(status_code=400, detail="Unsupported image format.")
                image.verify()
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc
    return content, suffix

app = FastAPI(
    title="RetinAI-DR API",
    description="Research prototype API for Diabetic Retinopathy classification from retinal fundus images.",
    version="0.2.0-research"
)

# Configure browser origins explicitly for deployed clients. Credentials are
# disabled because this API does not use cookie-based authentication.
cors_origins = [
    origin.strip()
    for origin in os.getenv("RETINAI_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
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
    temp_path = None
    try:
        content, ext = await _read_validated_image(file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        result = await run_in_threadpool(get_prediction, temp_path, generate_gradcam=gradcam)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs for details.") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/explain", response_model=ExplainResponse)
@app.post("/api/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)):
    temp_path = None
    try:
        content, ext = await _read_validated_image(file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        explain_result = await run_in_threadpool(get_gradcam_base64, temp_path)
        if not explain_result["accepted"]:
            raise HTTPException(status_code=422, detail=explain_result.get("error", "Quality check failed"))
        return explain_result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Grad-CAM request failed")
        raise HTTPException(status_code=500, detail="Grad-CAM generation failed. Check server logs for details.") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

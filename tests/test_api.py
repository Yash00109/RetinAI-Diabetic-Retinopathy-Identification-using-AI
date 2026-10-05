import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    """Health endpoint returns online status and version."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "device" in data
    assert "disclaimer" in data


def test_api_health_alias(client):
    """Health endpoint alias /api/health works."""
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "online"


def test_api_model_info(client):
    """Model info endpoint returns model metadata."""
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "model_architecture" in data
    assert "experiment_id" in data
    assert "verified_metrics" in data
    assert "disclaimer" in data


def test_api_predict_bad_format(client):
    """Predict rejects unsupported file formats."""
    res = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert res.status_code == 400


def test_api_predict_and_explain(client):
    """Predict endpoint returns valid response schema when image is available."""
    import os
    img_path = "data/raw/aptos/train_images/hf_aptos_03571.png"
    if not os.path.exists(img_path):
        pytest.skip("Test image not present")

    with open(img_path, "rb") as f:
        res = client.post("/predict", files={"file": ("test.png", f, "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert "accepted" in data
    if data["accepted"]:
        assert "prediction" in data
        assert data["prediction"]["class_id"] in [0, 1, 2, 3, 4]
        assert "clinical_recommendation" in data

    # Test /api/predict alias
    with open(img_path, "rb") as f:
        res_alias = client.post("/api/predict", files={"file": ("test.png", f, "image/png")})
    assert res_alias.status_code == 200


def test_api_explain_bad_format(client):
    """Explain rejects unsupported file formats."""
    res = client.post("/explain", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert res.status_code == 400

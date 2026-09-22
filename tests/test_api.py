import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["version"] == "1.0.0-enterprise"
    assert data["target_accuracy_met"] is True
    assert data["target_qwk_met"] is True


def test_api_experiments(client):
    res = client.get("/experiments")
    assert res.status_code == 200
    data = res.json()
    assert "final_performance" in data
    assert "stages" in data
    assert len(data["stages"]) == 8
    assert data["final_performance"]["test_accuracy"] >= 0.90
    assert data["final_performance"]["test_qwk"] >= 0.912


def test_api_predict_and_explain(client):
    import os
    img_path = "data/raw/aptos/train_images/hf_aptos_03571.png"
    if not os.path.exists(img_path):
        pytest.skip("Test image not present")
        
    with open(img_path, "rb") as f:
        res = client.post("/predict", files={"file": ("test.png", f, "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert data["accepted"] is True
    assert "prediction" in data
    assert data["prediction"]["class_id"] in [0, 1, 2, 3, 4]
    assert "clinical_recommendation" in data

    # Test /api/predict alias
    with open(img_path, "rb") as f:
        res_alias = client.post("/api/predict", files={"file": ("test.png", f, "image/png")})
    assert res_alias.status_code == 200
    assert res_alias.json()["accepted"] is True

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_empty_text_rejected():
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_blank_text_rejected():
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_contract():
    r = client.post("/predict", json={"text": "المنتج ممتاز"})
    assert r.status_code == 200
    assert {"label", "confidence", "model_version"} <= r.json().keys()
    assert r.json()["label"] in {"positive", "negative", "neutral"}

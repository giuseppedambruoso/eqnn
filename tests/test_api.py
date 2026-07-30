import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def checkpoint_path(tmp_path):
    num_qubits, reps = 4, 1
    params = torch.empty(num_qubits * reps).uniform_(-0.1, 0.1)
    path = tmp_path / "final_model.pt"
    torch.save(
        {
            "params": params,
            "val_acc": 0.9,
            "config": {
                "device": "default.qubit",
                "num_qubits": num_qubits,
                "p_err": 0.0,
                "reps": reps,
                "architecture": "config1",
                "img_size": 4,
            },
        },
        path,
    )
    return path


@pytest.fixture()
def client(monkeypatch, checkpoint_path):
    monkeypatch.setenv("MODEL_PATH", str(checkpoint_path))
    from src.api import app

    with TestClient(app) as test_client:
        yield test_client


def _fake_png_bytes() -> bytes:
    image = Image.new("L", (8, 8), color=128)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["classes"] == [3, 4]
    assert body["img_size"] == 4
    assert body["architecture"] == "config1"
    assert body["val_acc"] == pytest.approx(0.9)


def test_predict_returns_valid_response(client):
    files = {"file": ("digit.png", _fake_png_bytes(), "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_digit"] in (3, 4)
    assert 0.0 <= body["probability_digit_4"] <= 1.0


def test_predict_rejects_invalid_file(client):
    files = {"file": ("not_an_image.txt", b"hello world", "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400

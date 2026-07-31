"""FastAPI inference service for trained EQNN checkpoints.

Serves a single binary classifier (MNIST digit "3" vs "4") loaded from a
self-contained `final_model.pt` checkpoint produced by `src.train.train_loop`
(weights + circuit hyperparameters, so no Hydra config lookup is needed).
"""

import io
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from src.ansatz_builder import build_qnn_from_spec
from src.data_encoding import embedding_unitary
from src.data_loading import L2Normalize
from src.qnn import create_qnn

logger = logging.getLogger(__name__)

_MODEL: dict[str, Any] = {}


class PredictResponse(BaseModel):
    predicted_digit: int
    probability_digit_4: float
    raw_expectation: float


class ModelInfoResponse(BaseModel):
    path: str
    val_acc: float | None
    img_size: int
    architecture: str
    classes: list[int]


def _find_default_checkpoint() -> Path | None:
    candidates = [
        *Path("outputs").glob("*/*/final_model.pt"),
        *Path("multirun").glob("*/*/*/final_model.pt"),
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(model_path: str | None = None) -> None:
    path = Path(model_path) if model_path else _find_default_checkpoint()
    if path is None or not path.exists():
        raise FileNotFoundError(
            "No model checkpoint found. Set MODEL_PATH to a final_model.pt file "
            "(e.g. outputs/<date>/<time>/final_model.pt), or train one first."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise ValueError(
            f"Checkpoint {path} has no embedded 'config' — it was trained with an "
            "older version of the pipeline. Re-train to get a self-contained checkpoint."
        )

    cfg = checkpoint["config"]
    if "circuit_spec" in cfg:
        qnn, _, _ = build_qnn_from_spec(
            cfg["device"], cfg["num_qubits"], cfg["p_err"], cfg["circuit_spec"]
        )
        architecture_label = "custom"
    else:
        architecture = cfg.get("architecture", "config1")
        qnn = create_qnn(
            cfg["device"], cfg["num_qubits"], cfg["p_err"], cfg["reps"], architecture
        )
        architecture_label = architecture

    _MODEL["qnn"] = qnn
    _MODEL["params"] = checkpoint["params"]
    _MODEL["phi"] = torch.tensor(0.0)
    _MODEL["img_size"] = cfg["img_size"]
    _MODEL["architecture"] = architecture_label
    _MODEL["val_acc"] = checkpoint.get("val_acc")
    _MODEL["path"] = str(path)
    logger.info(f"Loaded model from {path} (val_acc={_MODEL['val_acc']})")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_model(os.environ.get("MODEL_PATH"))
    yield
    _MODEL.clear()


app = FastAPI(
    title="EQNN Inference API",
    description="Binary classifier (digit 3 vs 4) served from a trained equivariant QNN checkpoint.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "qnn" in _MODEL}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if "qnn" not in _MODEL:
        raise HTTPException(503, "Model not loaded")
    return ModelInfoResponse(
        path=_MODEL["path"],
        val_acc=_MODEL["val_acc"],
        img_size=_MODEL["img_size"],
        architecture=_MODEL["architecture"],
        classes=[3, 4],
    )


def _preprocess(image_bytes: bytes, img_size: int) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            L2Normalize(),
            transforms.Lambda(lambda x: x.squeeze(0)),
        ]
    )
    return tfm(image)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:  # noqa: B008
    if "qnn" not in _MODEL:
        raise HTTPException(503, "Model not loaded")

    contents = await file.read()
    try:
        image = _preprocess(contents, _MODEL["img_size"])
    except Exception as exc:
        raise HTTPException(400, f"Could not process image: {exc}") from exc

    unitary = embedding_unitary(image)
    with torch.no_grad():
        raw = _MODEL["qnn"](unitary, _MODEL["params"], _MODEL["phi"])

    probability = float(torch.clamp((1.0 + raw) / 2.0, min=1e-7, max=1.0 - 1e-7))
    predicted_digit = 4 if probability > 0.5 else 3
    return PredictResponse(
        predicted_digit=predicted_digit,
        probability_digit_4=probability,
        raw_expectation=float(raw),
    )

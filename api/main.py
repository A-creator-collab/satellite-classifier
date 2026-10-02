from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.data import load_config
from src.model import load_checkpoint
from src.predict import predict_probabilities

app = FastAPI(title="satellite-classifier")
_state: dict[str, Any] = {}


def get_state() -> dict[str, Any]:
    if not _state:
        cfg = load_config("config.yaml")
        checkpoint = Path(cfg["paths"]["checkpoint"])
        if not checkpoint.exists():
            raise HTTPException(status_code=503, detail=f"Checkpoint not found: {checkpoint}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, class_names = load_checkpoint(checkpoint, device)
        _state.update(cfg=cfg, device=device, model=model, class_names=class_names)
    return _state


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    state = get_state()
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    probs = predict_probabilities(
        state["model"], state["class_names"], image, state["cfg"], state["device"]
    )
    return {"prediction": max(probs, key=probs.get), "probabilities": probs}

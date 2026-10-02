from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data import load_config  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.predict import predict_probabilities  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_resources():
    cfg = load_config(ROOT / "config.yaml")
    checkpoint = ROOT / cfg["paths"]["checkpoint"]
    if not checkpoint.exists():
        return cfg, None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(checkpoint, device)
    return cfg, model, class_names, device


st.title("Satellite image classifier")
cfg, model, class_names, device = load_resources()

if model is None:
    st.error("Checkpoint not found. Run `python -m src.train --config config.yaml` first.")
    st.stop()

upload = st.file_uploader("Upload a Sentinel-2 RGB patch", type=["jpg", "jpeg", "png", "tif"])
if upload is not None:
    image = Image.open(upload)
    st.image(image, caption="Input", width=256)
    probs = predict_probabilities(model, class_names, image, cfg, device)
    top = max(probs, key=probs.get)
    st.subheader(f"Prediction: {top} ({probs[top]:.1%})")
    st.bar_chart(probs)

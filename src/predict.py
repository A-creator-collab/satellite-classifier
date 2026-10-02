from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from src.data import get_eval_transform, load_config
from src.model import load_checkpoint


def preprocess(image: Image.Image, cfg: dict[str, Any]) -> torch.Tensor:
    return get_eval_transform(cfg)(image.convert("RGB")).unsqueeze(0)


def predict_probabilities(
    model: nn.Module,
    class_names: list[str],
    image: Image.Image,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    x = preprocess(image, cfg).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    return dict(zip(class_names, probs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(Path(cfg["paths"]["checkpoint"]), device)
    with Image.open(args.image) as img:
        probs = predict_probabilities(model, class_names, img, cfg, device)
    top = max(probs, key=probs.get)
    print(json.dumps({"prediction": top, "probabilities": probs}, indent=2))


if __name__ == "__main__":
    main()

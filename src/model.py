from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def save_checkpoint(
    model: nn.Module, path: Path, class_names: list[str], cfg: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "input_size": cfg["data"]["input_size"],
            "mean": cfg["data"]["mean"],
            "std": cfg["data"]["std"],
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["class_names"]

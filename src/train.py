from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import build_data, compute_class_weights, load_config, set_seed
from src.model import build_model, save_checkpoint, set_backbone_trainable


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def run_phase(
    name: str,
    model: nn.Module,
    params: list[nn.Parameter],
    lr: float,
    epochs: int,
    data: Any,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict[str, Any],
    state: dict[str, Any],
) -> None:
    optimizer = AdamW(params, lr=lr, weight_decay=cfg["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch(
            model, data.train_loader, criterion, device, optimizer, scaler, use_amp
        )
        va_loss, va_acc = run_epoch(model, data.val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()
        state["history"].append(
            {
                "phase": name,
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
            }
        )
        print(f"{name} epoch {epoch + 1}/{epochs} val_acc={va_acc:.4f} val_loss={va_loss:.4f}")
        if va_acc > state["best_val_acc"]:
            state["best_val_acc"] = va_acc
            save_checkpoint(model, Path(cfg["paths"]["checkpoint"]), data.class_names, cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    outputs = Path(cfg["paths"]["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_data(cfg)
    num_classes = len(data.class_names)
    weights = compute_class_weights(data.train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model(num_classes, cfg["model"]["pretrained"]).to(device)
    state: dict[str, Any] = {"best_val_acc": -1.0, "history": []}
    t = cfg["train"]

    set_backbone_trainable(model, False)
    head_params = [p for p in model.parameters() if p.requires_grad]
    run_phase("head", model, head_params, t["lr_head"], t["epochs_head"],
              data, criterion, device, cfg, state)

    set_backbone_trainable(model, True)
    run_phase("finetune", model, list(model.parameters()),
              t["lr_head"] * t["finetune_lr_factor"], t["epochs_finetune"],
              data, criterion, device, cfg, state)

    (outputs / "history.json").write_text(json.dumps(state["history"], indent=2))
    print(f"best val_acc={state['best_val_acc']:.4f} checkpoint={cfg['paths']['checkpoint']}")


if __name__ == "__main__":
    main()

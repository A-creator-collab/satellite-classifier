from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from src.data import build_data, load_config, set_seed  # noqa: E402
from src.model import load_checkpoint  # noqa: E402


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class_f1": {n: float(v) for n, v in zip(class_names, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(Path(cfg["paths"]["checkpoint"]), device)
    data = build_data(cfg)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in data.test_loader:
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            targets.append(y.numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(targets)

    metrics = compute_metrics(y_true, y_pred, class_names)
    outputs = Path(cfg["paths"]["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), class_names,
                          outputs / "confusion_matrix.png")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()

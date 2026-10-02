from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from src.data import build_data, load_config, set_seed  # noqa: E402
from src.model import load_checkpoint  # noqa: E402


def compute_gradcam(
    model: nn.Module, x: torch.Tensor, target_class: int | None = None
) -> tuple[torch.Tensor, int]:
    """Return a heatmap in [0, 1] with shape (H, W) and the class it explains.

    x has shape (1, 3, H, W). The target layer is the last residual stage of ResNet-18.
    """
    store: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        store["activations"] = out
        out.register_hook(lambda grad: store.__setitem__("gradients", grad))

    handle = model.layer4.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(x)
            cls = int(logits.argmax(1).item()) if target_class is None else target_class
            logits[0, cls].backward()
    finally:
        handle.remove()

    weights = store["gradients"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["activations"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.detach().cpu(), cls


def denormalize(x: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(Path(cfg["paths"]["checkpoint"]), device)
    data = build_data(cfg)

    out_dir = Path(cfg["paths"]["outputs_dir"]) / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(cfg["gradcam"]["num_samples"], len(data.test_dataset))
    for i in range(n):
        x, label = data.test_dataset[i]
        cam, pred = compute_gradcam(model, x.unsqueeze(0).to(device))
        image = denormalize(x, cfg["data"]["mean"], cfg["data"]["std"]).permute(1, 2, 0)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(image.numpy())
        axes[0].set_title(f"true: {class_names[label]}", fontsize=8)
        axes[1].imshow(image.numpy())
        axes[1].imshow(cam.numpy(), cmap="jet", alpha=0.45)
        axes[1].set_title(f"pred: {class_names[pred]}", fontsize=8)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:03d}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {n} overlays to {out_dir}")


if __name__ == "__main__":
    main()

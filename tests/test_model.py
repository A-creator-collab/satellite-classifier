import torch

from src.model import build_model, set_backbone_trainable


def test_forward_shape_cpu() -> None:
    model = build_model(num_classes=10, pretrained=False).eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_freeze_backbone_leaves_head_trainable() -> None:
    model = build_model(num_classes=10, pretrained=False)
    set_backbone_trainable(model, False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("fc.") for n in trainable)
    set_backbone_trainable(model, True)
    assert all(p.requires_grad for p in model.parameters())

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import (
    SatelliteDataset,
    compute_class_weights,
    find_dataset_root,
    get_eval_transform,
    list_samples,
    make_splits,
)

CFG = {"data": {"input_size": 128, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}}


def make_fake_dataset(root: Path, per_class: int = 10) -> Path:
    for c in range(10):
        class_dir = root / "images" / f"class_{c}"
        class_dir.mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (64, 64), color=(c * 10, i * 5, 100)).save(class_dir / f"{i}.jpg")
    return root


def test_find_dataset_root_and_list_samples(tmp_path: Path) -> None:
    make_fake_dataset(tmp_path)
    root = find_dataset_root(tmp_path)
    assert root == tmp_path / "images"
    samples, class_names = list_samples(root)
    assert len(samples) == 100
    assert len(class_names) == 10


def test_dataset_item_shape(tmp_path: Path) -> None:
    make_fake_dataset(tmp_path, per_class=2)
    samples, _ = list_samples(tmp_path / "images")
    ds = SatelliteDataset(samples, get_eval_transform(CFG))
    x, y = ds[0]
    assert x.shape == (3, 128, 128)
    assert isinstance(y, int)


def test_splits_are_disjoint_and_stratified() -> None:
    labels = np.repeat(np.arange(10), 20)
    splits = make_splits(labels, 0.15, 0.15, seed=0)
    all_idx = splits["train"] + splits["val"] + splits["test"]
    assert len(all_idx) == len(set(all_idx)) == len(labels)
    assert set(np.unique(labels[splits["test"]])) == set(range(10))


def test_class_weights_use_given_labels_only() -> None:
    train_labels = np.array([0] * 8 + [1] * 2)
    w = compute_class_weights(train_labels, num_classes=2)
    assert isinstance(w, torch.Tensor)
    assert w[1] > w[0]
    assert torch.isclose(w[0], torch.tensor(10 / (2 * 8)))

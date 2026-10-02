from __future__ import annotations

import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif"}
NUM_CLASSES = 10


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def find_dataset_root(data_dir: Path) -> Path | None:
    """Return the first directory under data_dir that has 10 class subfolders with images."""
    if not data_dir.exists():
        return None
    for path in sorted([data_dir, *data_dir.rglob("*")]):
        if not path.is_dir():
            continue
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if len(subdirs) != NUM_CLASSES:
            continue
        if all(any(f.suffix.lower() in IMAGE_SUFFIXES for f in d.iterdir()) for d in subdirs):
            return path
    return None


def download_dataset(cfg: dict[str, Any]) -> Path:
    data_dir = Path(cfg["paths"]["data_dir"])
    root = find_dataset_root(data_dir)
    if root is not None:
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "EuroSAT_RGB.zip"
    if not zip_path.exists():
        with requests.get(cfg["data"]["url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    root = find_dataset_root(data_dir)
    if root is None:
        raise RuntimeError("TODO: verify dataset URL and archive layout under data/")
    return root


def get_eval_transform(cfg: dict[str, Any]) -> Callable[[Image.Image], torch.Tensor]:
    d = cfg["data"]
    return transforms.Compose(
        [
            transforms.Resize((d["input_size"], d["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(d["mean"], d["std"]),
        ]
    )


def get_train_transform(cfg: dict[str, Any]) -> Callable[[Image.Image], torch.Tensor]:
    d = cfg["data"]
    return transforms.Compose(
        [
            transforms.Resize((d["input_size"], d["input_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(d["mean"], d["std"]),
        ]
    )


def list_samples(root: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    samples: list[tuple[Path, int]] = []
    for idx, name in enumerate(class_names):
        files = sorted(f for f in (root / name).iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        samples.extend((f, idx) for f in files)
    return samples, class_names


class SatelliteDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label


def make_splits(
    labels: np.ndarray, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[int]]:
    idx = np.arange(len(labels))
    train_val, test = train_test_split(
        idx, test_size=test_fraction, stratify=labels, random_state=seed
    )
    rel_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=labels[train_val], random_state=seed
    )
    return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


def compute_class_weights(train_labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    test_dataset: SatelliteDataset
    class_names: list[str]
    train_labels: np.ndarray


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_data(cfg: dict[str, Any]) -> DataBundle:
    import json

    root = download_dataset(cfg)
    samples, class_names = list_samples(root)
    labels = np.array([s[1] for s in samples])

    d = cfg["data"]
    splits = make_splits(labels, d["val_fraction"], d["test_fraction"], cfg["seed"])
    splits_path = Path(cfg["paths"]["splits"])
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits))

    train_tf, eval_tf = get_train_transform(cfg), get_eval_transform(cfg)
    train_ds = SatelliteDataset([samples[i] for i in splits["train"]], train_tf)
    val_ds = SatelliteDataset([samples[i] for i in splits["val"]], eval_tf)
    test_ds = SatelliteDataset([samples[i] for i in splits["test"]], eval_tf)

    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    bs, nw = cfg["train"]["batch_size"], d["num_workers"]
    common = {"batch_size": bs, "num_workers": nw, "worker_init_fn": _seed_worker}
    return DataBundle(
        train_loader=DataLoader(train_ds, shuffle=True, generator=g, **common),
        val_loader=DataLoader(val_ds, shuffle=False, **common),
        test_loader=DataLoader(test_ds, shuffle=False, **common),
        test_dataset=test_ds,
        class_names=class_names,
        train_labels=labels[splits["train"]],
    )

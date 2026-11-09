import gzip
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

DATA_DIR = "dados"
RAW_DIR = f"{DATA_DIR}/FashionMNIST/raw"
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
SEED = 42
VAL_SIZE = 10000


def ensure_data() -> None:
    if all(os.path.exists(f"{RAW_DIR}/{name}") for name in FILES.values()):
        return
    from torchvision.datasets import FashionMNIST
    FashionMNIST(DATA_DIR, train=True, download=True)
    FashionMNIST(DATA_DIR, train=False, download=True)


def _read_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, 28 * 28)


def _read_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8, offset=8)


def _to_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(images.astype(np.float32) / 255.0).to(device)


def load_data(device: torch.device) -> dict:
    ensure_data()
    x_full = _read_images(f"{RAW_DIR}/{FILES['x_train']}")
    y_full = _read_labels(f"{RAW_DIR}/{FILES['y_train']}")
    x_test = _read_images(f"{RAW_DIR}/{FILES['x_test']}")
    y_test = _read_labels(f"{RAW_DIR}/{FILES['y_test']}")

    idx_train, idx_val = train_test_split(np.arange(len(y_full)), test_size=VAL_SIZE, stratify=y_full,
                                          random_state=SEED)
    return {
        "x_train": _to_tensor(x_full[idx_train], device), "y_train": y_full[idx_train].copy(),
        "x_val": _to_tensor(x_full[idx_val], device), "y_val": y_full[idx_val].copy(),
        "x_test": _to_tensor(x_test, device), "y_test": y_test.copy(),
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

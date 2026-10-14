import os

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def results_dir(image_path: str, root: str = "resultados") -> str:
    return f"{root}/{os.path.splitext(os.path.basename(image_path))[0]}"


def list_images(folder: str = "imagens") -> list:
    return sorted(f"{folder}/{name}" for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTENSIONS))


def load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def image_to_pixels(image: np.ndarray) -> np.ndarray:
    return image.reshape(-1, 3).astype(np.float64) / 255.0


def pixels_to_image(pixels: np.ndarray, shape: tuple) -> np.ndarray:
    return np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8).reshape(shape)

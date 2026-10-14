import numpy as np

from imagem import pixels_to_image


def nearest_units(pixels: np.ndarray, weights: np.ndarray, k: int = 2, chunk: int = 20000) -> np.ndarray:
    weights_sq = (weights ** 2).sum(axis=1)
    nearest = np.empty((pixels.shape[0], k), dtype=np.int64)
    for start in range(0, pixels.shape[0], chunk):
        block = pixels[start:start + chunk]
        dist = (block ** 2).sum(axis=1)[:, None] - 2.0 * block @ weights.T + weights_sq[None, :]
        candidates = np.argpartition(dist, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(dist, candidates, axis=1), axis=1)
        nearest[start:start + chunk] = np.take_along_axis(candidates, order, axis=1)
    return nearest


def quantization_error(pixels: np.ndarray, weights: np.ndarray, first: np.ndarray) -> float:
    return float(np.linalg.norm(pixels - weights[first], axis=1).mean())


def evaluate(model, pixels: np.ndarray, image: np.ndarray) -> tuple:
    nearest = nearest_units(pixels, model.weights, k=2)
    first, second = nearest[:, 0], nearest[:, 1]
    quantized = pixels_to_image(model.weights[first], image.shape)
    metrics = {
        "qe": quantization_error(pixels, model.weights, first),
        "te": model.topographic_error(first, second),
        "colors_used": int(np.unique(first).size),
    }
    return metrics, quantized

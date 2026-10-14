import numpy as np

from metricas import nearest_units, quantization_error


class SOM:
    def __init__(self, rows: int, cols: int, alpha0: float = 0.5, alpha_f: float = 0.01,
                 sigma0: float = None, sigma_f: float = 0.5, iterations: int = 100000, seed: int = 42):
        self.rows = rows
        self.cols = cols
        self.n_units = rows * cols
        self.alpha0 = alpha0
        self.alpha_f = alpha_f
        self.sigma0 = sigma0 if sigma0 is not None else max(rows, cols) / 2
        self.sigma_f = sigma_f
        self.iterations = iterations
        self.seed = seed

        coords = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=np.float64)
        diff = coords[:, None, :] - coords[None, :, :]
        self.grid_d2 = (diff ** 2).sum(axis=2)

        self.weights = None
        self.history = []

    def fit(self, pixels: np.ndarray, monitor_pixels: np.ndarray = None, checkpoints: int = 50) -> "SOM":
        rng = np.random.default_rng(self.seed)
        n = pixels.shape[0]
        weights = pixels[rng.choice(n, self.n_units, replace=False)].copy()
        samples = pixels[rng.integers(0, n, self.iterations)]

        alpha_ratio = self.alpha_f / self.alpha0
        sigma_ratio = self.sigma_f / self.sigma0
        every = max(1, self.iterations // checkpoints)
        self.history = []

        for t in range(self.iterations):
            frac = t / self.iterations
            alpha = self.alpha0 * alpha_ratio ** frac
            sigma = self.sigma0 * sigma_ratio ** frac

            delta = weights - samples[t]
            bmu = np.einsum("ij,ij->i", delta, delta).argmin()
            h = np.exp(-self.grid_d2[bmu] / (2.0 * sigma * sigma))
            weights -= (alpha * h)[:, None] * delta

            if monitor_pixels is not None and t % every == 0:
                self.history.append((t, self._monitor(weights, monitor_pixels)))

        self.weights = weights
        if monitor_pixels is not None:
            self.history.append((self.iterations, self._monitor(weights, monitor_pixels)))
        return self

    def topographic_error(self, first: np.ndarray, second: np.ndarray) -> float:
        r1, c1 = np.divmod(first, self.cols)
        r2, c2 = np.divmod(second, self.cols)
        adjacent = np.maximum(np.abs(r1 - r2), np.abs(c1 - c2)) <= 1
        return float(1.0 - adjacent.mean())

    @staticmethod
    def _monitor(weights: np.ndarray, monitor_pixels: np.ndarray) -> float:
        first = nearest_units(monitor_pixels, weights, k=1)[:, 0]
        return quantization_error(monitor_pixels, weights, first)

import numpy as np

from metricas import nearest_units, quantization_error


class GNG:
    def __init__(self, max_units: int, eps_b: float = 0.2, eps_n: float = 0.006, age_max: int = 50,
                 lam: int = 100, alpha: float = 0.5, decay: float = 0.995,
                 tune_iterations: int = 100000, seed: int = 42):
        self.max_units = max_units
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.age_max = age_max
        self.lam = lam
        self.alpha = alpha
        self.decay = decay
        self.tune_iterations = tune_iterations
        self.seed = seed

        self.n_units = 0
        self.iterations = 0
        self.grown_at = None
        self.weights = None
        self.edges = None
        self.history = []

    def fit(self, pixels: np.ndarray, monitor_pixels: np.ndarray = None, monitor_every: int = 1000) -> "GNG":
        rng = np.random.default_rng(self.seed)
        n = pixels.shape[0]
        cap = self.max_units

        W = np.zeros((cap, 3))
        err = np.zeros(cap)
        age = np.full((cap, cap), -1, dtype=np.int32)
        mask = np.full(cap, np.inf)

        start = rng.choice(n, 2, replace=False)
        W[0], W[1] = pixels[start[0]], pixels[start[1]]
        mask[:2] = 0.0
        age[0, 1] = age[1, 0] = 0
        n_active = 2

        block_size = 100000
        block = None
        frozen = False
        frozen_at = 0
        limit = self.lam * cap * 4 + self.tune_iterations
        t = 0
        self.history = []

        while True:
            if t % block_size == 0:
                block = pixels[rng.integers(0, n, block_size)]
            x = block[t % block_size]
            t += 1

            delta = W - x
            dist = np.einsum("ij,ij->i", delta, delta) + mask
            two = np.argpartition(dist, 1)[:2]
            s1, s2 = int(two[0]), int(two[1])

            nbrs = np.flatnonzero(age[s1] >= 0)
            age[s1, nbrs] += 1
            age[nbrs, s1] += 1
            err[s1] += dist[s1]
            W[s1] -= self.eps_b * delta[s1]
            if nbrs.size:
                W[nbrs] -= self.eps_n * delta[nbrs]

            age[s1, s2] = 0
            age[s2, s1] = 0

            old = np.flatnonzero(age[s1] > self.age_max)
            if old.size:
                age[s1, old] = -1
                age[old, s1] = -1
                if not frozen:
                    for u in old:
                        if not (age[u] >= 0).any():
                            mask[u] = np.inf
                            n_active -= 1

            if not frozen and t % self.lam == 0 and n_active < cap:
                active = np.isfinite(mask)
                q = int(np.argmax(np.where(active, err, -1.0)))
                q_nbrs = np.flatnonzero(age[q] >= 0)
                if q_nbrs.size:
                    f = int(q_nbrs[np.argmax(err[q_nbrs])])
                    r = int(np.flatnonzero(~active)[0])
                    W[r] = 0.5 * (W[q] + W[f])
                    age[q, f] = age[f, q] = -1
                    age[q, r] = age[r, q] = 0
                    age[f, r] = age[r, f] = 0
                    err[q] *= self.alpha
                    err[f] *= self.alpha
                    err[r] = err[q]
                    mask[r] = 0.0
                    n_active += 1

            err *= self.decay

            if not frozen and n_active == cap:
                frozen = True
                frozen_at = t
                self.grown_at = t

            if monitor_pixels is not None and t % monitor_every == 0:
                self.history.append((t, self._monitor(W[mask == 0.0], monitor_pixels), n_active))

            if frozen and t - frozen_at >= self.tune_iterations:
                break
            if t >= limit:
                break

        keep = np.flatnonzero(np.isfinite(mask))
        self.weights = W[keep].copy()
        self.edges = age[np.ix_(keep, keep)] >= 0
        self.n_units = int(keep.size)
        self.iterations = t
        if monitor_pixels is not None:
            self.history.append((t, self._monitor(self.weights, monitor_pixels), self.n_units))
        return self

    def topographic_error(self, first: np.ndarray, second: np.ndarray) -> float:
        return float(1.0 - self.edges[first, second].mean())

    @staticmethod
    def _monitor(weights: np.ndarray, monitor_pixels: np.ndarray) -> float:
        first = nearest_units(monitor_pixels, weights, k=1)[:, 0]
        return quantization_error(monitor_pixels, weights, first)

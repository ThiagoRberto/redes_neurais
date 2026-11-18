import itertools
import sys

import busca

LOG_PATH = "resultados/busca_ampliada.csv"
WINDOWS = [12, 24]
CELLS = ["lstm", "gru"]
HIDDEN = [64, 128]
LAYERS = [2, 3]
LRS = [1e-3, 3e-3]
SEEDS = [42, 43]


def make_tasks() -> list:
    return [{"cell": cell, "window": window, "hidden": hidden, "layers": layers, "dropout": 0.0, "lr": lr, "seed": seed}
            for window, cell, hidden, layers, lr, seed in itertools.product(WINDOWS, CELLS, HIDDEN, LAYERS, LRS, SEEDS)]


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    busca.run_search(make_tasks(), LOG_PATH, n_workers)

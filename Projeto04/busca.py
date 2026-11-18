import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import torch

from dados import FEATURES, get_device, make_windows
from treino import build_model, count_parameters, set_seed, train

LOG_PATH = "resultados/busca_rnn.csv"

SEEDS = [42, 43]
WINDOWS = [24, 48]
CELLS = ["lstm", "gru"]
HIDDEN = [16, 32, 64]
LAYERS = [1, 2]
LRS = [1e-3, 3e-4]
DROPOUTS = [0.0, 0.2]
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8

KEYS = ["cell", "window", "hidden", "layers", "dropout", "lr", "seed"]

_state = {}


def _init_worker() -> None:
    torch.set_num_threads(1)
    _state["device"] = get_device()
    _state["data"] = {}


def _data(window: int) -> dict:
    if window not in _state["data"]:
        _state["data"][window] = make_windows(window, _state["device"])
    return _state["data"][window]


def make_tasks() -> list:
    tasks = []
    for window, cell, hidden, layers, lr, seed in itertools.product(WINDOWS, CELLS, HIDDEN, LAYERS, LRS, SEEDS):
        for dropout in (DROPOUTS if layers > 1 else [0.0]):
            tasks.append({"cell": cell, "window": window, "hidden": hidden, "layers": layers, "dropout": dropout,
                          "lr": lr, "seed": seed})
    return tasks


def run_task(task: dict) -> dict:
    data, device = _data(task["window"]), _state["device"]
    set_seed(task["seed"])
    model = build_model(len(FEATURES), task["hidden"], task["layers"], task["cell"], task["dropout"], device)
    start = time.perf_counter()
    model, history = train(model, data["x_train"], data["y_train"], data["x_val"], data["y_val"], task["lr"],
                           BATCH_SIZE, MAX_EPOCHS, PATIENCE, task["seed"])
    fit_time = time.perf_counter() - start
    best = history["best_epoch"] - 1
    return {**task, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "n_params": count_parameters(model), "train_loss": history["train_loss"][best],
            "val_loss": history["val_loss"][best], "best_epoch": history["best_epoch"],
            "epochs_trained": history["epochs_trained"], "fit_time": fit_time}


def _load_done(log_path: str) -> set:
    if not os.path.exists(log_path):
        return set()
    df = pd.read_csv(log_path)
    return set(df[KEYS].itertuples(index=False, name=None))


def _append_rows(log_path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)


def run_search(tasks: list, log_path: str, workers: int) -> None:
    done = _load_done(log_path)
    pending = [t for t in tasks if tuple(t[k] for k in KEYS) not in done]
    print(f"{len(pending)} tentativas pendentes ({len(tasks) - len(pending)} ja registradas em {log_path})", flush=True)
    if not pending:
        return

    buffer = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(run_task, t) for t in pending]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            buffer.append({"timestamp": datetime.now().isoformat(timespec="seconds"), **row})
            if len(buffer) >= 5 or i == len(futures):
                _append_rows(log_path, buffer)
                buffer = []
                print(f"  {i}/{len(futures)} concluidas ({time.perf_counter() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    run_search(make_tasks(), LOG_PATH, n_workers)

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from dados import get_device, load_data
from treino import build_model, count_parameters, set_seed, train

LOG_PATH = "resultados/busca_vae.csv"

SEEDS = [42, 43]
LATENT_DIMS = [2, 8]
HIDDEN_DIMS = [[256], [512], [512, 256], [1024, 512], [512, 256, 128], [1024, 512, 256]]
LEARNING_RATES = [1e-3, 5e-4]
BATCH_SIZES = [128, 256]
ACTIVATIONS = ["relu", "tanh"]
MAX_EPOCHS = 80
PATIENCE = 8

KEYS = ["latent_dim", "hidden_dims", "activation", "lr", "batch_size", "seed"]

_state = {}


def _init_worker() -> None:
    device = get_device()
    _state["device"] = device
    _state["data"] = load_data(device)


def make_tasks() -> list:
    tasks = []
    for latent_dim, hidden, activation, lr, batch_size, seed in itertools.product(
            LATENT_DIMS, HIDDEN_DIMS, ACTIVATIONS, LEARNING_RATES, BATCH_SIZES, SEEDS):
        tasks.append({"latent_dim": latent_dim, "hidden_dims": json.dumps(hidden), "activation": activation,
                      "lr": lr, "batch_size": batch_size, "seed": seed})
    return tasks


def run_task(task: dict) -> dict:
    data, device = _state["data"], _state["device"]
    set_seed(task["seed"])
    model = build_model(json.loads(task["hidden_dims"]), task["latent_dim"], task["activation"], device)
    start = time.perf_counter()
    model, history = train(model, data["x_train"], data["x_val"], task["lr"], task["batch_size"],
                           max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=task["seed"])
    fit_time = time.perf_counter() - start
    best = history["best_epoch"] - 1
    return {**task, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "n_params": count_parameters(model),
            "train_loss": history["train_loss"][best], "val_loss": history["val_loss"][best],
            "val_recon": history["val_recon"][best], "val_kl": history["val_kl"][best],
            "best_epoch": history["best_epoch"], "epochs_trained": history["epochs_trained"],
            "fit_time": fit_time}


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
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_search(make_tasks(), LOG_PATH, n_workers)

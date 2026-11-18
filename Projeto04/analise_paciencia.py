import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

from dados import FEATURES, get_device, make_windows
from metricas import evaluate_split, model_predictions
from treino import build_model, set_seed, train

OUT = "resultados/extras/paciencia.csv"
CONFIG = json.load(open("resultados/configs_finais.json", encoding="utf-8"))["gru"]
PATIENCES = [8, 15, 30]
SEEDS = [42, 43, 44, 45, 46]
_state = {}


def _init() -> None:
    torch.set_num_threads(1)
    _state["device"] = get_device()
    _state["data"] = make_windows(CONFIG["window"], _state["device"])


def run(task: tuple) -> dict:
    patience, seed = task
    c, d = CONFIG, _state["data"]
    set_seed(seed)
    model = build_model(len(FEATURES), c["hidden"], c["layers"], "gru", c["dropout"], _state["device"])
    model, h = train(model, d["x_train"], d["y_train"], d["x_val"], d["y_val"], c["lr"], c["batch_size"], 150,
                     patience, seed)
    row = {"paciencia": patience, "seed": seed, "best_epoch": h["best_epoch"], "epochs_trained": h["epochs_trained"],
           "best_val_loss": min(h["val_loss"])}
    for s in ["val", "test"]:
        for k, v in evaluate_split(model_predictions(model, d, s), d, s).items():
            row[f"{s}_{k}"] = v
    print(f"paciencia={patience} seed={seed} melhor_epoca={row['best_epoch']} de {row['epochs_trained']} "
          f"val_rmse={row['val_rmse']:.3f} test_rmse={row['test_rmse']:.3f}", flush=True)
    return row


if __name__ == "__main__":
    tasks = [(p, s) for p in PATIENCES for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 11, initializer=_init) as pool:
        rows = list(pool.map(run, tasks))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.groupby("paciencia")[["best_epoch", "epochs_trained", "val_rmse", "test_rmse", "test_skill_persistencia"]]
          .agg(["mean", "std"]).round(4).to_string())

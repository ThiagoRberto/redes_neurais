import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

import dados
from dados import FEATURES, get_device, make_windows
from metricas import evaluate_split, model_predictions, persistence_prediction
from plotting import plot_train_test_series
from treino import build_model, set_seed, train

OUT_DIR = "resultados/extras"
CONFIG = json.load(open("resultados/configs_finais.json", encoding="utf-8"))
SEEDS = [42, 43, 44]
VAL_LENGTHS = {"1 ano (2020)": 2020, "2 anos (2020-2021)": 2021, "3 anos (2020-2022)": 2022}
PERIODS = {"Verão": ("2018-01-10", "2018-01-24", "2023-01-10", "2023-01-24"),
           "Inverno": ("2018-07-10", "2018-07-24", "2023-07-10", "2023-07-24")}
_state = {}


def _init() -> None:
    torch.set_num_threads(1)
    _state["device"] = get_device()
    _state["data"] = {}


def _data(window: int, stride: int = None) -> dict:
    key = (window, stride)
    if key not in _state["data"]:
        old = dados.TRAIN_STRIDE
        if stride is not None:
            dados.TRAIN_STRIDE = stride
        _state["data"][key] = make_windows(window, _state["device"])
        dados.TRAIN_STRIDE = old
    return _state["data"][key]


def _model(cell: str, seed: int):
    c = CONFIG[cell]
    set_seed(seed)
    return build_model(len(FEATURES), c["hidden"], c["layers"], cell, c["dropout"], _state["device"]), c


def _metrics(model, data, splits) -> dict:
    out = {}
    for s in splits:
        for k, v in evaluate_split(model_predictions(model, data, s), data, s).items():
            out[f"{s}_{k}"] = v
    return out


def train_fixed(model, x, y, lr, batch_size, epochs, seed) -> list:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator()
    gen.manual_seed(seed)
    losses = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(x.size(0), generator=gen).to(x.device)
        total = 0.0
        for i in range(0, x.size(0), batch_size):
            idx = order[i:i + batch_size]
            loss = torch.nn.functional.mse_loss(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * idx.numel()
        losses.append(total / x.size(0))
    return losses


def run(task: dict) -> dict:
    cell, seed = task["cell"], task["seed"]
    model, c = _model(cell, seed)
    d = _data(c["window"])
    start = time.perf_counter()
    if task["kind"] == "val_len":
        keep = (d["meta_val"]["date"].dt.year <= task["last_year"]).to_numpy()
        keep_t = torch.from_numpy(keep).to(d["x_val"].device)
        model, h = train(model, d["x_train"], d["y_train"], d["x_val"][keep_t], d["y_val"][keep_t], c["lr"],
                         c["batch_size"], c["max_epochs"], c["patience"], seed)
        row = {"kind": "val_len", "variante": task["label"], "cell": cell, "seed": seed, "n_val": int(keep.sum()),
               "best_epoch": h["best_epoch"], "train_loss": h["train_loss"][h["best_epoch"] - 1],
               "val_loss": h["val_loss"][h["best_epoch"] - 1]}
        row.update(_metrics(model, d, ["test"]))
    elif task["kind"] == "trainval":
        x = torch.cat([d["x_train"], d["x_val"][::dados.TRAIN_STRIDE]])
        y = torch.cat([d["y_train"], d["y_val"][::dados.TRAIN_STRIDE]])
        losses = train_fixed(model, x, y, c["lr"], c["batch_size"], task["epochs"], seed)
        row = {"kind": "trainval", "variante": "treino + validação (2012-2022)", "cell": cell, "seed": seed,
               "n_train": int(x.size(0)), "epochs": task["epochs"], "train_loss": losses[-1]}
        row.update(_metrics(model, d, ["test"]))
    else:  # padrão: treino em 2012-2019; previsões no treino completo (passo 1) e no teste
        model, h = train(model, d["x_train"], d["y_train"], d["x_val"], d["y_val"], c["lr"], c["batch_size"],
                         c["max_epochs"], c["patience"], seed)
        full = _data(c["window"], stride=1)
        row = {"kind": "standard", "variante": "padrão (treino 2012-2019)", "cell": cell, "seed": seed,
               "best_epoch": h["best_epoch"]}
        row.update(_metrics(model, full, ["train", "val", "test"]))
        preds = {s: model_predictions(model, full, s) for s in ["train", "test"]}
        row["series"] = {s: (full[f"meta_{s}"], preds[s], persistence_prediction(full, s)) for s in preds}
    row["fit_time"] = time.perf_counter() - start
    print(f"{row['kind']} {row.get('variante')} {cell} seed={seed} test_rmse={row['test_rmse']:.3f} "
          f"({row['fit_time']:.0f}s)", flush=True)
    return row


def make_figure(rows: list) -> None:
    series = {r["cell"]: r["series"] for r in rows if r["kind"] == "standard"}
    panels = []
    for name, (t0, t1, s0, s1) in PERIODS.items():
        line = []
        for split, a, b, title in [("train", t0, t1, f"{name}: treino (2018)"), ("test", s0, s1, f"{name}: teste (2023)")]:
            meta, _, persist = series["gru"][split]
            sel = ((meta["target_date"] >= a) & (meta["target_date"] < b)).to_numpy()
            preds = [("Persistência", persist[sel])] + [(c.upper(), series[c][split][1][sel]) for c in ["lstm", "gru"]]
            line.append((title, meta["target_date"][sel].to_numpy(), meta["target"].to_numpy()[sel], preds))
        panels.append(line)
    plot_train_test_series(panels, "serie_treino_teste.png", OUT_DIR)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = pd.read_csv("resultados/final/metricas_por_seed.csv")
    epochs = int(round(runs[runs["modelo"] == "GRU"]["best_epoch"].mean()))
    tasks = [{"kind": "standard", "cell": c, "seed": 42} for c in ["lstm", "gru"]]
    for label, last in VAL_LENGTHS.items():
        tasks += [{"kind": "val_len", "label": label, "last_year": last, "cell": "gru", "seed": s} for s in SEEDS]
    tasks += [{"kind": "trainval", "cell": "gru", "seed": s, "epochs": epochs} for s in SEEDS]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    only_figure = len(sys.argv) > 2 and sys.argv[2] == "figura"
    if only_figure:
        tasks = tasks[:2]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        rows = list(pool.map(run, tasks))
    make_figure(rows)
    if only_figure:
        return
    flat = pd.DataFrame([{k: v for k, v in r.items() if k != "series"} for r in rows])
    flat.to_csv(f"{OUT_DIR}/analises_extras.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["variante", "cell", "seed", "best_epoch", "train_loss", "val_loss", "train_rmse", "val_rmse", "test_rmse",
            "test_mae", "test_skill_persistencia"]
    print(flat[[c for c in cols if c in flat]].round(4).to_string(index=False))
    print(f"\nepocas fixas no treino+validacao: {epochs}")
    print(flat[flat.kind != "standard"].groupby("variante")[["test_rmse", "test_mae", "test_r2",
                                                             "test_skill_persistencia"]]
          .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()

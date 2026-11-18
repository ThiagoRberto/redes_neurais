import json
import os
import time

import numpy as np
import pandas as pd
import torch

from dados import FEATURES, HORIZON, get_device, make_windows
from metricas import (climatology_prediction, evaluate_split, model_predictions, persistence_prediction,
                      ridge_baseline)
from plotting import (plot_comparison, plot_error_by_hour, plot_learning_curves, plot_scatter, plot_test_series)
from treino import build_model, count_parameters, set_seed, train

OUT_DIR = "resultados/final"
CONFIGS_PATH = "resultados/configs_finais.json"
SEEDS = [42, 43, 44, 45, 46]
MODEL_NAMES = {"lstm": "LSTM", "gru": "GRU"}
PERIODS = [("Verão (janeiro de 2023)", "2023-01-10", "2023-01-24"), ("Inverno (julho de 2023)", "2023-07-10", "2023-07-24")]


def load_configs() -> dict:
    with open(CONFIGS_PATH, encoding="utf-8") as f:
        return json.load(f)


def baseline_rows(data: dict) -> tuple:
    alpha, ridge = ridge_baseline(data)
    candidates = {
        f"Persistência (T daqui a {HORIZON} h = T agora)": {s: persistence_prediction(data, s) for s in ["val", "test"]},
        "Climatologia (média por mês e hora)": {s: climatology_prediction(data, s) for s in ["val", "test"]},
        f"Regressão Ridge (alpha={alpha:g})": ridge,
    }
    rows, preds = [], {}
    for name, p in candidates.items():
        row = {"modelo": name}
        for split in ["val", "test"]:
            for key, value in evaluate_split(p[split], data, split).items():
                row[f"{split}_{key}"] = value
        rows.append(row)
        preds[name] = p["test"]
    return rows, preds


def run_model(cell: str, config: dict, device: torch.device) -> tuple:
    data = make_windows(config["window"], device)
    runs, reference = [], None
    for seed in SEEDS:
        set_seed(seed)
        model = build_model(len(FEATURES), config["hidden"], config["layers"], cell, config["dropout"], device)
        start = time.perf_counter()
        model, history = train(model, data["x_train"], data["y_train"], data["x_val"], data["y_val"], config["lr"],
                               config["batch_size"], config["max_epochs"], config["patience"], seed)
        fit_time = time.perf_counter() - start
        pred = {s: model_predictions(model, data, s) for s in ["val", "test"]}
        row = {"modelo": MODEL_NAMES[cell], "seed": seed, "n_params": count_parameters(model),
               "best_epoch": history["best_epoch"], "epochs_trained": history["epochs_trained"],
               "fit_time": fit_time}
        for split in ["val", "test"]:
            for key, value in evaluate_split(pred[split], data, split).items():
                row[f"{split}_{key}"] = value
        runs.append(row)
        print(f"[{MODEL_NAMES[cell]}] seed={seed} val_rmse={row['val_rmse']:.3f} test_rmse={row['test_rmse']:.3f} "
              f"skill={row['test_skill_persistencia']:.3f} epocas={history['epochs_trained']} tempo={fit_time:.0f}s",
              flush=True)
        if seed == SEEDS[0]:
            reference = (history, pred["test"], data)
    return runs, reference


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in runs.columns if c.startswith(("val_", "test_"))] + ["epochs_trained", "fit_time"]
    rows = []
    for model, group in runs.groupby("modelo", sort=False):
        row = {"modelo": model, "n_params": int(group["n_params"].iloc[0])}
        for key in metrics:
            row[f"{key}_mean"], row[f"{key}_std"] = group[key].mean(), group[key].std()
        rows.append(row)
    return pd.DataFrame(rows)


def pipeline() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()
    configs = load_configs()
    print(f"Dispositivo: {device}; configs: {list(configs)}")

    all_runs, references = [], {}
    for cell, config in configs.items():
        runs, reference = run_model(cell, config, device)
        all_runs.extend(runs)
        references[cell] = reference

    first = next(iter(references.values()))[2]["meta_test"]["index"].to_numpy()
    for _, _, d in references.values():
        assert np.array_equal(first, d["meta_test"]["index"].to_numpy()), "conjuntos de teste diferem entre modelos"

    runs = pd.DataFrame(all_runs)
    runs.to_csv(f"{OUT_DIR}/metricas_por_seed.csv", index=False)
    table = summarize(runs)
    table.to_csv(f"{OUT_DIR}/tabela_final.csv", index=False)

    best_cell = min(configs, key=lambda c: runs[runs["modelo"] == MODEL_NAMES[c]]["val_mse"].mean())
    ref_data = references[best_cell][2]
    base_rows, base_preds = baseline_rows(ref_data)
    pd.DataFrame(base_rows).to_csv(f"{OUT_DIR}/baselines.csv", index=False)

    curves = [(f"{MODEL_NAMES[c]} (janela {configs[c]['window']} h, {configs[c]['layers']} camada(s), "
               f"{configs[c]['hidden']} unid.)", references[c][0]) for c in configs]
    plot_learning_curves(curves, "curvas_aprendizagem.png", OUT_DIR)

    meta = ref_data["meta_test"]
    y_true = meta["target"].to_numpy()
    persist = persistence_prediction(ref_data, "test")
    preds = [("Persistência", persist)] + [(MODEL_NAMES[c], references[c][1]) for c in configs]
    plot_test_series(meta["target_date"], y_true, preds, PERIODS, "serie_teste.png", OUT_DIR)
    plot_scatter(y_true, references[best_cell][1], MODEL_NAMES[best_cell], "dispersao_teste.png", OUT_DIR)
    plot_error_by_hour(meta["target_date"], [(n, p - y_true) for n, p in preds], "erro_por_hora.png", OUT_DIR)

    comparison = [(r["modelo"], r["test_rmse"], r["test_skill_persistencia"]) for r in base_rows]
    comparison += [(r["modelo"] + " (média)", r["test_rmse_mean"], r["test_skill_persistencia_mean"])
                   for _, r in table.iterrows()]
    plot_comparison(comparison, "comparacao_modelos.png", OUT_DIR)
    pd.DataFrame({"data_alvo": meta["target_date"], "real": y_true, **{n: p for n, p in preds}}).to_csv(
        f"{OUT_DIR}/previsoes_teste_seed42.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["modelo", "val_rmse_mean", "test_rmse_mean", "test_rmse_std", "test_mae_mean", "test_r2_mean",
            "test_skill_persistencia_mean", "test_erro_max_mean", "epochs_trained_mean", "n_params"]
    print(f"\nModelo escolhido pela validacao: {MODEL_NAMES[best_cell]}")
    print("\n=== Modelos (media sobre %d seeds) ===" % len(SEEDS))
    print(table[show].round(4).to_string(index=False))
    print("\n=== Baselines ===")
    print(pd.DataFrame(base_rows)[["modelo", "val_rmse", "test_rmse", "test_mae", "test_r2",
                                   "test_skill_persistencia", "test_erro_max"]].round(4).to_string(index=False))
    print("\nResultados salvos em", OUT_DIR)


if __name__ == "__main__":
    pipeline()

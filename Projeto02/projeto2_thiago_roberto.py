import json
import os
import sys
import time

import numpy as np
import pandas as pd

from gng import GNG
from imagem import image_to_pixels, load_image, results_dir
from metricas import evaluate
from plotting import (plot_comparison, plot_gng_convergence, plot_gng_graphs, plot_reconstructions,
                      plot_som_convergence, plot_som_maps, save_image)
from som import SOM

IMAGE_PATH = "imagens/bolinhas.jpeg"
SEEDS = [42, 43, 44, 45, 46]
MONITOR_SIZE = 5000

SOM_CONFIG = {
    4: dict(alpha0=0.5, sigma0=1.0, sigma_f=0.5, iterations=200000),
    8: dict(alpha0=0.5, sigma0=2.0, sigma_f=0.5, iterations=200000),
    16: dict(alpha0=0.5, sigma0=4.0, sigma_f=0.5, iterations=200000),
}

GNG_CONFIG = {
    16: dict(eps_b=0.05, eps_n=0.002, age_max=100, lam=100, tune_iterations=100000),
    64: dict(eps_b=0.05, eps_n=0.002, age_max=100, lam=100, tune_iterations=100000),
    256: dict(eps_b=0.05, eps_n=0.002, age_max=100, lam=100, tune_iterations=100000),
}


def load_configs(out_dir: str) -> tuple:
    path = f"{out_dir}/configs_finais.json"
    if not os.path.exists(path):
        print(f"{path} nao encontrado: usando SOM_CONFIG e GNG_CONFIG padrao")
        return SOM_CONFIG, GNG_CONFIG
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Hiperparametros lidos de {path}")
    return {int(k): v for k, v in data["SOM"].items()}, {int(k): v for k, v in data["GNG"].items()}


def train_som(size: int, config: dict, seed: int, pixels: np.ndarray, monitor: np.ndarray) -> SOM:
    return SOM(size, size, seed=seed, **config).fit(pixels, monitor)


def train_gng(units: int, config: dict, seed: int, pixels: np.ndarray, monitor: np.ndarray) -> GNG:
    return GNG(units, seed=seed, **config).fit(pixels, monitor)


def run_model(method: str, size: int, neurons: int, trainer, config: dict, pixels: np.ndarray,
              image: np.ndarray, monitor: np.ndarray) -> tuple:
    runs = []
    reference = None
    for seed in SEEDS:
        start = time.perf_counter()
        model = trainer(size, config, seed, pixels, monitor)
        fit_time = time.perf_counter() - start
        metrics, quantized = evaluate(model, pixels, image)
        runs.append({"metodo": method, "neuronios": neurons, "seed": seed, **metrics,
                     "iteracoes": model.iterations, "fit_time": fit_time})
        print(f"[{method} {neurons}] seed={seed} qe={metrics['qe']:.4f} te={metrics['te']:.4f} "
              f"cores={metrics['colors_used']} tempo={fit_time:.1f}s", flush=True)
        if seed == SEEDS[0]:
            reference = (model, quantized, metrics)
    return runs, reference


def summarize(runs: pd.DataFrame, configs: dict) -> pd.DataFrame:
    rows = []
    for (method, neurons), group in runs.groupby(["metodo", "neuronios"]):
        row = {"metodo": method, "neuronios": neurons}
        for key in ["qe", "te", "colors_used", "iteracoes", "fit_time"]:
            row[f"{key}_mean"] = group[key].mean()
            row[f"{key}_std"] = group[key].std()
        row["hiperparametros"] = str(configs[(method, neurons)])
        rows.append(row)
    return pd.DataFrame(rows)


def pipeline(image_path: str, show: bool = False) -> None:
    out_dir = results_dir(image_path)
    som_config, gng_config = load_configs(out_dir)
    image = load_image(image_path)
    pixels = image_to_pixels(image)
    rng = np.random.default_rng(0)
    monitor = pixels[rng.choice(pixels.shape[0], MONITOR_SIZE, replace=False)]
    print(f"Imagem {image_path}: {image.shape[1]}x{image.shape[0]} px, {pixels.shape[0]} amostras RGB")

    all_runs = []
    configs = {}
    som_items, som_models, som_curves = [], [], []
    gng_items, gng_models, gng_curves = [], [], []

    for size, config in som_config.items():
        neurons = size * size
        configs[("SOM", neurons)] = config
        runs, (model, quantized, metrics) = run_model("SOM", size, neurons, train_som, config, pixels, image, monitor)
        all_runs += runs
        save_image(quantized, f"som_{size}x{size}.png", out_dir)
        title = f"SOM {size}x{size} ({neurons} cores)\nQE={metrics['qe']:.4f} | TE={metrics['te']:.3f}"
        som_items.append((title, quantized))
        som_models.append((f"SOM {size}x{size}", model))
        som_curves.append((f"{size}x{size}", model.history))

    for units, config in gng_config.items():
        configs[("GNG", units)] = config
        runs, (model, quantized, metrics) = run_model("GNG", units, units, train_gng, config, pixels, image, monitor)
        all_runs += runs
        save_image(quantized, f"gng_{units}.png", out_dir)
        title = f"GNG {units} neurônios\nQE={metrics['qe']:.4f} | TE={metrics['te']:.3f}"
        gng_items.append((title, quantized))
        gng_models.append((f"GNG {units} neurônios", model))
        gng_curves.append((f"{units} neurônios", model.history))

    runs_df = pd.DataFrame(all_runs)
    table = summarize(runs_df, configs)
    os.makedirs(out_dir, exist_ok=True)
    runs_df.to_csv(f"{out_dir}/execucoes_finais.csv", index=False)
    table.to_csv(f"{out_dir}/tabela_resultados.csv", index=False)
    print(table.drop(columns=["hiperparametros"]).round(4).to_string(index=False))

    plot_reconstructions(image, som_items, "reconstrucao_som.png", out_dir, show)
    plot_reconstructions(image, gng_items, "reconstrucao_gng.png", out_dir, show)
    plot_som_maps(som_models, "mapas_som.png", out_dir, show)
    plot_gng_graphs(gng_models, "grafos_gng.png", out_dir, show)
    plot_som_convergence(som_curves, "convergencia_som.png", out_dir, show)
    plot_gng_convergence(gng_curves, "convergencia_gng.png", out_dir, show)
    plot_comparison(table, "comparacao_metricas.png", out_dir, show)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for path in (args or [IMAGE_PATH]):
        pipeline(path, show="--mostrar" in sys.argv)

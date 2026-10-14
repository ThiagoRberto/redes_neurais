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

from gng import GNG
from imagem import image_to_pixels, list_images, load_image, results_dir
from metricas import evaluate
from som import SOM

SEEDS = [42, 43, 44]

SOM_SIZES = [4, 8, 16]
SOM_ALPHA0 = [0.1, 0.3, 0.5, 0.9]
SOM_SIGMA0_REL = [0.25, 0.5, 1.0]
SOM_SIGMA_F = [0.3, 0.5, 1.0]
SOM_ITERATIONS = [50000, 200000]

GNG_UNITS = [16, 64, 256]
GNG_EPS_B = [0.01, 0.05, 0.2]
GNG_EPS_N = [0.0006, 0.002, 0.006]
GNG_AGE_MAX = [25, 50, 100]
GNG_LAMBDA = [50, 100, 200]
GNG_TUNE = 100000

SOM_KEYS = ["size", "alpha0", "sigma0", "sigma_f", "iterations", "seed"]
GNG_KEYS = ["units", "eps_b", "eps_n", "age_max", "lam", "tune_iterations", "seed"]

_state = {}


def _init_worker(image_path: str) -> None:
    image = load_image(image_path)
    _state["image"] = image
    _state["pixels"] = image_to_pixels(image)


def som_tasks() -> list:
    tasks = []
    for size, alpha0, rel, sigma_f, iterations, seed in itertools.product(
            SOM_SIZES, SOM_ALPHA0, SOM_SIGMA0_REL, SOM_SIGMA_F, SOM_ITERATIONS, SEEDS):
        tasks.append({"size": size, "alpha0": alpha0, "sigma0": size * rel, "sigma_f": sigma_f,
                      "iterations": iterations, "seed": seed})
    return tasks


def gng_tasks() -> list:
    tasks = []
    for units, eps_b, eps_n, age_max, lam, seed in itertools.product(
            GNG_UNITS, GNG_EPS_B, GNG_EPS_N, GNG_AGE_MAX, GNG_LAMBDA, SEEDS):
        tasks.append({"units": units, "eps_b": eps_b, "eps_n": eps_n, "age_max": age_max, "lam": lam,
                      "tune_iterations": GNG_TUNE, "seed": seed})
    return tasks


def run_som_task(task: dict) -> dict:
    start = time.perf_counter()
    model = SOM(task["size"], task["size"], alpha0=task["alpha0"], sigma0=task["sigma0"],
                sigma_f=task["sigma_f"], iterations=task["iterations"], seed=task["seed"])
    model.fit(_state["pixels"])
    fit_time = time.perf_counter() - start
    metrics, _ = evaluate(model, _state["pixels"], _state["image"])
    return {**task, **metrics, "fit_time": fit_time}


def run_gng_task(task: dict) -> dict:
    start = time.perf_counter()
    model = GNG(task["units"], eps_b=task["eps_b"], eps_n=task["eps_n"], age_max=task["age_max"],
                lam=task["lam"], tune_iterations=task["tune_iterations"], seed=task["seed"])
    model.fit(_state["pixels"])
    fit_time = time.perf_counter() - start
    metrics, _ = evaluate(model, _state["pixels"], _state["image"])
    return {**task, **metrics, "units_final": model.n_units, "iterations_total": model.iterations,
            "fit_time": fit_time}


def _load_done(log_path: str, keys: list) -> set:
    if not os.path.exists(log_path):
        return set()
    df = pd.read_csv(log_path)
    return set(df[keys].itertuples(index=False, name=None))


def _append_rows(log_path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)


def run_search(tasks: list, worker, log_path: str, keys: list, workers: int, image_path: str) -> None:
    done = _load_done(log_path, keys)
    pending = [t for t in tasks if tuple(t[k] for k in keys) not in done]
    print(f"{len(pending)} tentativas pendentes ({len(tasks) - len(pending)} ja registradas em {log_path})")
    if not pending:
        return

    buffer = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(image_path,)) as pool:
        futures = [pool.submit(worker, t) for t in pending]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            buffer.append({"timestamp": datetime.now().isoformat(timespec="seconds"), **row})
            if len(buffer) >= 20 or i == len(futures):
                _append_rows(log_path, buffer)
                buffer = []
                print(f"  {i}/{len(futures)} concluidas ({time.perf_counter() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ambos"
    images = sys.argv[2:] or list_images()
    n_workers = max(1, min(10, (os.cpu_count() or 2) - 2))
    for image_path in images:
        out_dir = results_dir(image_path)
        print(f"=== {image_path} -> {out_dir} ===", flush=True)
        if which in ("som", "ambos"):
            run_search(som_tasks(), run_som_task, f"{out_dir}/busca_som.csv", SOM_KEYS, n_workers, image_path)
        if which in ("gng", "ambos"):
            run_search(gng_tasks(), run_gng_task, f"{out_dir}/busca_gng.csv", GNG_KEYS, n_workers, image_path)

import json
import os
import time

import numpy as np
import pandas as pd
import torch

from dados import CLASS_NAMES, get_device, load_data
from espaco_latente import decode_grid, encode_dataset, fit_pca, latent_analysis, reconstruct, sample_prior
from plotting import (plot_active_units, plot_class_panels, plot_explained_variance, plot_latent_scatter,
                      plot_manifold, plot_reconstructions, plot_samples, plot_training_curves)
from treino import build_model, count_parameters, evaluate, set_seed, train

OUT_DIR = "resultados/final"
CONFIGS_PATH = "resultados/configs_finais.json"
SEEDS = [42, 43, 44, 45, 46]
N_EXAMPLES = 10

DEFAULT_CONFIG = {
    2: dict(hidden_dims=[512, 256], activation="relu", lr=1e-3, batch_size=256, max_epochs=100, patience=10),
    8: dict(hidden_dims=[512, 256], activation="relu", lr=1e-3, batch_size=256, max_epochs=100, patience=10),
}


def load_configs() -> dict:
    if not os.path.exists(CONFIGS_PATH):
        print(f"{CONFIGS_PATH} nao encontrado: usando DEFAULT_CONFIG")
        return DEFAULT_CONFIG
    with open(CONFIGS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Hiperparametros lidos de {CONFIGS_PATH}")
    return {int(k): v for k, v in data.items()}


def run_latent(latent_dim: int, config: dict, data: dict, device: torch.device) -> tuple:
    runs, analyses = [], []
    reference = None
    for seed in SEEDS:
        set_seed(seed)
        model = build_model(config["hidden_dims"], latent_dim, config["activation"], device)
        start = time.perf_counter()
        model, history = train(model, data["x_train"], data["x_val"], config["lr"], config["batch_size"],
                               max_epochs=config["max_epochs"], patience=config["patience"], seed=seed)
        fit_time = time.perf_counter() - start
        val_loss, val_recon, val_kl = evaluate(model, data["x_val"])
        test_loss, test_recon, test_kl = evaluate(model, data["x_test"])
        analysis = latent_analysis(model, data, seed)
        runs.append({"latent_dim": latent_dim, "seed": seed, "n_params": count_parameters(model),
                     "val_loss": val_loss, "val_recon": val_recon, "val_kl": val_kl,
                     "test_loss": test_loss, "test_recon": test_recon, "test_kl": test_kl,
                     "best_epoch": history["best_epoch"], "epochs_trained": history["epochs_trained"],
                     "active_units": analysis["active_units"], "explained_2d": analysis["explained_2d"],
                     "silhouette": analysis["latent"]["silhouette"], "knn_acc": analysis["latent"]["knn_acc"],
                     "kmeans_nmi": analysis["latent"]["kmeans_nmi"], "kmeans_ari": analysis["latent"]["kmeans_ari"],
                     "silhouette_2d": analysis["projection"]["silhouette"],
                     "knn_acc_2d": analysis["projection"]["knn_acc"],
                     "kmeans_nmi_2d": analysis["projection"]["kmeans_nmi"],
                     "kmeans_ari_2d": analysis["projection"]["kmeans_ari"],
                     "fit_time": fit_time})
        analyses.append(analysis)
        print(f"[latente {latent_dim}D] seed={seed} val={val_loss:.3f} teste={test_loss:.3f} "
              f"(recon={test_recon:.3f} kl={test_kl:.3f}) epocas={history['epochs_trained']} "
              f"ativas={analysis['active_units']} kNN={analysis['latent']['knn_acc']:.3f} "
              f"tempo={fit_time:.0f}s", flush=True)
        if seed == SEEDS[0]:
            reference = (model, history, analysis)
    return runs, analyses, reference


def summarize(runs: pd.DataFrame, configs: dict) -> pd.DataFrame:
    rows = []
    for latent_dim, group in runs.groupby("latent_dim"):
        row = {"latent_dim": latent_dim}
        for key in ["val_loss", "test_loss", "test_recon", "test_kl", "active_units", "explained_2d", "silhouette",
                    "knn_acc", "kmeans_nmi", "kmeans_ari", "silhouette_2d", "knn_acc_2d", "kmeans_nmi_2d",
                    "kmeans_ari_2d", "epochs_trained", "fit_time"]:
            row[f"{key}_mean"] = group[key].mean()
            row[f"{key}_std"] = group[key].std()
        row["n_params"] = int(group["n_params"].iloc[0])
        row["hiperparametros"] = json.dumps(configs[int(latent_dim)])
        rows.append(row)
    return pd.DataFrame(rows)


def class_recall_table(analyses: dict) -> pd.DataFrame:
    rows = []
    for latent_dim, items in analyses.items():
        recall = np.mean([a["latent"]["knn_recall"] for a in items], axis=0)
        rows.append(pd.DataFrame({"latent_dim": latent_dim, "classe": CLASS_NAMES, "knn_recall": recall}))
    return pd.concat(rows, ignore_index=True)


def figures(references: dict, data: dict, show: bool = False) -> None:
    y_test = data["y_test"]
    indices = [int(np.nonzero(y_test == c)[0][0]) for c in range(N_EXAMPLES)]
    originals = data["x_test"][indices]
    recon_items, sample_items, curve_items, unit_items = [], [], [], []

    for latent_dim, (model, history, analysis) in references.items():
        mu_train, _ = encode_dataset(model, data["x_train"])
        mu_test, _ = encode_dataset(model, data["x_test"])
        recon_items.append((f"Reconstrução (latente {latent_dim}D)", reconstruct(model, originals)))
        sample_items.append((f"Amostras de N(0, I) (latente {latent_dim}D)", sample_prior(model, 80)))
        curve_items.append((f"Latente {latent_dim}D", history))
        unit_items.append((f"Latente {latent_dim}D: {analysis['active_units']} unidades ativas",
                           analysis["kl_per_dim"], analysis["mu_variance"]))

        if latent_dim == 2:
            plot_latent_scatter(mu_test, y_test, "z1", "z2", "Espaço latente 2D (teste, média μ)",
                                "latente2d_rotulos.png", OUT_DIR, show)
            plot_class_panels(mu_test, y_test, "z1", "z2", "Espaço latente 2D por classe",
                              "latente2d_por_classe.png", OUT_DIR, show)
            images, axis = decode_grid(model)
            plot_manifold(images, axis, "variedade_2d.png", OUT_DIR, show)
        else:
            pca = fit_pca(mu_train, 2)
            proj = pca.transform(mu_test)
            ratios = analysis["explained_ratio"]
            plot_latent_scatter(proj, y_test, f"PC1 ({ratios[0] * 100:.1f}%)", f"PC2 ({ratios[1] * 100:.1f}%)",
                                f"Espaço latente {latent_dim}D projetado por PCA "
                                f"({analysis['explained_2d'] * 100:.1f}% da variância)",
                                f"latente{latent_dim}d_pca_rotulos.png", OUT_DIR, show)
            plot_class_panels(proj, y_test, "PC1", "PC2", f"Espaço latente {latent_dim}D (PCA) por classe",
                              f"latente{latent_dim}d_pca_por_classe.png", OUT_DIR, show)
            plot_explained_variance(np.array(ratios), f"pca_variancia_{latent_dim}d.png", OUT_DIR, show)

    plot_reconstructions(originals.cpu().numpy().reshape(-1, 28, 28), recon_items, "reconstrucoes.png",
                         OUT_DIR, show)
    plot_samples(sample_items, 8, 10, "amostras_prior.png", OUT_DIR, show)
    plot_training_curves(curve_items, "curvas_treino.png", OUT_DIR, show)
    plot_active_units(unit_items, "unidades_ativas.png", OUT_DIR, show)


def pipeline(show: bool = False) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()
    data = load_data(device)
    configs = load_configs()
    print(f"Dispositivo: {device}  treino={data['x_train'].size(0)} val={data['x_val'].size(0)} "
          f"teste={data['x_test'].size(0)}")

    all_runs, analyses, references = [], {}, {}
    for latent_dim in sorted(configs):
        runs, latent_analyses, reference = run_latent(latent_dim, configs[latent_dim], data, device)
        all_runs.extend(runs)
        analyses[latent_dim] = latent_analyses
        references[latent_dim] = reference
        torch.save(reference[0].state_dict(), f"{OUT_DIR}/vae_latente{latent_dim}_seed{SEEDS[0]}.pt")

    runs = pd.DataFrame(all_runs)
    runs.to_csv(f"{OUT_DIR}/metricas_por_seed.csv", index=False)
    table = summarize(runs, configs)
    table.to_csv(f"{OUT_DIR}/tabela_final.csv", index=False)
    class_recall_table(analyses).to_csv(f"{OUT_DIR}/recall_knn_por_classe.csv", index=False)
    ratios = {d: references[d][2]["explained_ratio"] for d in references}
    with open(f"{OUT_DIR}/variancia_explicada.json", "w", encoding="utf-8") as f:
        json.dump(ratios, f, indent=2)

    figures(references, data, show)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n=== Tabela final (media sobre %d seeds) ===" % len(SEEDS))
    print(table.drop(columns=["hiperparametros"]).round(3).T.to_string())
    print("\nResultados salvos em", OUT_DIR)


if __name__ == "__main__":
    pipeline()

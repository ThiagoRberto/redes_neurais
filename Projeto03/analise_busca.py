import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_PATH = "resultados/busca_vae.csv"
FIGURES_DIR = "resultados/figuras"
GROUP = ["latent_dim", "hidden_dims", "activation", "lr", "batch_size"]
HPS = ["hidden_dims", "activation", "lr", "batch_size"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(GROUP).agg(val_loss_mean=("val_loss", "mean"), val_loss_std=("val_loss", "std"),
                                 val_recon=("val_recon", "mean"), val_kl=("val_kl", "mean"),
                                 train_loss=("train_loss", "mean"), n_params=("n_params", "first"),
                                 best_epoch=("best_epoch", "mean"), fit_time=("fit_time", "mean"),
                                 runs=("seed", "count")).reset_index()


def sort_hidden(values) -> list:
    return sorted(values, key=lambda v: (len(json.loads(v)), sum(json.loads(v))))


def plot_sensitivity(df: pd.DataFrame, latent_dim: int, name: str) -> None:
    sub = df[df["latent_dim"] == latent_dim]
    fig, axes = plt.subplots(1, len(HPS), figsize=(3.6 * len(HPS), 3.2), squeeze=False)
    for ax, hp in zip(axes[0], HPS):
        marginal = sub.groupby(hp)["val_loss"].mean()
        if hp == "hidden_dims":
            marginal = marginal.loc[sort_hidden(marginal.index)]
        else:
            marginal = marginal.sort_index()
        xs = np.arange(len(marginal))
        ax.plot(xs, marginal.values, marker="o")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v).replace(" ", "") if hp == "hidden_dims" else f"{v:g}" if not isinstance(v, str) else v
                            for v in marginal.index], fontsize=7, rotation=30 if hp == "hidden_dims" else 0)
        ax.set_title(hp)
        ax.set_ylabel("Perda de validação (média)")
    fig.suptitle(f"VAE com espaço latente {latent_dim}D: sensibilidade aos hiperparâmetros")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(f"{FIGURES_DIR}/{name}", dpi=150)
    plt.close(fig)


def plot_topology(summary: pd.DataFrame, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, latent_dim in zip(axes, sorted(summary["latent_dim"].unique())):
        sub = summary[summary["latent_dim"] == latent_dim]
        sc = ax.scatter(sub["n_params"], sub["val_loss_mean"], c=np.log10(sub["lr"]), cmap="viridis", s=26)
        best = sub.loc[sub["val_loss_mean"].idxmin()]
        ax.scatter([best["n_params"]], [best["val_loss_mean"]], facecolors="none", edgecolors="red", s=120,
                   label="melhor configuração")
        ax.set_xscale("log")
        ax.set_xlabel("Parâmetros")
        ax.set_ylabel("Perda de validação (média)")
        ax.set_title(f"Espaço latente {latent_dim}D (cor = log10 lr)")
        ax.legend()
        fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(f"{FIGURES_DIR}/{name}", dpi=150)
    plt.close(fig)


def sensitivity_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for latent_dim, sub in raw.groupby("latent_dim"):
        for hp in HPS:
            marginal = sub.groupby(hp)["val_loss"].mean()
            rows.append({"latent_dim": latent_dim, "hiperparametro": hp,
                         "melhor": marginal.idxmin(), "val_loss_min": marginal.min(),
                         "val_loss_max": marginal.max(),
                         "variacao_pct": (marginal.max() - marginal.min()) / marginal.mean() * 100.0})
    return pd.DataFrame(rows)


def report(summary: pd.DataFrame, top: int = 8) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for latent_dim in sorted(summary["latent_dim"].unique()):
        sub = summary[summary["latent_dim"] == latent_dim].sort_values("val_loss_mean")
        print(f"\n=== latente {latent_dim}D: {top} menores perdas de validacao ===")
        print(sub.head(top)[GROUP[1:] + ["val_loss_mean", "val_loss_std", "val_recon", "val_kl", "n_params",
                                          "best_epoch", "fit_time"]].round(3).to_string(index=False))


def analyze() -> None:
    raw = pd.read_csv(LOG_PATH)
    summary = summarize(raw)
    summary.to_csv("resultados/resumo_busca.csv", index=False)
    sensitivity = sensitivity_summary(raw)
    sensitivity.to_csv("resultados/sensibilidade_resumo.csv", index=False)

    print(f"{len(raw)} execucoes, {len(summary)} configuracoes")
    truncated = (raw["epochs_trained"] >= raw["max_epochs"]).mean() * 100.0
    print(f"execucoes que atingiram max_epochs: {truncated:.1f}%")
    report(summary)
    print("\n=== sensibilidade (variacao da perda media entre valores do hiperparametro) ===")
    print(sensitivity.round(3).to_string(index=False))

    for latent_dim in sorted(raw["latent_dim"].unique()):
        plot_sensitivity(raw, latent_dim, f"sensibilidade_latente{latent_dim}.png")
    plot_topology(summary, "perda_vs_parametros.png")
    print("\nFiguras de analise salvas em", FIGURES_DIR)


if __name__ == "__main__":
    analyze()

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_PATH = "resultados/busca_rnn.csv"
EXTENDED_LOG_PATH = "resultados/busca_ampliada.csv"
FIGURES_DIR = "resultados/figuras"
GROUP = ["cell", "window", "hidden", "layers", "dropout", "lr"]
HPS = ["cell", "window", "hidden", "layers", "dropout", "lr"]


NOMES = {"cell": "tipo de célula", "window": "janela (h)", "hidden": "unidades ocultas", "layers": "camadas", "dropout": "dropout", "lr": "taxa de aprendizado"}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(GROUP).agg(val_loss_mean=("val_loss", "mean"), val_loss_std=("val_loss", "std"),
                                 train_loss=("train_loss", "mean"), n_params=("n_params", "first"),
                                 best_epoch=("best_epoch", "mean"), fit_time=("fit_time", "mean"),
                                 runs=("seed", "count")).reset_index()


def sensitivity_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hp in HPS:
        marginal = raw.groupby(hp)["val_loss"].mean()
        rows.append({"hiperparametro": hp, "melhor": marginal.idxmin(), "val_loss_min": marginal.min(),
                     "val_loss_max": marginal.max(),
                     "variacao_pct": (marginal.max() - marginal.min()) / marginal.mean() * 100.0})
    return pd.DataFrame(rows)


def plot_sensitivity(series: list, name: str) -> None:
    """series: [(rotulo, dataframe bruto de uma busca), ...]; cada busca tem as médias marginais da sua própria grade."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), squeeze=False)
    for ax, hp in zip(axes.ravel(), HPS):
        values = sorted(set().union(*[set(raw[hp]) for _, raw in series]))
        for (label, raw), marker in zip(series, ["o", "s"]):
            marginal = raw.groupby(hp)["val_loss"].mean().sort_index()
            xs = [values.index(v) for v in marginal.index]
            ax.plot(xs, marginal.values, marker=marker, label=label)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:g}" if not isinstance(v, str) else v for v in values])
        ax.set_xlim(-0.3, len(values) - 0.7)
        ax.set_title(NOMES.get(hp, hp))
        ax.set_ylabel("MSE de validação (média)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(f"{FIGURES_DIR}/{name}", dpi=150)
    plt.close(fig)


def plot_params(summary: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = summary["cell"].map({"lstm": "tab:blue", "gru": "tab:orange"})
    ax.scatter(summary["n_params"], summary["val_loss_mean"], c=colors, s=26)
    best = summary.loc[summary["val_loss_mean"].idxmin()]
    ax.scatter([best["n_params"]], [best["val_loss_mean"]], facecolors="none", edgecolors="red", s=120,
               label="melhor configuração")
    ax.scatter([], [], c="tab:blue", label="LSTM")
    ax.scatter([], [], c="tab:orange", label="GRU")
    ax.set_xscale("log")
    ax.set_xlabel("Parâmetros")
    ax.set_ylabel("MSE de validação (média)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(f"{FIGURES_DIR}/{name}", dpi=150)
    plt.close(fig)


def analyze() -> None:
    raw = pd.read_csv(LOG_PATH)
    summary = summarize(raw)
    summary.to_csv("resultados/resumo_busca.csv", index=False)
    sensitivity = sensitivity_summary(raw)
    sensitivity.to_csv("resultados/sensibilidade_resumo.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"{len(raw)} execucoes, {len(summary)} configuracoes")
    print(f"execucoes que atingiram max_epochs: {(raw['epochs_trained'] >= raw['max_epochs']).mean() * 100:.1f}%")
    print("\n=== 10 menores MSE de validacao ===")
    print(summary.sort_values("val_loss_mean").head(10).round(4).to_string(index=False))
    print("\n=== sensibilidade ===")
    print(sensitivity.round(4).to_string(index=False))
    series = [("busca inicial (144 treinos)", raw)]
    if os.path.exists(EXTENDED_LOG_PATH):
        extended = pd.read_csv(EXTENDED_LOG_PATH)
        series.append((f"busca ampliada ({len(extended)} treinos)", extended))
    plot_sensitivity(series, "sensibilidade.png")
    plot_params(summary, "perda_vs_parametros.png")


if __name__ == "__main__":
    analyze()

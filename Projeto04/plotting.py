import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, out_dir: str, name: str) -> str:
    figures_dir = f"{out_dir}/figuras"
    os.makedirs(figures_dir, exist_ok=True)
    path = f"{figures_dir}/{name}"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_learning_curves(items: list, name: str, out_dir: str) -> str:
    fig, axes = plt.subplots(1, len(items), figsize=(5.6 * len(items), 3.8), squeeze=False)
    for ax, (label, history) in zip(axes[0], items):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="treino")
        ax.plot(epochs, history["val_loss"], label="validação")
        ax.axvline(history["best_epoch"], color="gray", linestyle="--", linewidth=1, label="melhor época")
        ax.set_xlabel("Época")
        ax.set_ylabel("MSE (alvo padronizado)")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
    return _save(fig, out_dir, name)


def plot_test_series(target_dates: pd.Series, y_true: np.ndarray, preds: list, periods: list, name: str,
                     out_dir: str) -> str:
    fig, axes = plt.subplots(len(periods), 1, figsize=(11, 3.4 * len(periods)), squeeze=False)
    dates = pd.to_datetime(target_dates).to_numpy()
    for ax, (title, start, end) in zip(axes[:, 0], periods):
        sel = (dates >= np.datetime64(start)) & (dates < np.datetime64(end))
        ax.plot(dates[sel], y_true[sel], color="black", linewidth=1.6, label="real")
        for label, y_pred in preds:
            ax.plot(dates[sel], y_pred[sel], linewidth=1.1, label=label)
        ax.set_ylabel("Temperatura (°C)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, ncol=len(preds) + 1)
        ax.grid(alpha=0.25)
    return _save(fig, out_dir, name)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, label: str, name: str, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=3, alpha=0.25, rasterized=True)
    lim = (min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1)
    ax.plot(lim, lim, color="gray", linestyle="--", linewidth=1, label="previsão perfeita")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Temperatura real (°C)")
    ax.set_ylabel("Temperatura prevista (°C)")
    ax.set_title(f"{label}: previsto contra real (teste)")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, name)


def plot_comparison(rows: list, name: str, out_dir: str) -> str:
    labels = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(labels, [r[1] for r in rows], color="tab:blue")
    axes[0].set_xlabel("RMSE no teste (°C), 24 h à frente")
    axes[1].barh(labels, [r[2] * 100.0 for r in rows], color="tab:green")
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Ganho de MSE sobre a persistência (%)")
    for ax in axes:
        ax.invert_yaxis()
    return _save(fig, out_dir, name)


def plot_train_test_series(panels: list, name: str, out_dir: str) -> str:
    """panels: linhas de [(titulo, datas, real, [(rotulo, previsao), ...]), ...] (colunas: treino, teste)."""
    rows, cols = len(panels), len(panels[0])
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 3.3 * rows), squeeze=False)
    for r, row in enumerate(panels):
        low = min(min(p[2].min(), *[q.min() for _, q in p[3]]) for p in row) - 1
        high = max(max(p[2].max(), *[q.max() for _, q in p[3]]) for p in row) + 1
        for c, (title, dates, y_true, preds) in enumerate(row):
            ax = axes[r, c]
            ax.plot(dates, y_true, color="black", linewidth=1.6, label="real")
            for label, y_pred in preds:
                ax.plot(dates, y_pred, linewidth=1.1, label=label)
            ax.set_ylim(low, high)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.25)
            ax.tick_params(axis="x", labelsize=8, rotation=20)
            if c == 0:
                ax.set_ylabel("Temperatura (°C)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figures_dir = f"{out_dir}/figuras"
    os.makedirs(figures_dir, exist_ok=True)
    path = f"{figures_dir}/{name}"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_error_by_hour(target_dates: pd.Series, errors: list, name: str, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    hours = pd.to_datetime(target_dates).dt.hour.to_numpy()
    for label, err in errors:
        rmse = [np.sqrt(np.mean(err[hours == h] ** 2)) for h in range(24)]
        ax.plot(range(24), rmse, marker="o", markersize=3, label=label)
    ax.set_xlabel("Hora do dia do instante previsto")
    ax.set_ylabel("RMSE no teste (°C)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, out_dir, name)

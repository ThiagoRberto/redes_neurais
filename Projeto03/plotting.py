import os

import matplotlib.pyplot as plt
import numpy as np

from dados import CLASS_NAMES

CMAP = plt.get_cmap("tab10")


def _save(fig, out_dir: str, name: str, show: bool, tight: bool = True) -> str:
    figures_dir = f"{out_dir}/figuras"
    os.makedirs(figures_dir, exist_ok=True)
    path = f"{figures_dir}/{name}"
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path


def _scatter_labels(ax, points: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    for c, name in enumerate(CLASS_NAMES):
        sel = y == c
        ax.scatter(points[sel, 0], points[sel, 1], s=3, alpha=0.5, color=CMAP(c), label=name, rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)


def plot_latent_scatter(points: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, name: str,
                        out_dir: str, show: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    _scatter_labels(ax, points, y, xlabel, ylabel, title)
    leg = ax.legend(markerscale=4, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    for handle in leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles:
        handle.set_alpha(1.0)
    return _save(fig, out_dir, name, show)


def plot_class_panels(points: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, name: str,
                      out_dir: str, show: bool = False) -> str:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), sharex=True, sharey=True)
    for c, ax in enumerate(axes.ravel()):
        ax.scatter(points[:, 0], points[:, 1], s=1, color="lightgray", rasterized=True)
        sel = y == c
        ax.scatter(points[sel, 0], points[sel, 1], s=3, color=CMAP(c), rasterized=True)
        ax.set_title(CLASS_NAMES[c], fontsize=10)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    return _save(fig, out_dir, name, show)


def plot_explained_variance(ratios: np.ndarray, name: str, out_dir: str, show: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs = np.arange(1, len(ratios) + 1)
    ax.bar(xs, ratios * 100.0, color="tab:blue", label="Individual")
    ax.plot(xs, np.cumsum(ratios) * 100.0, marker="o", color="tab:red", label="Acumulada")
    for x, value in zip(xs, np.cumsum(ratios) * 100.0):
        ax.annotate(f"{value:.1f}%", (x, value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Variância explicada (%)")
    ax.set_ylim(0, 112)
    ax.set_title("PCA do espaço latente 8D")
    ax.legend(loc="center right")
    return _save(fig, out_dir, name, show)


def _grid(images: np.ndarray, rows: int, cols: int) -> np.ndarray:
    images = images[:rows * cols].reshape(rows, cols, 28, 28)
    return images.transpose(0, 2, 1, 3).reshape(rows * 28, cols * 28)


def plot_reconstructions(original: np.ndarray, items: list, name: str, out_dir: str, show: bool = False) -> str:
    n = original.shape[0]
    fig, axes = plt.subplots(len(items) + 1, 1, figsize=(1.1 * n, 1.3 * (len(items) + 1)))
    axes[0].imshow(_grid(original, 1, n), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original", fontsize=10)
    for ax, (title, images) in zip(axes[1:], items):
        ax.imshow(_grid(images, 1, n), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
    for ax in axes:
        ax.axis("off")
    return _save(fig, out_dir, name, show)


def plot_samples(items: list, rows: int, cols: int, name: str, out_dir: str, show: bool = False) -> str:
    fig, axes = plt.subplots(1, len(items), figsize=(0.62 * cols * len(items) + 0.4, 0.62 * rows + 0.6),
                             squeeze=False)
    for ax, (title, images) in zip(axes[0], items):
        ax.imshow(_grid(images, rows, cols), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    return _save(fig, out_dir, name, show)


def plot_manifold(images: np.ndarray, axis: np.ndarray, name: str, out_dir: str, show: bool = False) -> str:
    steps = images.shape[0]
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    canvas = images.transpose(0, 2, 1, 3).reshape(steps * 28, steps * 28)
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, extent=(axis[0], axis[-1], axis[0], axis[-1]))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("Variedade 2D decodificada")
    return _save(fig, out_dir, name, show)


def plot_training_curves(items: list, name: str, out_dir: str, show: bool = False) -> str:
    fig, axes = plt.subplots(len(items), 3, figsize=(14, 3.6 * len(items)), squeeze=False)
    for r, (label, history) in enumerate(items):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        for c, (key, title) in enumerate((("loss", "Perda total (-ELBO)"), ("recon", "Reconstrução (BCE)"),
                                          ("kl", "Divergência KL"))):
            ax = axes[r][c]
            ax.plot(epochs, history[f"train_{key}"], label="treino")
            ax.plot(epochs, history[f"val_{key}"], label="validação")
            ax.axvline(history["best_epoch"], color="gray", linestyle="--", linewidth=1, label="melhor época")
            ax.set_xlabel("Época")
            ax.set_ylabel("nats por imagem")
            ax.set_title(f"{label}: {title}", fontsize=10)
            ax.legend(fontsize=8)
    return _save(fig, out_dir, name, show)


def plot_active_units(items: list, name: str, out_dir: str, show: bool = False) -> str:
    fig, axes = plt.subplots(1, len(items), figsize=(5.2 * len(items), 3.8), squeeze=False)
    for ax, (label, kl_per_dim, mu_variance) in zip(axes[0], items):
        xs = np.arange(1, len(kl_per_dim) + 1)
        ax.bar(xs, kl_per_dim, color="tab:blue")
        ax.set_xticks(xs)
        ax.set_xlabel("Dimensão latente")
        ax.set_ylabel("KL média por dimensão (nats)")
        ax.set_title(label)
        twin = ax.twinx()
        twin.plot(xs, mu_variance, marker="o", color="tab:orange")
        twin.axhline(0.01, color="gray", linestyle="--", linewidth=1)
        twin.set_ylabel("Variância de μ", color="tab:orange")
        twin.set_yscale("log")
    return _save(fig, out_dir, name, show)

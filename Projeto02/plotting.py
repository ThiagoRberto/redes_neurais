import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image


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


def save_image(image: np.ndarray, name: str, out_dir: str) -> str:
    images_dir = f"{out_dir}/imagens"
    os.makedirs(images_dir, exist_ok=True)
    path = f"{images_dir}/{name}"
    Image.fromarray(image).save(path)
    return path


def plot_reconstructions(original: np.ndarray, items: list, name: str, out_dir: str, show: bool = False) -> str:
    fig, axes = plt.subplots(1, len(items) + 1, figsize=(4.6 * (len(items) + 1), 3.9))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    for ax, (title, image) in zip(axes[1:], items):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
    for ax in axes:
        ax.axis("off")
    return _save(fig, out_dir, name, show)


def plot_som_maps(items: list, name: str, out_dir: str, show: bool = False) -> str:
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 4.2))
    for ax, (title, model) in zip(np.atleast_1d(axes), items):
        ax.imshow(np.clip(model.weights.reshape(model.rows, model.cols, 3), 0, 1), interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    return _save(fig, out_dir, name, show)


def plot_gng_graphs(items: list, name: str, out_dir: str, show: bool = False) -> str:
    fig = plt.figure(figsize=(5.2 * len(items), 5.2))
    for i, (title, model) in enumerate(items, start=1):
        ax = fig.add_subplot(1, len(items), i, projection="3d")
        w = np.clip(model.weights, 0, 1)
        src, dst = np.nonzero(np.triu(model.edges, 1))
        if src.size:
            ax.add_collection3d(Line3DCollection(np.stack([w[src], w[dst]], axis=1),
                                                 colors="gray", linewidths=0.6, alpha=0.6))
        ax.scatter(w[:, 0], w[:, 1], w[:, 2], c=w, s=36, edgecolors="black", linewidths=0.3, depthshade=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        ax.set_title(f"{title}\n{int(src.size)} arestas")
    fig.subplots_adjust(left=0.01, right=0.97, bottom=0.04, top=0.86, wspace=0.08)
    return _save(fig, out_dir, name, show, tight=False)


def plot_som_convergence(items: list, name: str, out_dir: str, show: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, history in items:
        its, qes = zip(*history)
        ax.plot(its, qes, label=label)
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Erro de quantização (amostra de pixels)")
    ax.set_title("SOM: convergência do erro de quantização")
    ax.set_yscale("log")
    ax.legend()
    return _save(fig, out_dir, name, show)


def plot_gng_convergence(items: list, name: str, out_dir: str, show: bool = False) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, history in items:
        its, qes, units = zip(*history)
        axes[0].plot(its, qes, label=label)
        axes[1].plot(its, units, label=label)
    axes[0].set_xlabel("Iteração")
    axes[0].set_ylabel("Erro de quantização (amostra de pixels)")
    axes[0].set_title("GNG: convergência do erro de quantização")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].set_xlabel("Iteração")
    axes[1].set_ylabel("Neurônios ativos")
    axes[1].set_title("GNG: crescimento da rede")
    axes[1].set_xscale("log")
    axes[1].legend()
    return _save(fig, out_dir, name, show)


def plot_comparison(table, name: str, out_dir: str, show: bool = False) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    metrics = [("qe", "Erro de quantização"), ("te", "Erro topológico")]
    for ax, (key, label) in zip(axes, metrics):
        for method, group in table.groupby("metodo"):
            group = group.sort_values("neuronios")
            ax.errorbar(group["neuronios"], group[f"{key}_mean"], yerr=group[f"{key}_std"],
                        marker="o", capsize=3, label=method)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(table["neuronios"].unique()))
        ax.set_xticklabels(sorted(table["neuronios"].unique()))
        ax.set_xlabel("Neurônios (cores na paleta)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
    return _save(fig, out_dir, name, show)

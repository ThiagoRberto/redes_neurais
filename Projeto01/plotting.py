import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

FIGURES_DIR = "resultados/figuras"
TABLE_PATH = "resultados/tabela_resultados.csv"
SEARCH_LOG_PATH = "resultados/busca_baseline.csv"


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_training_curves(history: dict, model_name: str) -> str:
    save_path = f"{FIGURES_DIR}/{model_name}_curvas_treinamento.png"
    _ensure_dir(save_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="Treino")
    axes[0].plot(history["val_loss"], label="Validação")
    axes[0].axvline(history["best_epoch"] - 1, color="gray", linestyle="--", label="Melhor época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("MSE")
    axes[0].set_title(f"{model_name}: Perda (MSE)")
    axes[0].legend()

    axes[1].plot(history["train_mae"], label="Treino")
    axes[1].plot(history["val_mae"], label="Validação")
    axes[1].axvline(history["best_epoch"] - 1, color="gray", linestyle="--", label="Melhor época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MAE")
    axes[1].set_title(f"{model_name}: Métrica Auxiliar (MAE)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_parity(y_true, y_pred, model_name: str) -> str:
    save_path = f"{FIGURES_DIR}/{model_name}_parity_plot.png"
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="black", linestyle="--", label="y = x")

    ax.set_xlabel("Y real")
    ax.set_ylabel("Y predito")
    ax.set_title(f"{model_name}: Real vs. Predito (Teste)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_residuals(y_true, y_pred, model_name: str) -> str:
    save_path = f"{FIGURES_DIR}/{model_name}_residuos.png"
    _ensure_dir(save_path)

    residuals = (y_true - y_pred).ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Resíduo (y - ŷ)")
    ax.set_ylabel("Frequência")
    ax.set_title(f"{model_name}: Distribuição dos Resíduos (Teste)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def update_results_table(model_name: str, metrics: dict, epochs_trained: int) -> pd.DataFrame:
    _ensure_dir(TABLE_PATH)

    row = {
        "Modelo": model_name,
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"],
        "Épocas até Convergência": epochs_trained,
    }

    if os.path.exists(TABLE_PATH):
        df = pd.read_csv(TABLE_PATH)
        df = df[df["Modelo"] != model_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(TABLE_PATH, index=False)
    return df


def log_search_trial(config: dict, train_loss: float, val_loss: float,
                      epochs_trained: int, best_epoch: int, log_path: str = SEARCH_LOG_PATH) -> pd.DataFrame:
    _ensure_dir(log_path)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **config,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
    }

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(log_path, index=False)
    return df

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imagem import list_images, results_dir

SOM_GROUP = ["size", "alpha0", "sigma0", "sigma_f", "iterations"]
GNG_GROUP = ["units", "eps_b", "eps_n", "age_max", "lam", "tune_iterations"]
SOM_HPS = ["alpha0", "sigma0_rel", "sigma_f", "iterations"]
GNG_HPS = ["eps_b", "eps_n", "age_max", "lam"]


def summarize(df: pd.DataFrame, group: list) -> pd.DataFrame:
    agg = df.groupby(group).agg(qe_mean=("qe", "mean"), qe_std=("qe", "std"),
                                te_mean=("te", "mean"), te_std=("te", "std"),
                                fit_time=("fit_time", "mean"),
                                runs=("seed", "count")).reset_index()
    return agg


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for i, row in df.iterrows():
        dominated = ((df["qe_mean"] <= row["qe_mean"]) & (df["te_mean"] <= row["te_mean"]) &
                     ((df["qe_mean"] < row["qe_mean"]) | (df["te_mean"] < row["te_mean"]))).any()
        if not dominated:
            keep.append(i)
    return df.loc[keep].sort_values("qe_mean")


def plot_sensitivity(df: pd.DataFrame, size_col: str, hps: list, title: str, name: str, figures_dir: str) -> None:
    sizes = sorted(df[size_col].unique())
    fig, axes = plt.subplots(len(sizes), len(hps), figsize=(3.6 * len(hps), 2.9 * len(sizes)), squeeze=False)
    for r, size in enumerate(sizes):
        sub = df[df[size_col] == size]
        for c, hp in enumerate(hps):
            ax = axes[r][c]
            marginal = sub.groupby(hp)[["qe", "te"]].mean()
            xs = np.arange(len(marginal))
            ax.plot(xs, marginal["qe"], marker="o", color="tab:blue")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{v:g}" for v in marginal.index], fontsize=8)
            ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=8)
            twin = ax.twinx()
            twin.plot(xs, marginal["te"], marker="s", color="tab:orange")
            twin.tick_params(axis="y", labelcolor="tab:orange", labelsize=8)
            if r == 0:
                ax.set_title(hp)
            if c == 0:
                ax.set_ylabel(f"{size_col}={size}\nQE", color="tab:blue")
            if c == len(hps) - 1:
                twin.set_ylabel("TE", color="tab:orange")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(f"{figures_dir}/{name}", dpi=150)
    plt.close(fig)


def plot_tradeoff(som: pd.DataFrame, gng: pd.DataFrame, name: str, figures_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for c, size in enumerate(sorted(som["size"].unique())):
        sub = som[som["size"] == size]
        sc = axes[0][c].scatter(sub["te_mean"], sub["qe_mean"], c=sub["sigma_f"], cmap="viridis", s=22)
        front = pareto_front(sub)
        axes[0][c].plot(front["te_mean"], front["qe_mean"], color="red", linewidth=1, label="fronteira de Pareto")
        axes[0][c].set_title(f"SOM {size}x{size} (cor = sigma_f)")
        axes[0][c].set_xlabel("Erro topológico")
        axes[0][c].set_ylabel("Erro de quantização")
        axes[0][c].legend()
        fig.colorbar(sc, ax=axes[0][c])
    for c, units in enumerate(sorted(gng["units"].unique())):
        sub = gng[gng["units"] == units]
        sc = axes[1][c].scatter(sub["te_mean"], sub["qe_mean"], c=np.log10(sub["eps_b"]), cmap="viridis", s=22)
        front = pareto_front(sub)
        axes[1][c].plot(front["te_mean"], front["qe_mean"], color="red", linewidth=1, label="fronteira de Pareto")
        axes[1][c].set_title(f"GNG {units} neurônios (cor = log10 eps_b)")
        axes[1][c].set_xlabel("Erro topológico")
        axes[1][c].set_ylabel("Erro de quantização")
        axes[1][c].legend()
        fig.colorbar(sc, ax=axes[1][c])
    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(f"{figures_dir}/{name}", dpi=150)
    plt.close(fig)


def report(summary: pd.DataFrame, size_col: str, hp_cols: list, top: int = 5) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for size in sorted(summary[size_col].unique()):
        sub = summary[summary[size_col] == size]
        print(f"\n=== {size_col}={size}: {top} menores QE ===")
        print(sub.sort_values("qe_mean").head(top)[hp_cols + ["qe_mean", "qe_std", "te_mean", "fit_time"]]
              .to_string(index=False))
        print(f"--- {size_col}={size}: {top} menores TE ---")
        print(sub.sort_values("te_mean").head(top)[hp_cols + ["qe_mean", "te_mean", "te_std", "fit_time"]]
              .to_string(index=False))
        print(f"--- {size_col}={size}: fronteira de Pareto (QE x TE) ---")
        print(pareto_front(sub)[hp_cols + ["qe_mean", "te_mean"]].to_string(index=False))


def sensitivity_summary(som_raw: pd.DataFrame, gng_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm, df, size_col, hps in (("SOM", som_raw, "size", SOM_HPS), ("GNG", gng_raw, "units", GNG_HPS)):
        for hp in hps:
            qe_var, te_var = [], []
            for _, sub in df.groupby(size_col):
                marginal = sub.groupby(hp)[["qe", "te"]].mean()
                qe_var.append((marginal["qe"].max() - marginal["qe"].min()) / marginal["qe"].mean() * 100.0)
                te_var.append((marginal["te"].max() - marginal["te"].min()) / marginal["te"].mean() * 100.0)
            rows.append({"algoritmo": algorithm, "hiperparametro": hp,
                         "qe_var_pct": float(np.mean(qe_var)), "te_var_pct": float(np.mean(te_var))})
    return pd.DataFrame(rows)


def analyze(image_path: str) -> None:
    out_dir = results_dir(image_path)
    figures_dir = f"{out_dir}/figuras"
    som_raw = pd.read_csv(f"{out_dir}/busca_som.csv")
    gng_raw = pd.read_csv(f"{out_dir}/busca_gng.csv")
    som_raw["sigma0_rel"] = som_raw["sigma0"] / som_raw["size"]

    som = summarize(som_raw, SOM_GROUP)
    gng = summarize(gng_raw, GNG_GROUP)
    som.to_csv(f"{out_dir}/resumo_busca_som.csv", index=False)
    gng.to_csv(f"{out_dir}/resumo_busca_gng.csv", index=False)
    sensitivity_summary(som_raw, gng_raw).to_csv(f"{out_dir}/sensibilidade_resumo.csv", index=False)

    print(f"##### {image_path} #####")
    print(f"SOM: {len(som_raw)} execucoes, {len(som)} configuracoes")
    print(f"GNG: {len(gng_raw)} execucoes, {len(gng)} configuracoes")
    print("GNG com numero de neuronios diferente do alvo:", int((gng_raw["units_final"] != gng_raw["units"]).sum()))

    report(som, "size", SOM_GROUP[1:])
    report(gng, "units", GNG_GROUP[1:])

    plot_sensitivity(som_raw, "size", SOM_HPS, "SOM: sensibilidade aos hiperparâmetros (média das demais configurações)",
                     "sensibilidade_som.png", figures_dir)
    plot_sensitivity(gng_raw, "units", GNG_HPS, "GNG: sensibilidade aos hiperparâmetros (média das demais configurações)",
                     "sensibilidade_gng.png", figures_dir)
    plot_tradeoff(som, gng, "tradeoff_qe_te.png", figures_dir)
    print("\nFiguras de analise salvas em", figures_dir)


if __name__ == "__main__":
    for image_path in (sys.argv[1:] or list_images()):
        analyze(image_path)

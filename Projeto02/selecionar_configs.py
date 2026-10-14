import json
import sys

import numpy as np
import pandas as pd

from imagem import list_images, results_dir

SOM_TOLERANCE = 0.10
GNG_TOLERANCE = 0.02

SOM_REFERENCE = {"alpha0": 0.5, "sigma_f": 0.5, "iterations": 200000}
GNG_REFERENCE = {"eps_b": 0.05, "eps_n": 0.002, "age_max": 100, "lam": 100}


def pick(df: pd.DataFrame, tolerance: float) -> pd.Series:
    best = df["qe_mean"].min()
    candidates = df[df["qe_mean"] <= (1.0 + tolerance) * best]
    return candidates.sort_values(["te_mean", "qe_mean"]).iloc[0]


def match(df: pd.DataFrame, **conditions) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in conditions.items():
        mask &= np.isclose(df[key], value)
    return df[mask].iloc[0]


def select(image_path: str) -> None:
    out_dir = results_dir(image_path)
    som = pd.read_csv(f"{out_dir}/resumo_busca_som.csv")
    gng = pd.read_csv(f"{out_dir}/resumo_busca_gng.csv")
    configs = {"SOM": {}, "GNG": {}}
    comparison = []

    print(f"##### {image_path} #####")
    for size in sorted(som["size"].unique()):
        subset = som[som["size"] == size]
        row = pick(subset, SOM_TOLERANCE)
        ref = match(subset, sigma0=size / 4, **SOM_REFERENCE)
        configs["SOM"][int(size)] = {"alpha0": float(row["alpha0"]), "sigma0": float(row["sigma0"]),
                                     "sigma_f": float(row["sigma_f"]), "iterations": int(row["iterations"])}
        comparison.append({"metodo": "SOM", "neuronios": int(size) ** 2, "qe_ref": ref["qe_mean"],
                           "te_ref": ref["te_mean"], "qe_sel": row["qe_mean"], "te_sel": row["te_mean"]})
        print(f"SOM {int(size)}x{int(size)}: {configs['SOM'][int(size)]}  busca: QE={row['qe_mean']:.4f} TE={row['te_mean']:.4f}")

    for units in sorted(gng["units"].unique()):
        subset = gng[gng["units"] == units]
        row = pick(subset, GNG_TOLERANCE)
        ref = match(subset, **GNG_REFERENCE)
        configs["GNG"][int(units)] = {"eps_b": float(row["eps_b"]), "eps_n": float(row["eps_n"]),
                                      "age_max": int(row["age_max"]), "lam": int(row["lam"]),
                                      "tune_iterations": int(row["tune_iterations"])}
        comparison.append({"metodo": "GNG", "neuronios": int(units), "qe_ref": ref["qe_mean"],
                           "te_ref": ref["te_mean"], "qe_sel": row["qe_mean"], "te_sel": row["te_mean"]})
        print(f"GNG {int(units)}: {configs['GNG'][int(units)]}  busca: QE={row['qe_mean']:.4f} TE={row['te_mean']:.4f}")

    with open(f"{out_dir}/configs_finais.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    pd.DataFrame(comparison).to_csv(f"{out_dir}/comparacao_referencia.csv", index=False)


if __name__ == "__main__":
    for image_path in (sys.argv[1:] or list_images()):
        select(image_path)

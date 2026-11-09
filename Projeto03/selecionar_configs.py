import json

import pandas as pd

SUMMARY_PATH = "resultados/resumo_busca.csv"
CONFIGS_PATH = "resultados/configs_finais.json"


def select() -> dict:
    summary = pd.read_csv(SUMMARY_PATH)
    raw = pd.read_csv("resultados/busca_vae.csv")
    max_epochs = int(raw["max_epochs"].iloc[0])
    patience = int(raw["patience"].iloc[0])
    configs = {}
    for latent_dim in sorted(summary["latent_dim"].unique()):
        row = summary[summary["latent_dim"] == latent_dim].sort_values("val_loss_mean").iloc[0]
        configs[int(latent_dim)] = {"hidden_dims": json.loads(row["hidden_dims"]), "activation": row["activation"],
                                    "lr": float(row["lr"]), "batch_size": int(row["batch_size"]),
                                    "max_epochs": max_epochs, "patience": patience}
        print(f"latente {int(latent_dim)}D: {configs[int(latent_dim)]}  busca: val_loss={row['val_loss_mean']:.3f} "
              f"(+-{row['val_loss_std']:.3f}) params={int(row['n_params'])}")
    with open(CONFIGS_PATH, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    return configs


if __name__ == "__main__":
    select()

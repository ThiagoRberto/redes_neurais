import json

import pandas as pd

SUMMARY_PATH = "resultados/resumo_busca.csv"
CONFIGS_PATH = "resultados/configs_finais.json"
FINAL_MAX_EPOCHS = 150


def row_to_config(row: pd.Series, patience: int) -> dict:
    return {"cell": row["cell"], "window": int(row["window"]), "hidden": int(row["hidden"]),
            "layers": int(row["layers"]), "dropout": float(row["dropout"]), "lr": float(row["lr"]),
            "batch_size": 64, "max_epochs": FINAL_MAX_EPOCHS, "patience": patience,
            "val_loss_busca": float(row["val_loss_mean"])}


def select() -> dict:
    summary = pd.read_csv(SUMMARY_PATH)
    patience = int(pd.read_csv("resultados/busca_rnn.csv")["patience"].iloc[0])
    configs = {}
    for cell in ["lstm", "gru"]:
        sub = summary[summary["cell"] == cell].sort_values("val_loss_mean")
        configs[cell] = row_to_config(sub.iloc[0], patience)
        print(f"{cell}: {configs[cell]}  (+-{sub.iloc[0]['val_loss_std']:.4f}) params={int(sub.iloc[0]['n_params'])}")
    with open(CONFIGS_PATH, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    return configs


if __name__ == "__main__":
    select()

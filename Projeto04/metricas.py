import numpy as np
from sklearn.linear_model import Ridge

from dados import split_of
from treino import predict


def to_raw(pred_scaled: np.ndarray, data: dict) -> np.ndarray:
    return pred_scaled * data["y_std"] + data["y_mean"]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_persist: np.ndarray) -> dict:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mse_persist = float(np.mean((y_persist - y_true) ** 2))
    return {"mse": mse, "rmse": mse ** 0.5, "mae": float(np.mean(np.abs(err))),
            "r2": float(1.0 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
            "skill_persistencia": float(1.0 - mse / mse_persist), "erro_max": float(np.max(np.abs(err)))}


def evaluate_split(y_pred_raw: np.ndarray, data: dict, split: str) -> dict:
    meta = data[f"meta_{split}"]
    return regression_metrics(meta["target"].to_numpy(), y_pred_raw, meta["temp_now"].to_numpy())


def model_predictions(model, data: dict, split: str) -> np.ndarray:
    return to_raw(predict(model, data[f"x_{split}"]).cpu().numpy().astype(np.float64), data)


def persistence_prediction(data: dict, split: str) -> np.ndarray:
    return data[f"meta_{split}"]["temp_now"].to_numpy()


def climatology_prediction(data: dict, split: str) -> np.ndarray:
    frame = data["frame"]
    train = frame[split_of(frame.index) == 0]
    table = train.groupby([train.index.month, train.index.hour])["temp"].mean()
    target_dates = data[f"meta_{split}"]["target_date"]
    keys = list(zip(target_dates.dt.month, target_dates.dt.hour))
    return table.reindex(keys).to_numpy()


def ridge_baseline(data: dict) -> tuple:
    flat = {s: data[f"x_{s}"].cpu().numpy().reshape(data[f"x_{s}"].shape[0], -1) for s in ["train", "val", "test"]}
    y_train, y_val = data["y_train"].cpu().numpy(), data["y_val"].cpu().numpy()
    best = None
    for alpha in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]:
        reg = Ridge(alpha=alpha).fit(flat["train"], y_train)
        val_mse = float(np.mean((reg.predict(flat["val"]) - y_val) ** 2))
        if best is None or val_mse < best[0]:
            best = (val_mse, alpha, reg)
    _, alpha, reg = best
    return alpha, {s: to_raw(reg.predict(flat[s]).astype(np.float64), data) for s in ["val", "test"]}

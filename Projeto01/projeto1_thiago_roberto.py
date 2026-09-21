import copy
import random

import numpy as np
import pandas as pd
import torch

import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from mlp import MLP
from plotting import log_search_trial, plot_parity, plot_residuals, plot_training_curves, update_results_table

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_data(data_path: str) -> tuple:
    df = pd.read_csv(data_path)
    x_data = df.iloc[:, :-1].values
    y_data = df.iloc[:, -1].values
    X_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

    print(f"X_shape: {X_tensor.shape} - Y_shape: {y_tensor.shape}", )

    return (X_tensor, y_tensor)


def scale_data(X_train, X_val, X_test, y_train, y_val, y_test) -> tuple:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train.numpy())
    X_val_s = x_scaler.transform(X_val.numpy())
    X_test_s = x_scaler.transform(X_test.numpy())

    y_train_s = y_scaler.fit_transform(y_train.numpy())
    y_val_s = y_scaler.transform(y_val.numpy())
    y_test_s = y_scaler.transform(y_test.numpy())

    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)

    return (to_tensor(X_train_s), to_tensor(X_val_s), to_tensor(X_test_s),
            to_tensor(y_train_s), to_tensor(y_val_s), to_tensor(y_test_s),
            x_scaler, y_scaler)


def split_data(x: torch.Tensor, y: torch.Tensor, batch_size: int = 32) -> tuple:
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.90, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(8 / 9), random_state=SEED)

    X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler = scale_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_scaler, y_scaler


@torch.no_grad()
def _evaluate_loader(model, loader, mse_criterion, mae_criterion) -> tuple:
    model.eval()
    running_loss, running_mae, n_samples = 0.0, 0.0, 0
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        batch_size = X_batch.size(0)
        running_loss += mse_criterion(y_pred, y_batch).item() * batch_size
        running_mae += mae_criterion(y_pred, y_batch).item() * batch_size
        n_samples += batch_size
    return running_loss / n_samples, running_mae / n_samples


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, lr: float = 0.01,
          momentum: float = 0.0, weight_decay: float = 0.0, l1_lambda: float = 0.0,
          max_epochs: int = 100000, patience: int = 20) -> tuple:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, running_mae, n_samples = 0.0, 0.0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = mse_criterion(y_pred, y_batch)

            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            running_mae += mae_criterion(y_pred, y_batch).item() * batch_size
            n_samples += batch_size

        train_loss = running_loss / n_samples
        train_mae = running_mae / n_samples
        val_loss, val_mae = _evaluate_loader(model, val_loader, mse_criterion, mae_criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["epochs_trained"] = epoch

    return model, history


@torch.no_grad()
def test(model: nn.Module, test_loader: DataLoader, y_scaler: StandardScaler = None) -> tuple:
    model.eval()
    y_true_batches, y_pred_batches = [], []
    for X_batch, y_batch in test_loader:
        y_pred_batches.append(model(X_batch))
        y_true_batches.append(y_batch)

    y_true = torch.cat(y_true_batches).numpy()
    y_pred = torch.cat(y_pred_batches).numpy()

    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true)
        y_pred = y_scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

    return metrics, y_true, y_pred


def pipeline(data_path: str) -> None:
    set_seed(SEED)

    x, y = read_data(data_path)
    train_loader, val_loader, test_loader, x_scaler, y_scaler = split_data(x, y, batch_size=32)

    patience = 100
    max_epochs = 100000
    config = {"hidden_dims": "[256, 128, 64, 32]", "activation": "relu", "batch_size": 32,
              "lr": 0.001, "patience": patience, "max_epochs": max_epochs}

    model = MLP(input_dim=x.shape[1], hidden_dims=[256, 128, 64, 32], output_dim=1, activation="relu")

    model, history = train(model, train_loader, val_loader, lr=0.001, momentum=0.0,
                            weight_decay=0.0, l1_lambda=0.0, patience=patience, max_epochs=max_epochs)

    print(f"Melhor epoca: {history['best_epoch']} (de {history['epochs_trained']} treinadas)")

    metrics, y_true, y_pred = test(model, test_loader, y_scaler=y_scaler)
    print("Metricas no conjunto de teste:", metrics)

    model_name = "baseline_vanilla"

    log_search_trial(config, history['train_loss'][-1], history['val_loss'][-1], history['epochs_trained'], history['best_epoch'])
    plot_training_curves(history, model_name)
    plot_parity(y_true, y_pred, model_name)
    plot_residuals(y_true, y_pred, model_name)
    update_results_table(model_name, metrics, history["epochs_trained"])


if __name__ == "__main__":
    pipeline("dataset_projeto1.csv")

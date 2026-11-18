import copy
import random

import numpy as np
import torch
import torch.nn as nn

from modelo import RNNRegressor

SEED = 42
GRAD_CLIP = 1.0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(input_dim: int, hidden_dim: int, num_layers: int, cell: str, dropout: float,
                device: torch.device) -> RNNRegressor:
    return RNNRegressor(input_dim, hidden_dim, num_layers, cell, dropout).to(device)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, x.size(0), batch_size)])


def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(((predict(model, x) - y) ** 2).mean())


def train(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor,
          lr: float, batch_size: int = 64, max_epochs: int = 200, patience: int = 15, seed: int = SEED) -> tuple:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    shuffler = torch.Generator()
    shuffler.manual_seed(seed)
    n = x_train.size(0)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch, bad, epoch = float("inf"), copy.deepcopy(model.state_dict()), 0, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=shuffler).to(x_train.device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item() * idx.numel()
        val_loss = mse(model, x_val, y_val)
        history["train_loss"].append(total / n)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val, best_state, best_epoch, bad = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["epochs_trained"] = epoch
    return model, history

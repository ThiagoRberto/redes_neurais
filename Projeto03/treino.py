import copy
import random

import numpy as np
import torch

from vae import VAE, vae_loss

SEED = 42
HISTORY_KEYS = ["train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(hidden_dims: list, latent_dim: int, activation: str, device: torch.device) -> VAE:
    return VAE(input_dim=784, hidden_dims=hidden_dims, latent_dim=latent_dim, activation=activation).to(device)


@torch.no_grad()
def evaluate(model: VAE, x: torch.Tensor, batch_size: int = 5000, noise_seed: int = 0) -> tuple:
    model.eval()
    generator = torch.Generator(device=x.device)
    generator.manual_seed(noise_seed)
    total, recon, kl = 0.0, 0.0, 0.0
    for start in range(0, x.size(0), batch_size):
        xb = x[start:start + batch_size]
        mu, logvar = model.encode(xb)
        eps = torch.randn(mu.shape, generator=generator, device=x.device)
        logits = model.decode(model.reparameterize(mu, logvar, eps))
        loss, r, k = vae_loss(logits, xb, mu, logvar)
        total += loss.item()
        recon += r.item()
        kl += k.item()
    n = x.size(0)
    return total / n, recon / n, kl / n


def train(model: VAE, x_train: torch.Tensor, x_val: torch.Tensor, lr: float, batch_size: int,
          max_epochs: int = 100, patience: int = 10, seed: int = SEED) -> tuple:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    shuffler = torch.Generator()
    shuffler.manual_seed(seed)
    n = x_train.size(0)

    history = {key: [] for key in HISTORY_KEYS}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=shuffler).to(x_train.device)
        sums = torch.zeros(3, device=x_train.device)
        for start in range(0, n, batch_size):
            xb = x_train[order[start:start + batch_size]]
            logits, mu, logvar = model(xb)
            loss, recon, kl = vae_loss(logits, xb, mu, logvar)
            optimizer.zero_grad()
            (loss / xb.size(0)).backward()
            optimizer.step()
            sums += torch.stack([loss.detach(), recon.detach(), kl.detach()])

        train_loss, train_recon, train_kl = (sums / n).tolist()
        val_loss, val_recon, val_kl = evaluate(model, x_val)
        for key, value in zip(HISTORY_KEYS, (train_loss, train_recon, train_kl, val_loss, val_recon, val_kl)):
            history[key].append(value)

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

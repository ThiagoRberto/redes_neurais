import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from vae import VAE

ACTIVE_THRESHOLD = 0.01
KNN_NEIGHBORS = 5
N_CLASSES = 10


@torch.no_grad()
def encode_dataset(model: VAE, x: torch.Tensor, batch_size: int = 5000) -> tuple:
    model.eval()
    mus, logvars = [], []
    for start in range(0, x.size(0), batch_size):
        mu, logvar = model.encode(x[start:start + batch_size])
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    return torch.cat(mus).numpy(), torch.cat(logvars).numpy()


@torch.no_grad()
def reconstruct(model: VAE, x: torch.Tensor) -> np.ndarray:
    model.eval()
    mu, _ = model.encode(x)
    return torch.sigmoid(model.decode(mu)).cpu().numpy().reshape(-1, 28, 28)


@torch.no_grad()
def sample_prior(model: VAE, n: int, seed: int = 0) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z = torch.randn(n, model.latent_dim, generator=generator, device=device)
    return torch.sigmoid(model.decode(z)).cpu().numpy().reshape(-1, 28, 28)


@torch.no_grad()
def decode_grid(model: VAE, limit: float = 3.0, steps: int = 20) -> tuple:
    model.eval()
    device = next(model.parameters()).device
    axis = np.linspace(-limit, limit, steps)
    zx, zy = np.meshgrid(axis, axis[::-1])
    z = torch.tensor(np.stack([zx.ravel(), zy.ravel()], axis=1), dtype=torch.float32, device=device)
    images = torch.sigmoid(model.decode(z)).cpu().numpy().reshape(steps, steps, 28, 28)
    return images, axis


def fit_pca(mu_train: np.ndarray, n_components: int = 2) -> PCA:
    return PCA(n_components=n_components).fit(mu_train)


def explained_variance(mu_train: np.ndarray) -> np.ndarray:
    return PCA(n_components=mu_train.shape[1]).fit(mu_train).explained_variance_ratio_


def active_units(mu: np.ndarray) -> np.ndarray:
    return mu.var(axis=0) > ACTIVE_THRESHOLD


def per_dim_kl(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    return (-0.5 * (1.0 + logvar - mu ** 2 - np.exp(logvar))).mean(axis=0)


def separation_metrics(z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray,
                       seed: int = 42) -> dict:
    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS).fit(z_train, y_train)
    pred = knn.predict(z_test)
    recall = [float((pred[y_test == c] == c).mean()) for c in range(N_CLASSES)]
    clusters = KMeans(n_clusters=N_CLASSES, n_init=10, random_state=seed).fit_predict(z_test)
    return {"silhouette": float(silhouette_score(z_test, y_test)),
            "knn_acc": float((pred == y_test).mean()),
            "kmeans_nmi": float(normalized_mutual_info_score(y_test, clusters)),
            "kmeans_ari": float(adjusted_rand_score(y_test, clusters)),
            "knn_recall": recall}


def latent_analysis(model: VAE, data: dict, seed: int = 42) -> dict:
    mu_train, _ = encode_dataset(model, data["x_train"])
    mu_test, logvar_test = encode_dataset(model, data["x_test"])
    y_train, y_test = data["y_train"], data["y_test"]
    latent_dim = mu_train.shape[1]

    ratios = explained_variance(mu_train)
    pca = fit_pca(mu_train, 2)
    proj_train, proj_test = pca.transform(mu_train), pca.transform(mu_test)

    result = {"latent_dim": latent_dim,
              "active_units": int(active_units(mu_test).sum()),
              "mu_variance": mu_test.var(axis=0).tolist(),
              "kl_per_dim": per_dim_kl(mu_test, logvar_test).tolist(),
              "explained_ratio": ratios.tolist(),
              "explained_2d": float(pca.explained_variance_ratio_.sum()),
              "latent": separation_metrics(mu_train, y_train, mu_test, y_test, seed),
              "projection": separation_metrics(proj_train, y_train, proj_test, y_test, seed)}
    return result

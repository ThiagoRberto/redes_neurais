from mlp import MLP
from plotting import log_search_trial, plot_parity, plot_residuals, plot_training_curves, update_results_table
from projeto1_thiago_roberto import SEED, read_data, set_seed, split_data, test, train

HIDDEN_DIMS = [256, 128, 64, 32]
ACTIVATION = "relu"
BATCH_SIZE = 32
LR = 0.001
PATIENCE = 100
MAX_EPOCHS = 100000

ABLATION_SEARCH_LOG = "resultados/busca_ablacao.csv"


def run_search_trial(x, y, tipo: str, valor: float, dropout_p: float = 0.0, momentum: float = 0.0,
                      weight_decay: float = 0.0, l1_lambda: float = 0.0) -> dict:
    set_seed(SEED)

    train_loader, val_loader, test_loader, x_scaler, y_scaler = split_data(x, y, batch_size=BATCH_SIZE)
    model = MLP(input_dim=x.shape[1], hidden_dims=HIDDEN_DIMS, output_dim=1,
                activation=ACTIVATION, dropout_p=dropout_p)

    model, history = train(model, train_loader, val_loader, lr=LR, momentum=momentum,
                            weight_decay=weight_decay, l1_lambda=l1_lambda,
                            patience=PATIENCE, max_epochs=MAX_EPOCHS)

    config = {"tipo": tipo, "valor": valor, "hidden_dims": str(HIDDEN_DIMS), "activation": ACTIVATION,
              "batch_size": BATCH_SIZE, "lr": LR, "patience": PATIENCE, "max_epochs": MAX_EPOCHS}
    log_search_trial(config, history["train_loss"][-1], history["val_loss"][-1],
                      history["epochs_trained"], history["best_epoch"], log_path=ABLATION_SEARCH_LOG)

    print(f"[{tipo}={valor}] melhor epoca: {history['best_epoch']} (de {history['epochs_trained']}) "
          f"| val_loss={history['val_loss'][-1]:.4f}")

    return history


def run_ablation_search(x, y) -> None:
    for beta in [0.5, 0.7, 0.9, 0.95, 0.99]:
        run_search_trial(x, y, "momentum", beta, momentum=beta)

    for lam in [1e-2, 1e-3, 1e-4, 1e-5]:
        run_search_trial(x, y, "l2", lam, weight_decay=lam)

    for lam in [1e-2, 1e-3, 1e-4, 1e-5]:
        run_search_trial(x, y, "l1", lam, l1_lambda=lam)

    for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
        run_search_trial(x, y, "dropout", p, dropout_p=p)


SEARCH_MAX_EPOCHS = 20000


def run_combo_trial(x, y, tipo: str, hidden_dims: list = None, lr: float = None, dropout_p: float = 0.0,
                     momentum: float = 0.0, weight_decay: float = 0.0, l1_lambda: float = 0.0) -> dict:
    hidden_dims = hidden_dims if hidden_dims is not None else HIDDEN_DIMS
    lr = lr if lr is not None else LR

    set_seed(SEED)
    train_loader, val_loader, test_loader, x_scaler, y_scaler = split_data(x, y, batch_size=BATCH_SIZE)
    model = MLP(input_dim=x.shape[1], hidden_dims=hidden_dims, output_dim=1,
                activation=ACTIVATION, dropout_p=dropout_p)

    model, history = train(model, train_loader, val_loader, lr=lr, momentum=momentum,
                            weight_decay=weight_decay, l1_lambda=l1_lambda,
                            patience=PATIENCE, max_epochs=SEARCH_MAX_EPOCHS)

    config = {
        "tipo": tipo, "momentum": momentum, "weight_decay": weight_decay,
        "l1_lambda": l1_lambda, "dropout_p": dropout_p, "hidden_dims": str(hidden_dims),
        "activation": ACTIVATION, "batch_size": BATCH_SIZE, "lr": lr,
        "patience": PATIENCE, "max_epochs": SEARCH_MAX_EPOCHS,
    }
    log_search_trial(config, history["train_loss"][-1], history["val_loss"][-1],
                      history["epochs_trained"], history["best_epoch"], log_path=ABLATION_SEARCH_LOG)

    print(f"[{tipo}] mom={momentum} wd={weight_decay} l1={l1_lambda} drop={dropout_p} arch={hidden_dims} lr={lr} "
          f"-> melhor epoca {history['best_epoch']}/{history['epochs_trained']} | val_loss={history['val_loss'][-1]:.4f}")

    return history


def run_extended_battery(x, y) -> None:
    for lam in [2e-5, 5e-5, 2e-4, 3e-4, 5e-4, 7e-4]:
        run_combo_trial(x, y, "l2_fino", weight_decay=lam)

    for lam in [2e-5, 5e-5, 2e-4, 3e-4, 5e-4, 7e-4]:
        run_combo_trial(x, y, "l1_fino", l1_lambda=lam)

    for l1v, l2v in [(1e-4, 1e-4), (1e-4, 1e-5), (1e-5, 1e-4)]:
        run_combo_trial(x, y, "l1+l2", l1_lambda=l1v, weight_decay=l2v)

    for beta in [0.5, 0.9]:
        for lam in [1e-5, 1e-4, 1e-3]:
            run_combo_trial(x, y, "momentum+l2", momentum=beta, weight_decay=lam)

    for beta in [0.5, 0.9]:
        for lam in [1e-5, 1e-4, 1e-3]:
            run_combo_trial(x, y, "momentum+l1", momentum=beta, l1_lambda=lam)

    for p in [0.02, 0.05, 0.08]:
        run_combo_trial(x, y, "dropout_leve", dropout_p=p)

    small_arch = [128, 64]
    run_combo_trial(x, y, "arch_pequena+l2", hidden_dims=small_arch, lr=0.01, weight_decay=1e-3)
    run_combo_trial(x, y, "arch_pequena+l1", hidden_dims=small_arch, lr=0.01, l1_lambda=1e-4)
    run_combo_trial(x, y, "arch_pequena+dropout", hidden_dims=small_arch, lr=0.01, dropout_p=0.2)
    run_combo_trial(x, y, "arch_pequena+momentum", hidden_dims=small_arch, lr=0.01, momentum=0.9)


SMALL_HIDDEN_DIMS = [128, 64]
SMALL_LR = 0.01


def run_small_arch_search(x, y) -> None:
    for beta in [0.5, 0.7, 0.9, 0.95, 0.99]:
        run_combo_trial(x, y, "pequeno_momentum", hidden_dims=SMALL_HIDDEN_DIMS, lr=SMALL_LR, momentum=beta)

    for lam in [1e-4, 1e-3, 1e-2]:
        run_combo_trial(x, y, "pequeno_l2", hidden_dims=SMALL_HIDDEN_DIMS, lr=SMALL_LR, weight_decay=lam)

    for lam in [1e-5, 1e-4, 1e-3, 1e-2]:
        run_combo_trial(x, y, "pequeno_l1", hidden_dims=SMALL_HIDDEN_DIMS, lr=SMALL_LR, l1_lambda=lam)

    for p in [0.05, 0.1, 0.2, 0.3, 0.5]:
        run_combo_trial(x, y, "pequeno_dropout", hidden_dims=SMALL_HIDDEN_DIMS, lr=SMALL_LR, dropout_p=p)


def run_variant(x, y, model_name: str, hidden_dims: list = None, lr: float = None, dropout_p: float = 0.0,
                 momentum: float = 0.0, weight_decay: float = 0.0, l1_lambda: float = 0.0) -> dict:
    hidden_dims = hidden_dims if hidden_dims is not None else HIDDEN_DIMS
    lr = lr if lr is not None else LR

    set_seed(SEED)

    train_loader, val_loader, test_loader, x_scaler, y_scaler = split_data(x, y, batch_size=BATCH_SIZE)
    model = MLP(input_dim=x.shape[1], hidden_dims=hidden_dims, output_dim=1,
                activation=ACTIVATION, dropout_p=dropout_p)

    model, history = train(model, train_loader, val_loader, lr=lr, momentum=momentum,
                            weight_decay=weight_decay, l1_lambda=l1_lambda,
                            patience=PATIENCE, max_epochs=MAX_EPOCHS)

    print(f"[{model_name}] melhor epoca: {history['best_epoch']} (de {history['epochs_trained']} treinadas)")

    metrics, y_true, y_pred = test(model, test_loader, y_scaler=y_scaler)
    print(f"[{model_name}] metricas no teste:", metrics)

    plot_training_curves(history, model_name)
    plot_parity(y_true, y_pred, model_name)
    plot_residuals(y_true, y_pred, model_name)
    update_results_table(model_name, metrics, history["epochs_trained"])

    return metrics


if __name__ == "__main__":
    x, y = read_data("dataset_projeto1.csv")
    run_variant(x, y, "baseline_momentum", momentum=0.3)

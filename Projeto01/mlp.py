import torch

import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64], output_dim: int = 1,
                 activation: str = "relu", dropout_p: float = 0.0):
        super(MLP, self).__init__()
        act_cls = ACTIVATIONS[activation]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

import torch
import torch.nn as nn

CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}


class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, cell: str = "lstm",
                 dropout: float = 0.0):
        super(RNNRegressor, self).__init__()
        self.rnn = CELLS[cell](input_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)

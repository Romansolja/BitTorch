import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """LSTM model for return prediction (not price level)"""

    def __init__(self, n_features: int = 11, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


# Keep old class for backward compatibility if needed
BitcoinLSTM = LSTMRegressor
import torch
import torch.nn as nn

class BitcoinLSTM(nn.Module):
    def __init__(self, in_size=1, hid=128, layers=2, out_size=1, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hid, num_layers=layers,
                            batch_first=True, dropout=drop if layers > 1 else 0)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hid, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)
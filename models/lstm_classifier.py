import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = False
    ):

        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # 2 classes: long (0) or short (1)
        )

    def forward(self, x):

        lstm_out, (hidden, cell) = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        out = self.fc(last_output)

        return out
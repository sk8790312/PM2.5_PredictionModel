"""Small unidirectional LSTM for seven-day AQI regression."""
import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, lookback=7, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=False)
        # Explicit output dropout also works with a one-layer LSTM.
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.lookback, 1):
            raise ValueError(f'Expected [batch, {self.lookback}, 1] input.')
        # No persisted hidden state: each seven-day sample starts from zeros.
        _, (hidden, _) = self.lstm(x)
        return self.head(self.dropout(hidden[-1])).squeeze(-1)


def load_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = LSTMRegressor(**saved['model_config'])
    model.load_state_dict(saved['model_state'])
    model.eval()
    return model, saved


def predict_aqi(model, saved, history):
    """History must be seven consecutive observed AQIs, oldest first."""
    values = torch.as_tensor(history, dtype=torch.float32)
    if values.shape != (model.lookback,) or not torch.isfinite(values).all():
        raise ValueError(f'Provide {model.lookback} finite consecutive daily AQI values.')
    if (values < 0).any():
        raise ValueError('AQI history must be non-negative.')
    scaler = saved['scaler']
    x = ((values - scaler['mean']) / scaler['scale']).reshape(1, model.lookback, 1)
    model.eval()
    with torch.inference_mode():
        return float(model(x).item() * scaler['scale'] + scaler['mean'])

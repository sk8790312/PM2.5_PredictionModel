"""Small causal self-attention model for numerical next-day regression."""
import torch
from torch import nn


class CausalTransformer(nn.Module):
    def __init__(self, lookback=7, d_model=32, nhead=2, num_layers=1,
                 dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.normal_(self.position, std=0.02)
        # This building block has self-attention and no cross-attention.
        # The explicit mask makes the stack causal (decoder-only).
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer('causal_mask',
            torch.triu(torch.ones(lookback, lookback, dtype=torch.bool), diagonal=1))

    def hidden_states(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.lookback, 1):
            raise ValueError(f'Expected [batch, {self.lookback}, 1] input.')
        return self.blocks(self.projection(x) + self.position, mask=self.causal_mask)

    def forward(self, x):
        return self.head(self.hidden_states(x)[:, -1]).squeeze(-1)


def load_checkpoint(path):
    """Load only a state dictionary and plain configuration, on CPU."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = CausalTransformer(**saved['model_config'])
    model.load_state_dict(saved['model_state'])
    model.eval()
    return model, saved


def predict_aqi(model, saved, history):
    """History must be consecutive observed days, oldest first."""
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

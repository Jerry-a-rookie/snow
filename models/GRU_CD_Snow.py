import torch
import torch.nn as nn

from models.SnowNet import build_snow_block


class Model(nn.Module):
    """Channel-dependent GRU with Snow residual adapters.

    The GRU receives the complete multivariate sequence and can mix channels
    in its hidden state.  Snow adapters are applied before the GRU and to its
    multi-channel forecast.

    Input:  [B, L, C]
    Output: [B, P, C]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)

        hidden_dim = int(getattr(configs, "hidden_dim", 64))
        e_layers = int(getattr(configs, "e_layers", 1))
        dropout = float(getattr(configs, "dropout", 0.0))
        gru_dropout = dropout if e_layers > 1 else 0.0

        self.snow_input = build_snow_block(configs, self.seq_len)
        self.snow_output = build_snow_block(configs, self.pred_len)
        self.gru = nn.GRU(
            input_size=self.channels,
            hidden_size=hidden_dim,
            num_layers=e_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, self.pred_len * self.channels)

    def forward(self, x, *args, future_patch=None, base_pred=None, **kwargs):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.snow_input(x)
        _, h = self.gru(x)
        z = self.dropout(h[-1])
        out = self.proj(z).reshape(x.size(0), self.pred_len, self.channels)
        return self.snow_output(out) + seq_last

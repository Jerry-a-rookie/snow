import torch
import torch.nn as nn


class Model(nn.Module):
    """Channel-independent GRU baseline.

    Each channel is treated as an independent univariate sequence while all
    channels share the same GRU and projection parameters.

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

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=e_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, self.pred_len)

    def forward(self, x, *args, future_patch=None, base_pred=None, **kwargs):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        batch_size, seq_len, channels = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * channels, seq_len, 1)

        _, h = self.gru(x)
        z = self.dropout(h[-1])
        out = self.proj(z)
        out = out.reshape(batch_size, channels, self.pred_len).permute(0, 2, 1)

        return out + seq_last

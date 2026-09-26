import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(self, seq_len: int, channels: int, hidden_dim: int, dropout: float):
        super(MixerBlock, self).__init__()
        self.time_norm = nn.LayerNorm(channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
        )
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, max(channels * 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(channels * 2, 16), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.time_norm(x).permute(0, 2, 1)
        y = self.time_mlp(y).permute(0, 2, 1)
        x = x + y
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class Model(nn.Module):
    """
    Lightweight TSMixer-style baseline.

    Input:  [B, L, C]
    Output: [B, P, C]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.individual = bool(getattr(configs, "individual", False))

        hidden_dim = int(getattr(configs, "hidden_dim", max(64, min(256, self.seq_len * 2))))
        e_layers = int(getattr(configs, "e_layers", 2))
        dropout = float(getattr(configs, "dropout", 0.1))
        self.blocks = nn.Sequential(
            *[MixerBlock(self.seq_len, self.channels, hidden_dim, dropout) for _ in range(e_layers)]
        )
        if self.individual:
            self.head = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
        else:
            self.head = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, *args, future_patch=None, base_pred=None, **kwargs):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        z = self.blocks(x)

        if self.individual:
            out = torch.empty(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                out[:, :, i] = self.head[i](z[:, :, i])
        else:
            out = self.head(z.permute(0, 2, 1)).permute(0, 2, 1)

        return out + seq_last

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.padding > 0:
            y = y[:, :, :-self.padding]
        y = self.pointwise(y)
        y = self.dropout(self.act(self.norm(y)))
        return x + y


class Model(nn.Module):
    """
    Lightweight temporal convolution baseline.

    Input:  [B, L, C]
    Output: [B, P, C]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.individual = bool(getattr(configs, "individual", False))

        e_layers = int(getattr(configs, "e_layers", 3))
        kernel_size = int(getattr(configs, "kernel_size", 3))
        dropout = float(getattr(configs, "dropout", 0.1))
        block_channels = 1 if self.individual else self.channels
        self.blocks = nn.Sequential(
            *[
                TemporalBlock(
                    channels=block_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
                for i in range(e_layers)
            ]
        )
        if self.individual:
            self.head = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.head = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, *args, future_patch=None, base_pred=None, **kwargs):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            bsz, seq_len, channels = x.shape
            z = x.permute(0, 2, 1).reshape(bsz * channels, 1, seq_len)
            z = self.blocks(z).squeeze(1).view(bsz, channels, seq_len).permute(0, 2, 1)
            z = z.permute(0, 2, 1).reshape(bsz * channels, seq_len)
            out = self.head(z).view(bsz, channels, self.pred_len).permute(0, 2, 1)
            return out + seq_last

        z = self.blocks(x.permute(0, 2, 1))
        out = self.head(z).permute(0, 2, 1)
        return out + seq_last

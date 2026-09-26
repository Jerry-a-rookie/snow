import torch
import torch.nn as nn
from models.SnowNet import build_snow_block


class Model(nn.Module):
    """
    Reversible-normalization Linear baseline.

    Input:  [B, L, C]
    Output: [B, P, C]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.individual = bool(getattr(configs, "individual", False))
        self.eps = float(getattr(configs, "eps", 1e-5))
        self.snow_input = build_snow_block(configs, self.seq_len)
        self.snow_output = build_snow_block(configs, self.pred_len)
        if self.individual:
            self.Linear = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, *args, future_patch=None, base_pred=None, **kwargs):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(self.eps)
        x = (x - mean) / std
        x = self.snow_input(x)
        if self.individual:
            out = torch.empty(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                out[:, :, i] = self.Linear[i](x[:, :, i])
        else:
            out = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        return self.snow_output(out) * std + mean

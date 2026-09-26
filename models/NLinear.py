import torch
import torch.nn as nn


class Model(nn.Module):
    """
    NLinear baseline.

    It predicts the residual relative to the last observed value and then
    restores the level, which is useful for non-stationary series.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.individual = bool(configs.individual)
        if self.individual:
            self.Linear = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, *args, future_patch=None, base_pred=None, **kwargs):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            out = torch.empty(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                out[:, :, i] = self.Linear[i](x[:, :, i])
        else:
            out = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        return out + seq_last

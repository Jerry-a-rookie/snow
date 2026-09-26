import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Lightweight PatchTST-style encoder.

    Input:  [B, L, C]
    Output: [B, P, C]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.individual = bool(getattr(configs, "individual", False))

        self.patch_len = int(getattr(configs, "patch_len", min(16, self.seq_len)))
        self.stride = int(getattr(configs, "stride", max(1, self.patch_len // 2)))
        self.d_model = int(getattr(configs, "d_model", 64))
        self.n_heads = int(getattr(configs, "n_heads", 4))
        self.e_layers = int(getattr(configs, "e_layers", 2))
        self.dropout = float(getattr(configs, "dropout", 0.1))
        self.patch_ff = int(getattr(configs, "d_ff", self.d_model * 2))

        self.num_patches = max(1, (self.seq_len - self.patch_len) // self.stride + 1)
        self.patch_proj = nn.Linear(self.patch_len, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        self.encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_heads,
                    dim_feedforward=self.patch_ff,
                    dropout=self.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.e_layers)
            ]
        )

        if self.individual:
            self.head = nn.ModuleList(
                [nn.Linear(self.num_patches * self.d_model, self.pred_len) for _ in range(self.channels)]
            )
        else:
            self.head = nn.Linear(self.num_patches * self.d_model, self.pred_len)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < self.patch_len:
            pad_len = self.patch_len - x.size(1)
            pad = x[:, :1, :].repeat(1, pad_len, 1)
            x = torch.cat([pad, x], dim=1)
        x = x.permute(0, 2, 1)
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def forward(self, x, *args, future_patch=None, base_pred=None, **kwargs):
        seq_mean = x.mean(dim=1, keepdim=True).detach()
        seq_std = x.std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(1e-5)
        x_norm = (x - seq_mean) / seq_std

        patches = self._patchify(x_norm)
        bsz, channels, num_patches, patch_len = patches.shape
        patches = patches.reshape(bsz * channels, num_patches, patch_len)

        z = self.patch_proj(patches) + self.pos_embed[:, :num_patches, :]
        for layer in self.encoder:
            z = layer(z)
        z = z.reshape(bsz, channels, num_patches * self.d_model)

        if self.individual:
            out = torch.empty(bsz, self.pred_len, channels, dtype=x.dtype, device=x.device)
            for i in range(channels):
                out[:, :, i] = self.head[i](z[:, i, :])
        else:
            out = self.head(z).permute(0, 2, 1)

        return out * seq_std + seq_mean

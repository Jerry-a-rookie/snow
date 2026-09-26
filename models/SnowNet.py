import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer


class SnowNet(nn.Module):
    """Sample-adaptive signed low-rank channel mixer."""

    def __init__(self, d_series, d_core, n_clusters=8, dropout=0.0, temperature=1.0):
        super().__init__()
        if d_series <= 0 or d_core <= 0 or n_clusters <= 0:
            raise ValueError("d_series, d_core, and n_clusters must be positive")

        self.d_series = d_series
        self.d_core = d_core
        self.n_clusters = n_clusters
        self.temperature = temperature

        self.value_proj = nn.Sequential(
            nn.Linear(d_series, d_series),
            nn.GELU(),
            nn.Linear(d_series, d_core),
        )
        route_dim = d_series // 2 + 1
        self.route_proj = nn.Sequential(
            nn.Linear(route_dim, d_core),
            nn.GELU(),
            nn.Linear(d_core, n_clusters),
        )
        self.out = nn.Sequential(
            nn.Linear(d_series + d_core, d_series),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_series, d_series),
        )

    def _route(self, x):
        # CUDA half-precision cuFFT does not support non-power-of-two lengths
        # such as seq_len=6, so keep only the FFT computation in float32.
        route_feature = torch.abs(torch.fft.rfft(x.float(), dim=-1))
        logits = self.route_proj(route_feature) / max(self.temperature, 1e-6)
        magnitude = F.softmax(torch.abs(logits), dim=-1)
        return torch.sign(logits) * magnitude

    @staticmethod
    def _aggregate_local(values, route):
        numerator = torch.einsum("bnk,bnd->bkd", route, values)
        denominator = route.abs().sum(dim=1).unsqueeze(-1) + 1e-6
        return numerator / denominator

    @staticmethod
    def _write_back(local, route):
        return torch.einsum("bnk,bkd->bnd", route, local)

    def forward(self, input, *args, **kwargs):
        route = self._route(input)
        values = self.value_proj(input)
        local = self._aggregate_local(values, route)
        channel_context = self._write_back(local, route)
        output = self.out(torch.cat([input, channel_context], dim=-1))
        return output, route


class SnowResidualBlock(nn.Module):
    """Residual Snow adapter for [B, L, N] or [B, N, D] tensors."""

    def __init__(
        self,
        d_series,
        d_core,
        n_clusters=8,
        dropout=0.0,
        temperature=1.0,
        residual_scale=0.1,
        learnable_scale=True,
    ):
        super().__init__()
        self.snow = SnowNet(
            d_series=d_series,
            d_core=d_core,
            n_clusters=n_clusters,
            dropout=dropout,
            temperature=temperature,
        )
        scale = torch.tensor(float(residual_scale), dtype=torch.float32)
        if learnable_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    def forward_tokens(self, x):
        snow_output, _ = self.snow(x)
        return x + self.scale * (snow_output - x)

    def forward(self, x):
        tokens = x.permute(0, 2, 1)
        tokens = self.forward_tokens(tokens)
        return tokens.permute(0, 2, 1)


def build_snow_block(configs, d_series):
    return SnowResidualBlock(
        d_series=d_series,
        d_core=int(getattr(configs, "d_core", getattr(configs, "d_model", d_series))),
        n_clusters=int(getattr(configs, "snow_clusters", 8)),
        dropout=float(getattr(configs, "dropout", 0.0)),
        temperature=float(getattr(configs, "snow_temperature", 1.0)),
        residual_scale=float(getattr(configs, "snow_residual_scale", 0.1)),
        learnable_scale=bool(int(getattr(configs, "snow_learnable_scale", 1))),
    )


class Model(nn.Module):
    """Forecasting backbone whose channel-interaction layers are SnowNet."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = bool(configs.use_norm)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.dropout,
        )

        def build_interaction():
            return SnowNet(
                d_series=configs.d_model,
                d_core=configs.d_core,
                n_clusters=configs.snow_clusters,
                dropout=configs.dropout,
                temperature=configs.snow_temperature,
            )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    build_interaction(),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Lightweight standalone Amplifier-style forecaster.

    It amplifies mirrored frequency energy, forecasts seasonal/trend components,
    then restores part of the mirrored spectrum.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.hidden_size = int(getattr(configs, "hidden_size", 128))
        self.use_common_pattern = bool(getattr(configs, "SCI", False))

        kernel_size = int(getattr(configs, "kernel_size", 25))
        self.decomposition = series_decomp(kernel_size)
        self.mask_matrix = nn.Parameter(torch.ones(self.seq_len // 2 + 1, self.channels))
        self.freq_linear = nn.Linear(self.seq_len // 2 + 1, self.pred_len // 2 + 1).to(torch.cfloat)

        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

        self.extract_common_pattern = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.LeakyReLU(),
            nn.Linear(self.channels, 1),
        )
        self.model_common_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len),
        )
        self.model_specific_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len),
        )

    def forward(self, x, *args, future_patch=None, base_pred=None, **kwargs):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        x_norm = (x - mean) / std
        x_fft = torch.fft.rfft(x_norm, dim=1)
        x_inverse_fft = torch.flip(x_fft, dims=[1]) * self.mask_matrix
        x_amplified = torch.fft.irfft(x_fft + x_inverse_fft, n=self.seq_len, dim=1)

        if self.use_common_pattern:
            common = self.extract_common_pattern(x_amplified)
            common = self.model_common_pattern(common.permute(0, 2, 1)).permute(0, 2, 1)
            specific = x_amplified - common.repeat(1, 1, self.channels)
            specific = self.model_specific_pattern(specific.permute(0, 2, 1)).permute(0, 2, 1)
            x_amplified = specific + common.repeat(1, 1, self.channels)
        seasonal, trend = self.decomposition(x_amplified)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out_amp = seasonal + trend

        out_fft = torch.fft.rfft(out_amp, n=self.pred_len, dim=1)
        restored = self.freq_linear(x_inverse_fft.permute(0, 2, 1)).permute(0, 2, 1)
        out = torch.fft.irfft(out_fft - restored, n=self.pred_len, dim=1)
        return out * std + mean

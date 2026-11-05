"""Diagonal state-space convolutions for one- and multi-dimensional inputs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class S4DKernel(nn.Module):
    """Learned S4D-Lin-style diagonal kernel, generated for the requested length."""

    def __init__(self, channels: int):
        super().__init__()
        # Stable continuous-time poles; the imaginary parts use the S4D-Lin grid.
        n_modes = max(8, min(32, channels))
        self.log_dt = nn.Parameter(torch.empty(channels).uniform_(math.log(1e-3), math.log(1e-1)))
        self.log_decay = nn.Parameter(torch.full((channels, n_modes), math.log(0.5)))
        self.frequency = nn.Parameter(
            math.pi * torch.arange(1, n_modes + 1, dtype=torch.float32).expand(channels, -1).clone()
        )
        self.real = nn.Parameter(torch.randn(channels, n_modes) / math.sqrt(n_modes))
        self.imag = nn.Parameter(torch.randn(channels, n_modes) / math.sqrt(n_modes))

    def forward(self, length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        t = torch.arange(length, device=device, dtype=torch.float32)
        dt = self.log_dt.float().exp().view(-1, 1, 1)
        decay = self.log_decay.float().exp().unsqueeze(-1) * dt
        phase = self.frequency.float().unsqueeze(-1) * dt * t
        response = torch.exp(-decay * t) * (
            self.real.float().unsqueeze(-1) * phase.cos()
            - self.imag.float().unsqueeze(-1) * phase.sin()
        )
        # Sum diagonal modes into a real-valued impulse response.
        return response.sum(dim=1).to(dtype=dtype)


def fft_convolve_axis(x: Tensor, kernel: Tensor, axis: int) -> Tensor:
    """Apply centered same-length convolution without circular wrap.

    ``x`` has shape ``[B,C,...]`` and ``kernel`` has shape ``[C,K]``. Padding
    to ``2L-1`` is sufficient for the same-length output: only lags in
    ``[-(L-1), L-1]`` can contribute, so a full ``L+K-1`` convolution is
    unnecessary for long kernels.
    """
    axis %= x.ndim
    if axis < 2:
        raise ValueError("convolution axis must be a spatial axis")
    length = x.shape[axis]
    fft_length = 1 << (2 * length - 2).bit_length()
    center = (kernel.shape[-1] - 1) // 2
    lag_low = max(-center, -(length - 1))
    lag_high = min(kernel.shape[-1] - 1 - center, length - 1)
    source = kernel[:, center + lag_low:center + lag_high + 1]
    lag = torch.arange(lag_low, lag_high + 1, device=x.device).remainder(fft_length)
    centered_kernel = kernel.new_zeros((kernel.shape[0], fft_length)).index_copy(1, lag, source)
    moved = x.movedim(axis, -1)
    xf = torch.fft.rfft(moved.float(), n=fft_length, dim=-1)
    kf = torch.fft.rfft(centered_kernel.float(), n=fft_length, dim=-1)
    shape = [1] * xf.ndim
    shape[1] = kernel.shape[0]
    shape[-1] = kf.shape[-1]
    yf = xf * kf.reshape(shape)
    result = torch.fft.irfft(yf, n=fft_length, dim=-1)[..., :length]
    return result.movedim(-1, axis).to(dtype=x.dtype)


class S4DConv(nn.Module):
    """Bidirectional diagonal state-space convolution on one spatial axis."""

    def __init__(self, channels: int):
        super().__init__()
        self.kernel = S4DKernel(channels)
        self.skip = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor, axis: int = -1) -> Tensor:
        axis %= x.ndim
        length = x.shape[axis]
        forward = self.kernel(length, device=x.device, dtype=x.dtype)
        kernel = forward + forward.flip(-1)
        y = fft_convolve_axis(x, kernel, axis)
        shape = [1] * x.ndim
        shape[1] = x.shape[1]
        return y + x * self.skip.to(dtype=x.dtype).reshape(shape)


class S4NDConv(nn.Module):
    """Separable S4ND convolution; its multidimensional kernel is an outer product."""

    def __init__(self, channels: int, dimensions: int):
        super().__init__()
        self.axes = nn.ModuleList(S4DConv(channels) for _ in range(dimensions))

    def forward(self, x: Tensor) -> Tensor:
        for offset, layer in enumerate(self.axes, start=2):
            x = layer(x, axis=offset)
        return x


class GatedS4ND(nn.Module):
    """Gated, bidirectional separable S4ND mixer with a short local convolution."""

    def __init__(self, channels: int, dimensions: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv1d(channels, channels * 2, 1)
        self.short = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.s4nd = S4NDConv(channels, dimensions)
        self.out_proj = nn.Conv1d(channels, channels, 1)
        self.mlp_norm = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels * 4, 1), nn.GELU(), nn.Conv1d(channels * 4, channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pointwise gates are applied per flattened spatial position; the S4ND
        # operation below retains the original spatial geometry.
        residual = x
        z = self.norm(x)
        gate_src = z.flatten(2)
        value, gate = self.in_proj(gate_src).chunk(2, dim=1)
        short = self.short(value)
        value = (short + value) * torch.sigmoid(gate)
        value = value.reshape_as(x)
        value = self.s4nd(value)
        value = self.out_proj(value.flatten(2)).reshape_as(x)
        x = residual + value
        return x + self.mlp(self.mlp_norm(x).flatten(2)).reshape_as(x)

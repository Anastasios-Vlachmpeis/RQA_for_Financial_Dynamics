"""Batched torch representations for delay states and recurrence plots."""
from __future__ import annotations

import torch
from torch import Tensor


def delay_embed(x: Tensor, m: int, tau: int) -> Tensor:
    """Embed ``x[B,L]`` into delay states ``[B,N,m]`` dated by their last sample."""
    if x.ndim != 2:
        raise ValueError("x must have shape [B,L]")
    if m < 1 or tau < 1:
        raise ValueError("m and tau must be positive")
    n_states = x.shape[1] - (m - 1) * tau
    if n_states < 1:
        raise ValueError("window is too short for the requested embedding")
    return torch.stack([x[:, k * tau:k * tau + n_states] for k in range(m)], dim=-1)


def distance_matrix(s: Tensor) -> Tensor:
    """Return normalized Euclidean state distances ``[B,N,N]``."""
    if s.ndim != 3:
        raise ValueError("s must have shape [B,N,m]")
    d = torch.cdist(s, s, p=2)
    n = d.shape[-1]
    if n > 1:
        offdiag = ~torch.eye(n, dtype=torch.bool, device=s.device)
        mean = d[:, offdiag].mean(dim=1).clamp_min(torch.finfo(d.dtype).eps)
    else:
        mean = d.new_ones(d.shape[0])
    return d / mean[:, None, None]


def _series_channels(x: Tensor, m: int, tau: int, n_states: int) -> Tensor:
    dated = x[:, (m - 1) * tau:(m - 1) * tau + n_states]
    xi = dated[:, :, None].expand(-1, -1, n_states)
    xj = dated[:, None, :].expand(-1, n_states, -1)
    return torch.stack((xi, xj), dim=1)


def _threshold_rps(d: Tensor, rates: tuple[float, ...]) -> Tensor:
    n = d.shape[-1]
    index = torch.arange(n, device=d.device)
    off_diagonal = (index[:, None] - index[None, :]).abs() > 0
    values = d[:, off_diagonal]
    thresholds = torch.quantile(values, torch.as_tensor(rates, device=d.device, dtype=d.dtype), dim=1).transpose(0, 1)
    # [B,rr,N,N]; thresholds are per sample and use off-diagonal distances only.
    return (d[:, None] <= thresholds[:, :, None, None]).to(d.dtype)


def rp_input(x: Tensor, m: int, tau: int, kind: str) -> Tensor:
    """Build ``[B,C,N,N]`` inputs: distance/cosine, series-only, or per-sample RR-thresholded RPs.

    ``cos`` is pairwise cosine similarity of delay states. ``series`` is the two
    dated series channels alone (row value, column value), without distances.
    ``RP@rr`` adds one binary recurrence channel; append ``+series`` to include
    the dated series channels, as for ``D+series``.
    """
    s = delay_embed(x, m, tau)
    if kind == "cos":
        norms = torch.linalg.vector_norm(s, dim=-1).clamp_min(torch.finfo(s.dtype).eps)
        cosine = torch.matmul(s, s.transpose(1, 2)) / (norms[:, :, None] * norms[:, None, :])
        return cosine.clamp(-1.0, 1.0)[:, None]
    if kind == "series":
        return _series_channels(x, m, tau, s.shape[1])
    d = distance_matrix(s)
    if kind == "D":
        return d[:, None]
    if kind == "D+series":
        return torch.cat((d[:, None], _series_channels(x, m, tau, s.shape[1])), dim=1)
    if kind == "RP":
        return _threshold_rps(d, (0.1,))
    if kind.startswith("RP@"):
        rate_text = kind[3:].removesuffix("+series")
        try:
            rate = float(rate_text)
        except ValueError as exc:
            raise ValueError(f"invalid recurrence-rate kind {kind!r}") from exc
        if not 0.0 < rate < 1.0:
            raise ValueError("RP recurrence rate must be between 0 and 1")
        rp = _threshold_rps(d, (rate,))
        return torch.cat((rp, _series_channels(x, m, tau, s.shape[1])), dim=1) if kind.endswith("+series") else rp
    if kind == "RPstack":
        return _threshold_rps(d, (0.05, 0.1, 0.2, 0.4))
    if kind == "RPstack+series":
        return torch.cat((_threshold_rps(d, (0.05, 0.1, 0.2, 0.4)),
                          _series_channels(x, m, tau, s.shape[1])), dim=1)
    raise ValueError("unsupported kind; expected D, D+series, series, cos, RP, RP@<rr>[+series], or RPstack[+series]")


def torp(s: Tensor, measure: str) -> Tensor:
    """Return a three-way recurrence tensor ``[B,N,N,N]`` from states ``[B,N,m]``."""
    if s.ndim != 3:
        raise ValueError("s must have shape [B,N,m]")
    if measure == "perimeter":
        d = distance_matrix(s)
        perimeter = d[:, :, :, None] + d[:, :, None, :]
        perimeter.add_(d[:, None, :, :])
        return perimeter
    if measure == "gencos":
        # Contract the shared embedding dimension after pairwise multiplication.
        # This avoids materializing the much larger [B,N,N,N,m] broadcast product.
        with torch.autocast(device_type=s.device.type, enabled=False):
            pair_products = s[:, :, None, :] * s[:, None, :, :]
            numerator = torch.einsum("bijn,bkn->bijk", pair_products, s)
            norm = s.abs().pow(3).sum(dim=-1).pow(1.0 / 3.0)
            denominator = (norm[:, :, None, None] * norm[:, None, :, None]
                           * norm[:, None, None, :])
            return (numerator / denominator.clamp_min(torch.finfo(s.dtype).eps)).clamp(-1.0, 1.0)
    raise ValueError("measure must be 'perimeter' or 'gencos'")

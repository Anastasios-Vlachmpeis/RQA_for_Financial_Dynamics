"""Raw-series, recurrence-plot, and recurrence-attention encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .recattn import RecAttnEncoder
from .s4 import GatedS4ND, S4DConv

ENCODERS = [
    "raw_attn", "raw_conv", "raw_hyena", "rp_cnn", "rp_s4nd", "rp_vit",
    "recattn", "recattn_taylor", "torp_cnn3d", "torp_s4nd3d",
]


def _positions(position: Tensor, n: int) -> Tensor:
    if position.shape[1] == n:
        return position
    return F.interpolate(position.transpose(1, 2), size=n, mode="linear", align_corners=False).transpose(1, 2)


def _states(x: Tensor, m: int, tau: int, hide: Tensor | None) -> Tensor:
    from .representations import delay_embed
    states = delay_embed(x, m, tau)
    if hide is None:
        indicator = torch.zeros_like(states[..., :1])
    else:
        states = states.masked_fill(hide.unsqueeze(-1), 0)
        indicator = hide.to(dtype=states.dtype).unsqueeze(-1)
    return torch.cat((states, indicator), dim=-1)


class MixerBlock1D(nn.Module):
    def __init__(self, d: int, kind: str):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.kind = kind
        if kind == "conv":
            self.mixer = nn.Conv1d(d, d, 3, padding=1, groups=d)
            self.gate = None
        else:
            self.mixer = S4DConv(d)
            self.gate = nn.Conv1d(d, 2 * d, 1)
        self.out = nn.Conv1d(d, d, 1)
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x).transpose(1, 2)
        if self.kind == "conv":
            y = self.mixer(z)
        else:
            value, gate = self.gate(z).chunk(2, dim=1)
            y = self.mixer(value) * torch.sigmoid(gate)
        x = x + self.out(y).transpose(1, 2)
        return x + self.mlp(self.mlp_norm(x))


class RawAttnBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: Tensor, hide: Tensor | None) -> Tensor:
        z = self.norm(x)
        # Hidden states are already zeroed plus flagged. They remain queries, but
        # are not keys/values for visible tokens.
        key_padding = None
        if hide is not None:
            key_padding = hide.clone()
            key_padding[hide.all(dim=1)] = False
        y = self.attn(z, z, z, key_padding_mask=key_padding, need_weights=False)[0]
        x = x + y
        return x + self.mlp(self.mlp_norm(x))


class RawEncoder(nn.Module):
    def __init__(self, name: str, d: int, depth: int, n_states: int, m: int, tau: int, heads: int):
        super().__init__()
        self.name, self.d_model, self.m, self.tau = name, d, m, tau
        self.input = nn.Linear(m + 1, d)
        self.position = nn.Parameter(torch.randn(1, n_states, d) * 0.02)
        if name == "raw_attn":
            self.blocks = nn.ModuleList(RawAttnBlock(d, heads) for _ in range(depth))
        else:
            self.blocks = nn.ModuleList(MixerBlock1D(d, "hyena" if name == "raw_hyena" else "conv") for _ in range(depth))
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x: Tensor, hide: Tensor | None = None) -> Tensor:
        states = _states(x, self.m, self.tau, hide)
        z = self.input(states) + _positions(self.position, states.shape[1])
        for block in self.blocks:
            z = block(z, hide) if self.name == "raw_attn" else block(z)
        return self.final_norm(z)


def _coordinate_channels(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    row = torch.linspace(0, 1, n, device=device, dtype=dtype)
    col = torch.linspace(0, 1, n, device=device, dtype=dtype)
    ri, cj = torch.meshgrid(row, col, indexing="ij")
    return torch.stack((ri, cj, (ri - cj).abs()), dim=0)


def _rp_image(
    x: Tensor, m: int, tau: int, kind: str, hide: Tensor | None, cell_hide: Tensor | None = None
) -> Tensor:
    from .representations import delay_embed, rp_input
    image = rp_input(x, m, tau, kind)
    batch, n = image.shape[0], image.shape[-1]
    if hide is None:
        hidden_cells = torch.zeros(batch, n, n, dtype=torch.bool, device=image.device)
    else:
        hidden_cells = hide[:, :, None] | hide[:, None, :]
        # rp_input's sample-wide distance scale / recurrence-rate thresholds
        # include hidden states. Recompute those statistics from visible states
        # only so hidden raw values cannot alter visible-visible cells.
        states = delay_embed(x, m, tau).float()
        distances = torch.cdist(states, states)
        visible = ~hide
        off_diag = ~torch.eye(n, dtype=torch.bool, device=x.device)[None]
        index = torch.arange(n, device=x.device)
        off_theiler = (index[:, None] - index[None, :]).abs() > 1
        visible_pairs = visible[:, :, None] & visible[:, None, :] & off_diag
        visible_rp_pairs = visible[:, :, None] & visible[:, None, :] & off_theiler
        visible_scale = (distances * visible_pairs).sum((1, 2)) / visible_pairs.sum((1, 2)).clamp_min(1)
        if kind.startswith("D"):
            image[:, 0] = distances / visible_scale.clamp_min(1e-12)[:, None, None]
        if kind == "RP":
            rates = (0.1,)
        elif kind.startswith("RP@"):
            rate_text = kind[3:].removesuffix("+series")
            try:
                rates = (float(rate_text),)
            except ValueError as exc:
                raise ValueError(f"invalid recurrence-rate kind {kind!r}") from exc
        elif kind.startswith("RPstack"):
            rates = (0.05, 0.1, 0.2, 0.4)
        else:
            rates = ()
        for channel, rate in enumerate(rates):
            for batch_index in range(batch):
                values = distances[batch_index][visible_rp_pairs[batch_index]]
                if values.numel():
                    threshold = torch.quantile(values, rate)
                    image[batch_index, channel] = (distances[batch_index] <= threshold).to(image.dtype)
        if cell_hide is not None:
            if cell_hide.shape != (batch, n, n) or cell_hide.dtype != torch.bool:
                raise ValueError("cell_hide must be a bool tensor with shape [B,N,N]")
            hidden_cells = hidden_cells | cell_hide
    image = image.masked_fill(hidden_cells[:, None], 0)
    coords = _coordinate_channels(n, image.device, image.dtype).unsqueeze(0).expand(batch, -1, -1, -1)
    coords = coords.masked_fill(hidden_cells[:, None], 0)
    hidden_channel = hidden_cells.to(image.dtype).unsqueeze(1)
    return torch.cat((image, coords, hidden_channel), dim=1)


class CNNBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.mlp_norm = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(nn.Conv2d(channels, 4 * channels, 1), nn.GELU(), nn.Conv2d(4 * channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.conv(F.gelu(self.norm(x)))
        return x + self.mlp(F.gelu(self.mlp_norm(x)))


class RPCNN(nn.Module):
    def __init__(self, d: int, depth: int, n_states: int, m: int, tau: int, rp_kind: str):
        super().__init__()
        from .representations import rp_input
        self.name, self.rp_kind = "rp_cnn", rp_kind
        self.d_model, self.m, self.tau = d, m, tau
        channels = len(rp_input(torch.zeros(1, n_states + (m - 1) * tau), m, tau, rp_kind)[0]) + 4
        self.input = nn.Conv2d(channels, d, 1)
        self.blocks = nn.ModuleList(CNNBlock2D(d) for _ in range(depth))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor, hide: Tensor | None = None, *, cell_hide: Tensor | None = None,
                rp_kind: str | None = None) -> Tensor:
        image = _rp_image(x, self.m, self.tau, self.rp_kind if rp_kind is None else rp_kind, hide, cell_hide)
        z = self.input(image)
        for block in self.blocks:
            z = block(z)
        return self.norm(z.mean(dim=-1).transpose(1, 2))


class RPS4ND(nn.Module):
    def __init__(self, d: int, depth: int, m: int, tau: int, rp_kind: str):
        super().__init__()
        from .representations import rp_input
        self.name, self.rp_kind = "rp_s4nd", rp_kind
        self.d_model, self.m, self.tau = d, m, tau
        self.channels = len(rp_input(torch.zeros(1, 8 + (m - 1) * tau), m, tau, rp_kind)[0]) + 4
        self.input = nn.Conv2d(self.channels, d, 1)
        self.blocks = nn.ModuleList(GatedS4ND(d, 2) for _ in range(depth))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor, hide: Tensor | None = None, *, cell_hide: Tensor | None = None,
                rp_kind: str | None = None) -> Tensor:
        kind = self.rp_kind if rp_kind is None else rp_kind
        z = self.input(_rp_image(x, self.m, self.tau, kind, hide, cell_hide))
        for block in self.blocks:
            z = block(z)
        return self.norm(z.mean(dim=-1).transpose(1, 2))


class ViTBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        x = x + self.attn(z, z, z, need_weights=False)[0]
        return x + self.mlp(self.mlp_norm(x))


class RPVisionTransformer(nn.Module):
    def __init__(self, d: int, depth: int, m: int, tau: int, rp_kind: str, patch: int, pos: str, heads: int):
        super().__init__()
        if pos not in ("grid", "diag"):
            raise ValueError("pos must be 'grid' or 'diag'")
        from .representations import rp_input
        self.name, self.rp_kind, self.patch, self.pos = "rp_vit", rp_kind, patch, pos
        self.d_model, self.m, self.tau = d, m, tau
        channels = len(rp_input(torch.zeros(1, 8 + (m - 1) * tau), m, tau, rp_kind)[0]) + 4
        self.embed = nn.Conv2d(channels, d, kernel_size=patch, stride=patch)
        self.row_pos = nn.Parameter(torch.randn(1, 256, d) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(1, 256, d) * 0.02)
        self.diag_pos = nn.Embedding(511, d)
        self.blocks = nn.ModuleList(ViTBlock(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor, hide: Tensor | None = None, *, cell_hide: Tensor | None = None,
                rp_kind: str | None = None) -> Tensor:
        kind = self.rp_kind if rp_kind is None else rp_kind
        image = _rp_image(x, self.m, self.tau, kind, hide, cell_hide)
        n = image.shape[-1]
        p = self.patch
        pad = (-n) % p
        if pad:
            image = F.pad(image, (0, pad, 0, pad))
        z = self.embed(image)
        b, d, nr, nc = z.shape
        rows = _positions(self.row_pos[:, :nr], nr).unsqueeze(2)
        cols = _positions(self.col_pos[:, :nc], nc).unsqueeze(1)
        z = z.flatten(2).transpose(1, 2) + (rows + cols).expand(b, nr, nc, d).reshape(b, nr * nc, d)
        if self.pos == "diag":
            r = torch.arange(nr, device=x.device)
            c = torch.arange(nc, device=x.device)
            offset = (r[:, None] - c[None, :] + 255).clamp(0, 510)
            z = z + self.diag_pos(offset.reshape(-1)).unsqueeze(0)
        for block in self.blocks:
            z = block(z)
        z = self.norm(z).reshape(b, nr, nc, d).mean(dim=2)
        z = z.repeat_interleave(p, dim=1)
        return z[:, :n]


def _torp_input(x: Tensor, m: int, tau: int, measure: str, hide: Tensor | None) -> Tensor:
    from .representations import delay_embed, torp
    states = delay_embed(x, m, tau)
    volume = torp(states, measure)
    n = volume.shape[-1]
    if hide is None:
        hidden = volume.new_zeros((volume.shape[0], 1, n, n, n))
    else:
        hidden_cells = hide[:, :, None, None] | hide[:, None, :, None] | hide[:, None, None, :]
        hidden = hidden_cells.to(volume.dtype).unsqueeze(1)
        if measure == "perimeter":
            distances = torch.cdist(states.float(), states.float())
            off_diag = ~torch.eye(n, dtype=torch.bool, device=x.device)[None]
            visible = ~hide
            visible_pairs = visible[:, :, None] & visible[:, None, :] & off_diag
            visible_scale = (distances * visible_pairs).sum((1, 2)) / visible_pairs.sum((1, 2)).clamp_min(1)
            volume = (
                distances[:, :, :, None] + distances[:, :, None, :] + distances[:, None, :, :]
            ) / visible_scale.clamp_min(1e-12)[:, None, None, None]
        volume = volume.masked_fill(hidden_cells, 0)
    return torch.cat((volume.unsqueeze(1), hidden), dim=1)


class CNNBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, 1),
        )
        self.mlp_norm = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, 2 * channels, 1), nn.GELU(), nn.Conv3d(2 * channels, channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.conv(F.gelu(self.norm(x)))
        return x + self.mlp(F.gelu(self.mlp_norm(x)))


class TORPCNN3D(nn.Module):
    def __init__(self, d: int, depth: int, m: int, tau: int, measure: str):
        super().__init__()
        self.name, self.measure, self.m, self.tau = "torp_cnn3d", measure, m, tau
        # N^3 activations dominate at the batch sizes used for training; keep
        # the TORP mixer narrow and project its row tokens back to d.
        width = max(8, d // 4)
        self.input = nn.Conv3d(2, width, 1)
        self.blocks = nn.ModuleList(CNNBlock3D(width) for _ in range(depth))
        self.project = nn.Linear(width, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor, hide: Tensor | None = None) -> Tensor:
        volume = _torp_input(x, self.m, self.tau, self.measure, hide)
        # N=32 profiling favored channels-last 3D over dense and axis-separable variants.
        z = self.input(volume.contiguous(memory_format=torch.channels_last_3d))
        for block in self.blocks:
            z = block(z)
        return self.norm(self.project(z.mean(dim=(3, 4)).transpose(1, 2)))


class TORPS4ND3D(nn.Module):
    def __init__(self, d: int, depth: int, m: int, tau: int, measure: str):
        super().__init__()
        self.name, self.measure, self.m, self.tau = "torp_s4nd3d", measure, m, tau
        width = max(4, d // 8)
        self.input = nn.Conv3d(2, width, 1)
        self.blocks = nn.ModuleList(GatedS4ND(width, 3) for _ in range(depth))
        self.project = nn.Linear(width, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor, hide: Tensor | None = None) -> Tensor:
        z = self.input(_torp_input(x, self.m, self.tau, self.measure, hide))
        for block in self.blocks:
            z = block(z)
        return self.norm(self.project(z.mean(dim=(3, 4)).transpose(1, 2)))


def make_encoder(name: str, d_model: int, depth: int, n_states: int, m: int = 3, tau: int = 1, **kw) -> nn.Module:
    """Construct an encoder returning [batch, n_states, d_model].

    ``recattn`` uses exact Gaussian kernels by default because the quadratic
    Taylor approximation smooths the sharp recurrence peaks needed for recall;
    ``recattn_taylor`` explicitly selects that linear-time approximation.
    """
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {ENCODERS}")
    if name == "raw_attn":
        return RawEncoder(name, d_model, depth, n_states, m, tau, kw.pop("heads", 4))
    if name in ("raw_conv", "raw_hyena"):
        return RawEncoder(name, d_model, depth, n_states, m, tau, kw.pop("heads", 4))
    if name in ("recattn", "recattn_taylor"):
        if name == "recattn":
            kw.setdefault("exact", True)
        else:
            kw["exact"], kw["taylor"] = False, True
        encoder = RecAttnEncoder(d_model, depth, n_states, m, tau, **kw)
        encoder.name = name
        return encoder
    if name == "rp_cnn":
        return RPCNN(d_model, depth, n_states, m, tau, kw.pop("rp_kind", "D+series"))
    if name == "rp_s4nd":
        return RPS4ND(d_model, depth, m, tau, kw.pop("rp_kind", "D+series"))
    if name == "rp_vit":
        return RPVisionTransformer(d_model, depth, m, tau, kw.pop("rp_kind", "D+series"), kw.pop("patch", 8), kw.pop("pos", "grid"), kw.pop("heads", 4))
    if name == "torp_cnn3d":
        return TORPCNN3D(d_model, depth, m, tau, kw.pop("measure", "perimeter")).to(
            memory_format=torch.channels_last_3d)
    return TORPS4ND3D(d_model, depth, m, tau, kw.pop("measure", "perimeter"))


def n_params(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_flops(model: nn.Module, L: int, batch: int = 1) -> int:
    """Count forward FLOPs with PyTorch's flop counter for a zero input window."""
    from torch.utils.flop_counter import FlopCounterMode
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x = torch.zeros(batch, L, device=device, dtype=dtype)
    n = L - (getattr(model, "m", 3) - 1) * getattr(model, "tau", 1)
    hide = torch.zeros(batch, n, device=device, dtype=torch.bool)
    was_training = model.training
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x, hide)
    model.train(was_training)
    return int(counter.get_total_flops())

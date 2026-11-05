"""Linear-time Gaussian recurrence attention with an exact inspection path."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn



def _features(u: Tensor) -> Tensor:
    """Second-order Based features [1, u, vec(uuᵀ)/sqrt(2)]."""
    outer = torch.einsum("bhnd,bhne->bhnde", u, u).flatten(-2) / math.sqrt(2.0)
    return torch.cat((torch.ones_like(u[..., :1]), u, outer), dim=-1)


class RecAttnBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, window: int, exact: bool, taylor: bool,
                 kernel_dim: int | None = None, eps_init: float = 1.0):
        super().__init__()
        self.heads, self.d_k = heads, d_k
        self.window, self.exact, self.taylor = window, exact, taylor
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model if kernel_dim is None else kernel_dim, heads * d_k, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.eps_log = nn.Parameter(torch.full((heads,), math.log(eps_init)))
        self.local_q = nn.Linear(d_model, heads * d_k, bias=False)
        self.local_k = nn.Linear(d_model, heads * d_k, bias=False)
        self.local_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model * 2, d_model)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        for layer in (self.local_q, self.local_k):
            nn.init.normal_(layer.weight, std=0.005)

    def _project(self, x: Tensor, kernel_x: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        kernel_input = x if kernel_x is None else kernel_x
        b, n, _ = kernel_input.shape
        projected = self.proj(kernel_input).view(b, n, self.heads, self.d_k).transpose(1, 2)
        v = self.v(x).view(b, n, self.heads, -1).transpose(1, 2)
        return projected, projected, v

    def _gaussian(self, q: Tensor, k: Tensor) -> Tensor:
        eps = self.eps_log.exp().view(1, self.heads, 1, 1).clamp_min(1e-6)
        delta = q.unsqueeze(-2) - k.unsqueeze(-3)
        return torch.exp(-delta.square().sum(dim=-1, keepdim=False) / (2 * eps))

    def kernel(self, q: Tensor, k: Tensor, hide: Tensor | None, *, exact: bool) -> Tensor:
        eps = self.eps_log.exp().view(1, self.heads, 1, 1).clamp_min(1e-6)
        if exact:
            weights = self._gaussian(q.float(), k.float())
        else:
            # Diagnostic materialization only; the Taylor forward path below
            # computes these contractions without allocating N×N weights.
            a, b = q.float() / eps.sqrt(), k.float() / eps.sqrt()
            weights = (_features(a) @ _features(b).transpose(-1, -2))
            qfactor = torch.exp(-q.float().square().sum(-1, keepdim=True) / (2 * eps))
            kfactor = torch.exp(-k.float().square().sum(-1).unsqueeze(-2) / (2 * eps))
            weights = (weights * qfactor * kfactor).clamp_min(0)
        if hide is not None:
            weights = weights.masked_fill(hide[:, None, None, :], 0)
        return weights

    def forward(self, x: Tensor, hide: Tensor | None, kernel_x: Tensor | None = None) -> Tensor:
        residual = x
        z = self.norm(x)
        q, k, v = self._project(z, kernel_x)
        qf, kf, vf = q.float(), k.float(), v.float()
        eps = self.eps_log.float().exp().view(1, self.heads, 1, 1).clamp_min(1e-6)
        if self.exact or not self.taylor:
            weights = self._gaussian(qf, kf)
            if hide is not None:
                weights = weights.masked_fill(hide[:, None, None, :], 0)
            global_out = torch.matmul(weights, vf) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            a, b = qf / eps.sqrt(), kf / eps.sqrt()
            qphi, kphi = _features(a), _features(b)
            qfactor = torch.exp(-qf.square().sum(-1, keepdim=True) / (2 * eps))
            kfactor = torch.exp(-kf.square().sum(-1, keepdim=True) / (2 * eps))
            if hide is not None:
                kfactor = kfactor.masked_fill(hide[:, None, :, None], 0)
            # sum_j phi(k_j) * factor_j * v_j, then query contraction
            kv = torch.einsum("bhkf,bhkv->bhfv", kphi * kfactor, vf)
            zsum = torch.einsum("bhkf,bhk->bhf", kphi * kfactor, torch.ones_like(kfactor[..., 0]))
            numerator = torch.einsum("bhqf,bhfv->bhqv", qphi, kv)
            denominator = torch.einsum("bhqf,bhf->bhq", qphi, zsum).unsqueeze(-1)
            global_out = numerator / denominator.clamp_min(1e-8)
        # A small exact sliding-window path augments the global recurrence path.
        lq = self.local_q(z).view(z.shape[0], z.shape[1], self.heads, self.d_k).transpose(1, 2)
        lk = self.local_k(z).view(z.shape[0], z.shape[1], self.heads, self.d_k).transpose(1, 2)
        lv = self.local_v(z).view(z.shape[0], z.shape[1], self.heads, -1).transpose(1, 2)
        scores = torch.matmul(lq.float(), lk.float().transpose(-1, -2)) / math.sqrt(self.d_k)
        n = x.shape[1]
        index = torch.arange(n, device=x.device)
        left, right = (self.window - 1) // 2, self.window // 2
        offset = index[None, :] - index[:, None]
        local_mask = (offset < -left) | (offset > right)
        scores = scores.masked_fill(local_mask[None, None], -torch.inf)
        if hide is not None:
            scores = scores.masked_fill(hide[:, None, None, :], -torch.inf)
        local_weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        local_out = torch.matmul(local_weights, lv.float())
        merged = torch.cat((global_out, local_out), dim=-1).transpose(1, 2).reshape(x.shape[0], n, -1)
        x = residual + self.out(merged.to(dtype=x.dtype))
        return x + self.mlp(self.mlp_norm(x))


class RecAttnEncoder(nn.Module):
    """Raw-state recurrence encoder; exact kernels preserve sharp recall peaks.

    The quadratic Taylor approximation can smooth those peaks, so it is exposed
    separately as ``recattn_taylor`` rather than used by the default encoder.
    """

    def __init__(self, d_model: int, depth: int, n_states: int, m: int = 3, tau: int = 1,
                 heads: int = 4, d_k: int = 8, window: int = 16, exact: bool = True,
                 taylor: bool = True):
        super().__init__()
        self.d_model, self.n_states, self.m, self.tau = d_model, n_states, m, tau
        self.name = "recattn"
        self.input = nn.Linear(m + 1, d_model)
        self.position = nn.Parameter(torch.randn(1, n_states, d_model) * 0.02)
        blocks = []
        for i in range(depth):
            kernel_dim = m + 1 if i == 0 else None
            target_median = 0.3 if i == 0 else 0.2
            expected_sq_distance = (2.0 * m * d_k / (m + 1)) if i == 0 else (2.0 * d_k)
            chi_median_factor = (1.0 - 2.0 / (9.0 * d_k)) ** 3
            median_sq_distance = expected_sq_distance * chi_median_factor
            eps_init = median_sq_distance / (2.0 * math.log(1.0 / target_median))
            blocks.append(
                RecAttnBlock(d_model, heads, d_k, window, exact, taylor, kernel_dim, eps_init)
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(d_model)

    def _states(self, x: Tensor, hide: Tensor | None) -> tuple[Tensor, Tensor | None]:
        from .representations import delay_embed
        states = delay_embed(x, self.m, self.tau)
        if hide is not None:
            states = states.masked_fill(hide.unsqueeze(-1), 0)
            indicator = hide.to(dtype=states.dtype).unsqueeze(-1)
            states = torch.cat((states, indicator), dim=-1)
        else:
            states = torch.cat((states, torch.zeros_like(states[..., :1])), dim=-1)
        return states, hide

    def forward(self, x: Tensor, hide: Tensor | None = None) -> Tensor:
        states, hide = self._states(x, hide)
        z = self.input(states) + self.position[:, :states.shape[1]]
        for i, block in enumerate(self.blocks):
            z = block(z, hide, states if i == 0 else None)
        return self.final_norm(z)

    def kernel_matrix(self, x: Tensor, hide: Tensor | None = None, exact: bool | None = None) -> Tensor:
        """Return the first block's (optionally Taylor-approximated) Gaussian RP."""
        states, hide = self._states(x, hide)
        z = self.input(states) + self.position[:, :states.shape[1]]
        block = self.blocks[0]
        q, k, _ = block._project(block.norm(z), states)
        return block.kernel(q, k, hide, exact=block.exact if exact is None else exact)

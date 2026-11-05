"""JEPA and masked-state pretraining for recurrence encoders."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .train import _autocast, _make_encoder, _seed, pool

MASKS = ["future", "block", "random", "cell_asym", "cell_sym"]
# cell_asym/cell_sym require rp_* encoders with cell_hide support.
CELL_MASKS = {"cell_asym", "cell_sym"}

def _state_mask(batch: int, n_states: int, mask: str, device: torch.device,
                generator: torch.Generator) -> torch.Tensor:
    """Sample target state indices, returning a boolean ``[B, N]`` mask."""
    result = torch.zeros((batch, n_states), dtype=torch.bool, device=device)
    if mask == "random":
        result = torch.rand((batch, n_states), device=device, generator=generator) < 0.25
        empty = ~result.any(dim=1)
        if empty.any():
            rows = empty.nonzero(as_tuple=False).flatten()
            columns = torch.randint(n_states, (len(rows),), device=device, generator=generator)
            result[rows, columns] = True
        return result
    if mask not in ("future", "block"):
        raise ValueError(f"{mask!r} is not a temporal state mask")
    low = max(1, n_states // 8)
    high = max(low, n_states // 4)
    for i in range(batch):
        length = int(torch.randint(low, high + 1, (), device=device, generator=generator).item())
        if mask == "future":
            start = n_states - length
        else:
            start = int(torch.randint(0, n_states - length + 1, (), device=device, generator=generator).item())
        result[i, start:start + length] = True
    return result


def _cell_state_mask(batch: int, n_states: int, mask: str, device: torch.device,
                     generator: torch.Generator, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample two rectangular cell blocks and return target rows and dilated cells."""
    base_cells = torch.zeros((batch, n_states, n_states), dtype=torch.bool, device=device)
    target_rows = torch.zeros((batch, n_states), dtype=torch.bool, device=device)
    low = max(1, n_states // 8)
    high = max(low, n_states // 4)
    for sample in range(batch):
        for _ in range(2):
            i_length = int(torch.randint(low, high + 1, (), device=device, generator=generator).item())
            j_length = int(torch.randint(low, high + 1, (), device=device, generator=generator).item())
            i_start = int(torch.randint(0, n_states - i_length + 1, (), device=device, generator=generator).item())
            j_start = int(torch.randint(0, n_states - j_length + 1, (), device=device, generator=generator).item())
            i_end, j_end = i_start + i_length, j_start + j_length
            base_cells[sample, i_start:i_end, j_start:j_end] = True
            target_rows[sample, i_start:i_end] = True
            if mask == "cell_sym":
                target_rows[sample, j_start:j_end] = True
    if mask == "cell_sym":
        base_cells = base_cells | base_cells.transpose(1, 2)
    if radius:
        kernel = 2 * radius + 1
        hidden_cells = F.max_pool2d(base_cells.float().unsqueeze(1), kernel_size=kernel,
                                    stride=1, padding=radius).squeeze(1).bool()
    else:
        hidden_cells = base_cells
    assert bool(hidden_cells[base_cells].all()), "cell_hide must contain every sampled block"
    return target_rows, hidden_cells

def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate state masks symmetrically, clipping at sequence boundaries."""
    if radius <= 0:
        return mask.clone()
    kernel = 2 * radius + 1
    return F.max_pool1d(mask.float().unsqueeze(1), kernel_size=kernel, stride=1,
                        padding=radius).squeeze(1).bool()


def _make_transformer(d_model: int, depth: int) -> nn.TransformerEncoder:
    if d_model % 4:
        raise ValueError("d_model must be divisible by 4 for the four-head predictor/decoder")
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
                                       batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


def _target_for_mae(x: torch.Tensor, m: int, tau: int, rp_encoder: bool) -> torch.Tensor:
    from .representations import delay_embed, distance_matrix

    states = delay_embed(x, m=m, tau=tau)
    return distance_matrix(states) if rp_encoder else states


def pretrain_jepa(enc_name: str, d_model: int, depth: int, x_pre: np.ndarray, *,
                  n_states: int, m: int, tau: int, mask: str = "future",
                  steps: int = 1000, batch: int = 128, lr: float = 1e-3,
                  ema: tuple[float, float] = (0.996, 1.0), pred_depth: int = 2,
                  seed: int = 0, device: str = "cuda", enc_kw: dict[str, Any] | None = None,
                  objective: str = "jepa", scales: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05),
                  verbose: bool = False) -> tuple[nn.Module, pd.DataFrame]:
    """Pretrain and return the EMA target encoder (JEPA) or online encoder (MAE).

    ``jepa_multiscale`` is available for RP CNN/S4ND/ViT encoders. Its context
    branch uses the encoder's configured ``rp_kind``; each target branch uses
    ``RP@<rate>`` and preserves two series channels when ``rp_kind`` ends in
    ``+series``. Coarse-to-fine scales are appended as extra predictor tokens.
    """
    if steps < 1 or batch < 1 or pred_depth < 1:
        raise ValueError("steps, batch, and pred_depth must be positive")
    if objective not in ("jepa", "jepa_multiscale", "mae"):
        raise ValueError("objective must be 'jepa', 'jepa_multiscale', or 'mae'")
    if objective == "jepa_multiscale":
        if enc_name not in ("rp_cnn", "rp_s4nd", "rp_vit"):
            raise ValueError("jepa_multiscale requires an rp_cnn, rp_s4nd, or rp_vit encoder")
        if not scales or any(not 0.0 < rate < 1.0 for rate in scales):
            raise ValueError("scales must contain recurrence rates between 0 and 1")
        if any(a <= b for a, b in zip(scales, scales[1:])):
            raise ValueError("scales must be ordered coarse-to-fine (strictly decreasing)")
    if mask not in MASKS:
        raise ValueError(f"unknown mask {mask!r}; choose one of {MASKS}")
    if not 0.0 <= ema[0] <= 1.0 or not 0.0 <= ema[1] <= 1.0:
        raise ValueError("EMA momenta must be in [0, 1]")

    _seed(seed)
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    encoder = _make_encoder(enc_name, d_model, depth, n_states, m, tau, enc_kw).to(dev)
    target_encoder = copy.deepcopy(encoder).to(dev) if objective in ("jepa", "jepa_multiscale") else None
    if target_encoder is not None:
        target_encoder.eval()
        for parameter in target_encoder.parameters():
            parameter.requires_grad_(False)
    rp_encoder = bool(getattr(encoder, "rp_kind", None)) or enc_name.startswith("rp_")
    if objective == "jepa_multiscale":
        rp_kind = encoder.rp_kind
        has_series = rp_kind.endswith("+series")
        base_kind = rp_kind.removesuffix("+series")
        if base_kind not in ("D", "cos", "RP") and not base_kind.startswith("RP@"):
            raise ValueError("jepa_multiscale requires a single-channel RP/D/cos rp_kind, optionally with +series")
        if has_series and base_kind == "cos":
            raise ValueError("cos+series is not an available rp_kind")
        scale_kinds = tuple(f"RP@{rate:g}" + ("+series" if has_series else "") for rate in scales)
    else:
        scale_kinds = ()
    if mask in CELL_MASKS and not rp_encoder:
        raise ValueError(f"mask={mask!r} is available only for rp_* encoders with cell_hide support")

    predictor = _make_transformer(d_model, pred_depth if objective != "mae" else 2).to(dev)
    mask_token = nn.Parameter(torch.zeros(1, 1, d_model, device=dev))
    position = nn.Parameter(torch.empty(1, n_states, d_model, device=dev))
    nn.init.normal_(mask_token, std=0.02)
    nn.init.normal_(position, std=0.02)
    if objective in ("jepa", "jepa_multiscale"):
        output_layer = nn.Identity()
    else:
        target_width = n_states if rp_encoder else m
        output_layer = nn.Linear(d_model, target_width).to(dev)
    learned = [mask_token, position]
    trainable = list(encoder.parameters()) + list(predictor.parameters()) + learned
    if objective == "mae":
        trainable += list(output_layer.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.05)
    x = np.asarray(x_pre, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("x_pre must be a nonempty [n, L] array")
    rng = np.random.default_rng(seed)
    mask_generator = torch.Generator(device=dev).manual_seed(seed + 1)
    rows: list[dict[str, float | int]] = []
    shown_hidden = False

    for step in range(steps):
        indices = rng.integers(0, len(x), size=batch)
        xb = torch.as_tensor(x[indices], dtype=torch.float32, device=dev)
        dilation_radius = (m - 1) * tau
        if mask in CELL_MASKS:
            target_mask, cell_hide = _cell_state_mask(batch, n_states, mask, dev,
                                                       mask_generator, dilation_radius)
            context_hide = None
            assert bool(cell_hide[0].any()), "cell_hide must contain masked RP cells"
        else:
            target_mask = _state_mask(batch, n_states, mask, dev, mask_generator)
            dilated_targets = _dilate(target_mask, dilation_radius)
            context_hide = dilated_targets
            cell_hide = None
            assert bool(context_hide[dilated_targets].all()), "context hide must contain all dilated target states"
        if verbose and not shown_hidden:
            if cell_hide is None:
                hidden = context_hide[0].nonzero().flatten().cpu().tolist()
                print(f"JEPA leakage check: target={target_mask[0].nonzero().flatten().cpu().tolist()} "
                      f"context_hide={hidden}")
            else:
                cells = cell_hide[0].nonzero().cpu().tolist()
                print(f"JEPA cell-mask check: target_rows={target_mask[0].nonzero().flatten().cpu().tolist()} "
                      f"cell_hide={cells}")
            shown_hidden = True
        encoder.train()
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(dev):
            if cell_hide is None:
                context_tokens = encoder(xb, hide=context_hide)
            else:
                context_tokens = encoder(xb, hide=context_hide, cell_hide=cell_hide)
            prediction_input = torch.where(target_mask.unsqueeze(-1), mask_token + position,
                                           context_tokens + position)
            if objective in ("jepa", "jepa_multiscale"):
                assert target_encoder is not None
                target_encoder.eval()
                if objective == "jepa":
                    predictions = output_layer(predictor(prediction_input))
                    with torch.no_grad():
                        target_tokens = target_encoder(xb, hide=None)
                        normalized_targets = F.layer_norm(target_tokens.float(), (d_model,))
                    error = F.smooth_l1_loss(predictions.float(), normalized_targets, reduction="none").mean(dim=-1)
                    loss = error[target_mask].mean()
                    with torch.no_grad():
                        embedding_std = pool(target_tokens.float()).std(dim=0, unbiased=False).mean()
                else:
                    level_losses = []
                    level_stds = []
                    previous_prediction = None
                    for kind in scale_kinds:
                        level_input = prediction_input
                        if previous_prediction is not None:
                            level_input = torch.cat((level_input, previous_prediction), dim=1)
                        all_predictions = output_layer(predictor(level_input))
                        predictions = all_predictions[:, :n_states]
                        with torch.no_grad():
                            target_tokens = target_encoder(xb, hide=None, rp_kind=kind)
                            normalized_targets = F.layer_norm(target_tokens.float(), (d_model,))
                            level_stds.append(pool(target_tokens.float()).std(dim=0, unbiased=False).mean())
                        error = F.smooth_l1_loss(predictions.float(), normalized_targets, reduction="none").mean(dim=-1)
                        level_losses.append(error[target_mask].mean())
                        previous_prediction = predictions
                    loss = torch.stack(level_losses).mean()
                    with torch.no_grad():
                        embedding_std = torch.stack(level_stds).mean()
            else:
                predictions = output_layer(predictor(prediction_input))
                targets = _target_for_mae(xb, m, tau, rp_encoder).float()
                element_error = (predictions.float() - targets) ** 2
                expanded_mask = target_mask.unsqueeze(-1).expand_as(element_error)
                loss = element_error[expanded_mask].mean()
                with torch.no_grad():
                    embedding_std = pool(context_tokens.float()).std(dim=0, unbiased=False).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if target_encoder is not None:
            momentum = ema[0] + (ema[1] - ema[0]) * (step / max(1, steps - 1))
            with torch.no_grad():
                for target_parameter, online_parameter in zip(target_encoder.parameters(), encoder.parameters()):
                    target_parameter.lerp_(online_parameter, 1.0 - momentum)
                for target_buffer, online_buffer in zip(target_encoder.buffers(), encoder.buffers()):
                    if target_buffer.is_floating_point():
                        target_buffer.lerp_(online_buffer, 1.0 - momentum)
                    else:
                        target_buffer.copy_(online_buffer)
        if verbose and (step + 1) % 50 == 0:
            print(f"step={step + 1} loss={float(loss.detach().item()):.6f} emb_std={float(embedding_std.item()):.6f}")
        row = {"step": step + 1, "loss": float(loss.detach().item()),
               "emb_std": float(embedding_std.item())}
        if objective == "jepa_multiscale":
            row.update({f"loss_level_{index}": float(level_loss.detach().item())
                        for index, level_loss in enumerate(level_losses)})
        rows.append(row)

    returned_encoder = target_encoder if target_encoder is not None else encoder
    assert returned_encoder is not None
    returned_encoder.eval()
    columns = ["step", "loss", "emb_std"]
    if objective == "jepa_multiscale":
        columns.extend(f"loss_level_{index}" for index in range(len(scale_kinds)))
    return returned_encoder, pd.DataFrame(rows, columns=columns)

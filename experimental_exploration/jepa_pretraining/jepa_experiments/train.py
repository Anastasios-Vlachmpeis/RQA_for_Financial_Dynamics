"""Supervised training utilities for recurrence-plot representation studies."""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn


def pool(tokens: torch.Tensor) -> torch.Tensor:
    """Pool ``[B, N, d]`` time tokens as ``[mean(tokens), last token]``."""
    if tokens.ndim != 3 or tokens.shape[1] == 0:
        raise ValueError("tokens must have shape [B, N, d] with N > 0")
    return torch.cat((tokens.mean(dim=1), tokens[:, -1]), dim=-1)


def _make_encoder(enc_name: str, d_model: int, depth: int, n_states: int,
                  m: int, tau: int, enc_kw: dict[str, Any] | None) -> nn.Module:
    from rpjepa.encoders import make_encoder

    return make_encoder(enc_name, d_model, depth, n_states, m=m, tau=tau,
                        **(enc_kw or {}))


def _seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_init_encoder(enc_name: str, d_model: int, depth: int, *, n_states: int,
                        m: int = 3, tau: int = 1, seed: int = 0,
                        enc_kw: dict[str, Any] | None = None,
                        device: str | torch.device = "cpu") -> nn.Module:
    """Construct a deterministically seeded, untrained encoder baseline."""
    _seed(seed)
    encoder = _make_encoder(enc_name, d_model, depth, n_states, m, tau, enc_kw)
    return encoder.to(device)


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                          enabled=device.type == "cuda")


def _metric(pred: torch.Tensor, target: torch.Tensor, kind: str) -> float:
    if kind == "cls":
        return float((pred.argmax(dim=1) == target.long()).float().mean().item())
    y = target.float()
    p = pred.reshape_as(y).float()
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot.item() <= torch.finfo(y.dtype).eps:
        return float("nan")
    return float((1.0 - ((y - p) ** 2).sum() / ss_tot).item())


def _evaluate(encoder: nn.Module, head: nn.Module, x: np.ndarray, y: np.ndarray,
              kind: str, device: torch.device, batch: int,
              y_mean: float, y_std: float) -> float:
    encoder.eval()
    head.eval()
    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.as_tensor(x[start:start + batch], dtype=torch.float32, device=device)
            with _autocast(device):
                out = head(pool(encoder(xb, hide=None)))
            pred = out.float()
            if kind == "reg":
                pred = pred * y_std + y_mean
            pred_chunks.append(pred.cpu())
            target_chunks.append(torch.as_tensor(y[start:start + batch], dtype=torch.float32 if kind == "reg" else torch.long))
    return _metric(torch.cat(pred_chunks), torch.cat(target_chunks), kind)


def train_supervised(enc_name: str, d_model: int, depth: int, train, val, test, *,
                     n_states: int, m: int, tau: int, epochs: int = 30,
                     batch: int = 128, lr: float = 1e-3, seed: int = 0, patience: int = 5,
                     device: str = "cuda", enc_kw: dict[str, Any] | None = None) -> dict:
    """Fit an encoder and linear pooled head, selecting weights by validation score.

    Training stops after `patience` epochs without a validation improvement.
    """
    if epochs < 1 or batch < 1 or patience < 1:
        raise ValueError("epochs, batch and patience must be positive")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _seed(seed)
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    encoder = _make_encoder(enc_name, d_model, depth, n_states, m, tau, enc_kw).to(dev)
    out_dim = int(train.n_classes) if train.kind == "cls" else 1
    head = nn.Linear(2 * d_model, out_dim).to(dev)
    model_params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=0.05)
    n_steps = epochs * math.ceil(len(train.x) / batch)
    warmup_steps = max(1, int(0.05 * n_steps))
    step = 0

    def lr_scale(step_index: int) -> float:
        if step_index < warmup_steps:
            return (step_index + 1) / warmup_steps
        progress = min(1.0, (step_index - warmup_steps) / max(1, n_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    if train.kind == "cls":
        criterion: nn.Module = nn.CrossEntropyLoss()
        y_mean, y_std = 0.0, 1.0
        y_train = np.asarray(train.y, dtype=np.int64)
    elif train.kind == "reg":
        criterion = nn.MSELoss()
        y_train = np.asarray(train.y, dtype=np.float32)
        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
        if not math.isfinite(y_std) or y_std <= 1e-8:
            y_std = 1.0
        y_train = (y_train - y_mean) / y_std
    else:
        raise ValueError(f"unknown task kind: {train.kind!r}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_score = -float("inf")
    best_state = None
    best_epoch = -1
    curve: list[float] = []
    epochs_run = 0
    start_time = time.perf_counter()
    for epoch in range(epochs):
        encoder.train()
        head.train()
        permutation = torch.randperm(len(train.x), generator=generator).numpy()
        for start in range(0, len(permutation), batch):
            idx = permutation[start:start + batch]
            xb = torch.as_tensor(train.x[idx], dtype=torch.float32, device=dev)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32 if train.kind == "reg" else torch.long, device=dev)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(dev):
                logits = head(pool(encoder(xb, hide=None)))
                loss = criterion(logits.reshape(-1) if train.kind == "reg" else logits, yb.reshape(-1) if train.kind == "reg" else yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

        val_score = _evaluate(encoder, head, val.x, val.y, train.kind, dev, batch, y_mean, y_std)
        curve.append(val_score)
        epochs_run = epoch + 1
        if np.isfinite(val_score) and val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict()))
        elif epoch - best_epoch >= patience:
            break
    train_s = time.perf_counter() - start_time
    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    test_score = _evaluate(encoder, head, test.x, test.y, train.kind, dev, batch, y_mean, y_std)
    from rpjepa.encoders import count_flops, n_params

    length = int(train.x.shape[1])
    params = n_params(encoder)
    flops = count_flops(encoder, length, batch=1)
    return {
        "metric": test_score,
        "val_metric": best_score,
        "params": params,
        "flops": flops,
        "train_s": train_s,
        "epochs_run": epochs_run,
        "curve": curve,
    }

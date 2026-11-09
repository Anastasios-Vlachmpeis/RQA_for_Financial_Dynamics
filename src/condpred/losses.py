"""Forecast loss functions and regression target transforms."""
from __future__ import annotations

import numpy as np


def qlike(y, f) -> np.ndarray:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return y / f - np.log(y / f) - 1


def log_ratio(y, f) -> np.ndarray:
    return np.log(np.asarray(y, float) / np.asarray(f, float))


def sq_norm(y, f, var) -> np.ndarray:
    return (np.asarray(y, float) - np.asarray(f, float)) ** 2 / float(var)


def fisher_abs(rho_real, rho_hat) -> np.ndarray:
    r, h = np.clip(np.asarray(rho_real, float), -0.999, 0.999), np.clip(np.asarray(rho_hat, float), -0.999, 0.999)
    return np.abs(np.arctanh(r) - np.arctanh(h))


def loss_transform(L, kind: str = "log", c: float = 1e-4):
    L = np.asarray(L, float)
    if kind == "log":
        return np.log(L + c)
    raise ValueError(kind)

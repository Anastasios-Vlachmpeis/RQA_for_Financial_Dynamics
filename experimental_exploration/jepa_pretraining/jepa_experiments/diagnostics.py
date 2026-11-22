"""Diagnostic orientation tasks kept separate from the frozen benchmark tasks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    TaskData,
    _garch_abs_batch,
    _henon_batch,
    _logistic_batch,
    _zscore,
    window_length,
)

DIAGNOSTIC_TASKS: dict[str, str] = {"sign": "cls", "direction": "cls"}
_SIGN_FAMILIES = ("logistic", "henon", "garch_abs")


def make_diagnostic(task: str, n: int, n_states: int, seed: int,
                    m: int = 3, tau: int = 1) -> TaskData:
    """Generate paired sign-flip or time-reversal windows as ``TaskData``.

    The sign task uses z-scored logistic, Hénon-x, and GARCH-absolute-return
    windows with small noise; label 1 is the negated window. The direction task
    uses the same irreversible systems; label 1 is the time-reversed window.
    """
    if task not in DIAGNOSTIC_TASKS:
        raise ValueError(f"unknown diagnostic task {task!r}; choose from {tuple(DIAGNOSTIC_TASKS)}")
    if n < 1:
        raise ValueError("n must be positive")
    length = window_length(n_states, m, tau)
    rng = np.random.default_rng(seed)
    n_base = (n + 1) // 2
    family_codes = rng.integers(0, len(_SIGN_FAMILIES), n_base)
    families = np.asarray(_SIGN_FAMILIES)[family_codes]
    parameters = np.empty(n_base, dtype=np.float64)
    raw = np.empty((n_base, length), dtype=np.float64)

    for code, family in enumerate(_SIGN_FAMILIES):
        selected = family_codes == code
        count = int(selected.sum())
        if not count:
            continue
        if family == "logistic":
            parameters[selected] = rng.uniform(3.8, 4.0, count)
            raw[selected] = _logistic_batch(parameters[selected], length, rng)
        elif family == "henon":
            parameters[selected] = rng.uniform(1.3, 1.4, count)
            raw[selected] = _henon_batch(parameters[selected], length, rng)
        else:
            parameters[selected] = rng.uniform(0.7, 0.98, count)
            raw[selected] = _garch_abs_batch(parameters[selected], length, rng)

    noise = np.full(n_base, 0.02, dtype=np.float64)
    raw += rng.standard_normal(raw.shape) * (noise[:, None] * np.maximum(raw.std(axis=1, keepdims=True), 1e-8))
    base = _zscore(raw)
    oriented = base[:, ::-1] if task == "direction" else -base
    paired = np.stack((base, oriented), axis=1).reshape(2 * n_base, length)
    labels = np.tile(np.asarray((0, 1), dtype=np.int64), n_base)
    orientations = ("forward", "reversed" if task == "direction" else "sign_flipped")
    meta = pd.DataFrame({
        "family": np.repeat(families, 2),
        "pair_id": np.repeat(np.arange(n_base), 2),
        "parameter": np.repeat(parameters, 2),
        "noise": np.repeat(noise, 2),
        "orientation": np.tile(orientations, n_base),
    })
    order = rng.permutation(len(labels))[:n]
    return TaskData(paired[order].astype(np.float32, copy=False), labels[order], "cls", 2,
                    meta.iloc[order].reset_index(drop=True))

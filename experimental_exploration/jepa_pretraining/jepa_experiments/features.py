"""CPU feature extraction from recurrence plots and raw time-series statistics."""
from __future__ import annotations

import __main__
import math
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the sibling RecPlay source tree without requiring an editable install.
_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from recplay.recurrence import recurrence_matrix
from recplay.rqa import rqa


def _permutation_entropy(x: np.ndarray, order: int = 4) -> float:
    n = len(x) - order + 1
    if n < 1:
        return float("nan")
    patterns = np.argsort(np.lib.stride_tricks.sliding_window_view(x, order), axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(order)))


def _feature_row(args) -> dict[str, float]:
    row, m, tau, rrs = args
    x = np.asarray(row, dtype=np.float64)
    sd = x.std()
    if sd <= 1e-12:
        x = x - x.mean()
    else:
        x = (x - x.mean()) / sd
    n_states = len(x) - (m - 1) * tau
    states = np.column_stack([x[k * tau:k * tau + n_states] for k in range(m)])
    result: dict[str, float] = {}
    for rr in rrs:
        R = recurrence_matrix(states, float(rr), theiler=1)
        for key, value in rqa(R).items():
            result[f"{key}_rr{rr:g}"] = float(value)
    diff = np.diff(x)
    result["diff_std"] = float(diff.std())
    for lag in range(1, 6):
        result[f"acf{lag}"] = float(np.mean(x[:-lag] * x[lag:]))
    result["permutation_entropy4"] = _permutation_entropy(x)
    for k, value in enumerate(x[-m:]):
        result[f"last_{k}"] = float(value)
    return result


def rqa_features(x: np.ndarray, m: int = 3, tau: int = 1,
                 rrs=(0.05, 0.1, 0.2)) -> pd.DataFrame:
    """Compute RQA measures, scalar statistics, and final samples for each [n,L] row.

    The only expected NaNs are ``K2_lines`` when too few diagonal lines are
    available to estimate its slope on a short/noise-like window.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must have shape [n,L]")
    if m < 1 or tau < 1 or x.shape[1] - (m - 1) * tau < 2:
        raise ValueError("embedding requires positive m/tau and at least two states")
    rrs = tuple(float(rr) for rr in rrs)
    if not rrs or any(not 0.0 < rr < 1.0 for rr in rrs):
        raise ValueError("rrs must contain recurrence rates strictly between zero and one")
    args = ((row, m, tau, rrs) for row in x)
    rows = list(args)
    if len(rows) < 2:
        computed = [_feature_row(rows[0])] if rows else []
    else:
        # Spawned workers need an importable, guarded main module on Windows;
        # interactive callers fall back to the same top-level worker serially.
        main_file = getattr(__main__, "__file__", None)
        if os.name == "nt" and (not main_file or str(main_file).startswith("<")):
            computed = [_feature_row(arg) for arg in rows]
        else:
            workers = min(len(rows), max(1, min(cpu_count(), 8)))
            with Pool(processes=workers) as pool:
                computed = pool.map(_feature_row, rows, chunksize=max(1, len(rows) // (workers * 8)))
    return pd.DataFrame.from_records(computed)

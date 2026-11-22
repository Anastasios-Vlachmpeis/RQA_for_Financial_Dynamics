"""Sliding-window RQA features. The row for month t uses only the W months ending at t.

`series` maps a column prefix to a column of `data` (e.g. {"cpi": "INFLATION"}); `joints` maps a prefix to
two series prefixes whose RPs are AND-ed (e.g. {"jun": ("cpi", "un")}). `distance(z, m, tau)` turns a
z-scored window into a pairwise distance or divergence matrix of its embedded states.
"""
import numpy as np
import pandas as pd

from recplay.embedding import embed
from recplay.recurrence import euclidean_distances, symmetric_beta_matrix, threshold_by_rate
from recplay.rqa import rqa


def window_z(x: np.ndarray, end: int, W: int) -> np.ndarray:
    """x[end-W+1 : end+1] z-scored with its own mean and std (zeros if constant)."""
    w = x[end - W + 1 : end + 1]
    sd = w.std()
    return (w - w.mean()) / sd if sd > 0 else np.zeros_like(w)


def euclidean_distance(z: np.ndarray, m: int, tau: int) -> np.ndarray:
    return euclidean_distances(embed(z, m, tau))


def beta_distance(z: np.ndarray, m: int, tau: int, beta: float) -> np.ndarray:
    """Symmetric β-divergence between embedded exp(z) states (exp makes them strictly positive)."""
    return symmetric_beta_matrix(embed(np.exp(z), m, tau), beta)


def exp_euclidean_distance(z: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Euclidean distance between embedded exp(z) states, the states beta_distance uses."""
    return euclidean_distances(embed(np.exp(z), m, tau))


def rqa_frame(data: pd.DataFrame, series: dict, joints: dict, W: int, m: int, tau: int, rr: float,
              distance, theiler: int = 1, step: int = 1) -> pd.DataFrame:
    """rqa() of every single-series RP (thresholded at rate rr) and every joint RP, columns '<prefix>_<measure>'.

    Also RR_last: the recurrence rate of the window's last state (the current month) with the rest of the window.
    Windows end every `step` rows, starting at row W - 1.
    """
    arrays = {p: data[c].to_numpy(dtype=float) for p, c in series.items()}
    rows = []
    for end in range(W - 1, len(data), step):
        R = {p: threshold_by_rate(distance(window_z(x, end, W), m, tau), rr, theiler) for p, x in arrays.items()}
        R.update({j: R[a] & R[b] for j, (a, b) in joints.items()})
        rows.append({f"{p}_{k}": v for p, Rp in R.items()
                     for k, v in {**rqa(Rp), "RR_last": Rp[-1].sum() / (len(Rp) - theiler)}.items()})
    return pd.DataFrame(rows, index=data.index[W - 1 :: step])

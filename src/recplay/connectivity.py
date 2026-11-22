"""Rank recurrence pairs for joint recurrence plots."""

import numpy as np
from scipy.stats import rankdata


def omega_ranks(D: np.ndarray, theiler: int = 1) -> np.ndarray:
    """Rank of each pair's D within Ω (ties share the highest rank); 0 outside Ω.

    Pair (i, j) is in the RP thresholded at rate q iff its rank <= q|Ω|, so the elementwise max of two such
    matrices orders pairs by when they enter the joint RP.
    """
    n = len(D)
    I, J = np.triu_indices(n, theiler)
    out = np.zeros(D.shape)
    out[I, J] = rankdata(D[I, J], method="max")
    return out

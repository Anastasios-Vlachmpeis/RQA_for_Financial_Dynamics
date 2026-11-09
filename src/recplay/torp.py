"""Third-order recurrence: triplet dissimilarity tensors (Boussé et al. 2023) and their thresholding."""
import numpy as np


def cross_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between every row of a and every row of b: shape (len(a), len(b))."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))


def perimeter_tensor(a: np.ndarray, b: np.ndarray, c: np.ndarray, distance=cross_distances,
                     dtype=np.float32) -> np.ndarray:
    """T[i, j, k] = d(a_i, b_j) + d(a_i, c_k) + d(b_j, c_k) for three state arrays of equal dimension.

    `distance(p, q)` returns the (len(p), len(q)) dissimilarity matrix. With a = b = c this is the
    single-series TORP: T[i, i, k] = 2 d(a_i, a_k) and T[i, i, i] = 0.
    """
    d_ab, d_ac, d_bc = (distance(p, q).astype(dtype) for p, q in ((a, b), (a, c), (b, c)))
    t = d_ab[:, :, None] + d_ac[:, None, :]
    t += d_bc[None, :, :]
    return t


def threshold_tensor_by_rate(t: np.ndarray, rr: float):
    """Binary tensor t <= ε with ε the rr-quantile of all entries, so the recurrence rate is about rr.

    Returns (R, eps). No Theiler band: across three systems, equal indices are not self-matches.
    """
    eps = float(np.quantile(t, rr))
    return t <= eps, eps


def whitening_matrix(features: np.ndarray) -> np.ndarray:
    """Σ^(-1/2) of the feature covariance, so Euclidean distance after `features @ matrix` is Mahalanobis."""
    vals, vecs = np.linalg.eigh(np.cov(np.asarray(features, dtype=float), rowvar=False))
    if vals.min() <= 1e-12 * vals.max():
        raise ValueError("singular feature covariance; drop a redundant feature")
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def same_time_stats(t: np.ndarray, rr: float) -> dict:
    """Same-time diagonal T[t, t, t] against the whole tensor: mean ratio and recurrence-rate enrichment at rate rr."""
    idx = np.arange(len(t))
    diag = t[idx, idx, idx].astype(float)
    R, eps = threshold_tensor_by_rate(t, rr)
    d_all = float(t.mean(dtype=np.float64))
    rr_sync = float(R[idx, idx, idx].mean())
    return {"D_sync": float(diag.mean()), "D_all": d_all, "sync_ratio": float(diag.mean()) / d_all,
            "eps": eps, "RR_sync": rr_sync, "sync_enrichment": rr_sync / rr}

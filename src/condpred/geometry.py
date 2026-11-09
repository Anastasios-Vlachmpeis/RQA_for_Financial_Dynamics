"""Window scaling, distance, and threshold arms for recurrence plots."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recplay.embedding import embed
from recplay.recurrence import euclidean_distances, symmetric_beta_matrix, threshold_by_rate


def window_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    sd = x.std()
    if sd == 0 or not np.isfinite(sd):
        return x - x.mean()
    return (x - x.mean()) / sd


@dataclass
class Geometry:
    scaling: str = "wz"
    distance: str = "euclid"
    threshold: str = "rr"
    rr: float = 0.10
    beta: float = 2.0
    theiler: int | None = None
    m: int = 3
    tau: int = 1
    train_mean: float | None = None
    train_sd: float | None = None
    train_median: float | None = None
    eps_train: float | None = None
    eps_pool: list | None = field(default=None, repr=False)

    def theiler_band(self) -> int:
        if self.theiler is not None:
            return self.theiler
        return max(1, (self.m - 1) * self.tau)

    def banded_ok(self) -> bool:
        """Whether the banded RP path (§01b §4.1) matches ``recurrence`` for this arm."""
        if self.scaling == "rank":
            return False
        if self.threshold == "eps_train" and self.scaling in ("wz", "wmad"):
            return False
        if self.distance == "beta_exp" and self.scaling in ("wz", "wmad"):
            return False
        if self.distance == "ordinal":
            return True
        if self.distance in ("beta_pos", "power_ctrl"):
            return True
        if self.distance == "euclid" and self.scaling in ("wz", "wmad", "gz"):
            return self.threshold == "rr" or (self.scaling == "gz" and self.threshold == "eps_train")
        return False

    def scale_window(self, win: np.ndarray) -> np.ndarray:
        win = np.asarray(win, float)
        if self.scaling == "wz":
            return window_z(win)
        if self.scaling == "gz":
            if self.train_mean is None or self.train_sd is None:
                raise ValueError("gz scaling needs train_mean and train_sd")
            sd = self.train_sd if self.train_sd > 0 else 1.0
            return (win - self.train_mean) / sd
        if self.scaling == "wmad":
            med = np.median(win)
            mad = np.median(np.abs(win - med))
            return (win - med) / (mad if mad > 0 else 1.0)
        if self.scaling == "rank":
            order = np.argsort(np.argsort(win, kind="stable"), kind="stable")
            return order / max(len(win) - 1, 1)
        raise ValueError(self.scaling)

    def distance_matrix(self, states: np.ndarray, positive_raw: np.ndarray | None = None) -> np.ndarray:
        states = np.asarray(states, float)
        d = self.distance
        if d == "euclid":
            return euclidean_distances(states)
        if d == "beta_exp":
            exp_s = np.exp(states)
            return symmetric_beta_matrix(exp_s, self.beta)
        if d == "beta_pos":
            if positive_raw is None:
                raise ValueError("beta_pos needs positive_raw")
            pos = np.asarray(positive_raw, float)
            if self.train_median is not None and self.train_median > 0:
                pos = pos / self.train_median
            st = embed(pos, self.m, self.tau)
            return symmetric_beta_matrix(st, self.beta)
        if d == "power_ctrl":
            if positive_raw is None:
                raise ValueError("power_ctrl needs positive_raw")
            pos = np.asarray(positive_raw, float)
            if self.train_median is not None and self.train_median > 0:
                pos = pos / self.train_median
            if self.beta == 0:
                tr = np.log(pos)
            else:
                tr = pos ** (self.beta / 2.0)
            st = embed(tr, self.m, self.tau)
            return euclidean_distances(st)
        if d == "ordinal":
            return _ordinal_matrix(states)
        raise ValueError(d)

    def recurrence(self, win: np.ndarray, positive_raw: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (distance matrix, thresholded RP) for a scaled window."""
        scaled = self.scale_window(win)
        states = embed(scaled, self.m, self.tau)
        raw_pos = positive_raw if positive_raw is not None else win
        if self.distance in ("beta_pos", "power_ctrl"):
            D = self.distance_matrix(states, positive_raw=raw_pos)
        else:
            D = self.distance_matrix(states)
        if self.distance == "ordinal":
            R = D.astype(bool)
            n = len(R)
            band = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < self.theiler_band()
            R[band] = False
            return D, R
        if self.threshold == "rr":
            R = threshold_by_rate(D, self.rr, self.theiler_band())
        elif self.threshold == "eps_train":
            if self.eps_train is None:
                raise ValueError("eps_train arm needs eps_train set")
            R = D <= self.eps_train
            n = len(R)
            band = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < self.theiler_band()
            R[band] = False
        else:
            raise ValueError(self.threshold)
        return D, R

    def fit_eps_train(self, windows: list[np.ndarray], *, positive_windows: list[np.ndarray] | None = None):
        """Pool off-diagonal distances on training windows; ``eps_train`` = ``rr``-quantile (§4.10)."""
        dists = []
        for i, w in enumerate(windows):
            w = np.asarray(w, float)
            pos_w = None
            if positive_windows is not None:
                pos_w = np.asarray(positive_windows[i], float)
            elif self.distance in ("beta_pos", "power_ctrl"):
                pos_w = w
            scaled = self.scale_window(w)
            states = embed(scaled, self.m, self.tau)
            if self.distance in ("beta_pos", "power_ctrl"):
                D = self.distance_matrix(states, positive_raw=pos_w if pos_w is not None else w)
            else:
                D = self.distance_matrix(states)
            n = len(D)
            th = self.theiler_band()
            dists.append(D[np.triu_indices(n, th)])
        pool = np.concatenate(dists)
        self.eps_pool = pool.tolist()
        self.eps_train = float(np.quantile(pool, self.rr))


def _ordinal_matrix(states: np.ndarray, order: int | None = None) -> np.ndarray:
    order = order or states.shape[1]
    patterns = np.argsort(states, axis=1)
    n = len(states)
    same = patterns[:, None, :] == patterns[None, :, :]
    R = same.all(axis=2).astype(float)
    return R

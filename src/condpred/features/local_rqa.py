"""Current-state (local) RQA features; row at t uses the window ending at t only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from condpred.geometry import Geometry


def local_rqa(x, end: int, W: int, m: int, tau: int, geom: Geometry, H: int = 8, w: int | None = None) -> dict:
    start = max(0, end - W + 1)
    win = np.asarray(x[start : end + 1], float)
    if len(win) < W:
        return _nan_out()
    geom = Geometry(**{**geom.__dict__})
    geom.m, geom.tau = m, tau
    pos = win if geom.distance in ("beta_pos", "power_ctrl") else None
    _, R = geom.recurrence(win, positive_raw=pos)
    return local_rqa_matrix(R, theiler=geom.theiler_band(), H=H, w=w)


def _lrqa_from_packed(packed: bytes, n: int, theiler: int, H: int, w: int | None) -> dict:
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[: n * n]
    R = bits.reshape(n, n).astype(bool)
    return local_rqa_matrix(R, theiler=theiler, H=H, w=w)


def local_rqa_matrix(R: np.ndarray, theiler: int = 1, H: int = 8, w: int | None = None) -> dict:
    R = np.asarray(R, bool)
    n = len(R)
    w = w if w is not None else theiler
    cur = n - 1
    neigh = [i for i in range(n - w) if R[i, cur]]
    if not neigh:
        return _nan_out()
    rr_last = len(neigh) / max(n - w, 1)
    b_lens = [_backward_match(R, i, cur) for i in neigh]
    lb_mean, lb_max = float(np.mean(b_lens)), float(np.max(b_lens))
    det_last = float(np.mean([b >= 2 for b in b_lens]))
    lam_last, tt_last = _lam_tt_last_col(R, cur, w)
    pairs_f = []
    for ii, i in enumerate(neigh):
        for j in neigh[ii + 1 :]:
            if j + H < n:
                pairs_f.append(_forward_match(R, i, j, H))
    lf_nb = float(np.mean(pairs_f)) if pairs_f else np.nan
    k2_local = _k2_local(pairs_f, H)
    return {
        "RR_last": rr_last,
        "Lb_mean": lb_mean,
        "Lb_max": lb_max,
        "DET_last": det_last,
        "Lf_nb": lf_nb,
        "K2_local": k2_local,
        "LAM_last": lam_last,
        "TT_last": tt_last,
    }


def local_rqa_frame(
    x,
    W,
    m,
    tau,
    geom: Geometry,
    step=1,
    n_jobs=-1,
    *,
    ends=None,
    backend: str = "numpy",
    device=None,
    positive_raw=None,
    H: int = 8,
) -> pd.DataFrame:
    x = np.asarray(x, float)
    if ends is None:
        ends = np.arange(W - 1 + (m - 1) * tau, len(x), step, dtype=int)
    else:
        ends = np.asarray(ends, dtype=int)
    geom = Geometry(**{**geom.__dict__})
    geom.m, geom.tau = m, tau
    if backend == "torch":
        from condpred import rp_batch

        if rp_batch.arm_falls_back(geom, W):
            import warnings

            warnings.warn(
                f"local_rqa_frame: {rp_batch.geometry_arm_key(geom, W)} using numpy fallback",
                stacklevel=2,
            )
            backend = "numpy"
    if backend == "numpy":
        rows = Parallel(n_jobs=n_jobs)(
            delayed(local_rqa)(x, int(t), W, m, tau, geom, H=H) for t in ends
        )
        return pd.DataFrame(rows, index=list(ends))

    from condpred import rp_batch
    n_state = W - (m - 1) * tau
    th = geom.theiler_band()
    pos = np.asarray(positive_raw, float) if positive_raw is not None else None
    all_rows: list[dict] = []
    all_idx: list[int] = []
    for eb, Rb, _ in rp_batch.window_rps(
        x, ends, W, geom, positive_raw=pos, torch_device=device
    ):
        jobs = [(int(e), np.packbits(Rb[j].ravel()).tobytes()) for j, e in enumerate(eb)]
        batch_rows = Parallel(n_jobs=n_jobs)(
            delayed(_lrqa_from_packed)(p, n_state, th, H, None) for _, p in jobs
        )
        all_rows.extend(batch_rows)
        all_idx.extend(e for e, _ in jobs)
    return pd.DataFrame(all_rows, index=all_idx)


def _nan_out():
    return {k: np.nan for k in ["RR_last", "Lb_mean", "Lb_max", "DET_last", "Lf_nb", "K2_local", "LAM_last", "TT_last"]}


def _backward_match(R, i, j, max_l=1000):
    l = 0
    while i - l >= 0 and j - l >= 0 and R[i - l, j - l]:
        l += 1
    return l


def _forward_match(R, i, j, H):
    l = 0
    while l < H and i + l < len(R) and j + l < len(R) and R[i + l, j + l]:
        l += 1
    return l


def _vertical_runs_in_col(col: np.ndarray) -> list[tuple[int, int]]:
    """Return (start_row, length) for each vertical run of True in ``col``."""
    runs = []
    i, n = 0, len(col)
    while i < n:
        if col[i]:
            j = i
            while j < n and col[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


def _lam_tt_last_col(R: np.ndarray, cur: int, w: int) -> tuple[float, float]:
    """LAM_last / TT_last from vertical runs in ``R[:n-w, cur]`` (spec §4.12)."""
    n = len(R)
    col = R[: max(0, n - w), cur]
    if not col.any():
        return np.nan, np.nan
    run_lens = [length for _, length in _vertical_runs_in_col(col)]
    if not run_lens:
        return np.nan, np.nan
    total = float(sum(run_lens))
    long = [r for r in run_lens if r >= 2]
    lam_last = float(sum(long) / total) if total > 0 else np.nan
    tt_last = float(np.mean(long)) if long else np.nan
    return lam_last, tt_last


def _k2_local(pairs_f, H):
    if len(pairs_f) < 10:
        return np.nan
    ls = np.arange(1, min(7, H + 1))
    shares = [np.mean([f >= l for f in pairs_f]) for l in ls]
    ok = [s > 0 for s in shares]
    if sum(ok) < 3:
        return np.nan
    ls_ok = ls[ok]
    sh_ok = np.log(np.asarray(shares)[ok])
    return -float(np.polyfit(ls_ok, sh_ok, 1)[0])

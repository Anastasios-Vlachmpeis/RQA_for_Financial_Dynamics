"""Window-level RQA features (feature row dated by window end t; uses samples ≤ t)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from condpred.geometry import Geometry
from recplay.rqa import line_lengths, rqa


def _shannon_entropies(lengths):
    if len(lengths) == 0:
        return np.nan
    vals = lengths[lengths >= 2]
    if len(vals) == 0:
        return np.nan
    counts = np.bincount(vals)[2:]
    p = counts[counts > 0] / counts[counts > 0].sum()
    return float(-(p * np.log(p)).sum())


def _keys():
    return ["DET", "L", "ENTR", "LAM", "DET3_DET2", "K2_lines", "TT", "Lmax", "Vmax", "ENTR_v", "DET_LAM", "RR"]


def _geom_slim(geom: Geometry) -> dict:
    return {
        "scaling": geom.scaling,
        "distance": geom.distance,
        "threshold": geom.threshold,
        "rr": geom.rr,
        "beta": geom.beta,
        "theiler": geom.theiler,
        "m": geom.m,
        "tau": geom.tau,
        "train_mean": geom.train_mean,
        "train_sd": geom.train_sd,
        "train_median": geom.train_median,
        "eps_train": geom.eps_train,
    }


def _k2_from_diag(diag: np.ndarray, n: int, ls=None, min_count: int = 10) -> float:
    """Same value as ``recplay.rqa.k2_lines``; pairs use the closed form ``(n-l)(n-l+1)/2``."""
    if ls is None:
        ls = np.arange(1, 7)
    ls = np.asarray(ls)
    counts = np.array([np.clip(np.asarray(diag) - l + 1, 0, None).sum() for l in ls], float)
    pairs = np.array([(n - int(l)) * (n - int(l) + 1) / 2.0 for l in ls], float)
    ok = counts >= min_count
    if int(ok.sum()) < 3:
        return np.nan
    return -np.polyfit(ls[ok], np.log(counts[ok] / pairs[ok]), 1)[0]


def _rqa_base_from_lines(diag: np.ndarray, vert: np.ndarray, R: np.ndarray) -> dict:
    """``recplay.rqa.rqa`` scalars from one ``line_lengths`` result."""
    pts = diag.sum()
    det = lambda lmin: diag[diag >= lmin].sum() / pts if pts else np.nan
    long = diag[diag >= 2]
    p = np.bincount(long)[2:] if len(long) else np.array([], int)
    p = p[p > 0] / p.sum() if len(p) and p.sum() else np.array([])
    return {
        "RR": float(np.asarray(R).mean()),
        "DET": det(2),
        "L": float(long.mean()) if len(long) else 0.0,
        "ENTR": float(-(p * np.log(p)).sum()) if len(p) else 0.0,
        "LAM": float(vert[vert >= 2].sum() / max(vert.sum(), 1)),
        "DET3_DET2": det(3) / max(det(2), 1e-12) if pts else np.nan,
        "K2_lines": _k2_from_diag(diag, len(R)),
    }


def _rqa_from_lines(
    diag: np.ndarray,
    vert: np.ndarray,
    R: np.ndarray,
    geom: Geometry,
    rr: float | None = None,
) -> dict:
    """RQA scalars without a second ``line_lengths`` call."""
    pts = diag.sum()
    det = lambda lmin: diag[diag >= lmin].sum() / pts if pts else np.nan
    long = diag[diag >= 2]
    p = np.bincount(long)[2:] if len(long) else np.array([], int)
    p = p[p > 0] / p.sum() if len(p) and p.sum() else np.array([])
    out = {
        "DET": det(2),
        "L": float(long.mean()) if len(long) else 0.0,
        "ENTR": float(-(p * np.log(p)).sum()) if len(p) else 0.0,
        "LAM": float(vert[vert >= 2].sum() / max(vert.sum(), 1)),
        "DET3_DET2": det(3) / max(det(2), 1e-12) if pts else np.nan,
        "K2_lines": _k2_from_diag(diag, len(R)),
    }
    if not (geom.threshold == "rr" and geom.distance != "ordinal"):
        out["RR"] = float(np.asarray(R).mean() if rr is None else rr)
    out["TT"] = float(vert[vert >= 2].mean()) if (vert >= 2).any() else np.nan
    out["Lmax"] = float(diag.max()) if len(diag) else np.nan
    out["Vmax"] = float(vert.max()) if len(vert) else np.nan
    out["ENTR_v"] = _shannon_entropies(vert)
    lam, det_v = out.get("LAM", np.nan), out.get("DET", np.nan)
    out["DET_LAM"] = det_v / lam if lam and np.isfinite(lam) and lam != 0 else np.nan
    return out


def _rqa_extras(R: np.ndarray, geom: Geometry) -> dict:
    diag, vert = line_lengths(R)
    return _rqa_from_lines(diag, vert, R, geom)


def window_rqa(x, end: int, W: int, m: int, tau: int, geom: Geometry) -> dict:
    """RQA of the window ``x[end-W+1:end+1]``; row dated by ``end``."""
    geom = Geometry(**{**geom.__dict__}) if geom.m != m or geom.tau != tau else geom
    geom.m, geom.tau = m, tau
    start = max(0, end - W + 1)
    win = np.asarray(x[start : end + 1], float)
    if len(win) < W:
        return {k: np.nan for k in _keys()}
    pos = win if geom.distance in ("beta_pos", "power_ctrl") else None
    _, R = geom.recurrence(win, positive_raw=pos)
    return _rqa_extras(R, geom)


def _wrqa_from_packed(packed: bytes, n: int, geom_kwargs: dict) -> dict:
    """Worker entry. ``geom_kwargs`` is only ``threshold`` and ``distance`` (no ``eps_pool``)."""
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[: n * n]
    R = bits.reshape(n, n).astype(bool)
    geom = Geometry(threshold=geom_kwargs["threshold"], distance=geom_kwargs["distance"])
    return _rqa_extras(R, geom)


def _default_ends(x, W, m, tau, step):
    return np.arange(W - 1 + (m - 1) * tau, len(x), step, dtype=int)


def window_rqa_frame(
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
) -> pd.DataFrame:
    x = np.asarray(x, float)
    if ends is None:
        ends = _default_ends(x, W, m, tau, step)
    else:
        ends = np.asarray(ends, dtype=int)
    geom = Geometry(**{**geom.__dict__})
    geom.m, geom.tau = m, tau
    if backend == "torch":
        from condpred import rp_batch

        if rp_batch.arm_falls_back(geom, W):
            import warnings

            warnings.warn(
                f"window_rqa_frame: {rp_batch.geometry_arm_key(geom, W)} using numpy fallback",
                stacklevel=2,
            )
            backend = "numpy"
    if backend == "numpy":
        rows = Parallel(n_jobs=n_jobs)(
            delayed(window_rqa)(x, int(t), W, m, tau, geom) for t in ends
        )
        return pd.DataFrame(rows, index=list(ends))

    from condpred import rp_batch
    from condpred.rp_rqa_stage2 import line_hist_torch, rqa_from_histograms

    n_state = W - (m - 1) * tau
    pos = np.asarray(positive_raw, float) if positive_raw is not None else None
    all_rows: list[dict] = []
    all_idx: list[int] = []
    for eb, Rb, _ in rp_batch.window_rps(
        x, ends, W, geom, positive_raw=pos, torch_device=device
    ):
        hist_dev = device if device is not None else rp_batch.device()
        dh, vh = line_hist_torch(Rb, device=hist_dev)
        for j, end in enumerate(eb):
            rr = float(Rb[j].mean())
            all_rows.append(rqa_from_histograms(dh[j], vh[j], n_state, geom, rr_from_R=rr))
            all_idx.append(int(end))
    return pd.DataFrame(all_rows, index=all_idx)


def wrqa_lrqa_frames(
    x,
    W,
    m,
    tau,
    geom: Geometry,
    *,
    ends=None,
    step=1,
    n_jobs=-1,
    backend: str = "numpy",
    device=None,
    positive_raw=None,
    H: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single RP batch feeding WRQA (stage-2 GPU hist) and LRQA (packed CPU workers)."""
    x = np.asarray(x, float)
    if ends is None:
        ends = _default_ends(x, W, m, tau, step)
    else:
        ends = np.asarray(ends, dtype=int)
    geom = Geometry(**{**geom.__dict__})
    geom.m, geom.tau = m, tau
    if backend == "torch":
        from condpred import rp_batch

        if rp_batch.arm_falls_back(geom, W):
            import warnings

            warnings.warn(
                f"wrqa_lrqa_frames: {rp_batch.geometry_arm_key(geom, W)} using numpy fallback",
                stacklevel=2,
            )
            backend = "numpy"
    if backend == "numpy":
        wr = window_rqa_frame(
            x, W, m, tau, geom, step=step, n_jobs=n_jobs, ends=ends, backend="numpy", positive_raw=positive_raw
        )
        from condpred.features.local_rqa import local_rqa_frame

        lr = local_rqa_frame(
            x,
            W,
            m,
            tau,
            geom,
            step=step,
            n_jobs=n_jobs,
            ends=ends,
            backend="numpy",
            H=H,
            positive_raw=positive_raw,
        )
        return wr, lr

    from condpred import rp_batch
    from condpred.features.local_rqa import _lrqa_from_packed
    from condpred.rp_rqa_stage2 import line_hist_torch, rqa_from_histograms
    n_state = W - (m - 1) * tau
    th = geom.theiler_band()
    pos = np.asarray(positive_raw, float) if positive_raw is not None else None
    wr_rows, lr_rows = [], []
    wr_idx, lr_idx = [], []
    for eb, Rb, _ in rp_batch.window_rps(
        x, ends, W, geom, positive_raw=pos, torch_device=device
    ):
        hist_dev = device if device is not None else rp_batch.device()
        dh, vh = line_hist_torch(Rb, device=hist_dev)
        pack_jobs = []
        for j, end in enumerate(eb):
            rr = float(Rb[j].mean())
            wr_rows.append(rqa_from_histograms(dh[j], vh[j], n_state, geom, rr_from_R=rr))
            wr_idx.append(int(end))
            pack_jobs.append((int(end), np.packbits(Rb[j].ravel()).tobytes()))
        batch_lr = Parallel(n_jobs=n_jobs)(
            delayed(_lrqa_from_packed)(p, n_state, th, H, None) for _, p in pack_jobs
        )
        lr_rows.extend(batch_lr)
        lr_idx.extend(e for e, _ in pack_jobs)
    return pd.DataFrame(wr_rows, index=wr_idx), pd.DataFrame(lr_rows, index=lr_idx)

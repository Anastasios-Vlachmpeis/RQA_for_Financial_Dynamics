"""Batched recurrence plots via PyTorch (GPU when available)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np

from condpred.geometry import Geometry
from recplay.embedding import embed

_DEVICE_ONCE = None
_FALLBACK_ARMS: set[str] = set()


def _torch():
    import torch

    return torch


def device(prefer: str = "cuda"):
    """Return cuda if available else cpu; print once."""
    global _DEVICE_ONCE
    torch = _torch()
    dev = torch.device("cuda" if prefer == "cuda" and torch.cuda.is_available() else "cpu")
    if _DEVICE_ONCE is None:
        _DEVICE_ONCE = dev
        ver = torch.__version__
        if dev.type == "cuda":
            name = torch.cuda.get_device_name(dev)
            print(f"condpred.rp_batch: backend=torch device={dev} ({name}) torch={ver}")
        else:
            print(f"condpred.rp_batch: backend=torch device=cpu torch={ver}")
    return dev


def batch_budget(
    n_states: int,
    dtype,
    *,
    torch_device=None,
    safety: float = 0.5,
    cap_bytes: int = 2 * 2**30,
) -> int:
    """Max windows per batch given peak tensor memory for [B,n,n] distances and masks."""
    torch = _torch()
    if torch_device is not None:
        dev = torch.device(torch_device)
    else:
        dev = device()
    elem = torch.finfo(dtype).bits // 8 if dtype in (torch.float32, torch.float64) else 4
    per_win = n_states * n_states * elem * 6
    if per_win <= 0:
        return 1
    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info(dev)
        budget = int(min(cap_bytes, safety * free))
    else:
        budget = cap_bytes
    return max(1, budget // max(per_win, 1))


def _lerp_eps_from_upper(D_upper, rr: float):
    """Per-row ε matching ``np.quantile(..., rr)`` (linear interpolation)."""
    torch = _torch()
    M = D_upper.shape[1]
    if M == 0:
        return torch.zeros(D_upper.shape[0], dtype=D_upper.dtype, device=D_upper.device)
    pos = rr * (M - 1)
    lo = int(np.floor(pos))
    g = float(pos - lo)
    eps_lo = torch.kthvalue(D_upper, lo + 1, dim=1).values
    if g == 0.0:
        return eps_lo
    hi_k = min(lo + 2, M)
    eps_hi = (
        torch.kthvalue(D_upper, hi_k, dim=1).values
        if hi_k > lo + 1
        else eps_lo
    )
    d = eps_hi - eps_lo
    if g < 0.5:
        return eps_lo + d * g
    return eps_hi - d * (1.0 - g)


def threshold_rr(D, rr: float, theiler: int):
    """Fixed-RR threshold matching numpy ``threshold_by_rate``."""
    torch = _torch()
    B, n, _ = D.shape
    mask = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=D.device), diagonal=theiler
    )
    D_upper = D.reshape(B, -1)[:, mask.reshape(-1)]
    eps = _lerp_eps_from_upper(D_upper, rr)
    band = (
        torch.abs(torch.arange(n, device=D.device)[:, None] - torch.arange(n, device=D.device)[None, :])
        < theiler
    )
    R = (D <= eps[:, None, None]) & ~band[None, :, :]
    return R


def threshold_eps(D, eps: float, theiler: int):
    torch = _torch()
    n = D.shape[-1]
    band = (
        torch.abs(torch.arange(n, device=D.device)[:, None] - torch.arange(n, device=D.device)[None, :])
        < theiler
    )
    R = D <= eps
    if D.ndim == 3:
        R = R & ~band[None, :, :]
    else:
        R = R & ~band
    return R


def recurrence_rate(R, theiler: int) -> np.ndarray:
    torch = _torch()
    if R.ndim == 2:
        R = R[None]
    n = R.shape[-1]
    mask = torch.abs(torch.arange(n, device=R.device)[:, None] - torch.arange(n, device=R.device)[None, :]) >= theiler
    counts = (R & mask[None]).sum(dim=(1, 2)).to(dtype=torch.float64)
    omega = mask.sum().to(dtype=torch.float64)
    return (counts / omega).detach().cpu().numpy()


def joint(Ra, Rb):
    return Ra & Rb


def jaccard(Ra, Rb, theiler: int):
    torch = _torch()
    if Ra.ndim == 2:
        Ra, Rb = Ra[None], Rb[None]
    n = Ra.shape[-1]
    mask = torch.abs(torch.arange(n, device=Ra.device)[:, None] - torch.arange(n, device=Ra.device)[None, :]) >= theiler
    m = mask[None]
    inter = (Ra & Rb & m).sum(dim=(1, 2)).to(dtype=torch.float64)
    union = ((Ra | Rb) & m).sum(dim=(1, 2)).to(dtype=torch.float64)
    return (inter / union.clamp(min=1)).detach().cpu().numpy()


def _beta_sequential(S, beta: float):
    """Coordinate-by-coordinate β-divergence. ``S`` is ``[B, n, m]`` and strictly positive.

    β = 1 uses ``numpy.log`` (torch.log is not bit-identical) and the sum stays on device.
    """
    torch = _torch()
    if not torch.isfinite(S).all() or (S <= 0).any():
        raise ValueError("β-divergence states must be finite and strictly positive")
    if beta == 1:
        tr = torch.as_tensor(np.log(S.detach().cpu().numpy()), device=S.device, dtype=S.dtype)
    else:
        tr = S ** (beta - 1) / (beta - 1)
    B, n, m = S.shape
    acc = torch.zeros(B, n, n, dtype=S.dtype, device=S.device)
    for c in range(m):
        ds = S[:, :, c].unsqueeze(2) - S[:, :, c].unsqueeze(1)
        dt = tr[:, :, c].unsqueeze(2) - tr[:, :, c].unsqueeze(1)
        acc = acc + 0.5 * ds * dt
    if not torch.isfinite(acc).all():
        raise ValueError("β-divergence overflow; check the input scale")
    return acc


def _euclid_sequential(S):
    """CUDA Euclidean distances: ``acc = acc + diff * diff`` per coordinate, then sqrt."""
    torch = _torch()
    B, n, m = S.shape
    acc = torch.zeros(B, n, n, dtype=S.dtype, device=S.device)
    for c in range(m):
        diff = S[:, :, c].unsqueeze(2) - S[:, :, c].unsqueeze(1)
        acc = acc + diff * diff
    return torch.sqrt(acc)


def _ordinal_match_matrix(states):
    """Pairwise ordinal pattern equality (matches ``geometry._ordinal_matrix``)."""
    torch = _torch()
    if states.ndim == 2:
        states = states.unsqueeze(0)
    p = torch.argsort(states, dim=-1, stable=True)
    same = (p.unsqueeze(2) == p.unsqueeze(1)).all(dim=-1)
    return same.float()


def _beta_pairs_rows(a, b, beta: float):
    """Lag-k β rows. CPU uses numpy; CUDA accumulates one coordinate at a time."""
    torch = _torch()
    if a.device.type != "cuda":
        aa, bb = a.detach().cpu().numpy(), b.detach().cpu().numpy()
        if beta == 1:
            ta, tb = np.log(aa), np.log(bb)
        else:
            ta = aa ** (beta - 1) / (beta - 1)
            tb = bb ** (beta - 1) / (beta - 1)
        out = 0.5 * ((aa - bb) * (ta - tb)).sum(axis=-1)
        return torch.as_tensor(out, device=a.device, dtype=a.dtype)
    if beta == 1:
        ta = torch.as_tensor(np.log(a.detach().cpu().numpy()), device=a.device, dtype=a.dtype)
        tb = torch.as_tensor(np.log(b.detach().cpu().numpy()), device=b.device, dtype=b.dtype)
    else:
        ta = a ** (beta - 1) / (beta - 1)
        tb = b ** (beta - 1) / (beta - 1)
    acc = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
    for c in range(a.shape[-1]):
        acc = acc + 0.5 * (a[:, c] - b[:, c]) * (ta[:, c] - tb[:, c])
    return acc


def _band_matrix(states, n: int, *, dist: str = "euclid", beta: float = 2.0):
    torch = _torch()
    N = len(states)
    out = torch.zeros((N, n), dtype=states.dtype, device=states.device)
    if dist == "euclid":
        out[:, 0] = 0.0
        for k in range(1, n):
            if k >= N:
                break
            out[: N - k, k] = torch.linalg.vector_norm(states[: N - k] - states[k:N], dim=-1)
    elif dist == "beta":
        out[:, 0] = 0.0
        for k in range(1, n):
            if k >= N:
                break
            out[: N - k, k] = _beta_pairs_rows(states[: N - k], states[k:N], beta)
    elif dist == "ordinal":
        out[:, 0] = 0.0
        p = torch.argsort(states, dim=-1, stable=True)
        for k in range(1, n):
            if k >= N:
                break
            out[: N - k, k] = (p[: N - k] == p[k:N]).all(dim=-1).float()
    else:
        raise ValueError(dist)
    return out


def _gather_window_d(D_band, g0: np.ndarray, n: int):
    torch = _torch()
    dev = D_band.device
    N = D_band.shape[0]
    B = len(g0)
    g0_t = torch.as_tensor(g0, device=dev, dtype=torch.long)
    ii = torch.arange(n, device=dev)
    jj = torch.arange(n, device=dev)
    mini = torch.minimum(ii[:, None], jj[None, :])
    off = torch.abs(jj[None, :] - ii[:, None])
    rows = g0_t[:, None, None] + mini[None, :, :]
    cols = off[None, :, :].expand(B, n, n)
    idx_ok = (rows + cols) < N
    rflat = rows.clamp(0, N - 1).reshape(-1)
    cflat = cols.clamp(0, n - 1).reshape(-1)
    vals = D_band[rflat, cflat].reshape(B, n, n)
    vals = torch.where(idx_ok, vals, torch.full_like(vals, float("inf")))
    return vals


def _scale_series_for_band(x: np.ndarray, geom: Geometry) -> np.ndarray:
    x = np.asarray(x, float)
    if geom.scaling == "gz":
        if geom.train_mean is None or geom.train_sd is None:
            raise ValueError("gz scaling needs train_mean and train_sd")
        sd = geom.train_sd if geom.train_sd > 0 else 1.0
        return (x - geom.train_mean) / sd
    if geom.distance in ("beta_pos", "power_ctrl"):
        return np.asarray(x, float)
    return x


def banded_ok(geom: Geometry) -> bool:
    return geom.banded_ok()


def _direct_distances(
    windows: np.ndarray, geom: Geometry, positive: np.ndarray | None, dtype, *, dev=None
):
    torch = _torch()
    dev = dev if dev is not None else device()
    W = windows.shape[-1]
    B = len(windows)
    m, tau = geom.m, geom.tau
    n = W - (m - 1) * tau
    scaled = []
    pos_batch = []
    for i in range(B):
        w = windows[i]
        scaled.append(geom.scale_window(w))
        if positive is not None:
            pos_batch.append(positive[i])
        elif geom.distance in ("beta_pos", "power_ctrl"):
            pos_batch.append(w)
    S = torch.as_tensor(
        np.stack([embed(scaled[i], m, tau) for i in range(B)]), device=dev, dtype=dtype
    )
    d = geom.distance
    on_cuda = dev.type == "cuda"
    if d == "euclid":
        if on_cuda:
            D = _euclid_sequential(S)
        else:
            from recplay.recurrence import euclidean_distances

            D = torch.as_tensor(
                np.stack([euclidean_distances(S[i].detach().cpu().numpy()) for i in range(B)]),
                device=dev,
                dtype=dtype,
            )
    elif d == "beta_exp":
        exp_np = np.exp(S.detach().cpu().numpy())
        if on_cuda:
            D = _beta_sequential(torch.as_tensor(exp_np, device=dev, dtype=dtype), geom.beta)
        else:
            from recplay.recurrence import symmetric_beta_matrix

            D = torch.as_tensor(
                np.stack([symmetric_beta_matrix(exp_np[i], geom.beta) for i in range(B)]),
                device=dev,
                dtype=dtype,
            )
    elif d == "beta_pos":
        pos_np = np.stack([np.asarray(p, float) for p in pos_batch])
        if geom.train_median is not None and geom.train_median > 0:
            pos_np = pos_np / float(geom.train_median)
        Sp = np.stack([embed(pos_np[i], m, tau) for i in range(B)])
        if on_cuda:
            D = _beta_sequential(torch.as_tensor(Sp, device=dev, dtype=dtype), geom.beta)
        else:
            from recplay.recurrence import symmetric_beta_matrix

            D = torch.as_tensor(
                np.stack([symmetric_beta_matrix(Sp[i], geom.beta) for i in range(B)]),
                device=dev,
                dtype=dtype,
            )
    elif d == "power_ctrl":
        pos_np = np.stack([np.asarray(p, float) for p in pos_batch])
        if geom.train_median is not None and geom.train_median > 0:
            pos_np = pos_np / float(geom.train_median)
        tr_np = np.log(pos_np) if geom.beta == 0 else pos_np ** (geom.beta / 2.0)
        St = np.stack([embed(tr_np[i], m, tau) for i in range(B)])
        if on_cuda:
            D = _euclid_sequential(torch.as_tensor(St, device=dev, dtype=dtype))
        else:
            from recplay.recurrence import euclidean_distances

            D = torch.as_tensor(
                np.stack([euclidean_distances(St[i]) for i in range(B)]),
                device=dev,
                dtype=dtype,
            )
    elif d == "ordinal":
        D = _ordinal_match_matrix(S)
    else:
        raise ValueError(d)
    return D, n


def _banded_series_states(
    x: np.ndarray, W: int, geom: Geometry, dtype, *, dev=None, positive_raw=None
):
    """Embed full series once and build the distance band (reuse across window batches)."""
    torch = _torch()
    dev = dev if dev is not None else device()
    m, tau = geom.m, geom.tau
    n = W - (m - 1) * tau
    xs = _scale_series_for_band(x, geom)
    if geom.distance in ("beta_pos", "power_ctrl"):
        pos = np.asarray(positive_raw if positive_raw is not None else x, float)
        if geom.train_median is not None and geom.train_median > 0:
            pos = pos / geom.train_median
        if geom.distance == "power_ctrl":
            xs = np.log(pos) if geom.beta == 0 else pos ** (geom.beta / 2.0)
        else:
            xs = pos
    states = torch.as_tensor(embed(xs, m, tau), device=dev, dtype=dtype)
    if geom.distance == "ordinal":
        bd, bbeta = "ordinal", geom.beta
    elif geom.distance in ("beta_pos",):
        bd, bbeta = "beta", geom.beta
    else:
        bd, bbeta = "euclid", geom.beta
    D_band = _band_matrix(states, n, dist=bd, beta=bbeta)
    return D_band, n, m, tau


def _banded_distances_from_band(D_band, ends: np.ndarray, W: int, m: int, tau: int, n: int):
    g0 = np.array([int(e) - W + 1 for e in ends], dtype=np.int64)
    return _gather_window_d(D_band, g0, n), n


def window_rps(
    x,
    ends,
    W: int,
    geom: Geometry,
    *,
    positive_raw=None,
    torch_device=None,
    dtype=None,
    force_path: str | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, None]]:
    """Yield (ends_batch, R_batch, None) thresholded RPs identical to ``geom.recurrence``."""
    torch = _torch()
    dev = torch.device(torch_device) if torch_device is not None else device()
    if dtype is None:
        dtype = torch.float64
    x = np.asarray(x, float)
    ends = np.asarray(ends, int)
    m, tau = geom.m, geom.tau
    min_e = W - 1
    max_e = len(x) - 1
    ends = ends[(ends >= min_e) & (ends <= max_e)]
    if len(ends) == 0:
        return
    n = W - (m - 1) * tau
    if n < 1:
        raise ValueError("window shorter than one embedded state")
    budget = batch_budget(n, dtype, torch_device=torch_device or str(dev))
    pos = np.asarray(positive_raw, float) if positive_raw is not None else None
    use_banded = force_path != "direct" and banded_ok(geom)
    if force_path == "banded" and not banded_ok(geom):
        raise ValueError(f"banded path invalid for arm {geom}")

    th = geom.theiler_band()
    band_cache = None
    band_n = n
    band_m, band_tau = m, tau
    if use_banded:
        band_dev = dev
        D_band, band_n, band_m, band_tau = _banded_series_states(
            x, W, geom, dtype, dev=band_dev, positive_raw=pos if pos is not None else None
        )
        band_cache = D_band.to(dev) if band_dev.type != dev.type else D_band
    for i in range(0, len(ends), budget):
        eb = ends[i : i + budget]
        if use_banded:
            D, _ = _banded_distances_from_band(band_cache, eb, W, band_m, band_tau, band_n)
        else:
            wins = np.stack([x[max(0, int(e) - W + 1) : int(e) + 1] for e in eb])
            pbatch = None
            if pos is not None:
                pbatch = np.stack([pos[max(0, int(e) - W + 1) : int(e) + 1] for e in eb])
            D, _ = _direct_distances(wins, geom, pbatch, dtype, dev=dev)
        if geom.distance == "ordinal":
            R = D.bool()
            th_mask = (
                torch.abs(torch.arange(n, device=dev)[:, None] - torch.arange(n, device=dev)[None, :])
                < th
            )
            R = R & ~th_mask
        elif geom.threshold == "rr":
            R = threshold_rr(D, geom.rr, th)
        elif geom.threshold == "eps_train":
            if geom.eps_train is None:
                raise ValueError("eps_train not set")
            R = threshold_eps(D, float(geom.eps_train), th)
        else:
            raise ValueError(geom.threshold)
        R_np = R.detach().cpu().numpy()
        yield eb, R_np, None


def pooled_quantile(windows, geom: Geometry, rr: float, *, torch_device=None) -> float:
    """Training-pooled ε quantile (same rule as ``Geometry.fit_eps_train``)."""
    torch = _torch()
    dev = torch.device(torch_device) if torch_device is not None else device()
    dt = torch.float64
    chunks = []
    for w in windows:
        w = np.asarray(w, float)
        pos_w = w if geom.distance in ("beta_pos", "power_ctrl") else None
        wins = w[None, :]
        D, n = _direct_distances(
            wins, geom, pos_w[None, :] if pos_w is not None else None, dt, dev=dev
        )
        th = geom.theiler_band()
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=th)
        chunks.append(D[0][mask].reshape(-1))
    pool = torch.cat(chunks)
    eps = _lerp_eps_from_upper(pool.reshape(1, -1), rr)
    return float(eps[0].detach().cpu())


def geometry_arm_key(geom: Geometry, W: int) -> str:
    return (
        f"{geom.scaling}|{geom.distance}|b={geom.beta}|{geom.threshold}"
        f"|m={geom.m}|t={geom.tau}|W={int(W)}"
    )


def _fallback_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed" / "condpred" / "rp_fallback_arms.json"


def _load_fallback_file() -> set[str]:
    path = _fallback_path()
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return set(data) if isinstance(data, list) else set()


def _save_fallback_file(arms: set[str]) -> None:
    path = _fallback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(arms)), encoding="utf-8")


def register_fallback(arm_key: str, reason: str):
    _FALLBACK_ARMS.add(arm_key)
    warnings.warn(f"rp_batch: arm {arm_key!r} fell back to numpy ({reason})", stacklevel=2)


def fallback_arms() -> frozenset[str]:
    return frozenset(_FALLBACK_ARMS | _load_fallback_file())


def arm_falls_back(geom: Geometry, W: int) -> bool:
    return geometry_arm_key(geom, W) in fallback_arms()


def commit_fallback_registry(tested: set[str], failed: set[str]) -> None:
    """Drop arms this run passed; keep failures. Untested registry entries stay."""
    merged = (_load_fallback_file() - set(tested)) | set(failed)
    _save_fallback_file(merged)
    _FALLBACK_ARMS.clear()
    _FALLBACK_ARMS.update(merged)


def persist_fallbacks_into_spec(spec: dict) -> dict:
    """Record ``rp_fallback_arms`` (possibly empty) on a spec that uses the torch path."""
    out = dict(spec)
    out["rp_fallback_arms"] = sorted(fallback_arms())
    return out


def direct_rps(windows, geom: Geometry, *, positive=None, torch_device=None, dtype=None):
    """Direct-path RPs for explicit windows (circular-shift nulls are not a series band)."""
    torch = _torch()
    dev = torch.device(torch_device) if torch_device is not None else device()
    if dtype is None:
        dtype = torch.float64
    windows = np.asarray(windows, float)
    if windows.ndim == 1:
        windows = windows[None, :]
    pos = None if positive is None else np.asarray(positive, float)
    D, n = _direct_distances(windows, geom, pos, dtype, dev=dev)
    th = geom.theiler_band()
    if geom.distance == "ordinal":
        R = D.bool()
        th_mask = (
            torch.abs(torch.arange(n, device=dev)[:, None] - torch.arange(n, device=dev)[None, :]) < th
        )
        R = R & ~th_mask
    elif geom.threshold == "rr":
        R = threshold_rr(D, geom.rr, th)
    elif geom.threshold == "eps_train":
        if geom.eps_train is None:
            raise ValueError("eps_train not set")
        R = threshold_eps(D, float(geom.eps_train), th)
    else:
        raise ValueError(geom.threshold)
    return R.detach().cpu().numpy()

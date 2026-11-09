"""Joint and cross-recurrence features for pairs (E10). Row at t uses windows ending at t only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from condpred.features.local_rqa import _nan_out, local_rqa_matrix
from condpred.geometry import Geometry
from recplay.embedding import embed
from recplay.rqa import rqa


_NULL_SHIFTS = (60, 90, 125, 160, 190)


def joint_rqa(xa, xb, end: int, W: int, m: int, tau: int, geom: Geometry, pos_fn=None) -> dict:
    start = end - W + 1
    wa = np.asarray(xa[start : end + 1], float)
    wb = np.asarray(xb[start : end + 1], float)
    if len(wa) < W:
        return {k: np.nan for k in _jrqa_keys()}
    g = Geometry(**{**geom.__dict__, "m": m, "tau": tau})
    def _pos(w):
        return pos_fn(w) if pos_fn is not None else None
    _, Ra = g.recurrence(wa, positive_raw=_pos(wa))
    _, Rb = g.recurrence(wb, positive_raw=_pos(wb))
    Rj = Ra & Rb
    base = rqa(Rj)
    out = {f"joint_{k}": v for k, v in base.items()}
    null_stats = []
    for shift in _NULL_SHIFTS:
        wb_null = np.roll(wb, shift)
        _, Rbn = g.recurrence(wb_null, positive_raw=_pos(wb_null))
        null_stats.append(rqa(Ra & Rbn))
    for k in base:
        null_mean = np.mean([ns[k] for ns in null_stats if np.isfinite(ns.get(k, np.nan))])
        out[f"excess_{k}"] = base[k] - null_mean if np.isfinite(null_mean) else np.nan
    cr = cross_recurrence(xa, xb, end, W, m, tau, geom)
    out.update(cr)
    return out


def joint_local_rqa(xa, xb, end: int, W: int, m: int, tau: int, geom: Geometry) -> dict:
    start = end - W + 1
    wa = np.asarray(xa[start : end + 1], float)
    wb = np.asarray(xb[start : end + 1], float)
    if len(wa) < W:
        return {f"jloc_{k}": v for k, v in _nan_out().items()}
    g = Geometry(**{**geom.__dict__, "m": m, "tau": tau})
    _, Ra = g.recurrence(wa)
    _, Rb = g.recurrence(wb)
    loc = local_rqa_matrix(Ra & Rb, theiler=geom.theiler_band())
    return {f"jloc_{k}": v for k, v in loc.items()}


def cross_recurrence(xa, xb, end: int, W: int, m: int, tau: int, geom: Geometry) -> dict:
    """Cross-RP on scaled states; ε from off-diagonal cross pairs (not diagonal quantile)."""
    g = Geometry(**{**geom.__dict__, "m": m, "tau": tau})
    start = end - W + 1
    raw_a = np.asarray(xa[start : end + 1], float)
    raw_b = np.asarray(xb[start : end + 1], float)
    wa, wb = g.scale_window(raw_a), g.scale_window(raw_b)
    Sa, Sb = embed(wa, m, tau), embed(wb, m, tau)
    n = len(Sa)
    cross_d = np.sqrt(((Sa[:, None, :] - Sb[None, :, :]) ** 2).sum(axis=-1))
    th = g.theiler_band()
    if g.threshold == "eps_train" and g.eps_train is not None:
        eps = g.eps_train
    else:
        eps = float(np.quantile(cross_d[np.triu_indices(n, th)], g.rr))
    sync_d = np.sqrt(((Sa - Sb) ** 2).sum(axis=1))
    hit = sync_d <= eps

    def _crr(slice_hit: np.ndarray) -> float:
        return float(np.mean(slice_hit)) if len(slice_hit) else np.nan

    crr = _crr(hit)
    if len(hit) >= 44:
        chg22 = _crr(hit[-22:]) - _crr(hit[-44:-22])
    elif len(hit) >= 22:
        chg22 = _crr(hit[-22:]) - _crr(hit[:-22])
    else:
        chg22 = np.nan
    return {"CRR_diag": crr, "CRR_diag_chg22": chg22}


def _unpack_rp(packed: bytes, n: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[: n * n]
    return bits.reshape(n, n).astype(bool)


def _joint_from_packed(packed_j, packed_nulls, n: int, gkw: dict, wa, wb) -> dict:
    """CPU worker: one line-length pass per joint RP, plus the cross-recurrence scalars."""
    from condpred.features.window_rqa import _rqa_base_from_lines
    from recplay.rqa import line_lengths

    def _one(packed: bytes) -> dict:
        R = _unpack_rp(packed, n)
        diag, vert = line_lengths(R)
        return _rqa_base_from_lines(diag, vert, R)

    base = _one(packed_j)
    out = {f"joint_{k}": v for k, v in base.items()}
    null_stats = [_one(pn) for pn in packed_nulls]
    for k in base:
        null_mean = np.mean([ns[k] for ns in null_stats if np.isfinite(ns.get(k, np.nan))])
        out[f"excess_{k}"] = base[k] - null_mean if np.isfinite(null_mean) else np.nan
    g = Geometry(**gkw)
    out.update(cross_recurrence(wa, wb, len(wa) - 1, len(wa), g.m, g.tau, g))
    return out


def joint_rqa_frame(
    xa,
    xb,
    W,
    m,
    tau,
    geom: Geometry,
    step=1,
    n_jobs=-1,
    *,
    backend: str = "numpy",
    device=None,
    ends=None,
) -> pd.DataFrame:
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    if ends is not None:
        ends = np.asarray(list(ends), dtype=int)
    else:
        ends = np.arange(W - 1 + (m - 1) * tau, min(len(xa), len(xb)), step, dtype=int)
    g = Geometry(**{**geom.__dict__, "m": m, "tau": tau})
    if backend == "torch":
        from condpred import rp_batch

        if rp_batch.arm_falls_back(g, W):
            import warnings

            warnings.warn(
                f"joint_rqa_frame: {rp_batch.geometry_arm_key(g, W)} using numpy fallback",
                stacklevel=2,
            )
            backend = "numpy"
    if backend == "numpy":
        rows = Parallel(n_jobs=n_jobs)(
            delayed(joint_rqa)(xa, xb, int(t), W, m, tau, geom) for t in ends
        )
        return pd.DataFrame(rows, index=list(ends))

    from condpred import rp_batch
    from condpred.features.window_rqa import _geom_slim

    n_state = W - (m - 1) * tau
    gkw = _geom_slim(g)
    budget = rp_batch.batch_budget(n_state, __import__("torch").float64, torch_device=device)
    rows: list[dict] = []
    idx: list[int] = []
    for i in range(0, len(ends), budget):
        eb = ends[i : i + budget]
        ba = list(rp_batch.window_rps(xa, eb, W, g, torch_device=device))
        bb = list(rp_batch.window_rps(xb, eb, W, g, torch_device=device))
        for (eba, Ra, _), (_, Rb, _) in zip(ba, bb):
            B = len(eba)
            wa = np.stack([xa[int(t) - W + 1 : int(t) + 1] for t in eba])
            wb = np.stack([xb[int(t) - W + 1 : int(t) + 1] for t in eba])
            null_batch = []
            for shift in _NULL_SHIFTS:
                rolled = np.stack([np.roll(wb[j], shift) for j in range(B)])
                null_batch.append(rp_batch.direct_rps(rolled, g, torch_device=device))
            Rj = Ra & Rb
            jobs = []
            for j in range(B):
                jobs.append(
                    (
                        np.packbits(Rj[j].ravel()).tobytes(),
                        [np.packbits((Ra[j] & nR[j]).ravel()).tobytes() for nR in null_batch],
                        wa[j],
                        wb[j],
                    )
                )
            batch_rows = Parallel(n_jobs=n_jobs)(
                delayed(_joint_from_packed)(pj, pnulls, n_state, gkw, waj, wbj)
                for pj, pnulls, waj, wbj in jobs
            )
            rows.extend(batch_rows)
            idx.extend(int(e) for e in eba)
    return pd.DataFrame(rows, index=idx)


def _jrqa_keys():
    keys = []
    for prefix in ("joint_", "excess_"):
        for k in ("RR", "DET", "L", "ENTR", "LAM", "DET3_DET2", "K2_lines"):
            keys.append(f"{prefix}{k}")
    keys += ["CRR_diag", "CRR_diag_chg22"]
    return keys

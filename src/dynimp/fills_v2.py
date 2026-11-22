"""Recurrence fill v2 (D3): overlap rules, penalised distance, fallback, channel weights, from one distance pass.

One distance pass per (gap, C) computes, per candidate window and per channel, the sum of squared
differences and the number of shared observed entries. Every rule and every channel-weight vector below
is derived from those two arrays, so rules and weights add almost no cost. Import-only companion to
``dynimp.fills`` / ``dynimp.fills_ctx`` (never edited; their caches stay valid).

Rules (admission / distance); ``cnt`` = shared observed entries, ``N_q`` = entries observed in the query
window (the most a candidate can share), g = gap length, Lw = g + 2C, F = channels:

    old     cnt >= 0.3 * F * Lw                       sum / cnt     (exact D1; unweighted, no fallback)
    old_fb  as old                                    sum / cnt     + fallback
    ctxNN   cnt >= f * (F * Lw - g), f = NN/100       sum / cnt     + fallback
    penNN   cnt >= 1 (unweighted)                     (sum + lam * 2 * (N_q - cnt)) / N_q, lam = NN/10... see below

pen05 / pen10 use lam = 0.5 / 1.0. The penalised distance charges each entry observed in the query but
not in the candidate the expected squared difference of two unrelated standardised values (2), scaled
by lam; at full overlap it equals the plain distance. New rules use weighted sums and weighted counts
for channel weights w; ``old`` always uses the unweighted D1 sums. Fallback, in order: (1) same C with
rule pen10 (same weights); (2) linear interpolation (``condpred.completion.linear``). After the fallback
no gap is unfilled.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from condpred import completion
from dynimp.fills import nan_runs

CTX = {"ctx10": 0.1, "ctx20": 0.2, "ctx40": 0.4, "ctx60": 0.6}
PEN = {"pen05": 0.5, "pen10": 1.0}
RULES = ["old", "old_fb", *CTX, *PEN]
FALLBACK_RULES = [r for r in RULES if r != "old"]


def _distances(rule: str, sum_w, cnt_w, cnt_u, nq_w, F: int, Lw: int, g: int, w_target: float, w_sum: float):
    """Distance per candidate for one rule and one weight vector (inf where not admitted)."""
    if rule in ("old", "old_fb"):
        return np.where(cnt_u >= 0.3 * F * Lw, sum_w / np.maximum(cnt_u, 1), np.inf)  # sum_w is unweighted here
    if rule in CTX:
        thr = CTX[rule] * (w_sum * Lw - w_target * g)
        return np.where(cnt_w >= thr, sum_w / np.maximum(cnt_w, 1e-300), np.inf)
    lam = PEN[rule]
    return np.where(cnt_u >= 1, (sum_w + lam * 2.0 * (nq_w - cnt_w)) / max(nq_w, 1e-300), np.inf)


def recurrence_fills_v2(test, lib, c, C, rules=None, weight_vectors=None, k=8, runs=None, self_lib=False):
    """Fill the NaN runs of test[:, c] once per (rule, weight label), sharing one distance pass per gap.

    test, lib: [n, F] with NaN. weight_vectors: {label: per-channel weights}; the target channel weight
    must be 1. ``old`` ignores the weights (exact D1 behaviour, which is the w = 1 case used here).
    self_lib: lib is test; candidate windows overlapping the query window are excluded (V_loo).
    Returns {"fills": {(rule, w_label): filled copy of test[:, c]},
             "stats": {(rule, w_label): {n_gaps, cand_total, cand_rejected, rejected_share,
                                          unfilled_before_fallback, fb_pen, fb_linear}}}.
    Candidates must be fully observed in channel c over the gap segment, as in D1.
    """
    test, lib = np.asarray(test, float), np.asarray(lib, float)
    n, F = test.shape
    rules = list(RULES if rules is None else rules)
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        raise ValueError(f"unknown rules {unknown}")
    weight_vectors = {1.0: np.ones(F)} if weight_vectors is None else {
        float(wl): np.asarray(wv, float) for wl, wv in weight_vectors.items()}
    for wl, wv in weight_vectors.items():
        if wv.shape != (F,) or wv[c] != 1.0:
            raise ValueError(f"weight vector for label {wl} must have length F and target weight 1")
    base = test[:, c].copy()
    fills = {(r, wl): base.copy() for r in rules for wl in weight_vectors}
    stats = {(r, wl): {"n_gaps": 0, "cand_total": 0, "cand_rejected": 0, "unfilled_before_fallback": 0,
                       "fb_pen": 0, "fb_linear": 0} for r in rules for wl in weight_vectors}
    lin = None  # lazy: linear interpolation of the target channel, fallback step 2
    fin_c = np.concatenate([[0], np.cumsum(np.isfinite(lib[:, c]))])
    by_g: dict[int, list] = {}
    for a, b in nan_runs(test[:, c]) if runs is None else runs:
        by_g.setdefault(b - a, []).append((a, b))
    for g, group in by_g.items():
        Lw = g + 2 * C
        cand = sliding_window_view(lib, Lw, axis=0) if len(lib) >= Lw else None
        P = cand.shape[0] if cand is not None else 0
        starts = np.arange(P)
        valid = (fin_c[starts + C + g] - fin_c[starts + C]) == g if P else np.zeros(0, bool)
        # Per-channel library window matrices, built once per (g, C) and shared by every gap of this
        # length. With Q the query matrix of the group, per candidate and channel:
        #   cnt = Wfin @ Qfin.T ;  sumsq = Wsq @ Qfin.T + Wfin @ Qsq.T - 2 * W0 @ Q0.T
        # (BLAS; identical quantities to differencing every window, at a fraction of the cost.)
        qs = []
        for a, b in group:
            q = np.full((F, Lw), np.nan)
            lo, hi = max(a - C, 0), min(b + C, n)
            q[:, lo - (a - C): lo - (a - C) + hi - lo] = test[lo:hi].T
            qs.append(q)
        Gg = len(group)
        cnt_all = np.zeros((Gg, P, F))
        sum_all = np.zeros((Gg, P, F))
        if P:
            Q = np.stack(qs)  # [Gg, F, Lw]
            Qfin, Q0 = np.isfinite(Q).astype(float), np.nan_to_num(Q)
            for f in range(F):
                Wm = np.ascontiguousarray(cand[:, f, :], dtype=float)
                Wfin = np.isfinite(Wm).astype(float)
                W0 = np.where(Wfin > 0, Wm, 0.0)
                cnt_all[:, :, f] = (Wfin @ Qfin[:, f, :].T).T
                sum_all[:, :, f] = (W0 ** 2 @ Qfin[:, f, :].T + Wfin @ (Q0[:, f, :] ** 2).T
                                    - 2.0 * (W0 @ Q0[:, f, :].T)).T
            cnt_all = np.rint(cnt_all)
            np.maximum(sum_all, 0.0, out=sum_all)
        for gi, (a, b) in enumerate(group):
            q = qs[gi]
            ok = valid & (np.abs(starts - (a - C)) >= Lw) if self_lib else valid
            idx = np.flatnonzero(ok)
            q_fin = np.isfinite(q)
            dists: dict[tuple, np.ndarray] = {}
            sq_ch, cnt_ch = sum_all[gi], cnt_all[gi]
            for wl, wv in weight_vectors.items():
                cnt_u = cnt_ch.sum(axis=1)
                nq_w = float((wv[:, None] * q_fin).sum())
                cnt_w = cnt_ch @ wv
                sum_w = sq_ch @ wv
                sum_u = sq_ch.sum(axis=1)
                for r in rules:
                    key = (r, wl)
                    st = stats[key]
                    st["n_gaps"] += 1
                    if not len(idx):
                        st["unfilled_before_fallback"] += 1
                        dists[key] = np.full(P, np.inf)
                        continue
                    d = _distances(r, sum_u if r in ("old", "old_fb") else sum_w, cnt_w, cnt_u, nq_w,
                                   F, Lw, g, float(wv[c]), float(wv.sum()))
                    d = np.where(ok, d, np.inf)
                    dists[key] = d
                    st["cand_total"] += int(len(idx))
                    st["cand_rejected"] += int((~np.isfinite(d[idx])).sum())
                    if not np.isfinite(d).any():
                        st["unfilled_before_fallback"] += 1
            # fill, with fallback (pen10 at the same C, then linear) for every rule except old
            for (r, wl), out in fills.items():
                if r == "old" and wl != 1.0:
                    continue  # old is defined unweighted only; other labels share the w=1 fill below
                d = dists.get((r, wl), np.empty(0))
                src = d
                if not np.isfinite(d).any() and r != "old":
                    d_fb = dists.get(("pen10", wl)) if ("pen10", wl) in dists else None
                    if d_fb is not None and np.isfinite(d_fb).any():
                        src, fb = d_fb, "fb_pen"
                    else:
                        src, fb = None, "fb_linear"
                    stats[(r, wl)]["fb_pen" if fb == "fb_pen" else "fb_linear"] += 1
                    if src is None:
                        if lin is None:
                            lin = completion.linear(base)
                        out[a:b] = lin[a:b]
                        continue
                if not np.isfinite(src).any():
                    continue  # old with no admitted candidate: left NaN, scored as zero (D1)
                kk = min(k, int(np.isfinite(src).sum()))
                nn = np.argpartition(src, kk - 1)[:kk]
                wts = np.exp(-src[nn] / max(src[nn].min(), 1e-12))
                segs = np.stack([lib[j + C: j + C + g, c] for j in nn])
                out[a:b] = (wts[:, None] * segs).sum(0) / wts.sum()
    for (r, wl), out in fills.items():
        if r == "old" and wl != 1.0:
            fills[(r, wl)] = fills[("old", 1.0)]
            stats[(r, wl)] = dict(stats[("old", 1.0)])
    for st in stats.values():
        st["rejected_share"] = float(st["cand_rejected"] / st["cand_total"]) if st["cand_total"] else float("nan")
    return {"fills": fills, "stats": stats}

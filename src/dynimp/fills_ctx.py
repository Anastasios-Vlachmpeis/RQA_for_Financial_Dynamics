"""Recurrence fill variants: context-only overlap rule and multiple donor counts, from one distance pass per gap.

Import-only companion to ``dynimp.fills`` (which is not edited, so the D1/D2 caches stay valid).
Rule ("window", 0.3) reproduces ``fills.recurrence_fill`` with ``min_overlap=0.3``: a library window counts
only if the entries observed in both it and the query are at least ``0.3 * F * (g + 2C)``, where the denominator
includes the ``g`` target-channel entries inside the gap, which are never observed in the query.
Rule ("context", f) counts only entries that can be observed: ``cnt >= f * (F * (g + 2C) - g)``.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from dynimp.fills import nan_runs


def rule_name(rule) -> str:
    kind, frac = rule
    return f"{kind}_{frac:g}"


def recurrence_fill_multi(test, lib, c, C, rules, ks=(8,), weights=None, runs=None):
    """Fill the NaN runs of test[:, c] once per (rule, k), sharing one distance pass per gap.

    test, lib: [n, F] with NaN, as in ``fills.recurrence_fill``. rules: iterable of (kind, frac) with kind
    "window" or "context". Returns {"fills": {(rule_name, k): filled copy of test[:, c]},
    "stats": {rule_name: {"n_gaps", "unfilled", "cand_total", "cand_rejected", "rejected_share"}}}.
    Candidates are the library windows whose channel-c segment over the gap is fully observed (as in D1);
    rejected_share is the fraction of those candidates dropped by the rule's overlap threshold.
    """
    test, lib = np.asarray(test, float), np.asarray(lib, float)
    n, F = test.shape
    w_ch = np.ones(F) if weights is None else np.asarray(weights, float)
    rules = [(str(kind), float(frac)) for kind, frac in rules]
    for kind, _ in rules:
        if kind not in ("window", "context"):
            raise ValueError(f"unknown overlap rule {kind!r}")
    ks = tuple(int(k) for k in ks)
    base = test[:, c].copy()
    fills = {(rule_name(r), k): base.copy() for r in rules for k in ks}
    stats = {rule_name(r): {"n_gaps": 0, "unfilled": 0, "cand_total": 0, "cand_rejected": 0} for r in rules}
    fin_c = np.concatenate([[0], np.cumsum(np.isfinite(lib[:, c]))])
    by_g: dict[int, list] = {}
    for a, b in nan_runs(test[:, c]) if runs is None else runs:
        by_g.setdefault(b - a, []).append((a, b))
    for g, group in by_g.items():
        Lw = g + 2 * C
        if len(lib) < Lw + 1:
            for r in rules:
                stats[rule_name(r)]["n_gaps"] += len(group)
                stats[rule_name(r)]["unfilled"] += len(group)
            continue
        cand = sliding_window_view(lib, Lw, axis=0)  # [P, F, Lw]
        starts = np.arange(cand.shape[0])
        valid = (fin_c[starts + C + g] - fin_c[starts + C]) == g
        idx = np.flatnonzero(valid)
        thr = {rule_name((kind, frac)): (frac * F * Lw if kind == "window" else frac * (F * Lw - g))
               for kind, frac in rules}
        for a, b in group:
            q = np.full((F, Lw), np.nan)
            lo, hi = max(a - C, 0), min(b + C, n)
            q[:, lo - (a - C): lo - (a - C) + hi - lo] = test[lo:hi].T
            if len(idx) == 0:
                for r in rules:
                    stats[rule_name(r)]["n_gaps"] += 1
                    stats[rule_name(r)]["unfilled"] += 1
                continue
            sq = (cand[idx] - q) ** 2 * w_ch[None, :, None]
            cnt = np.isfinite(sq).sum(axis=(1, 2))
            sums = np.nansum(sq, axis=(1, 2))
            for kind, frac in rules:
                rn = rule_name((kind, frac))
                st = stats[rn]
                st["n_gaps"] += 1
                st["cand_total"] += int(len(idx))
                st["cand_rejected"] += int((cnt < thr[rn]).sum())
                dist = np.where(cnt >= thr[rn], sums / np.maximum(cnt, 1), np.inf)
                n_fin = int(np.isfinite(dist).sum())
                if n_fin == 0:
                    st["unfilled"] += 1
                    continue
                for k in ks:
                    kk = min(k, n_fin)
                    nn = np.argpartition(dist, kk - 1)[:kk]
                    w = np.exp(-dist[nn] / max(dist[nn].min(), 1e-12))
                    segs = np.stack([lib[idx[j] + C: idx[j] + C + g, c] for j in nn])
                    fills[(rn, k)][a:b] = (w[:, None] * segs).sum(0) / w.sum()
    for st in stats.values():
        st["rejected_share"] = float(st["cand_rejected"] / st["cand_total"]) if st["cand_total"] else float("nan")
    return {"fills": fills, "stats": stats}


def average_over_C(fills) -> np.ndarray:
    """Per-point mean over the C values that filled the point; a point no C filled stays NaN.

    fills: mapping C -> 1-D array, or an iterable of 1-D arrays of the same length.
    """
    arrs = list(fills.values()) if hasattr(fills, "values") else list(fills)
    if not arrs:
        raise ValueError("average_over_C needs at least one fill")
    stack = np.stack([np.asarray(a, float) for a in arrs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)

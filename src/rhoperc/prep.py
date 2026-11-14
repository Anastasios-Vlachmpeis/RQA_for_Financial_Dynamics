"""Standard preparation: transform, embedding, distances, Theiler rule, surrogates."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from rhoperc.core import rho_perc


def transform_series(x, transform="z"):
    x = np.asarray(x, dtype=float)
    if transform == "z":
        sd = x.std()
        if sd == 0:
            return np.zeros_like(x)
        return (x - x.mean()) / sd
    if transform == "rank":
        return rankdata(x, method="average") / (len(x) + 1)
    raise ValueError(f"unknown transform {transform!r}")


def theiler_rule(x, m=3, tau=1, n_states=None):
    """Article 4 Theiler rule. Returns dict(w, tau_e, too_persistent)."""
    x = np.asarray(x, dtype=float)
    N = int(n_states) if n_states is not None else len(x)
    base = (m - 1) * tau + 1
    max_l = max(1, N // 10)
    tau_e = None
    if len(x) > 3:
        xc = x - x.mean()
        denom = float(xc @ xc)
        if denom > 0:
            for L in range(1, max_l + 1):
                ac = float(xc[:-L] @ xc[L:]) / denom
                if ac < 1.0 / np.e:
                    tau_e = L
                    break
    if tau_e is None:
        return {"w": int(base), "tau_e": None, "too_persistent": True}
    return {"w": int(max(base, tau_e)), "tau_e": int(tau_e), "too_persistent": False}


def distance(x, transform="z", m=3, tau=1):
    """transform -> recplay.embedding.embed -> euclidean_distances; returns D."""
    from recplay.embedding import embed
    from recplay.recurrence import euclidean_distances
    xt = transform_series(x, transform)
    states = embed(xt, m, tau)
    return euclidean_distances(states)


def measure(x, transform="z", m=3, tau=1, theiler=None, delta=0.05, connectivity=8):
    """One call from series to rho_perc result dict (plus settings)."""
    D = distance(x, transform=transform, m=m, tau=tau)
    if theiler is None:
        rule = theiler_rule(x, m=m, tau=tau, n_states=len(D))
        w = rule["w"]
        too_persistent = rule["too_persistent"]
        tau_e = rule["tau_e"]
    else:
        w = int(theiler)
        rule = theiler_rule(x, m=m, tau=tau, n_states=len(D))
        too_persistent = rule["too_persistent"]
        tau_e = rule["tau_e"]
    out = rho_perc(D, w, delta=delta, connectivity=connectivity)
    out.update({"transform": transform, "m": int(m), "tau": int(tau),
                "tau_e": tau_e, "too_persistent": bool(too_persistent)})
    return out


def surrogate_stats(observed, surrogate_values):
    """d = observed - mean(surrogates); pct = share of surrogates <= observed."""
    vals = np.asarray(surrogate_values, dtype=float)
    if vals.size == 0:
        raise ValueError("no surrogate values")
    return {"d": float(observed - vals.mean()), "pct": float(np.mean(vals <= observed)),
            "mean_surrogate": float(vals.mean()), "n_surrogates": int(vals.size)}

"""P5 competitor set C (00_CONSTITUTION.md P5), all on the same embedded states.

Every competitor is computed from one distance matrix ``D`` of embedded
states, one Theiler window ``w`` and the (transformed) series ``x`` that was
embedded. ``rho_perc`` itself is not a competitor; its bounds ``rho_iso`` /
``rho_lag`` are, and they are taken from ``rhoperc.core.rho_perc`` (an
already-computed core result may be passed in to avoid recomputation).

Competitors (column names used in P5 ``features.csv``):
- ``DET``, ``LAM``, ``L``, ``ENTR``, ``K2_lines``: ``recplay.rqa.rqa`` of the
  recurrence plot at RR = 10 % (``recplay.recurrence.threshold_by_rate``
  with ``theiler=w``).
- ``lag1_autocorr``: lag-1 autocorrelation of ``x``,
  ``corr(x[:-1], x[1:])`` (Pearson, population formula of
  ``rhoperc.prep.theiler_rule``).
- ``tau_e``: Article 4 decorrelation lag of ``x``: smallest ``L >= 1`` with
  autocorrelation below ``1/e``, searched for ``L <= N/10`` (``N = len(D)``
  states). If no such lag exists the value is censored at ``N//10 + 1``
  (meaning "above the search limit"); see ``tau_e_censored`` flag in the
  returned dict is not a feature column, censoring is documented in the
  P5 spec.
- ``rho_iso``, ``rho_lag``: the two bounds of Article 3.4 (RR at
  ``eps_iso`` / ``eps_lag`` over Omega = {lag >= w}).
- ``rho_conn``: RR at the largest edge of the minimum spanning tree of
  ``D``. Precisely: the MST is taken over the complete graph on the N
  states with edge weights ``D`` (``scipy.sparse.csgraph.
  minimum_spanning_tree``); let ``e_conn`` be its largest edge weight (the
  threshold at which the recurrence network on the states becomes
  connected). ``rho_conn`` is then the recurrence rate in the same sense
  as ``rho_perc``: the share of Omega points (upper triangle, lag >= w)
  with ``D_ij <= e_conn``.
- ``largest_share``: share of recurrence points in the largest structure
  at RR = 10 %: the RR = 10 % recurrence plot (same plot as the RQA
  measures) is labelled with the 8-neighbourhood
  (``scipy.ndimage.label`` with a 3x3 structure); ``largest_share`` is
  (points in the largest labelled component) / (total recurrence points),
  0 if there are no recurrence points.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import minimum_spanning_tree

COMPETITORS = ["DET", "LAM", "L", "ENTR", "K2_lines", "lag1_autocorr",
               "tau_e", "rho_iso", "rho_lag", "rho_conn", "largest_share"]

RR_RQA = 0.10


def _autocorr(x, lag):
    xc = np.asarray(x, dtype=float) - np.mean(x)
    denom = float(xc @ xc)
    if denom == 0:
        return 0.0
    return float(xc[:-lag] @ xc[lag:]) / denom


def tau_e_of(x, n_states):
    """Article 4 tau_e on series x; censored at n_states//10 + 1 if none."""
    max_l = max(1, int(n_states) // 10)
    for lag in range(1, max_l + 1):
        if _autocorr(x, lag) < 1.0 / np.e:
            return float(lag), False
    return float(max_l + 1), True


def compute_competitors(D, w, x, core_result=None):
    """Return dict with every competitor in COMPETITORS for one run.

    D: (N, N) Euclidean distances between embedded states.
    w: Theiler window used for Omega, the RR=10% plot and rho_perc.
    x: the transformed 1D series that was embedded (for lag1_autocorr/tau_e).
    core_result: optional dict from rhoperc.core.rho_perc(D, w); computed
    here if not given (only its rho_iso/rho_lag bounds are used).
    """
    from recplay.recurrence import threshold_by_rate
    from recplay.rqa import rqa

    D = np.asarray(D, dtype=float)
    n = len(D)
    w = int(w)
    if core_result is None:
        from rhoperc.core import rho_perc
        core_result = rho_perc(D, w)

    R = threshold_by_rate(D, RR_RQA, theiler=w)
    q = rqa(R)

    # largest 8-neighbour structure share at RR = 10 %
    total = int(R.sum())
    if total > 0:
        lab, n_lab = ndimage.label(R, structure=np.ones((3, 3), bool))
        largest = int(np.bincount(lab.ravel(), minlength=n_lab + 1)[1:].max())
        largest_share = largest / total
    else:
        largest_share = 0.0

    # rho_conn: largest MST edge, then RR over Omega (lag >= w)
    mst = minimum_spanning_tree(D).toarray()
    e_conn = float(mst.max()) if mst.size else 0.0
    I, J = np.triu_indices(n, w)
    rho_conn = float(np.mean(D[I, J] <= e_conn))

    tau_e, _censored = tau_e_of(x, n)
    return {
        "DET": float(q["DET"]), "LAM": float(q["LAM"]), "L": float(q["L"]),
        "ENTR": float(q["ENTR"]), "K2_lines": float(q["K2_lines"]),
        "lag1_autocorr": _autocorr(x, 1), "tau_e": tau_e,
        "rho_iso": float(core_result["rho_iso"]),
        "rho_lag": float(core_result["rho_lag"]),
        "rho_conn": rho_conn, "largest_share": float(largest_share),
    }

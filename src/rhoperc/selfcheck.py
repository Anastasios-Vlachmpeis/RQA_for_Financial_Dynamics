"""Known-answer tests K1-K9 (constitution P0). python -m rhoperc.selfcheck"""
from __future__ import annotations

import sys

import numpy as np

from rhoperc.core import rho_perc, rho_perc_reference


def _dist(x):
    x = np.asarray(x, dtype=float)
    return np.abs(np.subtract.outer(x, x))


def main():
    results = {}
    rng = np.random.default_rng(20261113)

    try:
        r = np.random.default_rng(0)
        D = np.ones((100, 100)) + np.triu(r.random((100, 100)) * 1e-3, 1)
        D = np.maximum(D, D.T)
        D[0, 99] = D[99, 0] = 0.01
        from recplay.connectivity import rr_critical
        legacy = rr_critical(D, theiler=1, delta=0.05)
        cover = rho_perc(D, 1)["rho"]
        results["K1"] = bool(legacy < 0.001 and cover > 0.2)
    except Exception as e:
        print(f"K1 exception {e!r}")
        results["K1"] = False

    try:
        ok = True
        N = 300
        for p in (7, 40, 90):
            x = (np.arange(N) % p).astype(float)
            Dp = _dist(x)
            for w in (1, 5):
                got = rho_perc(Dp, w, delta=0.05)
                lags = np.arange(w, N)
                exp_rho = float((N - lags[lags % p == 0]).sum() / (N - lags).sum())
                ok &= (got["eps"] == 0.0) and abs(got["rho"] - exp_rho) <= 1e-12
        results["K2"] = bool(ok)
    except Exception as e:
        print(f"K2 exception {e!r}")
        results["K2"] = False

    k3 = k4 = k5 = k6 = True
    try:
        for t in range(200):
            n = int(rng.integers(30, 201))
            x = rng.integers(0, 10, n).astype(float) if t % 2 else rng.standard_normal(n)
            w = int(rng.choice([1, 3, 10]))
            delta = float(rng.choice([0.02, 0.05, 0.2]))
            D = _dist(x)
            f8 = rho_perc(D, w, delta=delta, connectivity=8)
            f4 = rho_perc(D, w, delta=delta, connectivity=4)
            r8 = rho_perc_reference(D, w, delta=delta, connectivity=8)
            r4 = rho_perc_reference(D, w, delta=delta, connectivity=4)
            k3 &= bool(np.isclose(f8["rho"], r8) and np.isclose(f4["rho"], r4))
            k4 &= bool(f8["eps_iso"] <= f8["eps"] <= f8["eps_lag"])
            k4 &= bool(f4["eps_iso"] <= f4["eps"])
            k5 &= bool(np.isclose(f8["rho"], rho_perc(D ** 2, w, delta=delta, connectivity=8)["rho"]))
            k5 &= bool(np.isclose(f8["rho"], rho_perc(np.log1p(D), w, delta=delta, connectivity=8)["rho"]))
            k5 &= bool(np.isclose(f8["rho"], rho_perc(_dist(x[::-1].copy()), w, delta=delta, connectivity=8)["rho"]))
            k6 &= bool(f8["rho"] <= f4["rho"] + 1e-12)
    except Exception as e:
        print(f"K3-K6 exception {e!r}")
        k3 = k4 = k5 = k6 = False
    results["K3"] = bool(k3)
    results["K4"] = bool(k4)
    results["K5"] = bool(k5)
    results["K6"] = bool(k6)

    try:
        N = 300
        step = (np.arange(N) >= N // 2).astype(float)
        results["K7"] = bool(rho_perc(_dist(step), 1)["rho"] == 1.0)
    except Exception as e:
        print(f"K7 exception {e!r}")
        results["K7"] = False

    try:
        c = rho_perc(_dist(np.zeros(50)), 1)["rho"]
        raised = False
        try:
            rho_perc(_dist(np.arange(20.0)), 11)
        except ValueError:
            raised = True
        results["K8"] = bool(c == 1.0 and raised)
    except Exception as e:
        print(f"K8 exception {e!r}")
        results["K8"] = False

    try:
        from recplay.connectivity import omega_ranks
        x = np.random.default_rng(1).standard_normal(80)
        D = _dist(x)
        w = 3
        joint = np.maximum(omega_ranks(D, w), omega_ranks(D, w))
        results["K9"] = bool(np.isclose(rho_perc(joint, w)["rho"], rho_perc(D, w)["rho"]))
    except Exception as e:
        print(f"K9 exception {e!r}")
        results["K9"] = False

    allpass = True
    for k in [f"K{i}" for i in range(1, 10)]:
        ok = results.get(k, False)
        allpass &= ok
        print(f"{k}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if allpass else "SOME FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())

"""Known-answer tests and leakage checks for condpred pipelines."""
from __future__ import annotations

import numpy as np

from condpred import systems
from condpred.completion import hankel_complete
from condpred.features import baselines, percolation
from condpred.features.local_rqa import local_rqa
from condpred.features.window_rqa import window_rqa
from condpred.forecasters import ARFixed, EWMAVariance, KNNAnalogue, LogHAR
from condpred.ftle import ftle_flow, ftle_map, logistic_jacobian, lorenz_jacobian, lorenz_rhs
from condpred.geometry import Geometry, window_z
from condpred.evaluation import compare, walk_forward
from condpred.losses import log_ratio, qlike
from condpred.stats import giacomini_white, hac_se, panel_date_bootstrap
from recplay.embedding import embed
from recplay.recurrence import euclidean_distances, threshold_by_rate


def run_all(rng=None) -> list[tuple[str, bool, str]]:
    rng = rng or np.random.default_rng(0)
    results = []
    results.append(_check_ftle_logistic())
    results.append(_check_ftle_lorenz(rng))
    results.append(_check_losses())
    results.append(_check_threshold_rr())
    results.append(_check_wz_invariance())
    results.append(_check_power_ctrl())
    results.append(_check_beta_pos())
    results.append(_check_hankel())
    results.append(_check_perc_span(rng))
    results.append(_check_loghar_call_order(rng))
    results.append(_check_hac())
    results.append(_check_panel_bootstrap(rng))
    results.append(_check_gw_null_size(rng))
    results.append(_check_compare_ci_contains_delta(rng))
    results.append(_check_walk_forward_panel_purge())
    results.append(_check_knn_k_equal_no_crash(rng))
    return results


def _check_ftle_logistic():
    x = systems.logistic(4.0, 50000, np.random.default_rng(1))
    ft = ftle_map(lambda s: logistic_jacobian(4.0, float(s)), x, h=50)
    m = np.nanmean(ft)
    ok = np.isfinite(m) and abs(m - np.log(2)) < 0.01
    return ("ftle_logistic", ok, f"mean={m:.4f}")


def _check_ftle_lorenz(rng):
    s0 = rng.normal(0, 1, 3) + np.array([1.0, 1.0, 20.0])
    expon, _ = ftle_flow(
        lorenz_rhs,
        lorenz_jacobian,
        s0,
        dt=0.01,
        sample_every=5,
        n=25_000,
        h=200,
        burn=5000,
    )
    m = float(np.nanmean(expon))
    ok = np.isfinite(m) and abs(m - 0.906) <= 0.05
    return ("ftle_lorenz", ok, f"mean={m:.4f}")


def _check_losses():
    y = np.array([1.0, 2.0, 3.0])
    ok0 = np.allclose(qlike(y, y), 0)
    lr = log_ratio(np.array([2.0]), np.array([1.0]))[0]
    ok1 = lr > 0
    return ("losses", ok0 and ok1, "")


def _check_threshold_rr():
    rng = np.random.default_rng(9)
    d = euclidean_distances(embed(rng.standard_normal(50), 3, 1))
    rr_target = 0.1
    theiler = 1
    R = threshold_by_rate(d, rr_target, theiler)
    n = len(d)
    I, J = np.triu_indices(n, theiler)
    achieved = float(R[I, J].mean())
    omega = len(I)
    ok = abs(achieved - rr_target) <= 1 / omega
    return ("threshold_rr", ok, f"achieved={achieved:.6f}")


def _check_wz_invariance():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(200)
    geom = Geometry()
    w = window_rqa(x, 150, 80, 3, 1, geom)
    x2 = 3.0 * x + 7.0
    w2 = window_rqa(x2, 150, 80, 3, 1, geom)
    keys = [k for k in w if k not in ("RR",)]
    ok = all(
        (np.isnan(w[k]) and np.isnan(w2[k]))
        or (np.isfinite(w[k]) and np.allclose(w[k], w2[k], rtol=1e-5, equal_nan=True))
        for k in keys
    )
    return ("wz_invariance", ok, "")


def _check_power_ctrl():
    pos = np.abs(np.random.default_rng(3).standard_normal(80)) + 0.5
    g_pc = Geometry(distance="power_ctrl", beta=2, threshold="rr", rr=0.1, m=3, tau=1)
    g_eu = Geometry(distance="euclid", threshold="rr", rr=0.1, m=3, tau=1)
    _, R1 = g_pc.recurrence(pos, positive_raw=pos)
    _, R2 = g_eu.recurrence(pos)
    ok = R1.shape == R2.shape and np.array_equal(R1, R2)
    return ("power_ctrl_beta2", ok, f"match={ok}")


def _check_beta_pos():
    pos = np.abs(np.random.default_rng(4).standard_normal(80)) + 0.5
    g1 = Geometry(distance="beta_pos", beta=2, m=3, tau=1, rr=0.1)
    g2 = Geometry(distance="euclid", m=3, tau=1, rr=0.1)
    _, R1 = g1.recurrence(pos, positive_raw=pos)
    z = window_z(pos)
    _, R2 = g2.recurrence(z)
    ok = np.mean(R1 == R2) > 0.9
    return ("beta_pos_beta2", ok, f"agreement={np.mean(R1==R2):.3f}")


def _check_hankel():
    t = np.linspace(0, 20, 200)
    signal = np.sin(t) + np.sin(2.5 * t)
    rng = np.random.default_rng(5)
    missing = rng.random(len(signal)) < 0.3
    x = signal.copy()
    x[missing] = np.nan
    rec = hankel_complete(x, m=10, tau=1, n_iter=400, rank=4)
    err = float(np.linalg.norm(rec[missing] - signal[missing]) / np.linalg.norm(signal[missing]))
    return ("hankel_complete", err < 1e-3, f"miss_rel_err={err:.2e}")


def _check_perc_span(rng):
    n = 40
    D = np.full((n, n), 50.0)
    np.fill_diagonal(D, 0.0)
    D[0, n - 1] = D[n - 1, 0] = 0.01
    rs = percolation.rho_perc_span(D, theiler=1, delta=0.05)
    spans = []
    for _ in range(5):
        x = rng.standard_normal(100)
        win = x[-min(80, len(x)) :]
        Dw = euclidean_distances(embed(window_z(win), 3, 1))
        spans.append(percolation.rho_perc_span(Dw, theiler=1, delta=0.05))
    spans = np.asarray(spans, float)
    span_std = float(np.std(spans))
    ok_var = span_std > 0.01 and not np.all(np.isclose(spans, 1.0, atol=1e-3))
    ok = rs > 0.2 and ok_var and np.all(np.isfinite(spans))
    return (
        "perc_span_rejects_corner",
        ok,
        f"span={rs:.4f} span_std={span_std:.4f} spans={spans.round(4).tolist()}",
    )


def _check_loghar_call_order(rng):
    """Later predict in same refit block must not change an earlier origin's forecast."""
    v = np.abs(rng.standard_normal(600)) + 0.1
    t1, t2, h = 64, 83, 5
    block = 21
    assert (t1 // block) * block == (t2 // block) * block
    har_a = LogHAR(refit_every=21, window=400, horizons=(1, 5, 22))
    har_a.fit(v)
    har_a.predict(t2, h)
    y_after_t2 = har_a.predict(t1, h)
    har_b = LogHAR(refit_every=21, window=400, horizons=(1, 5, 22))
    har_b.fit(v)
    y_fresh = har_b.predict(t1, h)
    ok = y_after_t2 == y_fresh and np.isfinite(y_fresh)
    return ("loghar_call_order", ok, f"t1={t1} t2={t2} h={h} match={ok}")


def _check_hac():
    rng = np.random.default_rng(6)
    x_iid = rng.standard_normal(500)
    se_hac = hac_se(x_iid)
    se_cl = float(np.std(x_iid, ddof=1) / np.sqrt(len(x_iid)))
    ok_iid = abs(se_hac - se_cl) / se_cl < 0.1
    x_ar = systems.ar1(0.8, 500, rng)
    se_ar = hac_se(x_ar, lags=30)
    se_ar_cl = float(np.std(x_ar, ddof=1) / np.sqrt(len(x_ar)))
    ratio = se_ar / se_ar_cl
    expect = np.sqrt((1 + 0.8) / (1 - 0.8))
    ok_ar = ratio > expect * 0.85
    return ("hac_se", ok_iid and ok_ar, f"iid_ratio={se_hac/se_cl:.3f} ar_ratio={ratio:.3f} expect={expect:.3f}")


def _check_panel_bootstrap(rng):
    import pandas as pd

    unique = [f"2020-{m:02d}-01" for m in range(1, 13)]
    assets = ["A", "B", "C"]
    frame = pd.DataFrame(
        [{"date": d, "asset": a} for d in unique for a in assets]
    )
    ok = True
    msg = ""
    for _ in range(40):
        picked = panel_date_bootstrap(unique, block=3, rng=rng)
        if len(picked) == 0 or not set(picked).issubset(set(unique)):
            ok = False
            msg = "bad dates"
            break
        sub = frame[frame["date"].isin(picked)]
        n_by_date = sub.groupby("date")["asset"].nunique()
        if not (n_by_date == len(assets)).all():
            ok = False
            msg = f"split assets on {n_by_date[n_by_date != len(assets)].index.tolist()}"
            break
    return ("panel_bootstrap", ok, msg or f"n_dates_example={len(picked)}")


def _check_gw_null_size(rng):
    """Giacomini–White null: MA(4) loss differential, h=5; ~8 % reject at 5 % is healthy."""
    h, n_rep, n_obs = 5, 200, 320
    ma_coef = np.array([1.0, 0.6, 0.35, 0.2])
    reject = 0
    for _ in range(n_rep):
        eps = rng.standard_normal(n_obs + len(ma_coef))
        y = np.zeros(n_obs)
        for t in range(len(ma_coef) - 1, n_obs):
            y[t] = float(np.dot(ma_coef, eps[t - len(ma_coef) + 1 : t + 1][::-1]))
        shock = rng.standard_normal(n_obs) * 0.05
        l1 = (y + shock) ** 2 + 1e-4
        l2 = (y - shock) ** 2 + 1e-4
        inst = rng.standard_normal(n_obs)
        _, p = giacomini_white(l1, l2, inst, h=h)
        if np.isfinite(p) and p < 0.05:
            reject += 1
    rate = reject / n_rep
    ok = 0.02 <= rate <= 0.10
    return ("gw_null_size", ok, f"reject_rate={rate:.3f} n_rep={n_rep}")


def _check_compare_ci_contains_delta(rng):
    import pandas as pd

    n = 240
    y = rng.standard_normal(n)
    M0 = y + rng.standard_normal(n) * 0.4
    M1 = y + rng.standard_normal(n) * 0.25
    y_loss = np.abs(rng.standard_normal(n)) + 0.1
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    out = compare(
        M0,
        M1,
        y,
        bootstrap={"rng": rng, "n_boot": 400, "block": 12},
        y_train_mean=float(np.mean(y[:60])),
        y_loss=y_loss,
        y_train_loss=y_loss[:60],
        dates=dates,
    )
    ok = True
    bad = []
    for name, row in out.items():
        d = row.get("delta")
        lo, hi = row.get("ci_low"), row.get("ci_high")
        if not (np.isfinite(d) and np.isfinite(lo) and np.isfinite(hi)):
            continue
        if not (lo <= d <= hi):
            ok = False
            bad.append(name)
    return ("compare_ci_delta", ok, f"bad={bad}" if bad else "all_finite_metrics_ok")


def _check_walk_forward_panel_purge():
    import pandas as pd

    uniq = pd.bdate_range("2000-01-03", periods=3200, freq="B")
    assets = [f"S{i}" for i in range(10)]
    row_dates = np.repeat(uniq, len(assets))
    first_year = 2010
    h = 22
    splits = list(walk_forward(row_dates, first_year, refit="yearly", h=h, purge=True))
    if not splits:
        return ("walk_forward_panel_purge", False, "no yearly splits")
    split = splits[0]
    train_idx, test_idx = split
    d_train = pd.Index(row_dates[train_idx]).unique().sort_values()
    d_test = pd.Index(row_dates[test_idx]).unique().sort_values()
    gap = len(pd.bdate_range(d_train.max() + pd.Timedelta(days=1), d_test.min(), freq="B"))
    ok = gap >= h
    return (
        "walk_forward_panel_purge",
        ok,
        f"unique_date_gap={gap} need>={h} n_train={len(train_idx)}",
    )


def _check_knn_k_equal_no_crash(rng):
    m, tau, k = 3, 1, 5
    x = rng.standard_normal(30)
    knn = KNNAnalogue(m, tau, k=k, horizons=(1, 5, 22))
    knn.fit(x)
    out = knn.predict(7)
    ok_knn = all(np.isfinite(v[0]) or np.isnan(v[0]) for v in out.values())
    sp = baselines.knn_spread(x, 15, m, tau, k=k, h=5, lookback=14)
    ok_sp = np.isfinite(sp) or np.isnan(sp)
    return ("knn_k_equal", ok_knn and ok_sp, f"knn={out} spread={sp}")


def _realized_sq_loss(arr: np.ndarray, h: int) -> np.ndarray:
    x = np.asarray(arr, float)
    out = np.full(len(x), np.nan)
    for s in range(len(x) - h):
        out[s] = float(np.mean(x[s + 1 : s + h + 1] ** 2))
    return out


def _ohlc_from_logret(arr: np.ndarray) -> dict:
    x = np.asarray(arr, float)
    c = np.exp(np.cumsum(x))
    o = np.empty_like(c)
    o[0] = c[0]
    o[1:] = c[:-1] * np.exp(0.3 * x[1:])
    return {"Open": o, "High": c * 1.005, "Low": c * 0.995, "Close": c}


def _frozen_equal(a, b) -> bool:
    if a == b:
        return True
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(_frozen_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return False


def leakage_test(feature_fn, x, t_list, rng, rtol=0, atol=0):
    """Return False if any feature changes when x[t+1:] is perturbed."""
    x = np.asarray(x, float).copy()
    for t in t_list:
        baseline = _freeze(feature_fn(x, t))
        x2 = x.copy()
        x2[t + 1 :] = rng.standard_normal(len(x) - t - 1)
        perturbed = _freeze(feature_fn(x2, t))
        if not _frozen_equal(baseline, perturbed):
            return False
    return True


def _freeze(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, np.ndarray):
        return tuple(np.asarray(obj, float).ravel().tolist())
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, (float, int, np.floating, np.integer)):
        return float(obj) if np.isscalar(obj) else tuple(float(v) for v in obj)
    return obj


def timing_table(rng=None) -> dict:
    import time

    rng = rng or np.random.default_rng(0)
    x = rng.standard_normal(2000)
    geom = Geometry()
    out = {}
    for W in (80, 250, 500):
        t0 = time.perf_counter()
        n = 20
        for end in range(W, W + n * 10, 10):
            window_rqa(x, end, W, 3, 1, geom)
            local_rqa(x, end, W, 3, 1, geom)
        elapsed = (time.perf_counter() - t0) / n
        out[f"W={W}"] = elapsed * 1000
    return out


def _leak_t_list(rng, n, W):
    mid = list(rng.integers(W + 10, n - 100, size=15))
    tail = list(range(n - 55, n - 5, 5))[:4]
    tail.append(max(W + 10, n - 2))
    return mid + tail


def run_leakage_suite(x, rng):
    """Leakage tests for WRQA, LRQA, baselines, geometry arms, PERC, forecasters."""
    x = np.asarray(x, float)
    W, m, tau = 80, 3, 1
    horizons = (1, 5, 22)
    t_list = _leak_t_list(rng, len(x), W)
    L_lib = min(int(min(t_list) * 0.8), len(x) - 50)
    def _add(name, fn):
        checks.append((name, leakage_test(fn, x, t_list, rng)))

    checks: list[tuple[str, bool]] = []

    def wrqa_fn(arr, t):
        g = Geometry()
        return window_rqa(arr, t, W, m, tau, g)

    def lrqa_fn(arr, t):
        g = Geometry()
        return local_rqa(arr, t, W, m, tau, g)

    _add("WRQA", wrqa_fn)
    _add("LRQA", lrqa_fn)

    def b_level(arr, t):
        return baselines.level_features(arr, t, W)

    def b_pe(arr, t):
        return baselines.perm_entropy(arr, W, t)

    def b_acf(arr, t):
        return baselines.rolling_acf1(arr, W, t)

    def b_var(arr, t):
        return baselines.rolling_var(arr, W, t)

    def b_shape(arr, t):
        return baselines.shape_features(arr, W, t)

    def b_jump(arr, t):
        return baselines.jump_features(arr, W, t, ohlc=_ohlc_from_logret(arr))

    def b_knn_spread(arr, t):
        lb = min(120, t + 1)
        return {
            h: baselines.knn_spread(arr, t, m, tau, k=5, h=h, lookback=lb) for h in horizons
        }

    def b_state(arr, t):
        return baselines.state_location(arr, t, m, tau)

    def b_recent(arr, t):
        return {
            h: baselines.recent_loss(_realized_sq_loss(arr, h), h, 5.0, t) for h in horizons
        }

    for name, fn in [
        ("B_level", b_level),
        ("B_pe", b_pe),
        ("B_acf", b_acf),
        ("B_var", b_var),
        ("B_shape", b_shape),
        ("B_jump", b_jump),
        ("B_knn_spread", b_knn_spread),
        ("B_state", b_state),
        ("B_recent", b_recent),
    ]:
        _add(name, fn)

    def geom_gz_fn(arr, t):
        g = Geometry(scaling="gz")
        train = arr[: t + 1]
        g.train_mean = float(np.mean(train))
        g.train_sd = float(np.std(train)) or 1.0
        return window_rqa(arr, t, W, m, tau, g)

    def geom_eps_fn(arr, t):
        g = Geometry(threshold="eps_train", rr=0.1, m=m, tau=tau)
        ends = range(W, t + 1, max(10, (t - W) // 5 or 1))
        wins = [arr[max(0, e - W + 1) : e + 1] for e in ends]
        if wins:
            g.fit_eps_train(wins)
        return window_rqa(arr, t, W, m, tau, g)

    def geom_beta_pos_fn(arr, t):
        p = np.abs(arr) + 0.1
        g = Geometry(distance="beta_pos", beta=2, m=m, tau=tau)
        return window_rqa(p, t, W, m, tau, g)

    def geom_power_ctrl_fn(arr, t):
        p = np.abs(arr) + 0.1
        g = Geometry(distance="power_ctrl", beta=2, m=m, tau=tau)
        return window_rqa(p, t, W, m, tau, g)

    def geom_ordinal_fn(arr, t):
        g = Geometry(distance="ordinal", m=m, tau=tau)
        return window_rqa(arr, t, W, m, tau, g)

    for name, fn in [
        ("GEOM_gz", geom_gz_fn),
        ("GEOM_eps_train", geom_eps_fn),
        ("GEOM_beta_pos", geom_beta_pos_fn),
        ("GEOM_power_ctrl", geom_power_ctrl_fn),
        ("GEOM_ordinal", geom_ordinal_fn),
    ]:
        _add(name, fn)

    def perc_fn(arr, t):
        win = arr[max(0, t - W + 1) : t + 1]
        D = euclidean_distances(embed(window_z(win), m, tau))
        return {"span": percolation.rho_perc_span(D, 1, 0.05)}

    _add("PERC", perc_fn)

    def fc_knn(arr, t):
        knn = KNNAnalogue(m, tau, k=5, horizons=horizons)
        knn.fit(arr)
        return knn.predict(t)

    def fc_knn_series(arr, t):
        knn = KNNAnalogue(m, tau, k=5, horizons=horizons)
        knn.fit(arr)
        return knn.predict(t, series=arr)

    def fc_knn_lib_prefix(arr, t):
        lib_n = min(300, len(arr) - 50)
        knn = KNNAnalogue(m, tau, k=5, horizons=horizons)
        knn.fit(arr[:lib_n])
        return knn.predict(t, series=arr)

    def fc_ar(arr, t):
        model = ARFixed(2)
        model.fit(arr[:L_lib])
        return {h: model.predict(t, h, series=arr) for h in horizons}

    def fc_ew(arr, t):
        ew = EWMAVariance()
        ew.fit(np.abs(arr) + 0.1)
        return {h: ew.predict(t, h) for h in horizons}

    def fc_har(arr, t):
        har = LogHAR(horizons=horizons)
        har.fit(np.abs(arr) + 0.1)
        return {h: har.predict(t, h) for h in horizons}

    for name, fn in [
        ("KNN", fc_knn),
        ("KNN_series", fc_knn_series),
        ("KNN_lib_prefix", fc_knn_lib_prefix),
        ("AR", fc_ar),
        ("EWMA", fc_ew),
        ("LogHAR", fc_har),
    ]:
        _add(name, fn)

    return checks


def _parity_arm_configs(smoke: bool) -> list[Geometry]:
    scalings = ["wz", "gz", "wmad"] if not smoke else ["wz"]
    distances = [
        ("euclid", {}),
    ]
    if not smoke:
        distances += [
            ("beta_exp", {"beta": 0.0}),
            ("beta_exp", {"beta": 1.0}),
            ("beta_exp", {"beta": 3.0}),
            ("beta_pos", {"beta": 0.0}),
            ("beta_pos", {"beta": 1.0}),
            ("beta_pos", {"beta": 3.0}),
            ("power_ctrl", {"beta": 2.0}),
            ("ordinal", {}),
        ]
    thresholds = ["rr"] if smoke else ["rr", "eps_train"]
    mt = [(1, 1), (3, 1)] if smoke else [(1, 1), (3, 1), (3, 2)]
    Ws = [80] if smoke else [80, 250]
    geoms = []
    for sc in scalings + ([] if smoke else ["rank"]):
        for dist, kw in distances:
            for th in thresholds:
                if th == "eps_train" and sc in ("wz", "wmad"):
                    continue
                if dist == "beta_exp" and sc in ("wz", "wmad"):
                    continue
                if sc == "rank" and dist != "euclid":
                    continue
                for m, tau in mt:
                    for W in Ws:
                        g = Geometry(
                            scaling=sc,
                            distance=dist,
                            threshold=th,
                            rr=0.10,
                            m=m,
                            tau=tau,
                            **kw,
                        )
                        geoms.append((g, W))
    return geoms


def _series_bank(rng, n: int) -> dict[str, np.ndarray]:
    from condpred import systems

    return {
        "lorenz": systems.lorenz63(n, rng)[1],
        "ar1": systems.ar1(0.9, n, rng),
        "garch": systems.garch11(0.95, n, rng),
    }


def _fit_geom_train(g: Geometry, x: np.ndarray, W: int, m: int, tau: int):
    g = Geometry(**{**g.__dict__})
    g.m, g.tau = m, tau
    if g.scaling == "gz":
        g.train_mean = float(np.mean(x))
        g.train_sd = float(np.std(x)) or 1.0
    if g.distance in ("beta_pos", "power_ctrl"):
        pos = np.abs(x) + 0.1
        g.train_median = float(np.median(pos))
    if g.threshold == "eps_train":
        ends = range(W, min(len(x), W + 400), max(5, W // 2))
        wins = [x[e - W + 1 : e + 1] for e in ends]
        g.fit_eps_train(wins)
    return g


def _parity_compare_helpers(ref_R: dict[int, np.ndarray], Rb: np.ndarray, eb, geom, dev_name: str) -> int:
    from condpred import rp_batch

    mism = 0
    for j, end in enumerate(eb):
        if not np.array_equal(ref_R[int(end)], Rb[j]):
            mism += 1
    if mism == 0 and len(eb):
        n = Rb.shape[-1]
        th = geom.theiler_band()
        Rn = ref_R[int(eb[0])]
        R2 = np.roll(Rn, 3, axis=0)
        import torch

        dev = torch.device(dev_name if dev_name == "cpu" or torch.cuda.is_available() else "cpu")
        ta = torch.as_tensor(Rn, device=dev)
        tb = torch.as_tensor(R2, device=dev)
        m = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) >= th
        jac_n = np.sum((Rn & R2) & m) / max(np.sum((Rn | R2) & m), 1)
        jac_t = float(rp_batch.jaccard(ta[None], tb[None], th)[0])
        if jac_t != jac_n:
            mism += 1
        rr_t = float(rp_batch.recurrence_rate(ta[None], th)[0])
        rr_n = float(Rn[m].mean())
        if abs(rr_t - rr_n) > 1e-15:
            mism += 1
        tj = torch.as_tensor(rp_batch.joint(ta[None], tb[None]).detach().cpu().numpy())
        if not np.array_equal(Rn & R2, tj[0]):
            mism += 1
    return mism

def _joint_frame_parity(rng, device_list) -> list[dict]:
    """Exact numpy-vs-torch equality of joint_rqa_frame (incl. excess nulls)."""
    import pandas as pd
    from condpred.features.joint_rqa import joint_rqa_frame
    rows = []
    W, m, tau = 80, 3, 1
    geom = Geometry(m=m, tau=tau)
    xa = rng.standard_normal(420)
    xb = 0.6 * xa + 0.8 * rng.standard_normal(420)
    ref = joint_rqa_frame(xa, xb, W, m, tau, geom, step=17, n_jobs=1, backend="numpy")
    for dev_name in device_list:
        try:
            import torch
            if dev_name == "cuda" and not torch.cuda.is_available():
                continue
        except ImportError:
            continue
        got = joint_rqa_frame(xa, xb, W, m, tau, geom, step=17, n_jobs=1, backend="torch", device=dev_name)
        mism = 0
        if list(ref.index) != list(got.index) or list(ref.columns) != list(got.columns):
            mism = max(len(ref), 1)
        else:
            for col in ref.columns:
                a, b = ref[col].to_numpy(float), got[col].to_numpy(float)
                both_nan = np.isnan(a) & np.isnan(b)
                mism += int(np.sum(~both_nan & (a != b)))
        rows.append({"arm": "joint_frame|wz|euclid|rr|m=3|t=1|W=80", "path": "joint_frame", "device": dev_name, "windows": len(ref), "mismatches": int(mism)})
    return rows


def rp_stage2_parity(rng=None, *, smoke: bool = False, device_list: list[str] | None = None) -> pd.DataFrame:
    """Histograms equal ``line_lengths``; scalars within 1e-12 of ``_rqa_extras``."""
    import pandas as pd

    from condpred.features.window_rqa import _rqa_extras
    from condpred.rp_rqa_stage2 import line_hist_torch, rqa_from_histograms
    from recplay.rqa import line_lengths

    rng = rng or np.random.default_rng(2026)
    device_list = device_list or ["cpu", "cuda"]
    n_each = 4 if smoke else 20
    x = rng.standard_normal(900)
    cases: list[tuple[np.ndarray, Geometry]] = []
    for geom, W in (
        (Geometry(m=3, tau=1), 80),
        (Geometry(distance="ordinal", m=3, tau=1), 80),
        (Geometry(m=1, tau=1), 40),
    ):
        for end in rng.integers(W - 1, len(x) - 1, size=n_each):
            win = x[int(end) - W + 1 : int(end) + 1]
            _, R = geom.recurrence(win)
            cases.append((np.asarray(R, bool), geom))
    for n in ((9, 17) if smoke else (9, 17, 40)):
        R = rng.random((n, n)) < 0.3
        np.fill_diagonal(R, False)
        cases.append((R, Geometry()))
    rows = []
    try:
        import torch
    except ImportError:
        return pd.DataFrame([{"device": "none", "windows": 0, "mismatches": -1}])
    for dev_name in device_list:
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        mism = 0
        for R, geom in cases:
            n = len(R)
            dg, vt = line_lengths(R)
            hd = np.bincount(dg.astype(int), minlength=n + 1)[: n + 1]
            hv = np.bincount(vt.astype(int), minlength=n + 1)[: n + 1]
            dh, vh = line_hist_torch(R, device=dev_name)
            if not (np.array_equal(dh[0], hd) and np.array_equal(vh[0], hv)):
                mism += 1
            ref = _rqa_extras(R, geom)
            sc = rqa_from_histograms(dh[0], vh[0], n, geom, rr_from_R=float(R.mean()))
            for k in set(ref) | set(sc):
                a, b = ref.get(k, np.nan), sc.get(k, np.nan)
                if np.isnan(a) and np.isnan(b):
                    continue
                if not (np.isfinite(a) and np.isfinite(b) and abs(float(a) - float(b)) <= 1e-12):
                    mism += 1
        rows.append({"device": dev_name, "windows": len(cases), "mismatches": mism})
    return pd.DataFrame(rows)


def rp_parity(
    rng=None,
    *,
    smoke: bool = False,
    device_list: list[str] | None = None,
    geom_indices: list[int] | None = None,
) -> pd.DataFrame:
    import pandas as pd

    from condpred import rp_batch

    rng = rng or np.random.default_rng(2026)
    n_win = 30 if smoke else 200
    device_list = device_list or ["cpu", "cuda"]
    rows = []
    try:
        import torch
    except ImportError:
        return pd.DataFrame([{"arm": "n/a", "path": "n/a", "device": "none", "windows": 0, "mismatches": -1}])

    configs = _parity_arm_configs(smoke)
    if geom_indices is not None:
        configs = [configs[i] for i in geom_indices if 0 <= i < len(configs)]
    required = [(Geometry(scaling="wz", distance="beta_exp", beta=2.0, rr=0.1, m=3, tau=1), 80)]
    if smoke:
        required += [
            (Geometry(scaling="gz", distance="beta_pos", beta=2.0, rr=0.1, m=3, tau=1), 80),
            (Geometry(scaling="gz", distance="power_ctrl", beta=2.0, rr=0.1, m=3, tau=1), 80),
            (Geometry(scaling="gz", distance="euclid", threshold="eps_train", rr=0.1, m=3, tau=1), 80),
        ]
    configs = list(configs) + required
    tested_keys: set[str] = set()
    failed_keys: set[str] = set()

    for g0, W in configs:
        m, tau = g0.m, g0.tau
        for name, x in _series_bank(rng, max(1200, W + n_win * 5)).items():
            pos_arm = g0.distance in ("beta_pos", "power_ctrl")
            if pos_arm or name == "garch":
                x_use = np.abs(x) + 0.1 if name != "garch" else np.abs(x) + 1e-3
                pos_raw = (x_use * 1.07 + 0.03) if pos_arm else x_use
            else:
                x_use, pos_raw = x, None
            g = _fit_geom_train(g0, x_use, W, m, tau)
            key = rp_batch.geometry_arm_key(g, W)
            tested_keys.add(key)
            min_end = W - 1 + (m - 1) * tau
            ends = rng.integers(min_end, len(x_use) - 1, size=n_win)
            arm = f"{g.scaling}|{g.distance}|b={g.beta}|{g.threshold}|m={m}|t={tau}|W={W}|{name}"
            ref_R: dict[int, np.ndarray] = {}
            for end in ends:
                end_i = int(end)
                win = x_use[end_i - W + 1 : end_i + 1]
                pos_ref = (pos_raw[end_i - W + 1 : end_i + 1] if pos_raw is not None else None)
                if pos_arm and pos_ref is None:
                    pos_ref = win
                _, ref_R[end_i] = g.recurrence(win, positive_raw=pos_ref if pos_arm else None)
            for dev_name in device_list:
                if dev_name == "cuda" and not torch.cuda.is_available():
                    continue
                torch_dev = dev_name
                if g.threshold == "eps_train" and g.eps_train is not None:
                    wins = [x_use[int(e) - W + 1 : int(e) + 1] for e in ends[: min(20, len(ends))]]
                    pq = rp_batch.pooled_quantile(wins, g, g.rr, torch_device=torch_dev)
                    g_ref = Geometry(**{**g.__dict__})
                    g_ref.fit_eps_train(wins)
                    rel = abs(pq - g_ref.eps_train) / max(abs(float(g_ref.eps_train)), 1e-12)
                    rows.append(
                        {
                            "arm": arm,
                            "path": "pooled_quantile",
                            "device": dev_name,
                            "windows": len(wins),
                            "mismatches": int(rel > 1e-12),
                        }
                    )
                    if rel > 1e-12:
                        failed_keys.add(key)
                        rp_batch.register_fallback(key, "pooled_quantile")
                if g.banded_ok():
                    path_tests = [None, "direct"]
                else:
                    path_tests = ["direct"]
                    try:
                        list(
                            rp_batch.window_rps(
                                x_use, ends[:2], W, g, positive_raw=pos_raw, force_path="banded", torch_device=torch_dev
                            )
                        )
                        rows.append(
                            {
                                "arm": arm + "|force_banded",
                                "path": "banded_invalid",
                                "device": dev_name,
                                "windows": 0,
                                "mismatches": -1,
                            }
                        )
                        continue
                    except ValueError:
                        rows.append(
                            {
                                "arm": arm + "|force_banded",
                                "path": "banded_invalid",
                                "device": dev_name,
                                "windows": 0,
                                "mismatches": 0,
                            }
                        )
                for path in path_tests:
                    mism = 0
                    path_used = path or "banded"
                    kwargs = {"force_path": path} if path else {}
                    try:
                        batches = list(
                            rp_batch.window_rps(
                                x_use,
                                ends,
                                W,
                                g,
                                positive_raw=pos_raw,
                                torch_device=torch_dev,
                                **kwargs,
                            )
                        )
                    except Exception as exc:
                        rp_batch.register_fallback(key, str(exc))
                        failed_keys.add(key)
                        mism = n_win
                        rows.append(
                            {"arm": arm, "path": path_used, "device": dev_name, "windows": n_win, "mismatches": mism}
                        )
                        continue
                    for eb, Rb, _ in batches:
                        mism += _parity_compare_helpers(ref_R, Rb, eb, g, dev_name)
                    if mism:
                        failed_keys.add(key)
                        rp_batch.register_fallback(key, f"{mism} mismatches on {dev_name} path={path_used}")
                    rows.append(
                        {
                            "arm": arm,
                            "path": path_used,
                            "device": dev_name,
                            "windows": n_win,
                            "mismatches": mism,
                        }
                    )
    rows.extend(_joint_frame_parity(rng, device_list))
    rp_batch.commit_fallback_registry(tested_keys, failed_keys)
    return pd.DataFrame(rows)


def timing_backend(rng=None, *, smoke: bool = False) -> pd.DataFrame:
    """Per-window milliseconds. (c) and (d) start from a precomputed RP; torch times packed R after warmup."""
    import time

    import pandas as pd

    from condpred import rp_batch
    from condpred.features.local_rqa import local_rqa_matrix
    from recplay.rqa import rqa

    rng = rng or np.random.default_rng(7)
    n_win = 40 if smoke else 200
    Ws = [80, 250] if smoke else [80, 250, 500]
    ms = [1, 3]
    x = rng.standard_normal(max(Ws) + n_win + 50)
    rows = []
    try:
        import torch
    except ImportError:
        torch = None
    for W in Ws:
        for m in ms:
            tau = 1
            geom = Geometry(m=m, tau=tau)
            start = W - 1 + (m - 1) * tau
            ends = np.arange(start, start + n_win, dtype=int)
            Rs = []
            t0 = time.perf_counter()
            for end in ends:
                _, R = geom.recurrence(x[int(end) - W + 1 : int(end) + 1])
                Rs.append(R)
            numpy_rp = (time.perf_counter() - t0) / len(ends) * 1000
            t0 = time.perf_counter()
            for R in Rs:
                rqa(R)
            cpu_rqa = (time.perf_counter() - t0) / len(ends) * 1000
            th = geom.theiler_band()
            t0 = time.perf_counter()
            for R in Rs:
                local_rqa_matrix(R, theiler=th)
            local_ms = (time.perf_counter() - t0) / len(ends) * 1000
            row = {
                "W": W,
                "m": m,
                "numpy_rp_ms": numpy_rp,
                "torch_cpu_rp_ms": np.nan,
                "torch_cuda_rp_ms": np.nan,
                "cpu_rqa_ms": cpu_rqa,
                "local_rqa_ms": local_ms,
            }
            if torch is None:
                rows.append(row)
                continue
            for dev in ("cpu", "cuda"):
                if dev == "cuda" and not torch.cuda.is_available():
                    continue
                list(rp_batch.window_rps(x, ends[:8], W, geom, torch_device=dev))
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for eb, Rb, _ in rp_batch.window_rps(x, ends, W, geom, torch_device=dev):
                    for j in range(len(eb)):
                        np.packbits(np.asarray(Rb[j]).ravel())
                if dev == "cuda":
                    torch.cuda.synchronize()
                row[f"torch_{dev}_rp_ms"] = (time.perf_counter() - t0) / len(ends) * 1000
            rows.append(row)
    return pd.DataFrame(rows)


def run_leakage_suite_backends(x, rng):
    """Numpy leakage suite plus torch WRQA/LRQA on CPU and, when present, CUDA."""
    W, m, tau = 80, 3, 1
    t_list = _leak_t_list(rng, len(x), W)[:4]
    out = list(run_leakage_suite(x, rng))

    def _frame(kind: str, dev: str):
        def fn(arr, t):
            g = Geometry()
            ends = np.array([t])
            if kind == "wrqa":
                from condpred.features.window_rqa import window_rqa_frame

                df = window_rqa_frame(
                    arr, W, m, tau, g, ends=ends, backend="torch", device=dev, n_jobs=1
                )
            else:
                from condpred.features.local_rqa import local_rqa_frame

                df = local_rqa_frame(
                    arr, W, m, tau, g, ends=ends, backend="torch", device=dev, n_jobs=1
                )
            return df.iloc[0].to_dict()

        return fn

    out.append(("WRQA_torch", leakage_test(_frame("wrqa", "cpu"), x, t_list, rng)))
    out.append(("LRQA_torch", leakage_test(_frame("lrqa", "cpu"), x, t_list, rng)))
    try:
        import torch

        if torch.cuda.is_available():
            out.append(("WRQA_torch_cuda", leakage_test(_frame("wrqa", "cuda"), x, t_list, rng)))
            out.append(("LRQA_torch_cuda", leakage_test(_frame("lrqa", "cuda"), x, t_list, rng)))
    except ImportError:
        pass
    return out


def save_parity_table(out_dir, *, smoke: bool = False) -> pd.DataFrame:
    """Run §6 parity grid (cpu + cuda) and write ``parity.csv`` under ``out_dir``."""
    from condpred import report

    df = rp_parity(smoke=smoke, device_list=["cpu", "cuda"])
    report.save_table(df, out_dir, "parity")
    return df


def save_timing_backend_table(out_dir, rng=None, *, smoke: bool = False) -> pd.DataFrame:
    """Run §7 backend timing and write ``timing_backend.csv``."""
    from condpred import report

    df = timing_backend(rng=rng, smoke=smoke)
    report.save_table(df, out_dir, "timing_backend")
    return df

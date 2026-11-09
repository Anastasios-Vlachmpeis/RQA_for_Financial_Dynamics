"""E6 — Vanheusden et al. DET/LAM extreme-volatility warning re-test.

Backend is numpy. Window scaling to [-1, 1] is done here; Geometry.scale_window is not used
(it would z-score after the paper's scale). joblib is capped at 4 workers.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from condpred.cache import cached
from condpred.data import yahoo_ohlc_snapshot
from condpred.features.local_rqa import local_rqa_matrix
from condpred.selfcheck import leakage_test
from condpred.stats import block_bootstrap_idx, holm
from recplay.embedding import average_mutual_information, embed, false_nearest_fraction, first_local_minimum
from recplay.recurrence import euclidean_distances, threshold_by_rate
from recplay.rqa import line_lengths, rqa

E6_MODULES = [
    "condpred.experiments.e6",
    "condpred.features.local_rqa",
    "condpred.data",
]

N_JOBS = 4
PAPER_PERIOD = ("2022-02-10", "2023-01-10")
W_RQA = 80
M_EMB, TAU = 1, 1
VOL_WINDOW = 5
H_PRIMARY = 42
MEASURES = ["TT", "DET", "LAM", "ENTR_v", "DET_LAM", "LAM_ENTR_v"]
INDICES = ["^GSPC", "^NDX", "^RUT"]
SNAPSHOT_NAME = "e6_crypto_top50_v3"
INDEX_SNAPSHOT = "e6_indices_2000"
CRYPTO_START = "2014-09-01"
INDEX_START = "2000-01-01"
COINGECKO_MARKETS = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=50&page=1&sparkline=false"
)
STABLE_CATEGORIES = [
    "stablecoins",
    "usd-stablecoin",
    "fiat-backed-stablecoin",
    "crypto-backed-stablecoin",
    "algorithmic-stablecoin",
    "yield-bearing-stablecoins",
    "bridged-stablecoins",
]
WRAPPED_CATEGORIES = [
    "wrapped-tokens",
    "liquid-staking",
    "liquid-staking-tokens",
    "liquid-staked-eth",
    "liquid-staked-btc",
    "liquid-staked-sol",
]
STABLE_SYMBOLS = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "USDD",
    "FRAX", "USD1", "USDG", "RLUSD", "GUSD", "LUSD", "USDP", "USDY", "USYC",
    "USDF", "EURC", "EURI", "UST", "USDN",
}
WRAPPED_SYMBOLS = {
    "WBTC", "WETH", "STETH", "WSTETH", "WBETH", "WEETH", "WBNB", "CBBTC", "BETH",
    "RETH", "CBETH", "METH", "MSOL", "JITOSOL", "EETH", "ETHX", "SWETH", "RSETH", "EZETH",
}
YAHOO_BY_ID = {"the-open-network": "TON-USD"}
COMPLETION_KEYS = frozenset({
    "tickers_frozen",
    "market_cap_usd",
    "market_cap_source",
    "universe_top50",
    "universe_excluded",
    "eligible_not_selected",
    "download_tickers",
    "category_errors",
})
LRQA_COLS = ["lrqa_RR_last", "lrqa_DET_last", "lrqa_K2_local", "lrqa_LAM_last"]
M0_FIELDS = ["vol5", "vol22", "vol80", "log_vol_ratio"]
SECONDARY_HORIZONS = [5, 10, 22, 42]
_YAHOO_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "yahoo"


def _pick(smoke_mode: bool, full, small):
    return small if smoke_mode else full


def build_initial_spec(smoke_mode: bool) -> dict:
    return {
        "experiment": "E6",
        "version": 4,
        "changelog": [
            "v2: primary pass rule unchanged. Arm A is not in the primary. Secondaries are Holm-corrected. "
            "[-1,1] scaling is explicit in e6.",
            "v3: primary pass rule unchanged (DET/LAM, h=42, B0, v at t+h, date-block 66, delta>0 and ci_low>0). "
            "Crypto universe is the CoinGecko top 50 by market cap at the snapshot fetch, excluding stablecoins "
            "and wrapped/staked tokens, then the 8 largest other names with >=5 years and >=1460 closes. "
            "Indices are a separate snapshot from 2000-01-01; first test year stays 2018. "
            "Placebo shifts and within-ticker-year AUCs are diagnostics and do not change the pass rule.",
            "v4: primary pass rule unchanged. Index prices are dropna()-ed before slicing/returns so "
            "weekend/holiday NaN rows on the joined crypto calendar no longer delete Monday/post-holiday "
            "index returns. category_errors from the frozen universe are copied into the spec. "
            "Diagnostic decision rows carry NaN ci_low/ci_high, one row per quantity.",
        ],
        "smoke": bool(smoke_mode),
        "backend": "numpy",
        "n_jobs": N_JOBS,
        "data_name": SNAPSHOT_NAME,
        "data_start": CRYPTO_START,
        "data_end": None,
        "index_data_name": INDEX_SNAPSHOT,
        "index_sample_start": INDEX_START,
        "data_retry_name": SNAPSHOT_NAME + "_retry",
        "download_tickers": None,
        "returns_note": "R_t = 100*(ln P_t - ln P_{t-1}); yfinance 00:00 UTC close; crypto Close, indices Adj Close",
        "preprocessing": {
            "wavelet_denoise": "sym8",
            "baseline_removal": "db11",
            "swt_level_cap": 5,
            "sigma_w": "MAD(finest detail) / 0.6745",
            "threshold": "sqrt(2 * sigma_w^2 * ln N), soft",
            "scale_arm_a_full_series": "[-1,1] on the whole series before SWT; RQA windows are not scaled again",
            "scale_arm_b_window": "[-1,1] inside each 80-day window; no Geometry z-score",
        },
        "rqa": {"W": W_RQA, "m": M_EMB, "tau": TAU, "rr": 0.10, "l_min": 2, "v_min": 2},
        "arm_a_period": list(PAPER_PERIOD),
        "arm_a_horizons": _pick(smoke_mode, list(range(5, 51)), [5, H_PRIMARY]),
        "arm_a_daily_step": _pick(smoke_mode, 1, 10),
        "arm_a_label": "v_at_t_plus_h",
        "arm_a_measures": MEASURES,
        "arm_a_repro_auc_ref": 0.605,
        "arm_a_repro_auc_tol": 0.05,
        "arm_a_repro_h": H_PRIMARY,
        "arm_a_repro_measure": "DET_LAM",
        "volatility_window": VOL_WINDOW,
        "event_quantile": 0.95,
        "arm_b_preprocessing_primary": "B0",
        "arm_b_horizon_primary": H_PRIMARY,
        "arm_b_measure_primary": "DET_LAM",
        "arm_b_sample_start": "2015-01-01",
        "arm_b_first_test_year": _pick(smoke_mode, 2018, 2022),
        "frame_step": _pick(smoke_mode, 1, 10),
        "arm_c_indices": INDICES,
        "arm_c_crypto_rule": (
            "CoinGecko top 50 by USD market cap at fetch; drop stablecoins and wrapped/staked tokens; "
            "keep names with >=5 years and >=1460 closes; take the 8 largest others by that market cap"
        ),
        "exclusion_rule": {
            "universe": "CoinGecko top 50 by market cap (USD) at the snapshot fetch",
            "source_url": COINGECKO_MARKETS,
            "stable_categories": list(STABLE_CATEGORIES),
            "wrapped_staked_categories": list(WRAPPED_CATEGORIES),
            "stable_symbols": sorted(STABLE_SYMBOLS),
            "stable_symbol_pattern": "symbol is in stable_symbols, or starts with USD, or ends with USD",
            "wrapped_staked_symbols": sorted(WRAPPED_SYMBOLS),
            "name_rules": "CoinGecko id/name contains wrapped or liquid staked, or id starts with staked-",
            "yahoo_alias_by_id": dict(YAHOO_BY_ID),
            "history_rule": ">=5 calendar years and >=1460 closes, then the 8 largest other names",
            "xmr_leo": "included only if they pass exclusion and history and rank in those 8; not forced in",
        },
        "tickers_frozen": None,
        "market_cap_usd": None,
        "market_cap_source": None,
        "universe_top50": None,
        "universe_excluded": None,
        "eligible_not_selected": None,
        "category_errors": None,
        "primary_test": {
            "delta_auc_M1_minus_M0": "DET_LAM h=42 v_at_t+h B0",
            "bootstrap_block_days": 66,
            "panel": True,
            "pass_rule": "delta_auc > 0 and ci_low > 0",
            "includes_arm_A": False,
        },
        "diagnostics": {
            "placebo_shifts": 30,
            "placebo_min_days": 365,
            "placebo_spacing_days": 21,
            "within_year": "event-weighted AUC of M0, M1, and raw vol22 inside each ticker-year",
            "changes_pass_rule": False,
        },
        "secondary_horizons": _pick(smoke_mode, SECONDARY_HORIZONS, [5, H_PRIMARY]),
        "n_boot_arm_a": _pick(smoke_mode, 400, 25),
        "n_boot_arm_bc": _pick(smoke_mode, 500, 40),
        "seed_arm_a": 6006,
        "seed_arm_bc": 6042,
    }

def assert_spec_out_dir(spec: dict, out_dir) -> None:
    name = Path(out_dir).name
    if spec["smoke"] and name != "smoke":
        raise RuntimeError(f"smoke spec must be written to a directory named smoke, not {out_dir}")
    if (not spec["smoke"]) and name == "smoke":
        raise RuntimeError("full spec must not be written under smoke/")


def verify_spec_immutable(out_dir: Path, spec: dict, modules: list[str]) -> bool:
    from condpred.report import write_spec

    bad = dict(spec)
    bad["volatility_window"] = spec["volatility_window"] + 1
    try:
        write_spec(bad, out_dir, modules=modules)
    except RuntimeError:
        return True
    raise AssertionError("spec should reject silent overwrite")


def _pywt_available() -> bool:
    try:
        import pywt  # noqa: F401

        return True
    except ImportError:
        return False


def _log(msg: str) -> None:
    print(msg, flush=True)


def _scale_minus1_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def _mad_sigma(detail: np.ndarray) -> float:
    detail = np.asarray(detail, float)
    detail = detail[np.isfinite(detail)]
    if len(detail) == 0:
        return 1.0
    mad = np.median(np.abs(detail - np.median(detail)))
    return float(mad / 0.6745) if mad > 0 else float(np.std(detail) or 1.0)


def vanheusden_preprocess(x: np.ndarray, *, level_cap: int = 5) -> np.ndarray:
    """Scale to [-1, 1], sym8 soft-threshold denoise, db11 baseline removal.

    pywt.iswt uses the coarsest approximation (swt list entry 0); that coefficient is zeroed.
    """
    import pywt

    x = _scale_minus1_1(np.asarray(x, float))
    if len(x) % 2 == 1:
        x = x[1:]
    n = len(x)
    if n < 16:
        return x
    level = min(int(pywt.swt_max_level(n)), int(level_cap))
    if level < 1:
        return x
    coeffs = pywt.swt(x, "sym8", level=level)
    sigma_w = _mad_sigma(coeffs[-1][1])
    th = float(np.sqrt(2.0 * sigma_w**2 * np.log(max(n, 2))))
    coeffs_th = [(ca, pywt.threshold(cd, th, mode="soft")) for ca, cd in coeffs]
    x1 = np.asarray(pywt.iswt(coeffs_th, "sym8"), float)
    if len(x1) % 2 == 1:
        x1 = x1[1:]
    level2 = min(int(pywt.swt_max_level(len(x1))), int(level_cap))
    if level2 < 1:
        return x1
    coeffs2 = pywt.swt(x1, "db11", level=level2)
    coeffs2_z = [(np.zeros_like(ca) if i == 0 else ca, cd) for i, (ca, cd) in enumerate(coeffs2)]
    return np.asarray(pywt.iswt(coeffs2_z, "db11"), float)


def _preprocess_full(values: np.ndarray) -> np.ndarray:
    """Even-length SWT, aligned to the original index (oldest sample dropped if N is odd)."""
    values = np.asarray(values, float)
    offset = 1 if len(values) % 2 == 1 else 0
    out = np.full(len(values), np.nan)
    pre = vanheusden_preprocess(values[offset:])
    out[offset : offset + len(pre)] = pre
    return out


def clean_panel(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Degenerate-bar and gap cleaning that keeps the plan's crypto histories.

    Shared quality_filter drops anything shorter than 12 years, which would remove the
    arm C crypto panel. E6 does not use that drop.
    """
    parts = []
    reports = []
    tickers = list(dict.fromkeys(raw.columns.get_level_values(0)))
    for ticker in tickers:
        sub = raw[ticker].copy()
        sub.columns = [str(c) for c in sub.columns]
        if "Close" not in sub.columns:
            reports.append({"ticker": ticker, "n": 0, "years": 0.0, "dropped": True, "reason": "no_close"})
            continue
        obs = sub.loc[sub["Close"].notna()].copy()
        if obs.empty:
            reports.append({"ticker": ticker, "n": 0, "years": 0.0, "dropped": True, "reason": "no_prices"})
            continue
        o, h, l, c = obs["Open"], obs["High"], obs["Low"], obs["Close"]
        deg = (h == l) | ((o == h) & (h == l) & (l == c)) | (h < np.maximum(o, c)) | (l > np.minimum(o, c))
        deg = deg.fillna(True)
        gaps = obs.index.to_series().diff().dt.days.fillna(0) > 5
        bad = deg | gaps
        price_cols = [col for col in obs.columns if col in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}]
        kept = obs.copy()
        kept.loc[bad, price_cols] = np.nan
        finite = kept["Close"].dropna()
        years = (obs.index.max() - obs.index.min()).days / 365.25
        reports.append(
            {
                "ticker": ticker,
                "n": int(len(finite)),
                "start": str(obs.index.min().date()),
                "end": str(obs.index.max().date()),
                "years": float(years),
                "degenerate_frac": float(deg.mean()),
                "dropped": bool(len(finite) == 0),
                "reason": "no_prices_after_clean" if len(finite) == 0 else "",
            }
        )
        if len(finite) == 0:
            continue
        block = kept.rename(columns=lambda col: (ticker, col))
        block.columns = pd.MultiIndex.from_tuples(block.columns)
        parts.append(block)
    report = pd.DataFrame(reports)
    if not parts:
        return raw.iloc[:, :0], report
    return pd.concat(parts, axis=1).sort_index(), report


def _cg_json(url: str):
    import json
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "recplay-e6", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode())


def _category_ids(category: str) -> tuple[set[str], str]:
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        f"?vs_currency=usd&category={category}&order=market_cap_desc&per_page=250&page=1&sparkline=false"
    )
    try:
        rows = _cg_json(url)
    except Exception as exc:
        return set(), f"{type(exc).__name__}: {exc}"
    return {row["id"] for row in rows if row.get("id")}, ""


def _stable_symbol(symbol: str) -> bool:
    return symbol in STABLE_SYMBOLS or symbol.startswith("USD") or symbol.endswith("USD")


def _wrapped_name(cid: str, name: str, symbol: str, wrap_ids: set[str]) -> bool:
    lid, lname = cid.lower(), name.lower()
    return (
        cid in wrap_ids
        or symbol in WRAPPED_SYMBOLS
        or lid.startswith("wrapped")
        or "wrapped" in lname
        or "liquid staked" in lname
        or lid.startswith("staked-")
    )


def _build_universe() -> dict:
    import time
    from datetime import datetime, timezone

    top = _cg_json(COINGECKO_MARKETS)
    if not isinstance(top, list) or len(top) < 50:
        raise RuntimeError(f"CoinGecko top-50 response has {0 if not isinstance(top, list) else len(top)} rows")
    cat_errors = {}
    stable_ids: set[str] = set()
    wrap_ids: set[str] = set()
    for cat in STABLE_CATEGORIES:
        ids, err = _category_ids(cat)
        stable_ids |= ids
        if err:
            cat_errors[cat] = err
        time.sleep(0.4)
    for cat in WRAPPED_CATEGORIES:
        ids, err = _category_ids(cat)
        wrap_ids |= ids
        if err:
            cat_errors[cat] = err
        time.sleep(0.4)
    excluded, candidates, top_rows = [], [], []
    for rank, row in enumerate(top[:50], 1):
        cid = str(row["id"])
        symbol = str(row.get("symbol") or "").upper()
        name = str(row.get("name") or "")
        cap = float(row["market_cap"]) if row.get("market_cap") else float("nan")
        yahoo = YAHOO_BY_ID.get(cid, f"{symbol}-USD")
        rec = {"rank": rank, "id": cid, "symbol": symbol, "name": name, "market_cap": cap, "yahoo": yahoo}
        top_rows.append(rec)
        reasons = []
        if cid in stable_ids or _stable_symbol(symbol):
            reasons.append("stablecoin")
        if _wrapped_name(cid, name, symbol, wrap_ids):
            reasons.append("wrapped_or_staked")
        if reasons:
            excluded.append({**rec, "reason": "+".join(reasons)})
        else:
            candidates.append(rec)
    return {
        "source": "CoinGecko coins/markets ordered by market_cap_desc, top 50, USD",
        "source_url": COINGECKO_MARKETS,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "top50": top_rows,
        "excluded": excluded,
        "candidates": candidates,
        "category_errors": cat_errors,
        "download_tickers": [c["yahoo"] for c in candidates],
    }


def _universe() -> dict:
    import json

    path = _YAHOO_DIR / "e6_crypto_top50_v3_universe.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    uni = _build_universe()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(uni), encoding="utf-8")
    tmp.replace(path)
    _log(f"wrote universe sidecar {path.name} with {len(uni['download_tickers'])} yahoo tickers")
    return uni


def _require_snapshot(raw: pd.DataFrame, name: str, earliest: str, label: str) -> None:
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError(f"{label} snapshot {name} is empty or not multi-ticker; not overwriting it")
    start = pd.to_datetime(raw.index.min())
    if start > pd.Timestamp(earliest):
        raise RuntimeError(f"{label} snapshot {name} starts {start.date()}, expected on or before {earliest}")


def _snapshot_files(name: str) -> list[Path]:
    hits = sorted(_YAHOO_DIR.glob(f"condpred_{name}_*.csv"))
    return [p for p in hits if "manifest" not in p.name]


def _quarantine_new(name: str, before: set[Path]) -> None:
    """Move a snapshot this call just wrote so a rejected download is not reused."""
    for path in _snapshot_files(name):
        if path in before:
            continue
        os.replace(path, path.with_name("rejected_" + path.name))
        man = path.with_name(path.stem + "_manifest.json")
        if man.exists():
            os.replace(man, man.with_name("rejected_" + man.name))
        _log(f"quarantined rejected snapshot {path.name}")


def _checked_snapshot(tickers, start, end, name, earliest, label, extra=None) -> pd.DataFrame:
    before = set(_snapshot_files(name))
    raw = yahoo_ohlc_snapshot(tickers, start=start, end=end, name=name, refresh=False)
    try:
        _require_snapshot(raw, name, earliest, label)
        if extra is not None:
            extra(raw)
    except RuntimeError:
        _quarantine_new(name, before)
        raise
    return raw


def _merge_retry(crypto: pd.DataFrame, missing: list[str], spec: dict) -> pd.DataFrame:
    """Second snapshot for tickers the batch download dropped. Never refreshes the main file."""
    name = spec["data_retry_name"]
    before = set(_snapshot_files(name))
    try:
        extra = yahoo_ohlc_snapshot(
            missing, start=spec["data_start"], end=spec.get("data_end"), name=name, refresh=False,
        )
    except Exception as exc:
        _log(f"retry download failed: {type(exc).__name__}: {exc}")
        return crypto
    if extra.empty or not isinstance(extra.columns, pd.MultiIndex):
        _quarantine_new(name, before)
        _log(f"retry snapshot {name} had no usable columns")
        return crypto
    return pd.concat([crypto, extra], axis=1).sort_index()


def load_yahoo_snapshot(spec: dict, *, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if refresh:
        raise RuntimeError("E6 never refreshes Yahoo snapshots")
    uni = _universe()
    def _has_btc(raw):
        if "BTC-USD" not in set(raw.columns.get_level_values(0)):
            raise RuntimeError("crypto snapshot has no BTC-USD; refusing to overwrite it")

    crypto = _checked_snapshot(
        uni["download_tickers"], spec["data_start"], spec.get("data_end"),
        spec["data_name"], "2014-10-01", "crypto", extra=_has_btc,
    )
    def _has_prices(frame, ticker):
        if ticker not in set(frame.columns.get_level_values(0)):
            return False
        sub = frame[ticker]
        col = "Close" if "Close" in getattr(sub, "columns", []) else sub.columns[0]
        return bool(sub[col].notna().any())

    missing_raw = [t for t in uni["download_tickers"] if not _has_prices(crypto, t)]
    priced = [t for t in crypto.columns.get_level_values(0).unique() if _has_prices(crypto, t)]
    crypto = crypto.loc[:, crypto.columns.get_level_values(0).isin(priced)]
    if missing_raw and any(_YAHOO_DIR.glob(f"rejected_condpred_{spec['data_retry_name']}_*.csv")):
        _log(f"retry snapshot already rejected; still no bars for {missing_raw}")
    elif missing_raw:
        _log(f"retrying {missing_raw} under {spec['data_retry_name']} without touching {spec['data_name']}")
        crypto = _merge_retry(crypto, missing_raw, spec)
    indices = _checked_snapshot(
        list(INDICES), spec["index_sample_start"], spec.get("data_end"),
        spec["index_data_name"], "2000-02-01", "index",
    )
    clean_c, rep_c = clean_panel(crypto)
    clean_i, rep_i = clean_panel(indices)
    rep_c["snapshot"] = spec["data_name"]
    rep_i["snapshot"] = spec["index_data_name"]
    have = set(clean_c.columns.get_level_values(0))
    missing = [t for t in uni["download_tickers"] if t not in have]
    if missing:
        _log(f"yahoo returned no bars for {missing}")
    return pd.concat([clean_c, clean_i], axis=1).sort_index(), pd.concat([rep_c, rep_i], ignore_index=True)


def _coin_status(symbol: str, chosen, not_sel, excluded, missing) -> str:
    yahoo = f"{symbol}-USD"
    if any(r["yahoo"] == yahoo for r in chosen):
        return "in the top 8"
    if any(r["yahoo"] == yahoo for r in not_sel):
        return "eligible on history but outside the top 8"
    hit = [r for r in excluded if r.get("symbol") == symbol or r.get("yahoo") == yahoo]
    if hit:
        return "excluded: " + hit[0].get("reason", "")
    hit = [r for r in missing if r.get("symbol") == symbol or r.get("yahoo") == yahoo]
    if hit:
        return "not eligible: " + hit[0].get("reason", "short_or_missing")
    return "not in the CoinGecko top 50"


def close_series(ohlc: pd.DataFrame, ticker: str) -> pd.Series:
    sub = ohlc[ticker] if isinstance(ohlc.columns, pd.MultiIndex) else ohlc
    col = "Adj Close" if ticker in INDICES and "Adj Close" in sub.columns else "Close"
    s = sub[col].astype(float).copy()
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s.sort_index()


def log_returns_pct(prices: pd.Series) -> pd.Series:
    p = prices.astype(float)
    return 100.0 * np.log(p / p.shift(1))


def vol5(returns: pd.Series) -> pd.Series:
    return returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()


def _history(prices: pd.Series) -> tuple[float, int]:
    s = prices.dropna()
    if s.empty:
        return 0.0, 0
    return (s.index.max() - s.index.min()).days / 365.25, int(len(s))


def freeze_panel_tickers(df: pd.DataFrame, spec: dict) -> dict:
    spec = dict(spec)
    uni = _universe()
    avail = set(df.columns.get_level_values(0)) if isinstance(df.columns, pd.MultiIndex) else set()
    eligible, missing = [], []
    for rec in uni["candidates"]:
        if rec["yahoo"] == "BTC-USD":
            continue
        if rec["yahoo"] not in avail:
            missing.append({**rec, "reason": "no_yfinance_column"})
            continue
        years, n = _history(close_series(df, rec["yahoo"]))
        item = {**rec, "years": years, "n_closes": n}
        if years >= 5.0 and n >= 1460 and np.isfinite(rec.get("market_cap", np.nan)):
            eligible.append(item)
        else:
            item["reason"] = "short_history"
            missing.append(item)
    eligible.sort(key=lambda r: (-r["market_cap"], r["yahoo"]))
    chosen, not_sel = eligible[:8], eligible[8:]
    indices = [t for t in INDICES if t in avail and _history(close_series(df, t))[1] > 200]
    panel = ["BTC-USD"] + [r["yahoo"] for r in chosen] + indices
    spec["download_tickers"] = list(uni["download_tickers"])
    spec["universe_top50"] = uni["top50"]
    spec["universe_excluded"] = list(uni["excluded"]) + missing
    spec["eligible_not_selected"] = not_sel
    spec["category_errors"] = dict(uni.get("category_errors") or {})
    spec["market_cap_usd"] = {r["yahoo"]: r["market_cap"] for r in uni["candidates"]}
    spec["market_cap_source"] = f"{uni['source']} at {uni['fetched_at']}"
    spec["tickers_frozen"] = ["BTC-USD"] if spec["smoke"] else panel
    note = f"XMR: {_coin_status('XMR', chosen, not_sel, uni['excluded'], missing)}; LEO: {_coin_status('LEO', chosen, not_sel, uni['excluded'], missing)}"
    _log(f"panel ({'smoke uses BTC only; ' if spec['smoke'] else ''}full selection {panel})")
    _log(note)
    return spec


def _shannon_vertical(lengths: np.ndarray) -> float:
    if len(lengths) == 0:
        return np.nan
    vals = np.asarray(lengths)
    vals = vals[vals >= 2]
    if len(vals) == 0:
        return np.nan
    counts = np.bincount(vals.astype(int))[2:]
    p = counts[counts > 0] / counts[counts > 0].sum()
    return float(-(p * np.log(p)).sum())


def _recurrence_prescaled(win: np.ndarray, m: int, tau: int, rr: float = 0.10) -> np.ndarray:
    """RP of a window already on the paper's scale. No z-score."""
    states = embed(np.asarray(win, float), int(m), int(tau))
    if len(states) < 5:
        raise ValueError("embedding too short")
    return threshold_by_rate(euclidean_distances(states), rr, max(1, (int(m) - 1) * int(tau)))


def window_rqa_measures(win: np.ndarray, m: int, tau: int, *, rr: float = 0.10) -> dict:
    rec = _recurrence_prescaled(win, m, tau, rr=rr)
    out = dict(rqa(rec))
    _diag, vert = line_lengths(rec)
    out["TT"] = float(vert[vert >= 2].mean()) if np.any(vert >= 2) else np.nan
    out["ENTR_v"] = _shannon_vertical(vert)
    lam, det = out.get("LAM", np.nan), out.get("DET", np.nan)
    out["DET_LAM"] = float(det / lam) if np.isfinite(lam) and lam not in (0, 0.0) and np.isfinite(det) else np.nan
    ent = out["ENTR_v"]
    out["LAM_ENTR_v"] = float(lam / ent) if np.isfinite(ent) and ent not in (0, 0.0) and np.isfinite(lam) else np.nan
    return out


def _measure_dict(feat: dict) -> dict:
    return {k: feat.get(k, np.nan) for k in MEASURES}


def _nan_measures() -> dict:
    return {k: np.nan for k in MEASURES}


def _std(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return np.nan
    return float(np.std(a, ddof=1))


def _rows_for_ends(
    x, v, dates_ns, ends, preprocess, m, tau, include_lrqa, horizons,
):
    dates = pd.to_datetime(np.asarray(dates_ns))
    rows = []
    horizons = list(horizons)
    h_max = max(horizons)
    for t in ends:
        t = int(t)
        if t + h_max >= len(v) or not np.isfinite(v[t]):
            continue
        raw = x[t - W_RQA + 1 : t + 1]
        if len(raw) < W_RQA or not np.isfinite(raw).all():
            continue
        try:
            if preprocess == "window_swt":
                win = vanheusden_preprocess(raw)
            else:
                win = _scale_minus1_1(raw)
            if len(win) < W_RQA or not np.isfinite(win).all():
                feat = _nan_measures()
                lrqa = {}
            else:
                if len(win) > W_RQA:
                    win = win[-W_RQA:]
                feat = _measure_dict(window_rqa_measures(win, m, tau))
                lrqa = {}
                if include_lrqa:
                    rec = _recurrence_prescaled(win, m, tau)
                    theiler = max(1, (int(m) - 1) * int(tau))
                    local = local_rqa_matrix(rec, theiler=theiler, H=8, w=theiler)
                    lrqa = {
                        "lrqa_RR_last": local.get("RR_last", np.nan),
                        "lrqa_DET_last": local.get("DET_last", np.nan),
                        "lrqa_K2_local": local.get("K2_local", np.nan),
                        "lrqa_LAM_last": local.get("LAM_last", np.nan),
                    }
        except Exception:
            feat = _nan_measures()
            lrqa = {c: np.nan for c in LRQA_COLS} if include_lrqa else {}
        v22 = _std(x[max(0, t - 21) : t + 1])
        v80 = _std(x[max(0, t - 79) : t + 1])
        ratio = np.log(v22 / v80) if np.isfinite(v22) and np.isfinite(v80) and v22 > 0 and v80 > 0 else np.nan
        ts = pd.Timestamp(dates[t])
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        row = {
            "date": ts,
            "year": int(ts.year),
            "t": t,
            "vol_at_t": float(v[t]),
            "vol5": float(v[t]),
            "vol22": v22,
            "vol80": v80,
            "log_vol_ratio": ratio,
            **feat,
            **lrqa,
        }
        for h in horizons:
            if t + h < len(v):
                row[f"vol_h{h}"] = float(v[t + h]) if np.isfinite(v[t + h]) else np.nan
                window = v[t + 1 : t + h + 1]
                row[f"vmax_{h}"] = float(np.nanmax(window)) if np.isfinite(window).any() else np.nan
        rows.append(row)
    return rows


def _parallel_map(fn_args: list) -> list:
    if len(fn_args) <= 1:
        return [_rows_for_ends(*fn_args[0])] if fn_args else []
    n_jobs = min(N_JOBS, len(fn_args))
    return Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_rows_for_ends)(*args) for args in fn_args)


def _embedding(x_train: np.ndarray, mode: str) -> tuple[int, int, int]:
    """Return (m, tau, fnn_m). Primary is m=tau=1. Secondary m=3 uses AMI tau; FNN m is recorded."""
    x_train = np.asarray(x_train, float)
    x_train = x_train[np.isfinite(x_train)]
    if mode == "primary" or len(x_train) < 200:
        return (M_EMB, TAU, M_EMB) if mode == "primary" else (3, 1, M_EMB)
    max_lag = min(40, max(5, len(x_train) // 20))
    ami = average_mutual_information(x_train, max_lag)
    tau = int(first_local_minimum(ami) or 1)
    tau = max(1, min(tau, 10))
    fnn = false_nearest_fraction(x_train, tau, max_m=8)
    fnn_m = 8
    for i, frac in enumerate(fnn):
        if frac < 0.05:
            fnn_m = i + 1
            break
    return 3, tau, int(fnn_m)


def build_packs(
    raw: pd.DataFrame,
    spec: dict,
    cache_exp: str,
    *,
    preprocess: str,
    embedding: str,
    include_lrqa: bool,
) -> dict[str, dict]:
    horizons = sorted(set(spec["secondary_horizons"]) | {spec["arm_b_horizon_primary"]})
    packs = {}
    for ticker in spec["tickers_frozen"]:
        if ticker not in raw.columns.get_level_values(0):
            continue
        prices = close_series(raw, ticker).dropna()
        start = spec["index_sample_start"] if ticker in INDICES else spec["arm_b_sample_start"]
        prices = prices.loc[prices.index >= start]
        ret = log_returns_pct(prices).dropna()
        if len(ret) < W_RQA + max(horizons) + 10:
            continue
        x = ret.to_numpy(float)
        v = vol5(ret).to_numpy(float)
        dates_ns = ret.index.to_numpy(dtype="datetime64[ns]")
        cut = ret.index.year < int(spec["arm_b_first_test_year"])
        m, tau, fnn_m = _embedding(x[cut], embedding)
        step = max(1, int(spec["frame_step"]))
        ends = list(range(W_RQA - 1, len(x) - max(horizons), step))

        def _compute(x=x, v=v, dates_ns=dates_ns, ends=ends, m=m, tau=tau):
            _log(f"frame {ticker} {preprocess} m={m} tau={tau} n_ends={len(ends)} lrqa={include_lrqa}")
            chunks = [c for c in np.array_split(np.asarray(ends, int), N_JOBS) if len(c)]
            args = [
                (x, v, dates_ns, chunk.tolist(), preprocess, m, tau, include_lrqa, horizons) for chunk in chunks
            ]
            parts = _parallel_map(args)
            rows = [row for part in parts for row in part]
            frame = pd.DataFrame(rows)
            if len(frame):
                frame = frame.sort_values("t").reset_index(drop=True)
            return {"frame": frame, "vol": v, "m": m, "tau": tau, "fnn_m": fnn_m}

        key = {
            "ticker": ticker,
            "preprocess": preprocess,
            "embedding": embedding,
            "m": m,
            "tau": tau,
            "lrqa": include_lrqa,
            "step": step,
            "horizons": horizons,
            "n": int(len(x)),
            "end": str(pd.Timestamp(dates_ns[-1]).date()),
            "sample_start": str(pd.Timestamp(start).date()),
            "checksum": float(np.nansum(x)),
        }
        packs[ticker] = cached(cache_exp, f"frame_{ticker}", key, _compute, modules=E6_MODULES)
        packs[ticker]["fnn_m"] = fnn_m
    return packs


def _recent_exceed(vol: np.ndarray, t: int, thr: float) -> float:
    if not np.isfinite(thr):
        return np.nan
    w = vol[max(0, t - W_RQA + 1) : t + 1]
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return np.nan
    return float(np.mean(w > thr))


def _splits(df: pd.DataFrame, first_year: int, horizon: int):
    if df.empty:
        return
    t = df["t"].to_numpy(int)
    year = df["year"].to_numpy(int)
    for y in range(int(first_year), int(year.max()) + 1):
        test = np.flatnonzero(year == y)
        if len(test) < 8:
            continue
        t0 = int(t[test].min())
        train = np.flatnonzero((year < y) & (t + int(horizon) < t0))
        if len(train) < 80:
            continue
        yield y, train, test


def _fill_allnan(Xtr, Xte):
    Xtr = np.array(Xtr, float, copy=True)
    Xte = np.array(Xte, float, copy=True)
    for j in range(Xtr.shape[1]):
        if not np.isfinite(Xtr[:, j]).any():
            Xtr[:, j] = 0.0
            Xte[:, j] = 0.0
    return Xtr, Xte


def _fit_proba(Xtr, ytr, Xte):
    ytr = np.asarray(ytr, int)
    if len(ytr) < 20 or len(np.unique(ytr)) < 2:
        return None
    Xtr, Xte = _fill_allnan(Xtr, Xte)
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=800, class_weight="balanced")),
        ]
    )
    try:
        pipe.fit(Xtr, ytr)
        return pipe.predict_proba(Xte)[:, 1]
    except Exception:
        return None


def _design(frame: pd.DataFrame, exceed: np.ndarray, extra: list[str]) -> np.ndarray:
    cols = [frame[c].to_numpy(float) for c in M0_FIELDS]
    cols.append(np.asarray(exceed, float))
    for c in extra:
        cols.append(frame[c].to_numpy(float))
    return np.column_stack(cols)


def collect_oos(packs: dict, spec: dict, *, horizon: int, label: str, models: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """Walk-forward logistic scores. ``models`` maps a name to columns added on top of M0."""
    h = int(horizon)
    ycol = f"vol_h{h}" if label == "point" else f"vmax_{h}"
    first = int(spec["arm_b_first_test_year"])
    buckets = {name: [] for name in models}
    for ticker, pack in packs.items():
        df = pack["frame"]
        vol = np.asarray(pack["vol"], float)
        if df.empty or ycol not in df.columns:
            continue
        sub = df.loc[np.isfinite(df[ycol].to_numpy(float))].reset_index(drop=True)
        for year, tr_i, te_i in _splits(sub, first, h):
            hist = sub.loc[sub["year"] < year, "vol_at_t"].to_numpy(float)
            if np.isfinite(hist).sum() < 50:
                continue
            thr = float(np.nanquantile(hist, spec["event_quantile"]))
            tr, te = sub.iloc[tr_i], sub.iloc[te_i]
            ytr = (tr[ycol].to_numpy(float) > thr).astype(int)
            yte = (te[ycol].to_numpy(float) > thr).astype(int)
            exc_tr = np.array([_recent_exceed(vol, int(t), thr) for t in tr["t"]], float)
            exc_te = np.array([_recent_exceed(vol, int(t), thr) for t in te["t"]], float)
            p0 = _fit_proba(_design(tr, exc_tr, []), ytr, _design(te, exc_te, []))
            if p0 is None:
                continue
            base = {
                "ticker": ticker,
                "date": pd.to_datetime(te["date"]).to_numpy(),
                "year": int(year),
                "y": yte,
                "p0": p0,
                "vol22": te["vol22"].to_numpy(float),
            }
            for name, extra in models.items():
                missing = [c for c in extra if c not in tr.columns]
                if missing:
                    continue
                p1 = _fit_proba(_design(tr, exc_tr, extra), ytr, _design(te, exc_te, extra))
                if p1 is None:
                    continue
                part = pd.DataFrame(base)
                part["p1"] = p1
                buckets[name].append(part)
    out = {}
    for name, parts in buckets.items():
        out[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def _auc(y, score) -> float:
    y = np.asarray(y, float)
    score = np.asarray(score, float)
    ok = np.isfinite(y) & np.isfinite(score)
    if ok.sum() < 10:
        return np.nan
    yy = y[ok]
    if yy.min() == yy.max():
        return np.nan
    return float(roc_auc_score(yy, score[ok]))


def _boot_delta(y, p0, p1, dates, rng, n_boot: int, block: int):
    y = np.asarray(y, float)
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    dates = pd.to_datetime(np.asarray(dates))
    a0, a1 = _auc(y, p0), _auc(y, p1)
    delta = a1 - a0 if np.isfinite(a0) and np.isfinite(a1) else np.nan
    empty = (a0, a1, delta, np.nan, np.nan, np.nan, int(len(y)))
    if not np.isfinite(delta) or len(y) < 10:
        return empty
    uniq = pd.Index(dates).unique().sort_values()
    codes = pd.Categorical(dates, categories=uniq).codes
    buckets = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    boots = []
    for _ in range(int(n_boot)):
        draw = block_bootstrap_idx(len(uniq), block, rng)
        bi = np.concatenate([buckets[int(j)] for j in draw])
        b0, b1 = _auc(y[bi], p0[bi]), _auc(y[bi], p1[bi])
        if np.isfinite(b0) and np.isfinite(b1):
            boots.append(b1 - b0)
    if len(boots) < max(20, int(n_boot) // 5):
        return a0, a1, delta, np.nan, np.nan, 1.0, int(len(y))
    arr = np.asarray(boots, float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p = (1.0 + float(np.sum(arr <= 0.0))) / (1.0 + len(arr))
    return a0, a1, float(delta), float(lo), float(hi), float(p), int(len(y))


def summarize_by_asset(oos: pd.DataFrame, spec: dict, *, seed: int) -> pd.DataFrame:
    if oos is None or oos.empty:
        return pd.DataFrame()
    block = int(spec["primary_test"]["bootstrap_block_days"])
    n_boot = int(spec["n_boot_arm_bc"])
    rows = []
    rng = np.random.default_rng(seed)
    for ticker, g in oos.groupby("ticker", sort=True):
        a0, a1, delta, lo, hi, p, n = _boot_delta(g["y"], g["p0"], g["p1"], g["date"], rng, n_boot, block)
        rows.append(
            {
                "ticker": ticker,
                "row_type": "asset",
                "test_year": "all",
                "auc_M0": a0,
                "auc_M1": a1,
                "delta_auc": delta,
                "ci_low": lo,
                "ci_high": hi,
                "p_boot": p,
                "n_test": n,
                "n_events": int(np.nansum(g["y"].to_numpy(float))),
            }
        )
        for year, gy in g.groupby("year"):
            rows.append(
                {
                    "ticker": ticker,
                    "row_type": "year",
                    "test_year": int(year),
                    "auc_M0": _auc(gy["y"], gy["p0"]),
                    "auc_M1": _auc(gy["y"], gy["p1"]),
                    "delta_auc": _auc(gy["y"], gy["p1"]) - _auc(gy["y"], gy["p0"])
                    if np.isfinite(_auc(gy["y"], gy["p1"])) and np.isfinite(_auc(gy["y"], gy["p0"]))
                    else np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_boot": np.nan,
                    "n_test": int(len(gy)),
                    "n_events": int(np.nansum(gy["y"].to_numpy(float))),
                }
            )
    return pd.DataFrame(rows)


def summarize_panel(oos: pd.DataFrame, spec: dict, *, seed: int) -> pd.DataFrame:
    if oos is None or oos.empty:
        return pd.DataFrame()
    block = int(spec["primary_test"]["bootstrap_block_days"])
    rng = np.random.default_rng(seed)
    a0, a1, delta, lo, hi, p, n = _boot_delta(
        oos["y"], oos["p0"], oos["p1"], oos["date"], rng, int(spec["n_boot_arm_bc"]), block
    )
    return pd.DataFrame(
        [
            {
                "panel_delta_auc": delta,
                "auc_M0": a0,
                "auc_M1": a1,
                "ci_low": lo,
                "ci_high": hi,
                "p_boot": p,
                "n": n,
                "n_events": int(np.nansum(oos["y"].to_numpy(float))),
                "n_tickers": int(oos["ticker"].nunique()),
            }
        ]
    )


def _nw_beta_t(y, x, lags: int) -> tuple[float, float]:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < max(int(lags) + 5, 20):
        return np.nan, np.nan
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    scores = X * resid[:, None]
    meat = scores.T @ scores / n
    L = max(int(lags), 0)
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        cj = scores[j:].T @ scores[:-j] / n
        meat = meat + w * (cj + cj.T)
    meat = meat * n
    try:
        bread = np.linalg.inv(X.T @ X)
        var = float((bread @ meat @ bread)[1, 1])
    except np.linalg.LinAlgError:
        return float(beta[1]), np.nan
    se = np.sqrt(max(var, 0.0))
    if not np.isfinite(se) or se == 0:
        return float(beta[1]), np.nan
    return float(beta[1]), float(beta[1] / se)


def _block_permute(y: np.ndarray, block: int, rng) -> np.ndarray:
    y = np.asarray(y)
    n = len(y)
    block = max(1, int(block))
    if block >= n:
        return rng.permutation(y)
    n_blocks = int(np.ceil(n / block))
    pad = n_blocks * block - n
    ypad = np.concatenate([y, np.full(pad, -1)]) if pad else y
    blocks = ypad.reshape(n_blocks, block)[rng.permutation(n_blocks)].ravel()
    return blocks[blocks >= 0][:n]


def _perm_p(y, score, rng, n_boot: int, block: int | None) -> float:
    obs = _auc(y, score)
    if not np.isfinite(obs):
        return np.nan
    count = 0
    valid = 0
    y = np.asarray(y, int)
    for _ in range(int(n_boot)):
        yp = rng.permutation(y) if block is None else _block_permute(y, block, rng)
        a = _auc(yp, score)
        if np.isfinite(a):
            valid += 1
            count += int(a >= obs)
    if valid < max(10, int(n_boot) // 5):
        return np.nan
    return (1.0 + count) / (1.0 + valid)


def _arm_a_chunk(full_pre, ends, m, tau):
    out = []
    for t in ends:
        t = int(t)
        win = full_pre[t - W_RQA + 1 : t + 1]
        if len(win) < W_RQA or not np.isfinite(win).all():
            out.append(_nan_measures())
            continue
        try:
            out.append(_measure_dict(window_rqa_measures(win, m, tau)))
        except Exception:
            out.append(_nan_measures())
    return out


def _arm_a_features(full_pre, ends, m, tau):
    ends = [int(t) for t in ends]
    if len(ends) < 40:
        return _arm_a_chunk(full_pre, ends, m, tau)
    chunks = [c.tolist() for c in np.array_split(np.asarray(ends, int), N_JOBS) if len(c)]
    parts = Parallel(n_jobs=min(N_JOBS, len(chunks)), prefer="processes")(
        delayed(_arm_a_chunk)(full_pre, c, m, tau) for c in chunks
    )
    return [feat for part in parts for feat in part]


def arm_a_grid(prices: pd.Series, spec: dict, cache_exp: str) -> pd.DataFrame:
    if not _pywt_available():
        return pd.DataFrame()

    def _compute():
        ret = log_returns_pct(prices).dropna()
        full_pre = _preprocess_full(ret.to_numpy(float))
        idx = ret.index
        period = pd.to_datetime(spec["arm_a_period"])
        mask = (idx >= period[0]) & (idx <= period[1])
        v = vol5(ret).to_numpy(float)
        in_period = v[np.asarray(mask)]
        thr = float(np.nanquantile(in_period[np.isfinite(in_period)], spec["event_quantile"]))
        step = max(1, int(spec["arm_a_daily_step"]))
        days = np.flatnonzero(np.asarray(mask))[::step]
        ends = [int(t) for t in days if t >= W_RQA - 1 and t < len(v)]
        _log(f"arm A windows {len(ends)} step {step}")
        feats = _arm_a_features(full_pre, ends, M_EMB, TAU)
        rows = []
        for t, feat in zip(ends, feats):
            for h in spec["arm_a_horizons"]:
                if t + h >= len(v) or not np.isfinite(v[t + h]):
                    continue
                for meas in spec["arm_a_measures"]:
                    rows.append(
                        {
                            "date": idx[t],
                            "horizon": int(h),
                            "measure": meas,
                            "score": feat.get(meas, np.nan),
                            "vol_future": float(v[t + h]),
                            "event": int(v[t + h] > thr),
                        }
                    )
        return pd.DataFrame(rows)

    key = {
        "arm": "A",
        "period": list(spec["arm_a_period"]),
        "horizons": list(spec["arm_a_horizons"]),
        "step": spec["arm_a_daily_step"],
        "end": str(prices.dropna().index.max().date()),
        "n": int(prices.dropna().shape[0]),
    }
    grid = cached(cache_exp, "arm_a_points", key, _compute, modules=E6_MODULES)
    if grid.empty:
        return grid
    rng = np.random.default_rng(int(spec["seed_arm_a"]))
    n_boot = int(spec["n_boot_arm_a"])
    rows = []
    grouped = list(grid.groupby(["measure", "horizon"], sort=True))
    for (meas, h), sub in grouped:
        score = sub["score"].to_numpy(float)
        vol = sub["vol_future"].to_numpy(float)
        event = sub["event"].to_numpy(int)
        ok = np.isfinite(score) & np.isfinite(vol)
        auc = _auc(event[ok], score[ok])
        beta, tnw = _nw_beta_t(vol, score, int(h))
        rows.append(
            {
                "measure": meas,
                "horizon": int(h),
                "auc": auc,
                "beta_nw": beta,
                "t_nw": tnw,
                "bootstrap_p": _perm_p(event[ok], score[ok], rng, n_boot, 33),
                "p_rand": _perm_p(event[ok], score[ok], rng, n_boot, None),
                "p_block66": _perm_p(event[ok], score[ok], rng, n_boot, 66),
                "n": int(ok.sum()),
                "n_events": int(event[ok].sum()) if ok.any() else 0,
            }
        )
    return pd.DataFrame(rows)


def _contrast_row(family, test, oos, spec, *, measure, horizon, label, preprocess, m, tau, seed, extra=None):
    row = {
        "family": family,
        "test": test,
        "measure": measure,
        "horizon": int(horizon),
        "label": label,
        "preprocess": preprocess,
        "m": m,
        "tau": tau,
        "delta_auc": np.nan,
        "auc_M0": np.nan,
        "auc_M1": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_raw": np.nan,
        "n": 0,
        "fnn_m_by_asset": extra or "",
    }
    if oos is None or oos.empty:
        return row
    rng = np.random.default_rng(seed)
    a0, a1, delta, lo, hi, p, n = _boot_delta(
        oos["y"], oos["p0"], oos["p1"], oos["date"], rng,
        int(spec["n_boot_arm_bc"]), int(spec["primary_test"]["bootstrap_block_days"]),
    )
    row.update({"auc_M0": a0, "auc_M1": a1, "delta_auc": delta, "ci_low": lo, "ci_high": hi, "p_raw": p, "n": n})
    return row


def _delta_of_deltas(oos_a, oos_b, spec, seed: int):
    """Bootstrap AUC(b)-AUC(a), both already stored as p1 against the same p0 dates.

    oos frames must be the same rows. Returns delta_b, delta_a, diff, lo, hi, p, n.
    """
    if oos_a is None or oos_b is None or len(oos_a) == 0 or len(oos_b) == 0:
        return (np.nan,) * 6 + (0,)
    merged = oos_a.merge(oos_b, on=["ticker", "date"], suffixes=("_a", "_b"))
    if merged.empty or "p0_a" not in merged.columns:
        return (np.nan,) * 6 + (0,)
    y = merged["y_a"].to_numpy(float)
    d_a = _auc(y, merged["p1_a"]) - _auc(y, merged["p0_a"])
    d_b = _auc(y, merged["p1_b"]) - _auc(y, merged["p0_b"])
    if not (np.isfinite(d_a) and np.isfinite(d_b)):
        return d_b, d_a, np.nan, np.nan, np.nan, np.nan, int(len(merged))
    dates = pd.to_datetime(merged["date"])
    uniq = pd.Index(dates).unique().sort_values()
    codes = pd.Categorical(dates, categories=uniq).codes
    buckets = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    boots = []
    n_boot = int(spec["n_boot_arm_bc"])
    block = int(spec["primary_test"]["bootstrap_block_days"])
    for _ in range(n_boot):
        bi = np.concatenate([buckets[int(j)] for j in block_bootstrap_idx(len(uniq), block, rng)])
        da = _auc(y[bi], merged["p1_a"].to_numpy(float)[bi]) - _auc(y[bi], merged["p0_a"].to_numpy(float)[bi])
        db = _auc(y[bi], merged["p1_b"].to_numpy(float)[bi]) - _auc(y[bi], merged["p0_b"].to_numpy(float)[bi])
        if np.isfinite(da) and np.isfinite(db):
            boots.append(db - da)
    if len(boots) < max(20, n_boot // 5):
        return d_b, d_a, d_b - d_a, np.nan, np.nan, 1.0, int(len(merged))
    arr = np.asarray(boots, float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    p = (1.0 + float(np.sum(arr <= 0.0))) / (1.0 + len(arr))
    return d_b, d_a, float(d_b - d_a), float(lo), float(hi), float(p), int(len(merged))


def run_secondary(raw, spec: dict, packs_b0: dict, cache_exp: str) -> pd.DataFrame:
    """Real secondary tests. Holm is applied inside each pre-registered family."""
    rows = []
    horizons = list(spec["secondary_horizons"])
    _log(f"secondary grid horizons {horizons}")
    for h in horizons:
        oos = collect_oos(
            packs_b0, spec, horizon=int(h), label="point", models={m: [m] for m in MEASURES},
        )
        for meas in MEASURES:
            rows.append(
                _contrast_row(
                    "measures_x_horizon", f"{meas}_h{h}", oos.get(meas), spec,
                    measure=meas, horizon=h, label="v_at_t_plus_h", preprocess="B0",
                    m=1, tau=1, seed=7000 + int(h) * 17 + sum(ord(c) for c in meas),
                )
            )
    if _pywt_available():
        packs_b1 = build_packs(raw, spec, cache_exp, preprocess="window_swt", embedding="primary", include_lrqa=False)
        oos_b1 = collect_oos(packs_b1, spec, horizon=H_PRIMARY, label="point", models={"DET_LAM": ["DET_LAM"]})
        rows.append(
            _contrast_row(
                "protocol_variants", "B1_DET_LAM_h42", oos_b1.get("DET_LAM"), spec,
                measure="DET_LAM", horizon=H_PRIMARY, label="v_at_t_plus_h", preprocess="B1",
                m=1, tau=1, seed=7101,
            )
        )
    else:
        rows.append(
            {
                "family": "protocol_variants", "test": "B1_DET_LAM_h42", "measure": "DET_LAM",
                "horizon": H_PRIMARY, "label": "v_at_t_plus_h", "preprocess": "B1",
                "m": 1, "tau": 1, "delta_auc": np.nan, "auc_M0": np.nan, "auc_M1": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "p_raw": np.nan, "n": 0,
                "fnn_m_by_asset": "not_run (dependency)",
            }
        )
    oos_any = collect_oos(
        packs_b0, spec, horizon=H_PRIMARY, label="any_day", models={"DET_LAM": ["DET_LAM"]},
    )
    rows.append(
        _contrast_row(
            "protocol_variants", "any_day_DET_LAM_h42", oos_any.get("DET_LAM"), spec,
            measure="DET_LAM", horizon=H_PRIMARY, label="any_day_in_(t,t+h]", preprocess="B0",
            m=1, tau=1, seed=7102,
        )
    )
    packs_m3 = build_packs(raw, spec, cache_exp, preprocess="window_scale", embedding="m3", include_lrqa=False)
    fnn_txt = ",".join(f"{t}:{packs_m3[t].get('fnn_m')}" for t in packs_m3)
    tau_txt = ",".join(f"{t}:{packs_m3[t].get('tau')}" for t in packs_m3)
    oos_m3 = collect_oos(packs_m3, spec, horizon=H_PRIMARY, label="point", models={"DET_LAM": ["DET_LAM"]})
    row_m = _contrast_row(
        "protocol_variants", "m3_DET_LAM_h42", oos_m3.get("DET_LAM"), spec,
        measure="DET_LAM", horizon=H_PRIMARY, label="v_at_t_plus_h", preprocess="B0",
        m=3, tau=tau_txt, seed=7103, extra=fnn_txt,
    )
    rows.append(row_m)
    oos_cmp = collect_oos(
        packs_b0, spec, horizon=H_PRIMARY, label="point",
        models={"window": ["DET_LAM"], "lrqa": LRQA_COLS},
    )
    d_lrqa, d_win, diff, lo, hi, p, n = _delta_of_deltas(oos_cmp.get("window", pd.DataFrame()), oos_cmp.get("lrqa", pd.DataFrame()), spec, 7104)
    rows.append(
        {
            "family": "protocol_variants",
            "test": "lrqa_minus_window",
            "measure": "LRQA_vs_DET_LAM",
            "horizon": H_PRIMARY,
            "label": "v_at_t_plus_h",
            "preprocess": "B0",
            "m": 1,
            "tau": 1,
            "delta_auc": diff,
            "auc_M0": d_win,
            "auc_M1": d_lrqa,
            "ci_low": lo,
            "ci_high": hi,
            "p_raw": p,
            "n": n,
            "fnn_m_by_asset": "auc_M0 is delta(window); auc_M1 is delta(LRQA); delta_auc is LRQA-window",
        }
    )
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _fam, idx in df.groupby("family").groups.items():
        p = df.loc[idx, "p_raw"].to_numpy(float)
        ok = np.isfinite(p)
        adj = np.full(len(p), np.nan)
        if ok.any():
            adj[np.flatnonzero(ok)] = holm(p[ok])
        df.loc[idx, "p_holm"] = adj
    return df


def _arm_a_verdict(grid: pd.DataFrame, spec: dict) -> tuple[str, dict]:
    if grid is None or grid.empty:
        return "not_run (dependency)", {}
    sub = grid[(grid["measure"] == spec["arm_a_repro_measure"]) & (grid["horizon"] == spec["arm_a_repro_h"])]
    if sub.empty:
        return "not reproduced", {}
    row = sub.iloc[0]
    auc_ok = np.isfinite(row["auc"]) and abs(float(row["auc"]) - spec["arm_a_repro_auc_ref"]) <= spec["arm_a_repro_auc_tol"]
    beta_ok = (
        np.isfinite(row["beta_nw"]) and float(row["beta_nw"]) > 0
        and np.isfinite(row["t_nw"]) and abs(float(row["t_nw"])) > 2
    )
    return ("reproduced" if auc_ok and beta_ok else "not reproduced"), row.to_dict()


def _shift_values(dates, values, days: int) -> np.ndarray:
    dates = pd.to_datetime(np.asarray(dates))
    values = np.asarray(values, float)
    order = np.argsort(dates.to_numpy())
    ordered_dates = dates.to_numpy()[order]
    ordered = values[order]
    target = (dates - pd.Timedelta(days=int(days))).to_numpy()
    pos = np.searchsorted(ordered_dates, target, side="right") - 1
    out = np.full(len(values), np.nan)
    good = pos >= 0
    out[np.flatnonzero(good)] = ordered[pos[good]]
    return out


def _point_delta(frame: pd.DataFrame) -> float:
    if frame is None or len(frame) == 0:
        return np.nan
    return _auc(frame["y"], frame["p1"]) - _auc(frame["y"], frame["p0"])


def _event_weighted(frame: pd.DataFrame, col: str) -> float:
    if frame is None or len(frame) == 0 or col not in frame.columns:
        return np.nan
    score = frame[col].to_numpy(float)
    weight = frame["n_events"].to_numpy(float)
    ok = np.isfinite(score) & np.isfinite(weight) & (weight > 0)
    if not ok.any():
        return np.nan
    return float(np.average(score[ok], weights=weight[ok]))


def run_diagnostics(packs: dict, spec: dict, observed_delta: float) -> dict:
    """Placebo shifts and within-ticker-year AUCs. Neither one is the pass rule."""
    cfg = spec["diagnostics"]
    n_shifts = int(cfg["placebo_shifts"])
    placebo_rows = []
    for i in range(n_shifts):
        days = int(cfg["placebo_min_days"]) + i * int(cfg["placebo_spacing_days"])
        shifted = {}
        for ticker, pack in packs.items():
            frame = pack["frame"].copy()
            if len(frame) and "DET_LAM" in frame.columns:
                frame["DET_LAM"] = _shift_values(frame["date"], frame["DET_LAM"], days)
            shifted[ticker] = {**pack, "frame": frame}
        scored = collect_oos(
            shifted, spec, horizon=int(spec["arm_b_horizon_primary"]), label="point", models={"DET_LAM": ["DET_LAM"]}
        )
        delta = _point_delta(scored.get("DET_LAM"))
        placebo_rows.append({"shift_days": days, "delta_auc": delta})
        _log(f"placebo shift {days}d delta={delta}")
    placebo = pd.DataFrame(placebo_rows)
    finite = placebo["delta_auc"].to_numpy(float)
    finite = finite[np.isfinite(finite)]
    observed = float(observed_delta) if observed_delta is not None else np.nan
    share = float(np.mean(finite >= observed)) if len(finite) and np.isfinite(observed) else np.nan
    scored = collect_oos(
        packs, spec, horizon=int(spec["arm_b_horizon_primary"]), label="point", models={"DET_LAM": ["DET_LAM"]}
    )
    base = scored.get("DET_LAM", pd.DataFrame())
    year_rows = []
    if len(base):
        for (ticker, year), g in base.groupby(["ticker", "year"], sort=True):
            year_rows.append(
                {
                    "ticker": ticker,
                    "year": int(year),
                    "n": int(len(g)),
                    "n_events": int(np.nansum(g["y"].to_numpy(float))),
                    "auc_M0": _auc(g["y"], g["p0"]),
                    "auc_M1": _auc(g["y"], g["p1"]),
                    "auc_vol22": _auc(g["y"], g["vol22"]),
                }
            )
    within = pd.DataFrame(year_rows)
    summary = {
        "n_placebo": int(len(finite)),
        "placebo_mean": float(np.mean(finite)) if len(finite) else np.nan,
        "placebo_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
        "placebo_share_ge_observed": share,
        "observed_delta": observed,
        "auc_M0": _event_weighted(within, "auc_M0"),
        "auc_M1": _event_weighted(within, "auc_M1"),
        "auc_vol22": _event_weighted(within, "auc_vol22"),
    }
    return {"placebo": placebo, "within_year": within, "summary": summary}


def _fmt(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{float(x):.4f}"


def build_decision(spec, arm_a, arm_b, arm_c, *, pywt_ok: bool, diagnostics=None) -> list[dict]:
    arm_a_verdict, arm_a_row = _arm_a_verdict(arm_a, spec)
    if not pywt_ok:
        arm_a_verdict = "not_run (dependency)"
    rows = [
        {
            "gate": "arm_A_reproduction",
            "in_primary": False,
            "asset": "BTC-USD",
            "horizon": spec["arm_a_repro_h"],
            "statistic": "DET_LAM_auc",
            "value": arm_a_row.get("auc", np.nan),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "rule": "not in primary; reproduced if AUC within ±0.05 of 0.605 and beta>0 with |t|>2",
            "verdict": arm_a_verdict,
        }
    ]
    delta = ci_lo = ci_hi = np.nan
    if arm_c is not None and len(arm_c) and "panel_delta_auc" in arm_c.columns:
        delta = float(arm_c["panel_delta_auc"].iloc[0])
        ci_lo = float(arm_c["ci_low"].iloc[0])
        ci_hi = float(arm_c["ci_high"].iloc[0])
    survives = bool(np.isfinite(delta) and delta > 0 and np.isfinite(ci_lo) and ci_lo > 0)
    bc = "precedent_survives" if survives else "precedent_fails"
    rows.append(
        {
            "gate": "primary",
            "in_primary": True,
            "asset": "BTC-USD" if spec["smoke"] else "panel",
            "horizon": spec["arm_b_horizon_primary"],
            "statistic": "delta_auc_M1_M0",
            "value": delta,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "rule": spec["primary_test"]["pass_rule"],
            "verdict": bc,
        }
    )
    if arm_a_verdict == "reproduced" and survives:
        reading = "precedent confirmed under controls"
    elif arm_a_verdict == "reproduced":
        reading = "effect depends on look-ahead preprocessing / selection / missing baseline"
    elif arm_a_verdict.startswith("not_run"):
        reading = "arm A not run (dependency); primary is arms B/C only"
    else:
        reading = "not reproducible from the paper's description"
    cite = "diagnostics not run"
    if diagnostics is not None:
        s = diagnostics["summary"]
        cite = (
            f"placebo n={s['n_placebo']} mean={_fmt(s['placebo_mean'])} sd={_fmt(s['placebo_sd'])} "
            f"share_ge_observed={_fmt(s['placebo_share_ge_observed'])}; "
            f"within-year event-weighted AUC M0={_fmt(s['auc_M0'])} M1={_fmt(s['auc_M1'])} "
            f"vol22={_fmt(s['auc_vol22'])}; diagnostics do not change the pass rule"
        )
        diag_defs = [
            ("placebo", "placebo_mean", s["placebo_mean"],
             ">=30 shifts of >=365 days; diagnostic only, not the pass rule"),
            ("placebo", "placebo_sd", s["placebo_sd"],
             ">=30 shifts of >=365 days; diagnostic only, not the pass rule"),
            ("placebo", "placebo_share_ge_observed", s["placebo_share_ge_observed"],
             ">=30 shifts of >=365 days; diagnostic only, not the pass rule"),
            ("within_year", "within_year_auc_M0", s["auc_M0"],
             "event-weighted within ticker-year AUC; diagnostic only, not the pass rule"),
            ("within_year", "within_year_auc_M1", s["auc_M1"],
             "event-weighted within ticker-year AUC; diagnostic only, not the pass rule"),
            ("within_year", "within_year_auc_vol22", s["auc_vol22"],
             "event-weighted within ticker-year AUC; diagnostic only, not the pass rule"),
        ]
        for gate, statistic, value, rule in diag_defs:
            rows.append(
                {
                    "gate": gate,
                    "in_primary": False,
                    "asset": "panel",
                    "horizon": spec["arm_b_horizon_primary"],
                    "statistic": statistic,
                    "value": value,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "rule": rule,
                    "verdict": "diagnostic",
                }
            )
    rows.append(
        {
            "gate": "reading",
            "in_primary": False,
            "asset": "",
            "horizon": spec["arm_b_horizon_primary"],
            "statistic": "A_and_BC",
            "value": delta,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "rule": "plan table: A reproduced x B/C survives; A is not the primary test; " + cite,
            "verdict": reading,
        }
    )
    if spec["smoke"]:
        for r in rows:
            r["smoke"] = True
    if sum(bool(r["in_primary"]) for r in rows) != 1:
        raise RuntimeError("decision must have exactly one primary row")
    return rows


ASSUMPTIONS_LINES = [
    "SWT level is min(pywt.swt_max_level(N), 5).",
    "sigma_w is the MAD of the finest sym8 detail divided by 0.6745; details are soft-thresholded.",
    "Baseline removal zeros the coarsest db11 approximation (swt list entry 0), which is the one iswt uses.",
    "If N is odd the oldest return is dropped so SWT length is even; windows stay aligned to the series end.",
    "epsilon is recplay threshold_by_rate at RR=10%; the paper's +/-0.1% band is not a second search.",
    "l_min = v_min = 2. ENTR_v is the natural-log Shannon entropy of vertical lines l>=2, matching recplay ENTR.",
    "TT is the mean vertical line length among lines with l>=2.",
    "Volatility v is the 5-day standard deviation of raw returns, not of the wavelet series.",
    "Arm A label: v at t+h above the 95th percentile of v on origins inside 2022-02-10..2023-01-10.",
    "Arm A origins lie in that span; v at t+h may fall after 2023-01-10. The threshold does not use those later days.",
    "Arm B/C threshold is the 95th percentile of v on training dates only, compared with v at t+h.",
    "Any-day secondary label: max of v on (t, t+h] above that same training percentile of v.",
    "Newey-West slope uses an intercept and Bartlett lags equal to the horizon h.",
    "Arm A p-values permute labels (iid, and non-overlapping blocks of 33 and 66). bootstrap_p is the 33-day one. They are not in the reproduction rule.",
    "Returns are yfinance daily closes, not CoinLore. Crypto uses Close; indices use Adj Close.",
    "Crypto prices are snapshot e6_crypto_top50_v3 from 2014-09-01. Tickers that came back empty are tried once more in e6_crypto_top50_v3_retry, which does not replace the main file. Index prices are snapshot e6_indices_2000 from 2000-01-01. None of these files is refreshed, and e6_crypto_equity is left untouched.",
    "Shared quality_filter would drop series shorter than 12 years. E6 cleans degenerate bars locally and keeps cryptos with at least 5 years and 1460 closes.",
    "[-1,1] scaling is applied in e6: on the whole series before arm A SWT, and inside each arm B window. There is no Geometry.scale_window patch and no second z-score.",
    "Arm A full-series SWT looks ahead. Its leakage result is expected to fail and arm A is not in the primary test.",
    "B0 scales each window to [-1,1] only. B1 runs the same SWT inside the 80-day window.",
    "Primary embedding is m=1, tau=1. The m=3 secondary uses AMI tau chosen on returns before the first test year; FNN m is recorded and not substituted for 3.",
    "Smoke uses step 10, arm A horizons 5 and 42, first test year 2022, and fewer bootstrap draws. Full uses step 1 and horizons 5..50.",
    "The crypto panel is BTC-USD plus the 8 largest other names from the CoinGecko top 50 after dropping stablecoins and wrapped or staked tokens, among names with at least 5 years and 1460 closes. Indices use their own sample start of 2000-01-01. The full run's first test year is 2018 for every ticker.",
    "Horizon and window lengths are observations: calendar days for crypto and trading days for indices.",
    "Placebo shifts (30, each at least 365 days) and within-ticker-year event-weighted AUCs are reported on the reading row. They are not the pass rule.",
    "The leakage check calls _rows_for_ends and drops future labels before scoring. Arm A is still expected to fail because the full-series transform sees the future.",
    "The Holm family measures_x_horizon is the six measures times the secondary horizons, arm B protocol, frozen panel, label v at t+h.",
    "B1, the any-day label, m=3, and LRQA minus the window ratio are a second Holm family (protocol_variants).",
    "Date-block bootstrap draws chronological unique dates in blocks of 66, keeping every asset on a drawn date. One-sided p is (1+#{delta<=0})/(1+B).",
    "M0 is vol5, vol22, vol80, log(vol22/vol80), and the share of the last 80 days with v above the training threshold. Logistic regression is median-imputed, standardized, and class-balanced.",
    "Backend is numpy. joblib uses 4 workers.",
]


def write_assumptions(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assumptions.md").write_text("\n".join(f"- {ln}" for ln in ASSUMPTIONS_LINES) + "\n", encoding="utf-8")


def make_figures(arm_a, arm_b, fig_dir, arm_c=None) -> None:
    from condpred import report

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if arm_a is not None and len(arm_a):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for meas, g in arm_a.groupby("measure"):
            g = g.sort_values("horizon")
            ax.plot(g["horizon"], g["auc"], marker="o", ms=3, label=meas)
        ax.axhline(0.605, ls="--", color="gray", label="paper 0.605")
        ax.set_xlabel("horizon (observations)")
        ax.set_ylabel("AUC")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        report.save_fig(fig, fig_dir, "arm_a_auc_vs_horizon")
        plt.close(fig)
    if arm_b is not None and len(arm_b) and "delta_auc" in arm_b.columns:
        sub = arm_b[arm_b["row_type"] == "asset"].copy() if "row_type" in arm_b.columns else arm_b.copy()
        if arm_c is not None and len(arm_c) and "panel_delta_auc" in arm_c.columns:
            sub = pd.concat(
                [
                    sub,
                    pd.DataFrame(
                        [
                            {
                                "ticker": "PANEL",
                                "delta_auc": arm_c["panel_delta_auc"].iloc[0],
                                "ci_low": arm_c["ci_low"].iloc[0],
                                "ci_high": arm_c["ci_high"].iloc[0],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        if len(sub):
            fig, ax = plt.subplots(figsize=(7, max(3.0, 0.35 * len(sub) + 1)))
            y = np.arange(len(sub))
            delta = sub["delta_auc"].to_numpy(float)
            lo = np.maximum(0.0, delta - sub["ci_low"].to_numpy(float))
            hi = np.maximum(0.0, sub["ci_high"].to_numpy(float) - delta)
            ax.barh(y, delta, xerr=[lo, hi], color="C0", alpha=0.85, capsize=2)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["ticker"])
            ax.axvline(0, color="k", lw=0.6)
            ax.set_xlabel("delta AUC (M1 - M0)")
            fig.tight_layout()
            report.save_fig(fig, fig_dir, "arm_bc_delta_auc")
            plt.close(fig)


def _causal_features(arr: np.ndarray, t: int, *, lookahead: bool) -> dict:
    x = np.asarray(arr, float)
    v = pd.Series(x).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().to_numpy(float)
    series = x
    if lookahead:
        series = _preprocess_full(x) if _pywt_available() else _scale_minus1_1(x)
        if len(series) != len(x):
            series = np.asarray(series, float)
            series = series[-len(x) :] if len(series) > len(x) else np.pad(series, (len(x) - len(series), 0))
    dates = pd.date_range("2010-01-01", periods=len(x), freq="D").to_numpy()
    rows = _rows_for_ends(series, v, dates, [int(t)], "window_scale", M_EMB, TAU, True, [5])
    if not rows:
        return {}
    row = rows[0]
    hist = v[: int(t) + 1]
    hist = hist[np.isfinite(hist)]
    thr = float(np.quantile(hist, 0.95)) if len(hist) >= 20 else np.nan
    out = {}
    for key, val in row.items():
        if key in ("date", "year", "t") or str(key).startswith("vol_h") or str(key).startswith("vmax_"):
            continue
        out[key] = val
    out["recent_exceed"] = _recent_exceed(v, int(t), thr) if np.isfinite(thr) else np.nan
    return out


def run_leakage_e6(rng: np.random.Generator) -> list[dict]:
    n = 400
    x = rng.standard_normal(n)
    t_list = list(range(W_RQA + 10, n - 20, 30))

    def arm_a_fn(arr, t):
        return _causal_features(arr, t, lookahead=True)

    def arm_b_fn(arr, t):
        return _causal_features(arr, t, lookahead=False)

    rows = [
        {
            "pipeline": "arm_A_lookahead",
            "pass": bool(leakage_test(arm_a_fn, x, t_list, rng)),
            "expected_pass": False,
            "in_primary": False,
            "note": "expected failure: full-series scale and SWT use future returns; features come from _rows_for_ends; not in the primary",
        },
        {
            "pipeline": "arm_B_causal",
            "pass": bool(leakage_test(arm_b_fn, x, t_list, rng)),
            "expected_pass": True,
            "in_primary": False,
            "note": "_rows_for_ends vol22, vol80, log-ratio, LRQA, and recent exceedance; future labels dropped",
        },
    ]
    for r in rows:
        r["verdict"] = "expected" if r["pass"] == r["expected_pass"] else "unexpected"
    return rows

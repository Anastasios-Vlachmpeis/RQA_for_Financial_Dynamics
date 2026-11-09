"""Data readers and snapshots (network downloads only when refresh=True)."""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_RAW = _REPO / "data" / "raw"

_OM_URL = (
    "https://raw.githubusercontent.com/jonathancornelissen/highfrequency/master/"
    "data-raw/oxfordmanrealizedvolatilityindices.csv"
)
_BINANCE_BASE = "https://data.binance.vision/data/spot/monthly/klines"
_BINANCE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
_ETF_TICKERS = {
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "GLD",
    "SLV",
    "USO",
    "XLE",
    "XLF",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_yahoo_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            return df
    except (ValueError, pd.errors.ParserError):
        pass
    return pd.read_csv(path, index_col=0, parse_dates=True)


def yahoo_ohlc_snapshot(tickers, start, end, name, refresh=False) -> pd.DataFrame:
    folder = _RAW / "yahoo"
    folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(folder.glob(f"condpred_{name}_*.csv"))
    existing = [p for p in existing if "manifest" not in p.name]
    if existing and not refresh:
        return _read_yahoo_csv(existing[-1])
    import yfinance as yf

    data = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker", progress=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = folder / f"condpred_{name}_{stamp}.csv"
    data.to_csv(path)
    digest = _sha256_file(path)
    manifest = {
        "tickers": list(tickers),
        "start": str(start),
        "end": str(end),
        "download_time": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "sha256": digest,
    }
    path.with_name(path.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return _read_yahoo_csv(path)


def _ticker_ohlc(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in df.columns.get_level_values(0):
            return None
        sub = df[ticker].copy()
        sub.columns = [str(c) for c in sub.columns]
    else:
        sub = df.copy()
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(sub.columns):
        return None
    vol_col = "Volume" if "Volume" in sub.columns else None
    return sub[list(need) + ([vol_col] if vol_col else [])]


def _degenerate_mask(sub: pd.DataFrame) -> pd.Series:
    o, h, l, c = sub["Open"], sub["High"], sub["Low"], sub["Close"]
    bad = (h == l) | ((o == h) & (h == l) & (l == c)) | (h < np.maximum(o, c)) | (l > np.minimum(o, c))
    return bad.fillna(True)


def _weekday_missing_fracs(dates: pd.Series) -> tuple[float, float]:
    """Raw vs outage missing shares on Mon–Fri between min and max observed dates."""
    dts = pd.to_datetime(dates).dt.normalize()
    if dts.empty:
        return 1.0, 1.0
    bdays = pd.bdate_range(dts.min(), dts.max())
    present = set(dts)
    missing = np.array([d not in present for d in bdays], dtype=bool)
    n = len(bdays)
    missing_frac_raw = float(missing.sum() / n) if n else 1.0
    outage = 0
    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        if j - i >= 3:
            outage += j - i
        i = j
    missing_frac_outage = float(outage / n) if n else 1.0
    return missing_frac_raw, missing_frac_outage


def _binance_open_time_to_ms(values: pd.Series) -> pd.Series:
    t = pd.to_numeric(values, errors="coerce")
    if t.notna().any() and float(t.max()) > 1e14:
        return (t // 1000).astype(np.int64)
    return t.astype(np.int64)


def quality_filter(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("quality_filter expects Yahoo MultiIndex OHLC columns")
    tickers = df.columns.get_level_values(0).unique().tolist()
    reports = []
    parts = []
    for ticker in tickers:
        sub = _ticker_ohlc(df, ticker)
        if sub is None or sub.empty:
            continue
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        bad = _degenerate_mask(sub)
        if "Volume" in sub.columns and ticker in _ETF_TICKERS:
            bad |= sub["Volume"].fillna(0) <= 0
        idx = sub.index
        gap_bad = pd.Series(False, index=idx)
        if len(idx) > 1:
            gaps = idx.to_series().diff().dt.days
            gap_bad = gaps > 5
        bad_row = bad | gap_bad
        start_year = None
        for year, g in sub.assign(bad=bad_row).groupby(sub.index.year):
            if g["bad"].mean() < 0.01:
                start_year = year
                break
        if start_year is None:
            reports.append(
                {
                    "ticker": ticker,
                    "degenerate_frac": float(bad.mean()),
                    "start_year": None,
                    "years": 0.0,
                    "dropped": True,
                    "reason": "no_year_under_1pct_degenerate",
                }
            )
            continue
        window = sub.loc[sub.index.year >= start_year]
        bad_win = bad_row.reindex(window.index, fill_value=False)
        deg_frac = float(bad.reindex(window.index, fill_value=True).mean()) if len(window) else np.nan
        kept = window.copy()
        price_cols = [c for c in kept.columns if c in {"Open", "High", "Low", "Close", "Volume", "Adj Close"}]
        kept.loc[bad_win, price_cols] = np.nan
        years = (kept.index.max() - kept.index.min()).days / 365.25 if len(kept) else 0.0
        dropped = years < 12
        reports.append(
            {
                "ticker": ticker,
                "degenerate_frac": deg_frac,
                "start_year": start_year,
                "years": float(years),
                "dropped": dropped,
                "reason": "short_history" if dropped else "",
            }
        )
        if not dropped and len(kept):
            block = kept.rename(columns=lambda c: (ticker, c))
            block.columns = pd.MultiIndex.from_tuples(block.columns)
            parts.append(block)
    report = pd.DataFrame(reports)
    if not parts:
        return df.iloc[:, :0], report
    clean = pd.concat(parts, axis=1).sort_index()
    return clean, report


def variance_proxy(df, kind="gk_overnight") -> pd.Series:
    if kind == "close_sq":
        c = df["Adj Close"] if "Adj Close" in df else df["Close"]
        prev = c.shift(1)
        lr = np.log(c / prev)
        out = (lr**2).where(c.notna() & prev.notna())
        return out.clip(lower=1e-10)
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    if kind == "parkinson":
        hl = np.log(h / l)
        out = (hl**2 / (4 * np.log(2))).where(h.notna() & l.notna() & (l > 0))
        return out.clip(lower=1e-10)
    prev_c = c.shift(1)
    co = np.log(o / prev_c)
    valid = o.notna() & h.notna() & l.notna() & c.notna() & prev_c.notna() & (o > 0) & (l > 0) & (c > 0)
    v = co**2 + 0.5 * (np.log(h / l) ** 2) - (2 * np.log(2) - 1) * (np.log(c / o) ** 2)
    return v.where(valid).clip(lower=1e-10)


def log_returns(df) -> pd.Series:
    c = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    return np.log(c / c.shift(1))


def fred_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def oecd_mei(path) -> pd.DataFrame:
    return pd.read_csv(path)


def realized_from_intraday(bars) -> tuple[float, float, float]:
    r = np.asarray(bars, float).ravel()
    rv = float(np.sum(r**2))
    bv = float((np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    rq = float((len(r) / 3) * np.sum(r**4))
    return rv, bv, rq


def oxford_man(refresh=False) -> pd.DataFrame:
    folder = _RAW / "oxford_man"
    folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(folder.glob("oxfordmanrealizedvolatilityindices_*.csv"))
    manifest_path = folder / "manifest.json"
    if existing and not refresh:
        path = existing[-1]
        digest = _sha256_file(path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert digest == manifest["sha256"], "Oxford-Man SHA-256 mismatch"
        else:
            raise FileNotFoundError("Oxford-Man manifest missing; pass refresh=True to download")
        df = pd.read_csv(path)
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
            df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10), format="%Y-%m-%d")
        assert not df.duplicated(["Symbol", "date"]).any()
        return df

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = folder / f"oxfordmanrealizedvolatilityindices_{stamp}.csv"
    with urlopen(_OM_URL) as resp:
        path.write_bytes(resp.read())
    digest = _sha256_file(path)
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10), format="%Y-%m-%d")
    assert not df.duplicated(["Symbol", "date"]).any()
    manifest = {
        "url": _OM_URL,
        "download_time": datetime.now(timezone.utc).isoformat(),
        "sha256": digest,
        "rows": len(df),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "symbols": sorted(df["Symbol"].unique().tolist()) if "Symbol" in df.columns else [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return df


def om_quality(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Symbol" not in df.columns:
        raise ValueError("expected long Oxford-Man format with Symbol column")
    reports = []
    kept_parts = []
    for sym, g in df.groupby("Symbol"):
        g = g.sort_values("date").copy()
        rv = g["rv5"] if "rv5" in g.columns else g.get("rv_5", pd.Series(np.nan, index=g.index))
        g = g[(rv > 0) & rv.notna()]
        if g.empty:
            reports.append(
                {
                    "symbol": sym,
                    "missing_frac_raw": 1.0,
                    "missing_frac_outage": 1.0,
                    "years": 0.0,
                    "kept": False,
                    "drop_reason": "no_valid_rv5",
                }
            )
            continue
        nobs = g["nobs"] if "nobs" in g.columns else pd.Series(np.nan, index=g.index)
        low_nobs = pd.Series(False, index=g.index)
        if nobs.notna().any():
            thr = float(nobs.quantile(0.05))
            low_nobs = nobs < thr
        g = g.assign(low_nobs=low_nobs.astype(bool))
        miss_raw, miss_out = _weekday_missing_fracs(g["date"])
        years = (g["date"].max() - g["date"].min()).days / 365.25
        reasons = []
        if years < 12:
            reasons.append("short_history")
        if miss_out > 0.05:
            reasons.append("high_outage")
        if miss_raw > 0.10:
            reasons.append("scattered_missing")
        ok = not reasons
        drop_reason = "" if ok else ";".join(reasons)
        reports.append(
            {
                "symbol": sym,
                "missing_frac_raw": miss_raw,
                "missing_frac_outage": miss_out,
                "years": float(years),
                "kept": ok,
                "drop_reason": drop_reason,
            }
        )
        if ok:
            kept_parts.append(g)
    report = pd.DataFrame(reports)
    clean = pd.concat(kept_parts, ignore_index=True) if kept_parts else df.iloc[:0]
    return clean, report


def _month_range(start: str, end: str):
    s = pd.Timestamp(start).to_period("M")
    e = pd.Timestamp(end).to_period("M")
    cur = s
    while cur <= e:
        yield cur.year, f"{cur.month:02d}"
        cur += 1


def _verify_binance_checksum(data: bytes, checksum_url: str) -> None:
    with urlopen(checksum_url, timeout=120) as resp:
        line = resp.read().decode("utf-8").strip().split()
    expected = line[0].lower()
    got = _sha256_bytes(data).lower()
    if expected != got:
        raise ValueError(f"Binance checksum mismatch: expected {expected[:12]}… got {got[:12]}…")


def binance_klines(symbols, start, end, interval="1m", refresh=False) -> pd.DataFrame:
    """Load or fetch Binance spot klines (monthly zips); checksum verified."""
    folder = _RAW / "binance"
    folder.mkdir(parents=True, exist_ok=True)
    sym_key = "-".join(symbols)
    cache_path = folder / f"klines_{sym_key}_{start}_{end}_{interval}.parquet"
    manifest_path = folder / f"klines_{sym_key}_{start}_{end}_{interval}_manifest.json"
    if cache_path.exists() and manifest_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    frames = []
    files_meta = []
    for symbol in symbols:
        for year, month in _month_range(start, end):
            zip_name = f"{symbol}-{interval}-{year}-{month}.zip"
            url = f"{_BINANCE_BASE}/{symbol}/{interval}/{zip_name}"
            checksum_url = f"{url}.CHECKSUM"
            local_zip = folder / zip_name
            if local_zip.exists() and not refresh:
                data = local_zip.read_bytes()
            else:
                try:
                    with urlopen(url, timeout=120) as resp:
                        data = resp.read()
                except Exception as exc:
                    if "404" in str(exc):
                        continue
                    raise
                _verify_binance_checksum(data, checksum_url)
                local_zip.write_bytes(data)
            files_meta.append({"symbol": symbol, "file": zip_name, "sha256": _sha256_bytes(data)})
            with zipfile.ZipFile(BytesIO(data)) as zf:
                member = zf.namelist()[0]
                with zf.open(member) as f:
                    chunk = pd.read_csv(f, header=None, names=_BINANCE_COLS)
            chunk["symbol"] = symbol
            chunk["open_time"] = _binance_open_time_to_ms(chunk["open_time"])
            if "close_time" in chunk.columns:
                chunk["close_time"] = _binance_open_time_to_ms(chunk["close_time"])
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=_BINANCE_COLS + ["symbol"])
    out = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["open_time"] = pd.to_numeric(out["open_time"], errors="coerce")
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC")
    ts = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out = out[(ts >= t0) & (ts < t1)]
    out.to_parquet(cache_path, index=False)
    manifest = {
        "symbols": list(symbols),
        "start": str(start),
        "end": str(end),
        "interval": interval,
        "download_time": datetime.now(timezone.utc).isoformat(),
        "files": files_meta,
        "rows": len(out),
        "parquet_sha256": _sha256_file(cache_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def to_5min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    agg = df.groupby("symbol").resample("5min").agg({"close": "last", "open": "first", "high": "max", "low": "min"})
    return agg.dropna(subset=["close"])

"""Deterministic CPU generators for recurrence-learning benchmark tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TASKS: dict[str, str] = {
    "system": "cls",
    "lyapunov": "reg",
    "noise": "reg",
    "changepoint": "reg",
    "recall": "reg",
    "forecast": "reg",
}
SYSTEM_CLASSES = (
    "white_noise", "ar1", "sine", "logistic", "henon", "lorenz",
    "rossler", "garch_abs", "regime_ar", "ft_lorenz", "ft_logistic",
)


@dataclass
class TaskData:
    """A generated task split: x is [n,L], y is [n], and meta has n rows."""

    x: np.ndarray
    y: np.ndarray
    kind: str
    n_classes: int | None
    meta: pd.DataFrame


def window_length(n_states: int, m: int, tau: int) -> int:
    """Return the raw window length that embeds to exactly ``n_states`` states."""
    if n_states < 1 or m < 1 or tau < 1:
        raise ValueError("n_states, m, and tau must be positive")
    return int(n_states + (m - 1) * tau)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return ((x - mu) / sd).astype(np.float32)


def _logistic_batch(r: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(r)
    burn = rng.integers(500, 1201, size=n)
    state = rng.uniform(0.15, 0.85, size=n)
    out = np.empty((n, length), dtype=np.float64)
    for t in range(int(burn.max()) + length):
        state = r * state * (1.0 - state)
        rows = np.flatnonzero((t >= burn) & (t < burn + length))
        if rows.size:
            out[rows, t - burn[rows]] = state[rows]
    return out


def _henon_batch(a: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(a)
    burn = rng.integers(500, 1201, size=n)
    x = 0.1 + rng.normal(0.0, 0.01, size=n)
    y = 0.1 + rng.normal(0.0, 0.01, size=n)
    out = np.empty((n, length), dtype=np.float64)
    for t in range(int(burn.max()) + length):
        xn = 1.0 - a * x * x + y
        y = 0.3 * x
        x = xn
        rows = np.flatnonzero((t >= burn) & (t < burn + length))
        if rows.size:
            out[rows, t - burn[rows]] = x[rows]
    return out


def _lorenz_rhs(s: np.ndarray) -> np.ndarray:
    x, y, z = s.T
    return np.column_stack((10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z))


def _rossler_rhs(s: np.ndarray) -> np.ndarray:
    x1, y1, z1, x2, y2, z2 = s.T
    w1, w2 = 1.02, 0.98
    mu = 0.04
    return np.column_stack((
        -w1 * y1 - z1 + mu * (x2 - x1), w1 * x1 + 0.2 * y1, 0.2 + z1 * (x1 - 5.7),
        -w2 * y2 - z2 + mu * (x1 - x2), w2 * x2 + 0.2 * y2, 0.2 + z2 * (x2 - 5.7),
    ))


def _rk4_batch(s: np.ndarray, rhs, h: float, substeps: int) -> np.ndarray:
    h = h / substeps
    for _ in range(substeps):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * h * k1)
        k3 = rhs(s + 0.5 * h * k2)
        k4 = rhs(s + h * k3)
        s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return s


def _flow_batch(family: str, n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    if family == "lorenz":
        burn = rng.integers(700, 1201, size=n)
        state = rng.normal(0.0, 0.15, (n, 3)) + np.array([1.0, 1.0, 20.0])
        rhs, dt, sub, col = _lorenz_rhs, 0.05, 2, 0
    else:
        burn = rng.integers(350, 601, size=n)
        state = rng.normal(0.0, 0.15, (n, 6)) + np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
        rhs, dt, sub, col = _rossler_rhs, 0.1, 2, 0
    out = np.empty((n, length), dtype=np.float64)
    for t in range(int(burn.max()) + length):
        state = _rk4_batch(state, rhs, dt, sub)
        rows = np.flatnonzero((t >= burn) & (t < burn + length))
        if rows.size:
            out[rows, t - burn[rows]] = state[rows, col]
    return out


def _ar_batch(phi: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(phi)
    burn = rng.integers(100, 301, size=n)
    total = int(burn.max()) + length
    e = rng.standard_normal((n, total))
    x = np.zeros(n, dtype=np.float64)
    out = np.empty((n, length), dtype=np.float64)
    for t in range(total):
        x = phi * x + e[:, t]
        rows = np.flatnonzero((t >= burn) & (t < burn + length))
        if rows.size:
            out[rows, t - burn[rows]] = x[rows]
    return out


def _garch_abs_batch(persistence: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(persistence)
    alpha = 0.08
    beta = np.maximum(0.0, persistence - alpha)
    omega = np.maximum(0.005, 1.0 - persistence)
    burn = rng.integers(300, 601, size=n)
    total = int(burn.max()) + length
    z = rng.standard_normal((n, total))
    variance = np.ones(n, dtype=np.float64)
    out = np.empty((n, length), dtype=np.float64)
    for t in range(total):
        r = np.sqrt(variance) * z[:, t]
        variance = omega + alpha * r * r + beta * variance
        rows = np.flatnonzero((t >= burn) & (t < burn + length))
        if rows.size:
            out[rows, t - burn[rows]] = np.abs(r[rows])
    return out


def _surrogate_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return real FT surrogates with each row's exact discrete amplitude spectrum."""
    spectrum = np.fft.rfft(x, axis=1)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    phase[:, 0] = 0.0
    if x.shape[1] % 2 == 0:
        phase[:, -1] = 0.0
    randomized = np.abs(spectrum) * np.exp(1j * phase)
    randomized[:, 0] = spectrum[:, 0]
    if x.shape[1] % 2 == 0:
        randomized[:, -1] = spectrum[:, -1].real
    return np.fft.irfft(randomized, n=x.shape[1], axis=1)


def _lyap_cache_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "cache" / f"lyap_{name}.npy"


def _logistic_lyapunov_table() -> np.ndarray:
    path = _lyap_cache_path("logistic_v1")
    if path.exists():
        table = np.load(path)
        if table.ndim == 2 and table.shape[0] == 2 and table.shape[1] >= 1001:
            return table
    grid = np.linspace(3.5, 4.0, 1001)
    x = np.full(grid.shape, 0.371, dtype=np.float64)
    for _ in range(4000):
        x = grid * x * (1.0 - x)
    total = np.zeros_like(grid)
    for _ in range(20000):
        x = grid * x * (1.0 - x)
        total += np.log(np.maximum(np.abs(grid * (1.0 - 2.0 * x)), 1e-300))
    table = np.vstack((grid, total / 20000.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, table)
    return table


def _henon_lyapunov_table() -> np.ndarray:
    path = _lyap_cache_path("henon_v1")
    if path.exists():
        table = np.load(path)
        if table.ndim == 2 and table.shape[0] == 2 and table.shape[1] >= 101:
            return table
    grid = np.linspace(1.2, 1.4, 201)
    x = np.full(grid.shape, 0.2, dtype=np.float64)
    y = np.zeros_like(x)
    vx, vy = np.ones_like(x), np.zeros_like(x)
    total = np.zeros_like(x)
    for t in range(12000):
        old_x = x
        xn = 1.0 - grid * old_x * old_x + y
        y = 0.3 * old_x
        x = xn
        vxn = -2.0 * grid * old_x * vx + vy
        vyn = 0.3 * vx
        norm = np.sqrt(vxn * vxn + vyn * vyn)
        vx, vy = vxn / np.maximum(norm, 1e-300), vyn / np.maximum(norm, 1e-300)
        if t >= 2000:
            total += np.log(np.maximum(norm, 1e-300))
    table = np.vstack((grid, total / 10000.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, table)
    return table


def _make_recall(n: int, length: int, rng: np.random.Generator, key_len: int, n_pairs: int, query_noise: float) -> TaskData:
    if key_len < 1 or n_pairs < 1 or query_noise < 0:
        raise ValueError("key_len/n_pairs must be positive and query_noise nonnegative")
    query_start = length - key_len
    block = key_len + 1
    n_starts = query_start - block + 1
    if n_starts < 1 or n_pairs * block > query_start:
        raise ValueError(f"{n_pairs} non-overlapping pairs do not fit before the {key_len}-sample query in L={length}")
    raw = rng.standard_normal((n, length))
    keys = rng.standard_normal((n, n_pairs, key_len))
    values = rng.standard_normal((n, n_pairs))
    for row in range(n):
        while np.unique(keys[row], axis=0).shape[0] != n_pairs:
            keys[row] = rng.standard_normal((n_pairs, key_len))
        while np.unique(values[row]).size != n_pairs:
            values[row] = rng.standard_normal(n_pairs)
    # Randomly distribute unused samples before, between, and after the blocks.
    # This produces disjoint positions with a different address layout per row.
    slack = query_start - n_pairs * block
    pair_positions = np.empty((n, n_pairs), dtype=np.int64)
    for row in range(n):
        gaps = rng.multinomial(slack, np.full(n_pairs + 1, 1.0 / (n_pairs + 1)))
        starts = np.cumsum(gaps[:-1]) + np.arange(n_pairs) * block
        pair_positions[row] = starts[rng.permutation(n_pairs)]
    target_pair = rng.integers(0, n_pairs, n)
    row = np.arange(n)
    key_samples = pair_positions[:, :, None] + np.arange(key_len)
    raw[row[:, None, None], key_samples] = keys
    raw[row[:, None], pair_positions + key_len] = values
    query = keys[row, target_pair]
    raw[:, query_start:] = query + query_noise * rng.standard_normal(query.shape)
    x = _zscore(raw)
    planted = x[row[:, None], pair_positions + key_len]
    y = planted[row, target_pair]
    y = ((y - y.mean()) / max(y.std(), 1e-12)).astype(np.float32)
    meta = pd.DataFrame({"family": "associative_recall", "n_pairs": n_pairs, "key_len": key_len,
                         "query_noise": query_noise, "target_pair": target_pair,
                         "target_pos": pair_positions[row, target_pair],
                         "pair_positions": list(pair_positions)})
    return TaskData(x, y, "reg", None, meta)


def _lyapunov(family: str, param: np.ndarray) -> np.ndarray:
    table = _logistic_lyapunov_table() if family == "logistic" else _henon_lyapunov_table()
    return np.interp(param, table[0], table[1])


def _make_system(n: int, length: int, rng: np.random.Generator) -> TaskData:
    labels = np.arange(n, dtype=np.int64) % len(SYSTEM_CLASSES)
    rng.shuffle(labels)
    x = np.empty((n, length), dtype=np.float64)
    rows_meta: list[dict] = [{} for _ in range(n)]
    for cls_idx, family in enumerate(SYSTEM_CLASSES):
        idx = np.flatnonzero(labels == cls_idx)
        count = len(idx)
        if not count:
            continue
        if family == "white_noise":
            raw = rng.standard_normal((count, length))
            params = np.zeros(count)
        elif family == "ar1":
            params = rng.uniform(-0.8, 0.95, count)
            raw = _ar_batch(params, length, rng)
        elif family == "sine":
            freq = rng.uniform(0.02, 0.22, count)
            phase = rng.uniform(0, 2 * np.pi, count)
            t = np.arange(length)[None, :]
            raw = np.sin(2 * np.pi * freq[:, None] * t + phase[:, None])
            raw += 0.4 * rng.standard_normal(raw.shape)
            params = freq
        elif family == "logistic":
            params = rng.uniform(3.55, 4.0, count)
            raw = _logistic_batch(params, length, rng)
        elif family == "henon":
            params = rng.uniform(1.25, 1.4, count)
            raw = _henon_batch(params, length, rng)
        elif family in ("lorenz", "ft_lorenz"):
            raw = _flow_batch("lorenz", count, length, rng)
            params = np.full(count, 10.0)
            if family == "ft_lorenz":
                raw = _surrogate_batch(raw, rng)
        elif family == "rossler":
            raw = _flow_batch("rossler", count, length, rng)
            params = np.full(count, 0.04)
        elif family == "garch_abs":
            params = rng.uniform(0.55, 0.98, count)
            raw = _garch_abs_batch(params, length, rng)
        elif family == "regime_ar":
            params = rng.uniform(0.2, 0.9, count)
            phi2 = rng.uniform(-0.85, 0.85, count)
            split = rng.integers(max(1, length // 4), max(2, 3 * length // 4), count)
            innovations = rng.standard_normal((count, length))
            raw = np.zeros_like(innovations)
            for t in range(1, length):
                phi = np.where(t < split, params, phi2)
                raw[:, t] = phi * raw[:, t - 1] + innovations[:, t]
            params = np.column_stack((params, phi2, split / length))
        elif family == "ft_logistic":
            params = rng.uniform(3.55, 4.0, count)
            raw = _surrogate_batch(_logistic_batch(params, length, rng), rng)
        else:  # pragma: no cover - class list and dispatch stay aligned
            raise RuntimeError(f"unhandled system family {family}")
        base = raw.std(axis=1, keepdims=True)
        sigma = rng.uniform(0.0, 0.3, count)
        raw = raw + rng.standard_normal(raw.shape) * (sigma[:, None] * np.maximum(base, 1e-8))
        x[idx] = raw
        for local, row in enumerate(idx):
            rows_meta[row] = {
                "family": family,
                "parameter": params[local].tolist() if np.ndim(params[local]) else float(params[local]),
                "noise": float(sigma[local]),
            }
    return TaskData(_zscore(x), labels, "cls", len(SYSTEM_CLASSES), pd.DataFrame(rows_meta))


def _make_lyapunov(n: int, length: int, rng: np.random.Generator) -> TaskData:
    family = rng.integers(0, 2, n)
    param = np.where(family == 0, rng.uniform(3.5, 4.0, n), rng.uniform(1.2, 1.4, n))
    raw = np.empty((n, length), dtype=np.float64)
    for mask, name in ((family == 0, "logistic"), (family == 1, "henon")):
        if mask.any():
            raw[mask] = (_logistic_batch(param[mask], length, rng) if name == "logistic"
                         else _henon_batch(param[mask], length, rng))
    sigma = rng.uniform(0.0, 0.03, n)
    raw += rng.standard_normal(raw.shape) * (sigma[:, None] * np.maximum(raw.std(axis=1, keepdims=True), 1e-8))
    y = np.empty(n, dtype=np.float32)
    y[family == 0] = _lyapunov("logistic", param[family == 0])
    y[family == 1] = _lyapunov("henon", param[family == 1])
    meta = pd.DataFrame({"family": np.where(family == 0, "logistic", "henon"), "parameter": param, "noise": sigma})
    return TaskData(_zscore(raw), y, "reg", None, meta)


def _make_noise(n: int, length: int, rng: np.random.Generator) -> TaskData:
    family = rng.integers(0, 2, n)
    raw = np.empty((n, length), dtype=np.float64)
    raw[family == 0] = _flow_batch("lorenz", int((family == 0).sum()), length, rng)
    raw[family == 1] = _flow_batch("rossler", int((family == 1).sum()), length, rng)
    signal_sd = raw.std(axis=1)
    ratio = np.exp(rng.uniform(np.log(0.02), np.log(0.5), n))
    raw += rng.standard_normal(raw.shape) * (ratio * signal_sd)[:, None]
    y = np.log(ratio).astype(np.float32)
    meta = pd.DataFrame({"family": np.where(family == 0, "lorenz", "rossler"), "noise_ratio": ratio})
    return TaskData(_zscore(raw), y, "reg", None, meta)


def _make_changepoint(n: int, length: int, rng: np.random.Generator) -> TaskData:
    split = rng.integers(max(1, int(0.2 * length)), max(2, int(0.8 * length)), n)
    phi1, phi2 = rng.uniform(-0.8, 0.8, n), rng.uniform(-0.8, 0.8, n)
    innovations = rng.standard_normal((n, length))
    raw = np.zeros((n, length), dtype=np.float64)
    for t in range(1, length):
        phi = np.where(t < split, phi1, phi2)
        raw[:, t] = phi * raw[:, t - 1] + innovations[:, t]
    meta = pd.DataFrame({"family": "regime_ar", "phi_before": phi1, "phi_after": phi2, "changepoint": split / length})
    return TaskData(_zscore(raw), (split / length).astype(np.float32), "reg", None, meta)




def _make_forecast(n: int, length: int, rng: np.random.Generator, horizon: int) -> TaskData:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    family = rng.integers(0, 4, n)
    total = length + horizon
    raw = np.empty((n, total), dtype=np.float64)
    param = np.empty(n, dtype=np.float64)
    for code, name in enumerate(("logistic", "henon", "lorenz", "rossler")):
        mask = family == code
        count = int(mask.sum())
        if not count:
            continue
        if name == "logistic":
            param[mask] = rng.uniform(3.7, 4.0, count)
            raw[mask] = _logistic_batch(param[mask], total, rng)
        elif name == "henon":
            param[mask] = rng.uniform(1.3, 1.4, count)
            raw[mask] = _henon_batch(param[mask], total, rng)
        else:
            param[mask] = np.nan
            raw[mask] = _flow_batch(name, count, total, rng)
    sigma = rng.uniform(0.0, 0.05, n)
    raw += rng.standard_normal(raw.shape) * (sigma[:, None] * np.maximum(raw[:, :length].std(axis=1, keepdims=True), 1e-8))
    mu, sd = raw[:, :length].mean(axis=1), raw[:, :length].std(axis=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    x = ((raw[:, :length] - mu[:, None]) / sd[:, None]).astype(np.float32)
    y = ((raw[:, length + horizon - 1] - mu) / sd).astype(np.float32)
    meta = pd.DataFrame({"family": np.asarray(("logistic", "henon", "lorenz", "rossler"))[family],
                         "parameter": param, "noise": sigma, "horizon": horizon})
    return TaskData(x, y, "reg", None, meta)




def make_task(task: str, n: int, n_states: int, seed: int, m: int = 3, tau: int = 1, **difficulty) -> TaskData:
    """Generate one benchmark task deterministically; returned x has shape [n,L]."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {tuple(TASKS)}")
    if n < 1:
        raise ValueError("n must be positive")
    length = window_length(n_states, m, tau)
    rng = np.random.default_rng(seed)
    if task == "system":
        return _make_system(n, length, rng)
    if task == "lyapunov":
        return _make_lyapunov(n, length, rng)
    if task == "noise":
        return _make_noise(n, length, rng)
    if task == "changepoint":
        return _make_changepoint(n, length, rng)
    if task == "recall":
        return _make_recall(n, length, rng, int(difficulty.get("key_len", m)),
                            int(difficulty.get("n_pairs", 4)), float(difficulty.get("query_noise", 0.0)))
    return _make_forecast(n, length, rng, int(difficulty.get("horizon", 3)))

def make_pretrain(n: int, n_states: int, seed: int, m: int = 3, tau: int = 1) -> np.ndarray:
    """Generate an unlabeled, approximately balanced mixture of all system classes."""
    return _make_system(n, window_length(n_states, m, tau), np.random.default_rng(seed)).x


def check_data() -> pd.DataFrame:
    """Run deterministic sanity checks and return observed values in a DataFrame."""
    checks: list[dict[str, object]] = []
    logtab = _logistic_lyapunov_table()
    lam4 = float(np.interp(4.0, logtab[0], logtab[1]))
    lam35 = float(np.interp(3.5, logtab[0], logtab[1]))
    checks.extend([
        {"check": "logistic lambda r=4", "value": lam4, "expected": "ln(2) ± 0.01"},
        {"check": "logistic lambda r=3.5", "value": lam35, "expected": "< 0"},
    ])
    rng = np.random.default_rng(27)
    original = _logistic_batch(np.full(16, 4.0), 128, rng)
    surrogate = _surrogate_batch(original, rng)
    amp_err = float(np.max(np.abs(np.abs(np.fft.rfft(original, axis=1)) - np.abs(np.fft.rfft(surrogate, axis=1))) /
                           np.maximum(np.abs(np.fft.rfft(original, axis=1)), 1e-12)))
    asym_orig = float(np.mean(np.diff(original, axis=1) ** 3))
    asym_surr = float(np.mean(np.diff(surrogate, axis=1) ** 3))
    checks.extend([
        {"check": "FT surrogate relative amplitude-spectrum error", "value": amp_err, "expected": "< 1e-6"},
        {"check": "FT surrogate nonlinear time-reversal statistic difference", "value": abs(asym_orig - asym_surr), "expected": "> 0"},
    ])
    recall = _make_recall(128, window_length(64, 3, 1), rng, 3, 4, 0.0)
    L = recall.x.shape[1]
    # Brute-force nearest key lookup on each noiseless normalized query.
    key_len = 3
    positions = np.stack(recall.meta["pair_positions"].to_numpy())
    rows = np.arange(len(recall.x))
    keys = recall.x[rows[:, None, None], positions[:, :, None] + np.arange(key_len)]
    query = recall.x[:, -key_len:]
    match = np.abs(keys - query[:, None, :]).max(axis=2).argmin(axis=1)
    planted = recall.x[rows[:, None], positions + key_len]
    recovered = planted[rows, match]
    # Apply the same global target scaling used when constructing y.
    targets_before_scale = planted[rows, recall.meta["target_pair"].to_numpy()]
    recovered = (recovered - targets_before_scale.mean()) / targets_before_scale.std()
    denom = float(np.sum((recall.y - recall.y.mean()) ** 2))
    r2 = float(1.0 - np.sum((recovered - recall.y) ** 2) / max(denom, 1e-12))
    checks.append({"check": "recall exact key match R2", "value": r2, "expected": "> 0.99"})
    system = _make_system(110, 64, rng)
    counts = np.bincount(system.y, minlength=len(SYSTEM_CLASSES))
    checks.append({"check": "system class count range", "value": int(counts.max() - counts.min()), "expected": "≤ 1"})
    result = pd.DataFrame(checks, columns=("check", "value", "expected"))
    failures = []
    if abs(lam4 - np.log(2.0)) >= 0.01:
        failures.append("lambda(4)")
    if lam35 >= 0:
        failures.append("lambda(3.5)")
    if amp_err >= 1e-6:
        failures.append("surrogate spectrum")
    if abs(asym_orig - asym_surr) <= 1e-12:
        failures.append("surrogate nonlinear statistic")
    if r2 <= 0.99:
        failures.append("recall matching")
    if counts.max() - counts.min() > 1:
        failures.append("class balance")
    if failures:
        raise AssertionError(f"data checks failed: {', '.join(failures)}\n{result}")
    return result

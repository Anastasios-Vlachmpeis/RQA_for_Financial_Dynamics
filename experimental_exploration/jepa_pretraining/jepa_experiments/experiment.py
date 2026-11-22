"""Deterministic data and result caches for JEPA benchmark experiments."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data import TaskData, make_task
from .diagnostics import DIAGNOSTIC_TASKS, make_diagnostic

SMOKE = os.environ.get("JEPA_SMOKE") == "1"


def _json_safe(value):
    """Convert values to strict JSON types, mapping non-finite floats to null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical(value) -> str:
    """Serialize cache identity values deterministically."""
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _split_seed(task: str, split: str, base_seed: int, difficulty: dict) -> int:
    identity = _canonical({"task": task, "split": split, "difficulty": difficulty,
                           "base_seed": base_seed})
    return int.from_bytes(hashlib.blake2b(identity.encode("utf-8"), digest_size=4).digest(), "big")


def get_split(task: str, split: str, n: int, n_states: int, m: int = 3,
              tau: int = 1, base_seed: int = 0, **difficulty) -> TaskData:
    """Load or generate a task split, cached as compressed arrays and CSV metadata.

    Split names include the standard supervised/probe splits plus dedicated
    ``pretrain`` and ``probe_val`` splits.
    """
    allowed = {"train", "val", "test", "probe_train", "probe_test", "pretrain", "probe_val"}
    identity = {"task": task, "split": split, "n": int(n), "n_states": int(n_states),
                "m": int(m), "tau": int(tau), "base_seed": int(base_seed),
                "difficulty": difficulty}
    if task in DIAGNOSTIC_TASKS:
        identity["diagnostic_version"] = 1
    root = Path(__file__).resolve().parents[1] / "cache" / "tasks"
    if SMOKE:
        root = root / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_{split}_{_digest(identity)[:16]}"
    array_path, meta_path = root / f"{stem}.npz", root / f"{stem}.csv"
    if array_path.exists() and meta_path.exists():
        with np.load(array_path, allow_pickle=False) as saved:
            x, y = saved["x"], saved["y"]
            kind = str(saved["kind"].item())
            n_classes = int(saved["n_classes"].item()) if bool(saved["has_n_classes"].item()) else None
        return TaskData(x, y, kind, n_classes, pd.read_csv(meta_path))

    split_seed = _split_seed(task, split, base_seed, difficulty)
    if task in DIAGNOSTIC_TASKS:
        data = make_diagnostic(task, n, n_states, split_seed, m=m, tau=tau)
    else:
        data = make_task(task, n, n_states, split_seed, m=m, tau=tau, **difficulty)
    tmp_array, tmp_meta = array_path.with_suffix(".tmp.npz"), meta_path.with_suffix(".tmp.csv")
    np.savez_compressed(tmp_array, x=data.x, y=data.y, kind=np.asarray(data.kind),
                        has_n_classes=np.asarray(data.n_classes is not None),
                        n_classes=np.asarray(data.n_classes if data.n_classes is not None else -1))
    data.meta.to_csv(tmp_meta, index=False)
    tmp_array.replace(array_path)
    tmp_meta.replace(meta_path)
    return data


def cached_run(store_dir: str | Path, key: dict, fn: Callable[[], dict]) -> dict:
    """Return cached JSON output for ``key`` or evaluate and atomically save ``fn``."""
    directory = Path(store_dir)
    if SMOKE and "smoke" not in directory.parts:
        directory = directory / "smoke"
    directory.mkdir(parents=True, exist_ok=True)
    normalized_key = json.loads(_canonical(key))
    path = directory / f"{_digest(normalized_key)}.json"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved.get("key") == normalized_key and isinstance(saved.get("result"), dict):
            return saved["result"]

    result = fn()
    if not isinstance(result, dict):
        raise TypeError("cached_run fn must return a dict")
    payload = {"key": normalized_key, "result": _json_safe(result)}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)
    return result


def summarise(df: pd.DataFrame, by, metric: str) -> pd.DataFrame:
    """Summarise ``metric`` by columns in ``by`` with sample SD and seed count."""
    groups = [by] if isinstance(by, str) else list(by)
    return (df.groupby(groups, dropna=False)[metric]
            .agg(mean="mean", sd="std", n="count").reset_index())

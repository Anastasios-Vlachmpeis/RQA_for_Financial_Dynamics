"""Fingerprinted on-disk cache for condpred experiments."""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from condpred.config import code_fingerprint

_REPO = Path(__file__).resolve().parents[2]
_ROOT = _REPO / "data" / "processed" / "condpred"


def _hash12(key: dict, modules: list[str] | None) -> str:
    payload = json.dumps(key, sort_keys=True, default=str)
    if modules:
        payload += code_fingerprint(modules)
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def cached(exp: str, name: str, key: dict, fn, *, fmt: str = "pickle", modules: list[str] | None = None):
    """Cache under ``data/processed/condpred/<exp>/<name>_<hash12>.<ext>``."""
    if not modules:
        raise ValueError("cached() requires modules for code fingerprinting")
    h = _hash12(key, modules)
    ext = "parquet" if fmt == "parquet" else "pkl"
    folder = _ROOT / exp
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_{h}.{ext}"
    manifest = folder / "MANIFEST.csv"
    if path.exists():
        if fmt == "parquet":
            return pd.read_parquet(path)
        with path.open("rb") as f:
            return pickle.load(f)

    obj = fn()
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=f".{ext}.tmp")
    os.close(fd)
    try:
        if fmt == "parquet":
            if not isinstance(obj, pd.DataFrame):
                raise TypeError("parquet cache requires a DataFrame")
            obj.to_parquet(tmp)
        else:
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        Path(tmp).replace(path)
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass

    fp = code_fingerprint(modules)
    row = {
        "name": name,
        "hash": h,
        "created": datetime.now(timezone.utc).isoformat(),
        "key": json.dumps(key, sort_keys=True, default=str),
        "fingerprint": fp,
    }
    header = not manifest.exists()
    with manifest.open("a", encoding="utf-8") as f:
        if header:
            f.write("name,hash,created,key,fingerprint\n")
        f.write(",".join(f'"{row[k]}"' if k in ("key",) else str(row[k]) for k in row) + "\n")
    return obj

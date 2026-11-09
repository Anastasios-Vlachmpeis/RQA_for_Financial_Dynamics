"""Spec helpers, code fingerprinting, and smoke-mode scaling."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path


def smoke(full, small):
    """Return ``small`` when ``CONDPRED_SMOKE=1``, else ``full``."""
    return small if os.environ.get("CONDPRED_SMOKE") == "1" else full


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()


def _module_source(name: str) -> str:
    mod = importlib.import_module(name)
    path = Path(inspect.getfile(mod))
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n")


def code_fingerprint(modules: list[str]) -> str:
    """SHA-256 over listed module sources (CRLF normalised to LF)."""
    h = hashlib.sha256()
    for name in sorted(modules):
        h.update(name.encode())
        h.update(b"\0")
        h.update(_module_source(name).encode())
    return h.hexdigest()

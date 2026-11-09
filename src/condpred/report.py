"""Pre-registration files, decision tables, figures."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from condpred.config import code_fingerprint, spec_hash


_META_KEYS = frozenset({"spec_hash", "code_fingerprint", "written_at"})


def _frozen_spec(spec: dict, exclude: frozenset | None = None) -> dict:
    skip = _META_KEYS | (exclude or frozenset())
    return {k: v for k, v in spec.items() if k not in skip}


def write_spec(spec: dict, out_dir, modules=None, *, completion_keys: frozenset | None = None):
    if not modules:
        raise ValueError("write_spec() requires modules for code fingerprinting")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "spec.json"
    payload = dict(spec)
    payload["spec_hash"] = spec_hash(spec)
    payload["code_fingerprint"] = code_fingerprint(modules)
    payload["written_at"] = datetime.now(timezone.utc).isoformat()
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old_ver = old.get("version")
        new_ver = spec.get("version")
        if old_ver == new_ver and old.get("spec_hash") != payload["spec_hash"]:
            allow = completion_keys or frozenset()
            if allow and spec_hash(_frozen_spec(old, allow)) == spec_hash(_frozen_spec(spec, allow)):
                pass
            else:
                raise RuntimeError("spec version unchanged but hash differs; bump SPEC version")
        if old_ver is not None and new_ver is not None and old_ver != new_ver:
            archive = out_dir / f"spec_v{old_ver}.json"
            if not archive.exists():
                archive.write_text(json.dumps(old, indent=2, default=str), encoding="utf-8")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_decision(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "decision.csv", index=False)


def save_table(df, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=(not isinstance(df.index, pd.RangeIndex)))


def save_fig(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=120, bbox_inches="tight")

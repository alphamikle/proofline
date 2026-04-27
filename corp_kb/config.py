from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        # Fall back to config.example.yaml if user has not copied it yet.
        example = Path(__file__).resolve().parents[1] / "config.example.yaml"
        if example.exists():
            path = example
        else:
            raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand_env(cfg)
    root = Path(cfg.get("workspace", "./data"))
    cfg["workspace"] = str(root)
    cfg.setdefault("storage", {})
    cfg["storage"].setdefault("duckdb_path", str(root / "kb.duckdb"))
    cfg["storage"].setdefault("sqlite_fts_path", str(root / "indexes" / "code_fts.sqlite"))
    cfg["storage"].setdefault("vector_index_path", str(root / "indexes" / "code_vectors.faiss"))
    cfg["storage"].setdefault("vector_meta_path", str(root / "indexes" / "code_vectors_meta.parquet"))
    cfg.setdefault("repos", {})
    cfg["repos"].setdefault("root", "./repos")
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    root = Path(cfg["workspace"])
    for sub in [
        root,
        root / "raw" / "repos",
        root / "raw" / "datadog" / "logs",
        root / "raw" / "datadog" / "spans",
        root / "raw" / "bigquery",
        root / "raw" / "confluence",
        root / "raw" / "jira",
        root / "normalized",
        root / "indexes",
        root / "reports",
    ]:
        sub.mkdir(parents=True, exist_ok=True)
    Path(cfg["storage"]["duckdb_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["storage"]["sqlite_fts_path"]).parent.mkdir(parents=True, exist_ok=True)

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

DEFAULT_CONFIG = "proofline.yaml"
CONFIG_ENV_VAR = "PROOFLINE_CONFIG"
CONFIG_SHAPE_VERSION = 2


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def example_config_path() -> Path:
    return package_root() / "proofline.example.yaml"


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


def missing_paths(base: Dict[str, Any], override: Dict[str, Any], prefix: str = "") -> List[str]:
    paths: List[str] = []
    for key, value in base.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in override:
            paths.append(path)
            continue
        if isinstance(value, dict) and isinstance(override.get(key), dict):
            paths.extend(missing_paths(value, override[key], path))
    return paths


def config_shape_diff(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Run `proofline init` or pass --config."
        )
    original_text = path.read_text(encoding="utf-8")
    original = yaml.safe_load(original_text) or {}
    defaults = default_config()
    missing = missing_paths(defaults, original)
    version_current = original.get("config_version") == CONFIG_SHAPE_VERSION
    return {
        "path": str(path),
        "original_text": original_text,
        "original": original,
        "defaults": defaults,
        "missing_paths": missing,
        "version_current": version_current,
        "current_version": original.get("config_version"),
        "target_version": CONFIG_SHAPE_VERSION,
        "needs_migration": bool(missing) or not version_current,
    }


def default_config() -> Dict[str, Any]:
    path = example_config_path()
    if not path.exists():
        return minimal_default_config()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["config_version"] = CONFIG_SHAPE_VERSION
    return cfg


def minimal_default_config() -> Dict[str, Any]:
    return {
        "config_version": CONFIG_SHAPE_VERSION,
        "workspace": "./data",
        "repos": {"root": "./repos", "update_existing": True},
        "storage": {},
        "git_history": {},
        "datadog": {"enabled": False},
        "bigquery": {"enabled": False},
        "confluence": {"enabled": False},
        "jira": {"enabled": False},
        "indexing": {
            "lexical_fts": True,
            "chunk_write_batch_size": 5000,
            "repo_max_workers": 1,
            "max_workers": 1,
            "ast_chunking": {
                "enabled": True,
                "source": "cgc",
                "fallback_regex": True,
                "keep_file_windows": True,
                "include_node_types": ["Function", "Class", "Method", "Module", "Struct", "Enum", "Interface"],
                "max_symbol_lines": 240,
                "symbol_window_lines": 160,
                "symbol_window_overlap": 30,
                "include_context_prefix": True,
                "dedupe_overlapping_chunks": True,
            },
            "embeddings": {"enabled": True},
        },
        "retrieval": {"reranker": {"enabled": True}},
        "graph_backend": {"enabled": False, "provider": "neo4j"},
        "code_graph": {"enabled": False},
        "visualization": {"output_path": ""},
        "agent": {"provider": "none"},
    }


def migrate_config_file(path: str | Path, *, quiet: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    path = Path(path)
    diff = config_shape_diff(path)
    original_text = diff["original_text"]
    original = diff["original"]
    defaults = diff["defaults"]
    missing = missing_paths(defaults, original)
    added = [path for path in missing if "." not in path]
    merged = deep_merge(defaults, original)
    if merged.get("config_version") != CONFIG_SHAPE_VERSION:
        merged["config_version"] = CONFIG_SHAPE_VERSION
        if "config_version" not in added:
            added.append("config_version")
    if added:
        migrated = migrate_config_text(original_text, original, merged)
        path.write_text(migrated, encoding="utf-8")
        if not quiet:
            shown = ", ".join(added[:12])
            suffix = f" (+{len(added) - 12} more)" if len(added) > 12 else ""
            print(f"Updated config shape in {path}: added {shown}{suffix}", file=sys.stderr)
            warnings = config_followup_warnings(merged)
            if warnings:
                print("Config follow-up: " + "; ".join(warnings[:4]), file=sys.stderr)
    return merged, added


def upgrade_config_file(
    path: str | Path,
    *,
    use_agent: bool = True,
    quiet: bool = False,
) -> Tuple[Dict[str, Any], Path | None, List[str]]:
    """Write a full merged config, preserving the old file beside it."""
    path = Path(path)
    diff = config_shape_diff(path)
    original = diff["original"]
    defaults = diff["defaults"]
    missing = list(diff["missing_paths"])
    merged = deep_merge(defaults, original)
    merged["config_version"] = CONFIG_SHAPE_VERSION

    if not diff["needs_migration"]:
        return merged, None, []

    backup = backup_config_path(path)
    backup.write_text(diff["original_text"], encoding="utf-8")

    text = None
    if use_agent:
        text = agent_merged_config_text(diff["original_text"], defaults, merged, original)
    if not text:
        text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=False)
    path.write_text(text, encoding="utf-8")
    if not quiet:
        shown = ", ".join(missing[:12])
        suffix = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
        print(f"Upgraded config in {path}: merged missing {shown}{suffix}", file=sys.stderr)
        print(f"Previous config saved as {backup}", file=sys.stderr)
        warnings = config_followup_warnings(merged)
        if warnings:
            print("Config follow-up: " + "; ".join(warnings[:4]), file=sys.stderr)
    return merged, backup, missing


def backup_config_path(path: str | Path) -> Path:
    path = Path(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.stem}_{stamp}_old{path.suffix or '.yaml'}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{counter}_old{path.suffix or '.yaml'}")
        counter += 1
    return candidate


def agent_merged_config_text(
    original_text: str,
    defaults: Dict[str, Any],
    merged: Dict[str, Any],
    original: Dict[str, Any],
) -> str | None:
    agent_cfg = original.get("agent") if isinstance(original, dict) else None
    if not isinstance(agent_cfg, dict):
        return None
    provider = str(agent_cfg.get("provider") or "").lower()
    agents = agent_cfg.get("agents")
    if provider in {"", "none"} and not agents:
        return None
    try:
        from proofline.agent.providers import complete_with_agent
    except Exception:
        return None

    system_prompt = (
        "You merge Proofline YAML config files. Return only YAML, no markdown. "
        "Preserve every existing user value exactly, add missing keys from the new template, "
        "and keep the result valid YAML."
    )
    user_prompt = (
        "OLD CONFIG:\n"
        + original_text
        + "\n\nNEW DEFAULT CONFIG YAML:\n"
        + yaml.safe_dump(defaults, sort_keys=False, allow_unicode=False)
        + "\n\nDETERMINISTIC MERGED CONFIG YAML:\n"
        + yaml.safe_dump(merged, sort_keys=False, allow_unicode=False)
    )
    try:
        bounded_agent_cfg = dict(agent_cfg)
        try:
            bounded_agent_cfg["request_timeout_seconds"] = min(int(agent_cfg.get("request_timeout_seconds") or 60), 60)
        except Exception:
            bounded_agent_cfg["request_timeout_seconds"] = 60
        text = complete_with_agent(system_prompt, user_prompt, {"agent": bounded_agent_cfg})
    except Exception:
        return None
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = yaml.safe_load(text) or {}
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed["config_version"] = CONFIG_SHAPE_VERSION
    if not _preserves_existing_values(original, parsed):
        return None
    completed = deep_merge(merged, parsed)
    completed["config_version"] = CONFIG_SHAPE_VERSION
    return yaml.safe_dump(completed, sort_keys=False, allow_unicode=False)


def _preserves_existing_values(original: Any, candidate: Any) -> bool:
    if isinstance(original, dict):
        if not isinstance(candidate, dict):
            return False
        for key, value in original.items():
            if key not in candidate:
                return False
            if not _preserves_existing_values(value, candidate[key]):
                return False
        return True
    return original == candidate


def migrate_config_text(text: str, original: Dict[str, Any], merged: Dict[str, Any]) -> str:
    template = example_config_path().read_text(encoding="utf-8") if example_config_path().exists() else yaml.safe_dump(default_config(), sort_keys=False)
    blocks = top_level_blocks(template)
    order = [key for key, _ in blocks]
    existing = set(original.keys())
    out = text
    if out and not out.endswith("\n"):
        out += "\n"
    if "config_version" in existing and original.get("config_version") != CONFIG_SHAPE_VERSION:
        out = replace_top_level_scalar(out, "config_version", CONFIG_SHAPE_VERSION)
    for key, block in blocks:
        if key in {"neo4j"}:
            continue
        if key not in existing:
            out = insert_top_level_block(out, key, block, order)
            existing.add(key)
    if "config_version" not in existing:
        block = f"config_version: {CONFIG_SHAPE_VERSION}\n"
        out = insert_top_level_block(out, "config_version", block, ["config_version"] + order)
    return out


def top_level_blocks(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines(keepends=True)
    starts: List[Tuple[str, int]] = []
    for i, line in enumerate(lines):
        if not line.strip() or line.startswith("#") or line[0].isspace():
            continue
        if ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        if key:
            starts.append((key, i))
    blocks: List[Tuple[str, str]] = []
    for idx, (key, start) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines)
        blocks.append((key, "".join(lines[start:end]).rstrip() + "\n"))
    return blocks


def insert_top_level_block(text: str, key: str, block: str, order: List[str]) -> str:
    lines = text.splitlines(keepends=True)
    existing_positions = top_level_positions(lines)
    anchor = next_later_existing_key(key, order, existing_positions)
    insert_at = existing_positions[anchor] if anchor else len(lines)
    prefix = "\n" if insert_at > 0 and "".join(lines[:insert_at]).strip() and (insert_at == 0 or lines[insert_at - 1].strip()) else ""
    suffix = "" if block.endswith("\n\n") else "\n"
    lines[insert_at:insert_at] = [prefix + block.rstrip() + "\n" + suffix]
    return "".join(lines)


def top_level_positions(lines: List[str]) -> Dict[str, int]:
    positions: Dict[str, int] = {}
    for i, line in enumerate(lines):
        if not line.strip() or line.startswith("#") or line[0].isspace() or ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        if key:
            positions[key] = i
    return positions


def next_later_existing_key(key: str, order: List[str], positions: Dict[str, int]) -> str:
    if key not in order:
        return ""
    for candidate in order[order.index(key) + 1:]:
        if candidate in positions:
            return candidate
    return ""


def replace_top_level_scalar(text: str, key: str, value: Any) -> str:
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            newline = "\n" if line.endswith("\n") else ""
            lines[i] = f"{key}: {value}{newline}"
            break
    return "".join(lines)


def config_followup_warnings(cfg: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if cfg.get("datadog", {}).get("enabled") and not (os.getenv("DD_API_KEY") and os.getenv("DD_APP_KEY")):
        warnings.append("Datadog is enabled but DD_API_KEY/DD_APP_KEY are not set")
    if cfg.get("bigquery", {}).get("enabled") and not (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CLOUD_PROJECT")):
        warnings.append("BigQuery is enabled but GOOGLE_APPLICATION_CREDENTIALS/GOOGLE_CLOUD_PROJECT are not set")
    for key, env_name in [("jira", "JIRA_BASE_URL"), ("confluence", "CONFLUENCE_BASE_URL")]:
        section = cfg.get(key, {})
        if section.get("enabled") and not os.getenv(env_name) and str(section.get("base_url") or "").startswith("${"):
            warnings.append(f"{key} is enabled but {env_name} is not set")
    agent = cfg.get("agent", {})
    agents = agent.get("agents")
    if isinstance(agents, list) and agents:
        active = str(agent.get("active") or "").strip()
        names = [str(a.get("name") or "") for a in agents if isinstance(a, dict)]
        if active and active not in names:
            warnings.append(f"agent.active={active} does not match any configured agent name")
        for item in agents:
            if not isinstance(item, dict):
                continue
            provider = item.get("provider", agent.get("provider"))
            if provider not in (None, "", "none") and not (item.get("model") or item.get("command") or agent.get("model") or agent.get("command")):
                warnings.append(f"agent {item.get('name') or '<unnamed>'} provider is enabled but model/command is empty")
                break
    elif agent.get("provider") not in (None, "", "none") and not (agent.get("model") or agent.get("command")):
        warnings.append("agent provider is enabled but model/command is empty")
    return warnings


def load_config(path: str | Path, *, quiet: bool = False) -> Dict[str, Any]:
    path = Path(path)
    cfg, _ = migrate_config_file(path, quiet=quiet)
    cfg = _expand_env(cfg)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    _normalize_graph_backend(cfg)
    root = Path(cfg.get("workspace", "./data"))
    cfg["workspace"] = str(root)
    cfg.setdefault("storage", {})
    cfg["storage"].setdefault("duckdb_path", str(root / "kb.duckdb"))
    cfg["storage"].setdefault("sqlite_fts_path", str(root / "indexes" / "code_fts.sqlite"))
    cfg["storage"].setdefault("vector_index_path", str(root / "indexes" / "code_vectors.faiss"))
    cfg["storage"].setdefault("vector_meta_path", str(root / "indexes" / "code_vectors_meta.parquet"))
    cfg.setdefault("visualization", {})
    cfg["visualization"].setdefault("output_path", str(root / "visualization" / "graph.json"))
    cfg.setdefault("repos", {})
    cfg["repos"].setdefault("root", "./repos")
    cfg.setdefault("git_history", {})
    gh = cfg["git_history"]
    gh.setdefault("enabled", True)
    gh.setdefault("metadata_days", None)
    gh.setdefault("max_commits_per_repo", None)
    gh.setdefault("rename_detection", True)
    gh.setdefault("patch_hunks", True)
    gh.setdefault("current_blame", True)
    gh.setdefault("use_blame_ignore_revs", True)
    gh.setdefault("write_commit_graph", True)
    gh.setdefault("cochange_window_days", 730)
    gh.setdefault("cochange_min_count", 1)
    gh.setdefault("max_hunk_chars", 20000)
    gh.setdefault("max_blame_files", 500)
    gh.setdefault("max_blame_rows", 50000)
    gh.setdefault("git_timeout_seconds", 300)
    return cfg


def _normalize_graph_backend(cfg: Dict[str, Any]) -> None:
    backend = dict(cfg.get("graph_backend") or {})
    if not backend:
        return
    provider = backend.get("provider", "neo4j")
    if provider != "neo4j":
        return
    neo4j = {k: v for k, v in backend.items() if k != "provider"}
    cfg["neo4j"] = deep_merge(dict(cfg.get("neo4j") or {}), neo4j)


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
        root / "visualization",
    ]:
        sub.mkdir(parents=True, exist_ok=True)
    Path(cfg["storage"]["duckdb_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["storage"]["sqlite_fts_path"]).parent.mkdir(parents=True, exist_ok=True)

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

from corp_kb.utils import now_iso, run_cmd, safe_read_text, stable_id, json_dumps

EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".go": "go", ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".rs": "rust", ".scala": "scala", ".swift": "swift", ".m": "objective-c", ".h": "c/c++", ".cpp": "c++",
    ".c": "c", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml",
    ".xml": "xml", ".gradle": "gradle", ".proto": "protobuf", ".graphql": "graphql", ".md": "markdown",
    ".tf": "terraform", ".sh": "shell", ".dart": "dart",
}

MANIFEST_NAMES = {
    "package.json", "pnpm-lock.yaml", "yarn.lock", "package-lock.json", "go.mod", "go.sum",
    "pom.xml", "build.gradle", "settings.gradle", "requirements.txt", "poetry.lock", "pyproject.toml",
    "Cargo.toml", "Cargo.lock", "Gemfile", "Gemfile.lock", "composer.json", "pubspec.yaml",
}
API_PATTERNS = ["openapi", "swagger", "asyncapi"]
K8S_MARKERS = ["deployment", "service", "ingress", "statefulset", "daemonset", "configmap", "secret"]
ROUTE_FILE_HINTS = ["controller", "routes", "router", "handler", "api"]


def find_git_repos(root: Path, exclude_dirs: Iterable[str] = ()) -> List[Path]:
    repos: List[Path] = []
    excludes = set(exclude_dirs)
    for dirpath, dirnames, _ in os.walk(root):
        if ".git" in dirnames:
            repos.append(Path(dirpath))
            # A discovered repo can contain large dependency/build trees. The
            # file scanner will handle this repo later with its own excludes.
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d not in excludes and d not in {".git", ".hg", ".svn"}
        ]
    return sorted(set(repos), key=lambda p: str(p).lower())


def repo_id_from_path(repo: Path) -> str:
    return repo.name


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def iter_files(repo: Path, exclude_dirs: Iterable[str], max_file_mb: float) -> Iterator[Path]:
    excludes = set(exclude_dirs)
    max_bytes = int(max_file_mb * 1024 * 1024)
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in excludes and not d.startswith(".") or d == ".github"]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if p.stat().st_size <= max_bytes:
                    yield p
            except Exception:
                continue


def detect_kind(path: Path, rel: str) -> str:
    name = path.name.lower()
    low = rel.lower()
    if name in MANIFEST_NAMES:
        return "manifest"
    if name in {"dockerfile", "containerfile"} or name.endswith(".dockerfile"):
        return "dockerfile"
    if any(x in name for x in API_PATTERNS) or path.suffix.lower() in {".proto", ".graphql"}:
        return "api_contract"
    if "/docs/" in low or name.startswith("readme") or path.suffix.lower() == ".md":
        return "doc"
    if path.suffix.lower() in {".yaml", ".yml", ".json", ".tf"} and any(m in low for m in ["helm", "k8s", "kubernetes", "terraform", "deploy", "argocd", "chart"]):
        return "deploy_config"
    if any(h in low for h in ROUTE_FILE_HINTS):
        return "source_route_hint"
    return "source" if path.suffix.lower() in EXT_LANG else "other"


def classify_repo(files: List[Dict[str, Any]], languages: Counter) -> str:
    names = {Path(f["rel_path"]).name.lower() for f in files}
    rels = [f["rel_path"].lower() for f in files]
    has_deploy = any(f["kind"] in {"dockerfile", "deploy_config"} for f in files)
    has_api = any(f["kind"] == "api_contract" for f in files)
    has_tf = any(r.endswith(".tf") for r in rels)
    has_frontend = "package.json" in names and any(x in " ".join(rels) for x in ["react", "next.config", "vue", "angular", "svelte"])
    has_job = any(x in " ".join(rels) for x in ["airflow", "dag", "cron", "worker", "job", "scheduler", "dbt"])
    if has_tf and not has_api and not any(f["kind"] == "source" for f in files):
        return "infra"
    if has_api or has_deploy:
        return "service"
    if has_frontend:
        return "frontend"
    if has_job:
        return "job_or_data_pipeline"
    if any(n in names for n in MANIFEST_NAMES):
        return "library"
    return "unknown"


def scan_repo(repo: Path, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    repo_id = repo_id_from_path(repo)
    exclude_dirs = cfg["repos"].get("exclude_dirs", [])
    max_file_mb = float(cfg["repos"].get("max_file_mb", 2))
    files: List[Dict[str, Any]] = []
    language_counts: Counter = Counter()
    for path in iter_files(repo, exclude_dirs, max_file_mb):
        rel = str(path.relative_to(repo))
        ext = path.suffix.lower() or path.name
        lang = EXT_LANG.get(ext)
        if lang:
            language_counts[lang] += 1
        kind = detect_kind(path, rel)
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        files.append({
            "repo_id": repo_id,
            "path": str(path),
            "rel_path": rel,
            "ext": ext,
            "size_bytes": size,
            "kind": kind,
            "sha1": file_sha1(path),
            "indexed_at": now_iso(),
        })
    primary_language = language_counts.most_common(1)[0][0] if language_counts else "unknown"
    names = {Path(f["rel_path"]).name.lower() for f in files}
    rels = [f["rel_path"].lower() for f in files]
    repo_url = run_cmd(["git", "config", "--get", "remote.origin.url"], cwd=repo)
    commit_sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo)
    default_branch = run_cmd(["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd=repo).replace("origin/", "")
    last_commit_at = run_cmd(["git", "log", "-1", "--format=%cI"], cwd=repo)
    total_size = sum(f["size_bytes"] for f in files) / (1024 * 1024)
    # Worktree + .git size can be expensive; keep approximate from scanned files.
    inventory = {
        "repo_id": repo_id,
        "repo_path": str(repo),
        "repo_url": repo_url,
        "default_branch": default_branch,
        "commit_sha": commit_sha,
        "primary_language": primary_language,
        "languages": json_dumps(dict(language_counts)),
        "probable_type": classify_repo(files, language_counts),
        "size_mb": round(total_size, 3),
        "worktree_size_mb": round(total_size, 3),
        "last_commit_at": last_commit_at,
        "has_codeowners": any("codeowners" == n for n in names),
        "has_readme": any(n.startswith("readme") for n in names),
        "has_openapi": any("openapi" in r or "swagger" in r for r in rels),
        "has_proto": any(r.endswith(".proto") for r in rels),
        "has_graphql": any(r.endswith(".graphql") or "schema.graphql" in r for r in rels),
        "has_asyncapi": any("asyncapi" in r for r in rels),
        "has_dockerfile": any(Path(r).name.lower() == "dockerfile" or r.endswith(".dockerfile") for r in rels),
        "has_k8s": any(any(m in r for m in ["k8s", "kubernetes", "helm", "chart", "deploy"]) for r in rels),
        "has_helm": any("chart.yaml" in r or "values.yaml" in r for r in rels),
        "has_terraform": any(r.endswith(".tf") for r in rels),
        "has_package_manifest": any(Path(r).name in MANIFEST_NAMES for r in rels),
        "indexed_at": now_iso(),
    }
    ownership = extract_ownership(repo, repo_id, files)
    history = extract_git_history(repo, repo_id, limit=200 if cfg["repos"].get("include_git_history_metadata", True) else 0)
    return inventory, files, ownership, history


def extract_ownership(repo: Path, repo_id: str, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    codeowners_files = [f for f in files if Path(f["rel_path"]).name.upper() == "CODEOWNERS"]
    for f in codeowners_files:
        text = safe_read_text(Path(f["path"])) or ""
        owners = set(re.findall(r"@[A-Za-z0-9_.\-/]+", text))
        out.append({
            "entity_id": f"repo:{repo_id}", "entity_type": "repo",
            "owner_team": ",".join(sorted(owners)), "owner_people": "",
            "source": "CODEOWNERS", "confidence": 0.8 if owners else 0.4,
            "evidence_ref": f["rel_path"],
        })
    authors = run_cmd(["git", "log", "--format=%ae", "-n", "200"], cwd=repo)
    if authors:
        top = Counter(a for a in authors.splitlines() if a).most_common(10)
        out.append({
            "entity_id": f"repo:{repo_id}", "entity_type": "repo",
            "owner_team": "", "owner_people": ",".join(a for a, _ in top),
            "source": "git_recent_authors", "confidence": 0.35,
            "evidence_ref": "git log -n 200",
        })
    return out


def extract_git_history(repo: Path, repo_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    fmt = "%H%x1f%an%x1f%ae%x1f%cI%x1f%s"
    raw = run_cmd(["git", "log", f"-n{limit}", f"--format={fmt}", "--name-only"], cwd=repo, timeout=60)
    rows: List[Dict[str, Any]] = []
    cur = None
    changed: List[str] = []
    for line in raw.splitlines():
        if "\x1f" in line:
            if cur:
                cur["changed_files"] = json_dumps(changed)
                rows.append(cur)
            parts = line.split("\x1f")
            cur = {
                "repo_id": repo_id, "commit_sha": parts[0], "author_name": parts[1],
                "author_email": parts[2], "commit_time": parts[3], "subject": parts[4],
                "changed_files": "[]",
            }
            changed = []
        elif line.strip():
            changed.append(line.strip())
    if cur:
        cur["changed_files"] = json_dumps(changed)
        rows.append(cur)
    return rows


def scan_all_repos(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    root = Path(cfg["repos"]["root"])
    repos = find_git_repos(root, cfg["repos"].get("exclude_dirs", []))
    inventories: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    ownership: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    for repo in repos:
        inv, fs, own, hist = scan_repo(repo, cfg)
        inventories.append(inv)
        files.extend(fs)
        ownership.extend(own)
        history.extend(hist)
    return {
        "repo_inventory": pd.DataFrame(inventories),
        "repo_files": pd.DataFrame(files),
        "ownership": pd.DataFrame(ownership),
        "repo_git_history": pd.DataFrame(history),
    }

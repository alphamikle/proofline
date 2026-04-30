from __future__ import annotations

import itertools
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from proofline.extractors.repo import detect_kind, find_git_repos, repo_id_from_path
from proofline.utils import json_dumps, normalize_name, now_iso, run_cmd, stable_id

JIRA_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
URL_RE = re.compile(r"https?://[^\s)>\"']+")
PR_RE = re.compile(r"(?:pull request|pull|pr|merge request|mr)[\s#:!]*([0-9]+)", re.I)
REVERT_SHA_RE = re.compile(r"This reverts commit ([0-9a-f]{7,40})", re.I)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
SERVICE_FILE_KINDS = {"api_contract", "deploy_config", "dockerfile", "manifest", "source_route_hint", "source"}

LOCKFILES = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "go.sum", "poetry.lock",
    "Cargo.lock", "Gemfile.lock", "composer.lock",
}


def build_git_history(cfg: Dict[str, Any], repo_inventory: pd.DataFrame | None = None) -> Dict[str, pd.DataFrame]:
    gh_cfg = _git_history_cfg(cfg)
    if not gh_cfg.get("enabled", True):
        return _empty_frames()
    repos = _repos_from_inventory(repo_inventory) if repo_inventory is not None and not repo_inventory.empty else [
        (repo_id_from_path(p), p) for p in find_git_repos(Path(cfg.get("repos", {}).get("root", "./repos")), cfg.get("repos", {}).get("exclude_dirs", []))
    ]
    rows = {name: [] for name in _empty_frames()}
    for repo_id, repo_path in repos:
        repo_rows = extract_repo_git_history(Path(repo_path), repo_id, gh_cfg)
        for key, values in repo_rows.items():
            rows[key].extend(values)
    return {key: pd.DataFrame(values) for key, values in rows.items()}


def extract_repo_git_history(
    repo: Path,
    repo_id: str,
    cfg: Dict[str, Any],
    *,
    progress_desc: str | None = None,
    progress_position: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    prepare_git_history(repo, cfg)
    commits = extract_commits(repo, repo_id, cfg)
    commit_by_sha = {str(c["commit_sha"]): c for c in commits}
    file_changes: List[Dict[str, Any]] = []
    hunks: List[Dict[str, Any]] = []
    semantic: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    reverts: List[Dict[str, Any]] = []

    commit_iter: Iterable[Dict[str, Any]] = commits
    if progress_desc:
        try:
            from tqdm.auto import tqdm

            commit_iter = tqdm(commits, total=len(commits), desc=progress_desc, unit="commit", position=progress_position, leave=False)
        except Exception:
            commit_iter = commits
    for commit in commit_iter:
        sha = str(commit["commit_sha"])
        file_changes.extend(extract_file_changes(repo, repo_id, sha, cfg))
        if cfg.get("patch_hunks", True):
            hs = extract_patch_hunks(repo, repo_id, sha, cfg)
            hunks.extend(hs)
            semantic.extend(extract_semantic_changes(repo_id, sha, hs))
        links.extend(detect_links(repo_id, sha, f"{commit.get('subject') or ''}\n{commit.get('body') or ''}"))
        if commit.get("is_revert") or commit.get("reverts_commit_sha"):
            reverts.append({
                "repo_id": repo_id,
                "revert_commit_sha": sha,
                "reverted_commit_sha": commit.get("reverts_commit_sha") or "",
                "confidence": 0.9 if commit.get("reverts_commit_sha") else 0.55,
                "evidence": commit.get("subject") or "",
            })

    blame = extract_current_blame(repo, repo_id, cfg)
    cochange = build_cochange_edges(repo_id, file_changes, commit_by_sha, cfg)
    return {
        "git_commits": commits,
        "git_file_changes": file_changes,
        "git_patch_hunks": hunks,
        "git_detected_links": links,
        "git_reverts": reverts,
        "git_blame_current": blame,
        "git_semantic_changes": semantic,
        "git_cochange_edges": cochange,
    }


def extract_repo_git_blame(
    repo: Path,
    repo_id: str,
    cfg: Dict[str, Any],
    *,
    progress_desc: str | None = None,
    progress_position: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    gh_cfg = dict(_git_history_cfg({"git_history": cfg}))
    gh_cfg["current_blame"] = True
    return {"git_blame_current": extract_current_blame(repo, repo_id, gh_cfg, progress_desc=progress_desc, progress_position=progress_position)}


def prepare_git_history(repo: Path, cfg: Dict[str, Any]) -> None:
    if cfg.get("write_commit_graph", True):
        subprocess.run(["git", "commit-graph", "write", "--reachable", "--changed-paths"], cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=120)


def extract_commits(repo: Path, repo_id: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    fmt = "%H%x1f%P%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%aI%x1f%cI%x1f%s%x1f%B%x1e"
    cmd = ["git", "log", f"--format={fmt}"]
    max_commits = cfg.get("max_commits_per_repo")
    if max_commits:
        cmd.insert(2, f"-n{int(max_commits)}")
    since = _since_arg(cfg.get("metadata_days"))
    if since:
        cmd.insert(2, since)
    raw = run_cmd(cmd, cwd=repo, timeout=int(cfg.get("git_timeout_seconds", 300)))
    rows: List[Dict[str, Any]] = []
    stop_commits = set(cfg.get("stop_commit_shas") or [])
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        parts = rec.split("\x1f", 9)
        if len(parts) < 10:
            continue
        if parts[0] in stop_commits:
            break
        subject = parts[8].strip()
        body = parts[9].strip()
        target = detect_revert_target(subject, body)
        rows.append({
            "repo_id": repo_id,
            "commit_sha": parts[0],
            "parent_shas": json_dumps([p for p in parts[1].split() if p]),
            "author_name": parts[2],
            "author_email": parts[3],
            "committer_name": parts[4],
            "committer_email": parts[5],
            "author_time": parts[6],
            "commit_time": parts[7],
            "subject": subject,
            "body": body,
            "is_merge": len([p for p in parts[1].split() if p]) > 1,
            "is_revert": is_revert_like(subject, body),
            "is_hotfix": bool(re.search(r"\b(hotfix|emergency|incident|rollback|roll back)\b", f"{subject}\n{body}", re.I)),
            "reverts_commit_sha": target,
            "detected_jira_keys": json_dumps(sorted(set(JIRA_RE.findall(f"{subject}\n{body}")))),
            "detected_urls": json_dumps(sorted(set(URL_RE.findall(f"{subject}\n{body}")))),
            "indexed_at": now_iso(),
        })
    return rows


def extract_file_changes(repo: Path, repo_id: str, commit_sha: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rename_flag = "-M" if cfg.get("rename_detection", True) else "--no-renames"
    raw = run_cmd(["git", "show", "--format=", "--numstat", "--name-status", rename_flag, commit_sha], cwd=repo, timeout=int(cfg.get("git_timeout_seconds", 300)))
    numstats: Dict[str, Tuple[int | None, int | None, bool]] = {}
    statuses: List[Tuple[str, str, str, int | None]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and _looks_numstat(parts[0], parts[1]):
            added = None if parts[0] == "-" else int(parts[0])
            deleted = None if parts[1] == "-" else int(parts[1])
            path = parts[-1]
            numstats[path] = (added, deleted, added is None or deleted is None)
            continue
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            statuses.append((status, parts[1], parts[2], _score(status)))
        elif status.startswith("C") and len(parts) >= 3:
            statuses.append((status, parts[1], parts[2], _score(status)))
        elif len(parts) >= 2:
            statuses.append((status, "", parts[1], None))
    rows = []
    for status, old_path, new_path, score in statuses:
        added, deleted, binary = numstats.get(new_path, (None, None, False))
        rows.append({
            "repo_id": repo_id,
            "commit_sha": commit_sha,
            "old_path": old_path,
            "new_path": new_path,
            "change_type": _change_type(status),
            "added_lines": added,
            "deleted_lines": deleted,
            "is_rename": status.startswith("R"),
            "rename_score": score,
            "is_copy": status.startswith("C"),
            "is_binary": binary,
            "file_extension": Path(new_path or old_path).suffix.lower(),
            "file_category": classify_history_file(new_path or old_path),
        })
    return rows


def extract_patch_hunks(repo: Path, repo_id: str, commit_sha: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rename_flag = "-M" if cfg.get("rename_detection", True) else "--no-renames"
    raw = run_cmd(["git", "show", "--format=", "--unified=3", "--no-ext-diff", rename_flag, commit_sha], cwd=repo, timeout=int(cfg.get("git_timeout_seconds", 300)))
    rows: List[Dict[str, Any]] = []
    current_file = ""
    old_file = ""
    hunk: Dict[str, Any] | None = None
    added: List[str] = []
    removed: List[str] = []
    context: List[str] = []

    def flush() -> None:
        nonlocal hunk, added, removed, context
        if not hunk:
            return
        text_size = sum(len(x) for x in added + removed + context)
        if text_size <= int(cfg.get("max_hunk_chars", 20000)):
            hunk["added_text"] = "\n".join(added)
            hunk["removed_text"] = "\n".join(removed)
            hunk["context_text"] = "\n".join(context)
            hunk["classification"] = classify_hunk(hunk["file_path"], hunk["added_text"], hunk["removed_text"])
            rows.append(hunk)
        hunk = None
        added = []
        removed = []
        context = []

    for line in raw.splitlines():
        if line.startswith("diff --git "):
            flush()
            current_file = ""
            old_file = ""
            continue
        if line.startswith("--- "):
            old_file = _patch_path(line[4:])
            continue
        if line.startswith("+++ "):
            current_file = _patch_path(line[4:])
            continue
        match = HUNK_RE.match(line)
        if match:
            flush()
            file_path = current_file or old_file
            if not should_index_patch(file_path, cfg):
                hunk = None
                continue
            old_start, old_lines, new_start, new_lines, header = match.groups()
            hunk_id = stable_id(repo_id, commit_sha, file_path, old_start, new_start, header)
            hunk = {
                "repo_id": repo_id,
                "commit_sha": commit_sha,
                "file_path": file_path,
                "hunk_id": hunk_id,
                "old_start": int(old_start),
                "old_lines": int(old_lines or "1"),
                "new_start": int(new_start),
                "new_lines": int(new_lines or "1"),
                "hunk_header": header,
                "added_text": "",
                "removed_text": "",
                "context_text": "",
                "classification": "",
            }
            continue
        if not hunk:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        elif line.startswith(" "):
            context.append(line[1:])
    flush()
    return rows


def extract_current_blame(
    repo: Path,
    repo_id: str,
    cfg: Dict[str, Any],
    *,
    progress_desc: str | None = None,
    progress_position: int = 1,
) -> List[Dict[str, Any]]:
    if not cfg.get("current_blame", True):
        return []
    ignore = repo / ".git-blame-ignore-revs"
    ignore_args = ["--ignore-revs-file", str(ignore)] if cfg.get("use_blame_ignore_revs", True) and ignore.exists() else []
    files = run_cmd(["git", "ls-files"], cwd=repo, timeout=120).splitlines()
    rows: List[Dict[str, Any]] = []
    max_files = int(cfg.get("max_blame_files", 500))
    candidates = [rel for rel in files if should_index_blame(rel)][:max_files]
    iterator: Iterable[str] = candidates
    if progress_desc:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(candidates, total=len(candidates), desc=progress_desc, unit="file", position=progress_position, leave=False)
        except Exception:
            iterator = candidates
    for rel in iterator:
        if len(rows) >= int(cfg.get("max_blame_rows", 50000)):
            break
        raw = run_cmd(["git", "blame", "--line-porcelain", *ignore_args, "--", rel], cwd=repo, timeout=120)
        rows.extend(compact_blame(repo_id, rel, raw, bool(ignore_args)))
    return rows


def compact_blame(repo_id: str, rel: str, raw: str, ignored: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cur_sha = ""
    cur_author = ""
    cur_time = ""
    line_no = 0
    last: Dict[str, Any] | None = None
    for line in raw.splitlines():
        if re.match(r"^[0-9a-f]{40} ", line):
            cur_sha = line.split()[0]
        elif line.startswith("author-mail "):
            cur_author = line.replace("author-mail ", "").strip("<>")
        elif line.startswith("author-time "):
            try:
                cur_time = datetime.fromtimestamp(int(line.split()[1]), timezone.utc).isoformat()
            except Exception:
                cur_time = ""
        elif line.startswith("\t"):
            line_no += 1
            if last and last["last_commit_sha"] == cur_sha and last["last_author_email"] == cur_author:
                last["line_end"] = line_no
            else:
                last = {
                    "repo_id": repo_id,
                    "file_path": rel,
                    "line_start": line_no,
                    "line_end": line_no,
                    "symbol_id": "",
                    "last_commit_sha": cur_sha,
                    "last_author_email": cur_author,
                    "last_commit_time": cur_time,
                    "ignored_revs_applied": ignored,
                }
                rows.append(last)
    return rows


def build_cochange_edges(repo_id: str, file_changes: List[Dict[str, Any]], commits: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    window_days = cfg.get("cochange_window_days")
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(window_days)) if window_days else None
    by_commit: Dict[str, List[str]] = defaultdict(list)
    by_jira: Dict[str, List[str]] = defaultdict(list)
    by_pr: Dict[str, List[str]] = defaultdict(list)
    last_seen: Dict[Tuple[str, str], str] = {}
    for fc in file_changes:
        sha = str(fc.get("commit_sha") or "")
        commit = commits.get(sha, {})
        if cutoff and not _after_cutoff(str(commit.get("commit_time") or ""), cutoff):
            continue
        path = str(fc.get("new_path") or fc.get("old_path") or "")
        if not path or fc.get("is_binary"):
            continue
        by_commit[sha].append(path)
        for key in _json_list(commit.get("detected_jira_keys")):
            by_jira[key].append(path)
        for pr in detect_pr_refs(f"{commit.get('subject') or ''}\n{commit.get('body') or ''}\n{commit.get('detected_urls') or ''}"):
            by_pr[pr].append(path)
    counts: Counter[Tuple[str, str]] = Counter()
    jira_counts: Counter[Tuple[str, str]] = Counter()
    pr_counts: Counter[Tuple[str, str]] = Counter()
    for sha, paths in by_commit.items():
        for a, b in _pairs(paths):
            counts[(a, b)] += 1
            last_seen[(a, b)] = str(commits.get(sha, {}).get("commit_time") or "")
    for paths in by_jira.values():
        for a, b in _pairs(paths):
            jira_counts[(a, b)] += 1
    for paths in by_pr.values():
        for a, b in _pairs(paths):
            pr_counts[(a, b)] += 1
    rows = []
    for (a, b), count in counts.items():
        confidence = min(0.95, 0.25 + count * 0.08 + jira_counts[(a, b)] * 0.08 + pr_counts[(a, b)] * 0.08)
        if count < int(cfg.get("cochange_min_count", 1)):
            continue
        rows.append({
            "from_entity": f"file:{repo_id}:{a}",
            "to_entity": f"file:{repo_id}:{b}",
            "entity_type": "file",
            "cochange_type": "same_commit",
            "same_commit_count": int(count),
            "same_pr_count": int(pr_counts[(a, b)]),
            "same_jira_count": int(jira_counts[(a, b)]),
            "same_release_count": 0,
            "last_cochanged_at": last_seen.get((a, b), ""),
            "window_days": int(window_days) if window_days else None,
            "confidence": confidence,
        })
    return rows


def extract_semantic_changes(repo_id: str, commit_sha: str, hunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for h in hunks:
        path = str(h.get("file_path") or "")
        added = str(h.get("added_text") or "")
        removed = str(h.get("removed_text") or "")
        context = str(h.get("context_text") or "")
        events = []
        if is_openapi_file(path):
            events.extend(openapi_events(path, added, removed, context))
        elif path.endswith(".proto"):
            events.extend(proto_events(path, added, removed, context))
        elif path.endswith(".graphql") or path.endswith(".gql"):
            events.extend(graphql_events(path, added, removed, context))
        elif is_migration_file(path):
            events.extend(sql_migration_events(path, added, removed, context))
        elif Path(path).name in {"package.json", "pyproject.toml", "go.mod", "pom.xml", "build.gradle", "Cargo.toml", "Gemfile"}:
            events.extend(package_events(path, added, removed, context))
        for ev in events:
            rows.append({
                "repo_id": repo_id,
                "service_id": normalize_name(repo_id),
                "commit_sha": commit_sha,
                "change_type": ev["change_type"],
                "entity_type": ev["entity_type"],
                "entity_id": ev["entity_id"],
                "before_value": ev.get("before_value", ""),
                "after_value": ev.get("after_value", ""),
                "breaking_risk": ev.get("breaking_risk", "unknown"),
                "confidence": ev.get("confidence", 0.45),
                "evidence_id": h.get("hunk_id"),
            })
    return rows


def openapi_events(path: str, added: str, removed: str, context: str) -> List[Dict[str, Any]]:
    events = []
    for method, api_path in _api_paths(added):
        events.append(_event("API_ENDPOINT_ADDED", "endpoint", f"{method.upper()} {api_path}", after=api_path, risk="low", conf=0.55))
    for method, api_path in _api_paths(removed):
        events.append(_event("API_ENDPOINT_REMOVED", "endpoint", f"{method.upper()} {api_path}", before=api_path, risk="breaking", conf=0.65))
    for field in _schema_fields(added):
        change = "API_REQUIRED_FIELD_ADDED" if field in added_required_fields(added) else "API_FIELD_ADDED"
        events.append(_event(change, "api_field", f"{path}:{field}", after=field, risk="breaking" if change.endswith("REQUIRED_FIELD_ADDED") else "low", conf=0.45))
    for field in _schema_fields(removed):
        events.append(_event("API_FIELD_REMOVED", "api_field", f"{path}:{field}", before=field, risk="breaking", conf=0.45))
    for value in _enum_values(removed):
        events.append(_event("API_ENUM_VALUE_REMOVED", "api_enum_value", f"{path}:{value}", before=value, risk="breaking", conf=0.5))
    return events


def proto_events(path: str, added: str, removed: str, context: str) -> List[Dict[str, Any]]:
    events = []
    for direction, text, change in [("after", added, "PROTO_FIELD_ADDED"), ("before", removed, "PROTO_FIELD_REMOVED")]:
        for typ, name, num in re.findall(r"\b(?:optional|required|repeated)?\s*([A-Za-z_][\w.]*)\s+([A-Za-z_]\w*)\s*=\s*(\d+)", text):
            risk = "breaking" if direction == "before" else ("medium" if "required" in text else "low")
            events.append(_event(change, "proto_field", f"{path}:{name}:{num}", **{direction: f"{typ} {name} = {num}"}, risk=risk, conf=0.6))
    added_nums = {n for _, _, n in re.findall(r"\b([A-Za-z_][\w.]*)\s+([A-Za-z_]\w*)\s*=\s*(\d+)", added)}
    removed_nums = {n for _, _, n in re.findall(r"\b([A-Za-z_][\w.]*)\s+([A-Za-z_]\w*)\s*=\s*(\d+)", removed)}
    for num in sorted(added_nums & removed_nums):
        events.append(_event("PROTO_FIELD_NUMBER_REUSED", "proto_field_number", f"{path}:{num}", before=num, after=num, risk="breaking", conf=0.5))
    return events


def graphql_events(path: str, added: str, removed: str, context: str) -> List[Dict[str, Any]]:
    events = []
    for name, typ in re.findall(r"^\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_][\w!\[\]]*)", added, re.M):
        events.append(_event("GRAPHQL_FIELD_ADDED", "graphql_field", f"{path}:{name}", after=f"{name}: {typ}", risk="low", conf=0.5))
    for name, typ in re.findall(r"^\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_][\w!\[\]]*)", removed, re.M):
        events.append(_event("GRAPHQL_FIELD_REMOVED", "graphql_field", f"{path}:{name}", before=f"{name}: {typ}", risk="breaking", conf=0.55))
    for enum in re.findall(r"^\s*([A-Z][A-Z0-9_]+)\s*$", removed, re.M):
        events.append(_event("GRAPHQL_ENUM_VALUE_REMOVED", "graphql_enum_value", f"{path}:{enum}", before=enum, risk="breaking", conf=0.45))
    return events


def sql_migration_events(path: str, added: str, removed: str, context: str) -> List[Dict[str, Any]]:
    events = []
    for text, direction in [(added, "after"), (removed, "before")]:
        for table in re.findall(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.`\"]+)", text, re.I):
            events.append(_event("DB_TABLE_CREATED" if direction == "after" else "DB_TABLE_CREATE_REMOVED", "db_table", table.strip("`\""), **{direction: table}, risk="medium", conf=0.55))
        for table in re.findall(r"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([A-Za-z0-9_.`\"]+)", text, re.I):
            events.append(_event("DB_TABLE_DROPPED" if direction == "after" else "DB_TABLE_DROP_REMOVED", "db_table", table.strip("`\""), **{direction: table}, risk="breaking", conf=0.65))
        for table, column in re.findall(r"\bALTER\s+TABLE\s+([A-Za-z0-9_.`\"]+)\s+DROP\s+COLUMN\s+([A-Za-z0-9_`\"]+)", text, re.I):
            clean_table = table.strip("`\"")
            clean_column = column.strip("`\"")
            events.append(_event("DB_COLUMN_DROPPED", "db_column", f"{clean_table}.{clean_column}", **{direction: clean_column}, risk="breaking", conf=0.65))
        for table, column in re.findall(r"\bALTER\s+TABLE\s+([A-Za-z0-9_.`\"]+)\s+ADD\s+COLUMN\s+([A-Za-z0-9_`\"]+)", text, re.I):
            clean_table = table.strip("`\"")
            clean_column = column.strip("`\"")
            events.append(_event("DB_COLUMN_ADDED", "db_column", f"{clean_table}.{clean_column}", **{direction: clean_column}, risk="low", conf=0.55))
    return events


def package_events(path: str, added: str, removed: str, context: str) -> List[Dict[str, Any]]:
    events = []
    added_deps = _dependency_lines(added)
    removed_deps = _dependency_lines(removed)
    for dep, version in added_deps.items():
        if dep in removed_deps:
            events.append(_event("PACKAGE_DEPENDENCY_UPGRADED", "package_dependency", dep, before=removed_deps[dep], after=version, risk="medium", conf=0.45))
        else:
            events.append(_event("PACKAGE_DEPENDENCY_ADDED", "package_dependency", dep, after=version, risk="medium", conf=0.45))
    for dep, version in removed_deps.items():
        if dep not in added_deps:
            events.append(_event("PACKAGE_DEPENDENCY_REMOVED", "package_dependency", dep, before=version, risk="medium", conf=0.45))
    return events


def detect_links(repo_id: str, commit_sha: str, text: str) -> List[Dict[str, Any]]:
    rows = []
    for key in sorted(set(JIRA_RE.findall(text))):
        rows.append({"repo_id": repo_id, "commit_sha": commit_sha, "link_type": "jira", "target": key, "source_text": key})
    for pr in detect_pr_refs(text):
        rows.append({"repo_id": repo_id, "commit_sha": commit_sha, "link_type": "pr", "target": pr, "source_text": pr})
    for url in sorted(set(URL_RE.findall(text))):
        low = url.lower()
        if "atlassian" in low and "/wiki/" in low:
            typ = "confluence"
        elif "datadog" in low:
            typ = "datadog"
        elif "/pull/" in low or "/merge_requests/" in low:
            typ = "pr"
        else:
            typ = "url"
        rows.append({"repo_id": repo_id, "commit_sha": commit_sha, "link_type": typ, "target": url, "source_text": url})
    return rows


def detect_pr_refs(text: str) -> List[str]:
    refs = set(PR_RE.findall(text))
    refs.update(re.findall(r"/pull/(\d+)", text, re.I))
    refs.update(re.findall(r"/merge_requests/(\d+)", text, re.I))
    return sorted(refs)


def _git_history_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "enabled": True,
        "metadata_days": None,
        "max_commits_per_repo": None,
        "rename_detection": True,
        "patch_hunks": True,
        "current_blame": True,
        "use_blame_ignore_revs": True,
        "write_commit_graph": True,
        "cochange_window_days": 730,
        "cochange_min_count": 1,
        "max_hunk_chars": 20000,
        "max_blame_files": 500,
        "max_blame_rows": 50000,
        "git_timeout_seconds": 300,
    }
    out.update(cfg.get("git_history") or {})
    return out


def _empty_frames() -> Dict[str, pd.DataFrame]:
    return {
        "git_commits": pd.DataFrame(),
        "git_file_changes": pd.DataFrame(),
        "git_patch_hunks": pd.DataFrame(),
        "git_detected_links": pd.DataFrame(),
        "git_reverts": pd.DataFrame(),
        "git_blame_current": pd.DataFrame(),
        "git_semantic_changes": pd.DataFrame(),
        "git_cochange_edges": pd.DataFrame(),
    }


def _repos_from_inventory(inv: pd.DataFrame) -> List[Tuple[str, Path]]:
    return [(str(r.get("repo_id") or ""), Path(str(r.get("repo_path") or ""))) for _, r in inv.iterrows() if str(r.get("repo_path") or "")]


def _since_arg(days: Any) -> str:
    if days in (None, "", 0):
        return ""
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
    return f"--since={cutoff.date().isoformat()}"


def _looks_numstat(a: str, b: str) -> bool:
    return (a == "-" or a.isdigit()) and (b == "-" or b.isdigit())


def _score(status: str) -> int | None:
    return int(status[1:]) if len(status) > 1 and status[1:].isdigit() else None


def _change_type(status: str) -> str:
    first = status[:1]
    return {"A": "added", "M": "modified", "D": "deleted", "R": "renamed", "C": "copied", "T": "type_changed"}.get(first, status.lower())


def classify_history_file(path: str) -> str:
    name = Path(path).name
    if name in LOCKFILES:
        return "lockfile"
    return detect_kind(Path(path), path)


def should_index_patch(path: str, cfg: Dict[str, Any]) -> bool:
    if not path or path == "/dev/null":
        return False
    p = Path(path)
    if p.name in LOCKFILES:
        return False
    if any(part in {"node_modules", "vendor", "dist", "build", "target"} for part in p.parts):
        return False
    return classify_history_file(path) in SERVICE_FILE_KINDS or is_migration_file(path)


def should_index_blame(path: str) -> bool:
    return should_index_patch(path, {})


def _patch_path(raw: str) -> str:
    if raw == "/dev/null":
        return raw
    if raw.startswith("a/") or raw.startswith("b/"):
        return raw[2:]
    return raw


def classify_hunk(path: str, added: str, removed: str) -> str:
    if is_openapi_file(path):
        return "api_contract_change"
    if path.endswith(".proto") or path.endswith(".graphql") or path.endswith(".gql"):
        return "schema_change"
    if is_migration_file(path):
        return "db_migration_change"
    if Path(path).name in LOCKFILES:
        return "metadata_only"
    if Path(path).name in {"package.json", "pyproject.toml", "go.mod", "pom.xml", "build.gradle", "Cargo.toml", "Gemfile"}:
        return "dependency_change"
    if re.search(r"\b[A-Z_][A-Z0-9_]*=", added + "\n" + removed):
        return "config_change"
    return "source_code_change"


def is_revert_like(subject: str, body: str) -> bool:
    return bool(re.search(r"\b(revert|rollback|roll back)\b", f"{subject}\n{body}", re.I))


def detect_revert_target(subject: str, body: str) -> str:
    m = REVERT_SHA_RE.search(f"{subject}\n{body}")
    return m.group(1) if m else ""


def _json_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        import orjson
        return [str(x) for x in orjson.loads(str(value))]
    except Exception:
        return []


def _after_cutoff(value: str, cutoff: datetime) -> bool:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")) >= cutoff
    except Exception:
        return True


def _pairs(paths: Iterable[str]) -> Iterable[Tuple[str, str]]:
    unique = sorted(set(paths))
    return itertools.combinations(unique, 2)


def is_openapi_file(path: str) -> bool:
    low = path.lower()
    return ("openapi" in low or "swagger" in low) and Path(path).suffix.lower() in {".yaml", ".yml", ".json"}


def is_migration_file(path: str) -> bool:
    low = path.lower()
    return Path(path).suffix.lower() == ".sql" and ("migration" in low or "/migrate" in low or "/migrations" in low)


def _api_paths(text: str) -> List[Tuple[str, str]]:
    out = []
    current_path = ""
    for line in text.splitlines():
        m_path = re.match(r"\s*(/[A-Za-z0-9_{}./:-]+)\s*:\s*$", line)
        if m_path:
            current_path = m_path.group(1)
            continue
        m_method = re.match(r"\s*(get|post|put|patch|delete|options|head)\s*:\s*$", line, re.I)
        if m_method and current_path:
            out.append((m_method.group(1).lower(), current_path))
    return out


def _schema_fields(text: str) -> List[str]:
    fields = []
    for name in re.findall(r"^\s{2,}([A-Za-z_][\w-]*)\s*:\s*$", text, re.M):
        if name not in {"properties", "responses", "requestBody", "schema", "content", "required", "enum"}:
            fields.append(name)
    return fields


def added_required_fields(text: str) -> set[str]:
    values = set()
    for line in text.splitlines():
        m = re.match(r"\s*-\s+([A-Za-z_][\w-]*)\s*$", line)
        if m:
            values.add(m.group(1))
    return values


def _enum_values(text: str) -> List[str]:
    return [m.group(1) for m in re.finditer(r"^\s*-\s+([A-Za-z0-9_.-]+)\s*$", text, re.M)]


def _dependency_lines(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r'\s*["\']?([@A-Za-z0-9_.\-/]+)["\']?\s*[:=]\s*["\']?([^,"\'\s]+)', line)
        if m and not m.group(1).lower() in {"name", "version", "description"}:
            out[m.group(1)] = m.group(2)
        m2 = re.match(r"\s*([A-Za-z0-9_.\-/]+)\s+v?([0-9][^\s]*)", line)
        if m2:
            out[m2.group(1)] = m2.group(2)
    return out


def _event(change: str, entity_type: str, entity: str, before: str = "", after: str = "", risk: str = "unknown", conf: float = 0.45) -> Dict[str, Any]:
    return {
        "change_type": change,
        "entity_type": entity_type,
        "entity_id": entity,
        "before_value": before,
        "after_value": after,
        "breaking_risk": risk,
        "confidence": conf,
    }

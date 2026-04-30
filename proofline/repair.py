from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import yaml

from proofline.config import default_config, ensure_dirs, load_config, package_root


@dataclass
class RepairStep:
    name: str
    ok: bool
    action: str
    details: str = ""


def run_repair(
    *,
    config_path: str | Path,
    bin_dir: Optional[str] = None,
    dry_run: bool = False,
    skip_python_deps: bool = False,
    skip_cgc: bool = False,
    skip_bin_links: bool = False,
) -> list[dict[str, Any]]:
    root = package_root()
    path = Path(config_path).expanduser()
    steps: list[RepairStep] = []

    cfg = ensure_config(path, dry_run=dry_run, steps=steps)
    if cfg is not None:
        ensure_workspace(cfg, dry_run=dry_run, steps=steps)

    if not skip_python_deps:
        install_python_dependencies(root, dry_run=dry_run, steps=steps)
    else:
        steps.append(RepairStep("python_dependencies", True, "skipped", "--skip-python-deps"))

    if not skip_bin_links:
        link_cli_binaries(bin_dir=bin_dir, dry_run=dry_run, steps=steps)
    else:
        steps.append(RepairStep("cli_links", True, "skipped", "--skip-bin-links"))

    if not skip_cgc:
        repair_cgc_stack(root, cfg or {}, dry_run=dry_run, steps=steps)
    else:
        steps.append(RepairStep("cgc_stack", True, "skipped", "--skip-cgc"))

    return [asdict(step) for step in steps]


def ensure_config(path: Path, *, dry_run: bool, steps: list[RepairStep]) -> dict[str, Any] | None:
    if not path.exists():
        if dry_run:
            steps.append(RepairStep("config", True, "would_create", str(path)))
            return default_config()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(default_config(), f, sort_keys=False, allow_unicode=False)
        steps.append(RepairStep("config", True, "created", str(path)))
    else:
        steps.append(RepairStep("config", True, "found", str(path)))

    try:
        cfg = load_config(path, quiet=True)
    except Exception as exc:
        steps.append(RepairStep("config_load", False, "failed", str(exc)))
        return None
    steps.append(RepairStep("config_load", True, "loaded", str(path)))
    return cfg


def ensure_workspace(cfg: dict[str, Any], *, dry_run: bool, steps: list[RepairStep]) -> None:
    workspace = str(cfg.get("workspace") or "./data")
    if dry_run:
        steps.append(RepairStep("workspace", True, "would_ensure_dirs", workspace))
        return
    ensure_dirs(cfg)
    steps.append(RepairStep("workspace", True, "ensured_dirs", workspace))


def install_python_dependencies(root: Path, *, dry_run: bool, steps: list[RepairStep]) -> None:
    requirements = root / "requirements.txt"
    if not requirements.exists():
        steps.append(RepairStep("python_dependencies", False, "missing_requirements", str(requirements)))
        return
    commands = [
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        [sys.executable, "-m", "pip", "install", "-e", str(root)],
    ]
    if dry_run:
        steps.append(
            RepairStep(
                "python_dependencies",
                True,
                "would_install",
                " && ".join(" ".join(cmd) for cmd in commands),
            )
        )
        return
    for cmd in commands:
        result = subprocess.run(cmd, cwd=str(root), check=False)
        if result.returncode != 0:
            steps.append(RepairStep("python_dependencies", False, "failed", " ".join(cmd)))
            return
    steps.append(RepairStep("python_dependencies", True, "installed", str(requirements)))


def link_cli_binaries(*, bin_dir: Optional[str], dry_run: bool, steps: list[RepairStep]) -> None:
    target_dir = Path(bin_dir or os.getenv("PROOFLINE_BIN_DIR") or Path.home() / ".local" / "bin").expanduser()
    scripts_dir = Path(sys.executable).parent
    created: list[str] = []
    missing: list[str] = []
    backups: list[str] = []
    for name in ("proofline", "pfl"):
        source = scripts_dir / name
        target = target_dir / name
        if dry_run:
            created.append(f"{target} -> {source}")
            continue
        if not source.exists():
            missing.append(str(source))
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                target.unlink()
            else:
                backup = target.with_name(f"{target.name}.repair.{int(time.time())}.bak")
                target.replace(backup)
                backups.append(f"{target} -> {backup}")
        target.symlink_to(source)
        created.append(f"{target} -> {source}")
    if missing and not created:
        steps.append(RepairStep("cli_links", False, "missing_console_scripts", "; ".join(missing)))
    elif dry_run:
        steps.append(RepairStep("cli_links", True, "would_link", "; ".join(created) or "already unavailable"))
    else:
        details = "; ".join(created)
        if backups:
            details += f"; backed up: {'; '.join(backups)}"
        if missing:
            details += f"; missing: {'; '.join(missing)}"
        steps.append(RepairStep("cli_links", True, "linked", details))


def repair_cgc_stack(root: Path, cfg: dict[str, Any], *, dry_run: bool, steps: list[RepairStep]) -> None:
    script = root / "scripts" / "cgc.sh"
    if not script.exists():
        steps.append(RepairStep("cgc_stack", False, "missing_script", str(script)))
        return

    env = os.environ.copy()
    env.update(cgc_environment(cfg))
    cmd = ["bash", str(script)]
    if dry_run:
        details = " ".join(f"{k}={v}" for k, v in sorted(cgc_environment(cfg).items()))
        steps.append(RepairStep("cgc_stack", True, "would_run", f"{details} {' '.join(cmd)}"))
        return

    result = subprocess.run(cmd, cwd=str(root), env=env, check=False)
    if result.returncode != 0:
        steps.append(RepairStep("cgc_stack", False, "failed", f"{' '.join(cmd)} exited {result.returncode}"))
        return
    steps.append(RepairStep("cgc_stack", True, "repaired", str(script)))
    verify_cgc_runtime(env, steps=steps)


def cgc_environment(cfg: dict[str, Any]) -> dict[str, str]:
    neo = dict(cfg.get("neo4j") or cfg.get("graph_backend") or {})
    uri = str(neo.get("uri") or "bolt://localhost:7687")
    username = str(neo.get("username") or "neo4j")
    password = str(neo.get("password") or "codegraphcontext")
    database = str(neo.get("database") or "neo4j")
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    bolt_port = str(parsed.port or 7687)
    http_port = str(neo.get("http_port") or os.getenv("NEO4J_HTTP_PORT") or "7474")
    if host not in {"localhost", "127.0.0.1", "::1"}:
        http_port = str(os.getenv("NEO4J_HTTP_PORT") or http_port)
    return {
        "NEO4J_URI": uri,
        "NEO4J_USER": username,
        "NEO4J_PASSWORD": password,
        "NEO4J_DATABASE": database,
        "NEO4J_BOLT_PORT": bolt_port,
        "NEO4J_HTTP_PORT": http_port,
    }


def verify_cgc_runtime(env: dict[str, str], *, steps: list[RepairStep]) -> None:
    cgc = shutil.which("cgc") or str(Path.home() / ".local" / "bin" / "cgc")
    if Path(cgc).exists() or shutil.which("cgc"):
        result = subprocess.run([cgc, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        details = (result.stdout or result.stderr or cgc).strip()
        steps.append(RepairStep("cgc", result.returncode == 0, "verified" if result.returncode == 0 else "failed", details))
    else:
        steps.append(RepairStep("cgc", False, "missing", "cgc is not on PATH and ~/.local/bin/cgc does not exist"))

    docker = shutil.which("docker")
    if not docker:
        steps.append(RepairStep("neo4j_docker", False, "missing", "docker"))
        return
    container_name = env.get("NEO4J_CONTAINER_NAME", "cgc-neo4j")
    result = subprocess.run(
        [docker, "ps", "--filter", f"name=^/{container_name}$", "--format", "{{.Names}} {{.Status}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    details = (result.stdout or result.stderr).strip()
    steps.append(RepairStep("neo4j_docker", result.returncode == 0 and bool(details), "verified", details or f"{container_name} not running"))

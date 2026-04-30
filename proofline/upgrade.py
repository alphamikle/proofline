from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_REPO = "https://github.com/alphamikle/proofline.git"
DEFAULT_REF = "main"

PROTECTED_NAMES = {
    ".git",
    ".venv",
    ".codegraphcontext",
    "__pycache__",
    "data",
    "old-data",
    "repos",
    "proofline.yaml",
    "config.yaml",
}


class UpgradeError(RuntimeError):
    pass


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_upgrade(
    *,
    repo: str | None = None,
    ref: str | None = None,
    install_dir: str | Path | None = None,
    bin_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    skip_deps: bool = False,
) -> None:
    root = Path(install_dir or os.getenv("PROOFLINE_DIR") or package_root()).expanduser().resolve()
    target_bin_dir = Path(bin_dir or os.getenv("PROOFLINE_BIN_DIR") or Path.home() / ".local" / "bin").expanduser().resolve()
    selected_ref = ref or os.getenv("PROOFLINE_REF") or DEFAULT_REF
    selected_repo = repo or os.getenv("PROOFLINE_REPO") or DEFAULT_REPO

    _log(f"Proofline install: {root}")
    _log(f"Binary dir: {target_bin_dir}")

    if source_dir:
        source = Path(source_dir).expanduser().resolve()
        _validate_source(source)
        _log(f"Upgrading from local source: {source}")
        if source == root:
            _log("Local source is the install directory; skipping file sync")
        else:
            _sync_source(source, root, dry_run=dry_run)
    elif (root / ".git").exists():
        _upgrade_git_checkout(root, selected_ref, force=force, dry_run=dry_run)
    else:
        _upgrade_non_git(root, selected_repo, selected_ref, force=force, dry_run=dry_run)

    if not skip_deps:
        _install_python(root, dry_run=dry_run)
    _link_binaries(root, target_bin_dir, dry_run=dry_run)
    if not dry_run:
        _verify(root)
    _log("Upgrade complete")


def _upgrade_git_checkout(root: Path, ref: str, *, force: bool, dry_run: bool) -> None:
    _log("Updating git checkout")
    if _dirty(root) and not force:
        raise UpgradeError("Git checkout has local changes. Commit/stash them or rerun with --force.")
    if force:
        _run(["git", "-C", str(root), "reset", "--hard"], dry_run=dry_run)
    _run(["git", "-C", str(root), "fetch", "--tags", "origin"], dry_run=dry_run)
    _run(["git", "-C", str(root), "checkout", "-q", ref], dry_run=dry_run)
    _run(["git", "-C", str(root), "pull", "--ff-only", "origin", ref], dry_run=dry_run)


def _upgrade_non_git(root: Path, repo: str, ref: str, *, force: bool, dry_run: bool) -> None:
    _log("Install is not a git checkout; upgrading from remote checkout")
    if root.exists() and not force:
        _backup_code(root, dry_run=dry_run)
    with tempfile.TemporaryDirectory(prefix="proofline-upgrade-") as tmp:
        checkout = Path(tmp) / "proofline"
        _run(["git", "clone", "--depth=1", "--branch", ref, repo, str(checkout)], dry_run=dry_run)
        if dry_run:
            _log(f"Would sync remote checkout into {root}")
            return
        _sync_source(checkout, root, dry_run=False)


def _backup_code(root: Path, *, dry_run: bool) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / ".upgrade-backups" / stamp
    _log(f"Backing up current code to {backup}")
    if dry_run:
        return
    backup.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        if _protected(item):
            continue
        dest = backup / item.name
        if item.is_dir():
            shutil.copytree(item, dest, ignore=_copy_ignore)
        else:
            shutil.copy2(item, dest)


def _sync_source(source: Path, root: Path, *, dry_run: bool) -> None:
    _validate_source(source)
    _log("Synchronizing code")
    if dry_run:
        _log(f"Would copy {source} -> {root} excluding local data/config")
        return
    root.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if _protected(item):
            continue
        dest = root / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest, ignore=_copy_ignore)
        else:
            shutil.copy2(item, dest)


def _install_python(root: Path, *, dry_run: bool) -> None:
    _validate_source(root)
    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        _run([sys.executable, "-m", "venv", str(root / ".venv")], dry_run=dry_run)
    python = str(venv_python)
    _run([python, "-m", "pip", "install", "--upgrade", "pip"], dry_run=dry_run)
    requirements = root / "requirements.txt"
    if requirements.exists():
        _run([python, "-m", "pip", "install", "-r", str(requirements)], dry_run=dry_run)
    _run([python, "-m", "pip", "install", "-e", str(root)], dry_run=dry_run)


def _link_binaries(root: Path, bin_dir: Path, *, dry_run: bool) -> None:
    _log("Linking CLI binaries")
    if dry_run:
        _log(f"Would link {bin_dir / 'pfl'} and {bin_dir / 'proofline'}")
        return
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("proofline", "pfl"):
        src = root / ".venv" / "bin" / name
        dest = bin_dir / name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src)


def _verify(root: Path) -> None:
    python = root / ".venv" / "bin" / "python"
    _run([str(python), "-m", "compileall", "-q", str(root / "proofline")], dry_run=False)
    _run([str(root / ".venv" / "bin" / "pfl"), "--help"], dry_run=False, quiet=True)


def _validate_source(path: Path) -> None:
    if not (path / "pyproject.toml").exists():
        raise UpgradeError(f"{path} is missing pyproject.toml")
    if not (path / "proofline").is_dir():
        raise UpgradeError(f"{path} is missing proofline package")


def _dirty(root: Path) -> bool:
    result = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return bool(result.stdout.strip())


def _protected(path: Path) -> bool:
    return path.name in PROTECTED_NAMES or path.name.endswith(".pyc")


def _copy_ignore(_dir: str, names: Iterable[str]) -> set[str]:
    return {name for name in names if name in PROTECTED_NAMES or name.endswith(".pyc")}


def _run(cmd: list[str], *, dry_run: bool, quiet: bool = False) -> None:
    if dry_run:
        _log("$ " + " ".join(cmd))
        return
    if not quiet:
        _log("$ " + " ".join(cmd))
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if quiet else {}
    result = subprocess.run(cmd, check=False, **kwargs)
    if result.returncode != 0:
        raise UpgradeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _log(message: str) -> None:
    print(f"[upgrade] {message}", file=sys.stderr, flush=True)

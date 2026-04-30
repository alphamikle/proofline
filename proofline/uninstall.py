from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List


PRESERVED_TOP_LEVEL = {
    "data",
    "repos",
    "old-data",
    "proofline.yaml",
    "config.yaml",
    ".codegraphcontext",
}

PROOFLINE_TOP_LEVEL = {
    ".gitignore",
    ".venv",
    ".upgrade-backups",
    "Makefile",
    "README.md",
    "ask.sh",
    "config.schema.json",
    "install.sh",
    "optional-requirements.txt",
    "proofline",
    "proofline.egg-info",
    "proofline.example.yaml",
    "pyproject.toml",
    "requirements.txt",
    "run.sh",
    "scripts",
    "tests",
}

PROOFLINE_BIN_NAMES = {"pfl", "proofline"}
CGC_BIN_NAMES = {
    "cgc",
    "scip",
    "scip-python",
    "scip-typescript",
    "scip-go",
    "scip-rust",
    "scip-java",
    "scip-clang",
    "scip-ruby",
    "scip-dotnet",
    "scip-dart",
    "scip-php",
}


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def uninstall_plan(
    *,
    install_dir: str | Path | None = None,
    bin_dir: str | Path | None = None,
    include_cgc: bool = False,
) -> Dict[str, Any]:
    root = Path(install_dir or os.getenv("PROOFLINE_DIR") or package_root()).expanduser().resolve()
    target_bin_dir = Path(bin_dir or os.getenv("PROOFLINE_BIN_DIR") or Path.home() / ".local" / "bin").expanduser().resolve()
    remove_paths: List[Path] = []
    keep_paths: List[Path] = []

    if root.exists():
        for item in root.iterdir():
            if item.name in PRESERVED_TOP_LEVEL:
                keep_paths.append(item)
            elif item.name in PROOFLINE_TOP_LEVEL or item.name.endswith(".pyc") or item.name == "__pycache__":
                remove_paths.append(item)

    for name in PROOFLINE_BIN_NAMES:
        candidate = target_bin_dir / name
        if _owned_by_install(candidate, root):
            remove_paths.append(candidate)

    if include_cgc:
        for name in CGC_BIN_NAMES:
            candidate = target_bin_dir / name
            if candidate.exists() or candidate.is_symlink():
                remove_paths.append(candidate)
        cgc_home = Path(os.getenv("CGC_HOME") or Path.home() / ".codegraphcontext").expanduser()
        for rel in ["cgc-venv", "node"]:
            candidate = cgc_home / rel
            if candidate.exists():
                remove_paths.append(candidate)
        keep_paths.append(cgc_home / ".env")
        keep_paths.append(cgc_home / "db")

    return {
        "install_dir": str(root),
        "bin_dir": str(target_bin_dir),
        "remove": [str(p) for p in _unique_existing(remove_paths)],
        "preserve": [str(p) for p in _unique_existing(keep_paths)],
        "include_cgc": include_cgc,
    }


def run_uninstall(
    *,
    install_dir: str | Path | None = None,
    bin_dir: str | Path | None = None,
    include_cgc: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    plan = uninstall_plan(install_dir=install_dir, bin_dir=bin_dir, include_cgc=include_cgc)
    if dry_run:
        return plan
    for raw in plan["remove"]:
        path = Path(raw)
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)
    return plan


def _owned_by_install(path: Path, root: Path) -> bool:
    if not (path.exists() or path.is_symlink()):
        return False
    try:
        return path.resolve().is_relative_to(root)
    except Exception:
        return False


def _unique_existing(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        if path.exists() or path.is_symlink():
            out.append(path)
            seen.add(key)
    return out

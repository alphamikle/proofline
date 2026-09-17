#!/usr/bin/env bash
set -euo pipefail

PROOFLINE_REPO="${PROOFLINE_REPO:-https://github.com/alphamikle/proofline.git}"
PROOFLINE_REF="${PROOFLINE_REF:-main}"
PROOFLINE_DIR="${PROOFLINE_DIR:-$HOME/.proofline}"
PROOFLINE_PYTHON="${PROOFLINE_PYTHON:-python3}"
PROOFLINE_BIN_DIR="${PROOFLINE_BIN_DIR:-$HOME/.local/bin}"
PROOFLINE_INSTALL_CGC="${PROOFLINE_INSTALL_CGC:-0}"
PROOFLINE_LOCAL_SOURCE="${PROOFLINE_LOCAL_SOURCE:-0}"
PROOFLINE_SOURCE_DIR="${PROOFLINE_SOURCE_DIR:-}"
# Minimum Python for `pip install -r requirements.txt` (mcp>=1.0.0 needs 3.10+).
PROOFLINE_MIN_PYTHON="${PROOFLINE_MIN_PYTHON:-3.10}"

log() {
  printf '\n==> %s\n' "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

require() {
  have "$1" || fail "Missing required command: $1"
}

copy_local_checkout() {
  local source_dir
  source_dir="$(cd "$PROOFLINE_SOURCE_DIR" && pwd)"
  [[ -f "$source_dir/pyproject.toml" ]] || fail "Local source is not a Proofline checkout: $source_dir"

  log "Installing Proofline from local checkout $source_dir"

  mkdir -p "$PROOFLINE_DIR"
  if have rsync; then
    rsync -a --delete \
      --exclude .git \
      --exclude .venv \
      --exclude .codegraphcontext \
      --exclude .idea \
      --exclude .vscode \
      --exclude data \
      --exclude repos \
      --exclude __pycache__ \
      --exclude '*.pyc' \
      --exclude '*.egg-info' \
      --exclude proofline.yaml \
      "$source_dir/" "$PROOFLINE_DIR/"
  else
    log "rsync not found; copying files without deleting stale code files"
    tar -C "$source_dir" \
      --exclude .git \
      --exclude .venv \
      --exclude .codegraphcontext \
      --exclude .idea \
      --exclude .vscode \
      --exclude data \
      --exclude repos \
      --exclude __pycache__ \
      --exclude '*.pyc' \
      --exclude '*.egg-info' \
      --exclude proofline.yaml \
      -cf - . | tar -C "$PROOFLINE_DIR" -xf -
  fi
}

checkout_proofline() {
  if [[ "$PROOFLINE_LOCAL_SOURCE" == "1" ]]; then
    [[ -n "$PROOFLINE_SOURCE_DIR" ]] || fail "PROOFLINE_SOURCE_DIR must be set when PROOFLINE_LOCAL_SOURCE=1"
    copy_local_checkout
    return
  fi

  if [[ -d "$PROOFLINE_DIR/.git" ]]; then
    log "Updating Proofline in $PROOFLINE_DIR"
    git -C "$PROOFLINE_DIR" fetch --tags origin
    git -C "$PROOFLINE_DIR" checkout -q "$PROOFLINE_REF" 2>/dev/null || git -C "$PROOFLINE_DIR" checkout -q "origin/$PROOFLINE_REF"
    git -C "$PROOFLINE_DIR" pull --ff-only origin "$PROOFLINE_REF" 2>/dev/null || true
    return
  fi

  if [[ -e "$PROOFLINE_DIR" ]]; then
    fail "$PROOFLINE_DIR exists but is not a git checkout. Set PROOFLINE_DIR to another path."
  fi

  log "Cloning Proofline into $PROOFLINE_DIR"
  git clone --depth=1 --branch "$PROOFLINE_REF" "$PROOFLINE_REPO" "$PROOFLINE_DIR"
}

validate_checkout() {
  [[ -f "$PROOFLINE_DIR/pyproject.toml" ]] || fail "Installed checkout at $PROOFLINE_DIR is missing pyproject.toml. PROOFLINE_REPO/PROOFLINE_REF probably points to an older Proofline revision."
  [[ -d "$PROOFLINE_DIR/proofline" ]] || fail "Installed checkout at $PROOFLINE_DIR is missing the proofline package."
}

# Pick a python3 >= PROOFLINE_MIN_PYTHON, else fail with install hints.
# macOS note: /usr/bin/python3 is the Xcode stub (3.9, no newer wheels for
# mcp>=1.0.0). Prefer Homebrew python3.14/3.13/3.12/3.11/3.10 when present.
resolve_python() {
  local candidates=()
  if [[ -n "${PROOFLINE_PYTHON:-}" && "$PROOFLINE_PYTHON" != "python3" ]]; then
    candidates+=("$PROOFLINE_PYTHON")
  fi
  candidates+=(
    python3.14 python3.13 python3.12 python3.11 python3.10
    /opt/homebrew/bin/python3.14 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12
    /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10
    /usr/local/bin/python3.14 /usr/local/bin/python3.13 /usr/local/bin/python3.12
    /usr/local/bin/python3.11 /usr/local/bin/python3.10
    python3
  )
  local cand ver
  for cand in "${candidates[@]}"; do
    if ! have "$cand" && [[ ! -x "$cand" ]]; then
      continue
    fi
    ver="$("$cand" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
    [[ -z "$ver" ]] && continue
    # ver is new enough when the minimum sorts first (min <= ver).
    if [[ "$(printf '%s\n%s\n' "$PROOFLINE_MIN_PYTHON" "$ver" | sort -V | head -n1)" == "$PROOFLINE_MIN_PYTHON" ]]; then
      PROOFLINE_PYTHON="$cand"
      log "Using Python $ver ($cand)"
      return 0
    fi
  done
  fail "No Python >= $PROOFLINE_MIN_PYTHON found (PROOFLINE_PYTHON=$PROOFLINE_PYTHON resolved to $("$PROOFLINE_PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo unknown)).

Install one of:
  brew install python@3.12        # then rerun install.sh (auto-detected)
  PROOFLINE_PYTHON=/path/to/python3.12 curl ... | bash

Why: requirements.txt needs mcp>=1.0.0, which has no Python 3.9 wheels.
The macOS Xcode /usr/bin/python3 (3.9) cannot install it."
}

install_python_env() {
  validate_checkout
  resolve_python
  log "Creating Python environment ($PROOFLINE_PYTHON)"
  "$PROOFLINE_PYTHON" -m venv "$PROOFLINE_DIR/.venv"
  "$PROOFLINE_DIR/.venv/bin/python" -m pip install --upgrade pip

  log "Installing Proofline"
  "$PROOFLINE_DIR/.venv/bin/pip" install -r "$PROOFLINE_DIR/requirements.txt"
  "$PROOFLINE_DIR/.venv/bin/pip" install -e "$PROOFLINE_DIR"
}

create_config() {
  if [[ ! -f "$PROOFLINE_DIR/proofline.yaml" ]]; then
    log "Creating proofline.yaml"
    (cd "$PROOFLINE_DIR" && "$PROOFLINE_DIR/.venv/bin/proofline" init --config "$PROOFLINE_DIR/proofline.yaml")
    "$PROOFLINE_DIR/.venv/bin/python" - "$PROOFLINE_DIR" <<'PY'
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1]).expanduser().resolve()
config_path = root / "proofline.yaml"
cfg = yaml.safe_load(config_path.read_text()) or {}
cfg["workspace"] = str(root / "data")
cfg.setdefault("repos", {})["root"] = str(root / "repos")
cfg.setdefault("storage", {})
cfg["storage"]["duckdb_path"] = str(root / "data" / "kb.duckdb")
cfg["storage"]["sqlite_fts_path"] = str(root / "data" / "indexes" / "code_fts.sqlite")
cfg["storage"]["vector_index_path"] = str(root / "data" / "indexes" / "code_vectors.faiss")
cfg["storage"]["vector_meta_path"] = str(root / "data" / "indexes" / "code_vectors_meta.parquet")
cfg.setdefault("confluence", {})["output_dir"] = str(root / "data" / "raw" / "confluence")
cfg.setdefault("jira", {})["output_dir"] = str(root / "data" / "raw" / "jira")
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  fi
  mkdir -p "$PROOFLINE_DIR/repos" "$PROOFLINE_DIR/data"
}

install_cgc_stack() {
  if [[ "$PROOFLINE_INSTALL_CGC" != "1" ]]; then
    return
  fi
  log "Installing optional CodeGraphContext stack"
  (cd "$PROOFLINE_DIR" && INSTALL_CGC_STACK=1 ./scripts/bootstrap.sh)
}

link_binary() {
  mkdir -p "$PROOFLINE_BIN_DIR"
  ln -sfn "$PROOFLINE_DIR/.venv/bin/proofline" "$PROOFLINE_BIN_DIR/proofline"
  ln -sfn "$PROOFLINE_DIR/.venv/bin/pfl" "$PROOFLINE_BIN_DIR/pfl"
}

print_next_steps() {
  cat <<EOF

Proofline installed.

Binary:
  $PROOFLINE_BIN_DIR/proofline
  $PROOFLINE_BIN_DIR/pfl

Config:
  $PROOFLINE_DIR/proofline.yaml

Try:
  proofline doctor --config "$PROOFLINE_DIR/proofline.yaml"
  pfl doctor --config "$PROOFLINE_DIR/proofline.yaml"

If your shell cannot find proofline or pfl, add this to your shell profile:
  export PATH="$PROOFLINE_BIN_DIR:\$PATH"

EOF
}

main() {
  require git
  require "$PROOFLINE_PYTHON"
  checkout_proofline
  install_python_env
  create_config
  install_cgc_stack
  link_binary
  print_next_steps
}

main "$@"

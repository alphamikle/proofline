#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '\n==> %s\n' "$*"
}

CGC_BOOTSTRAP_SCRIPT="${CGC_BOOTSTRAP_SCRIPT:-./bootstrap/cgc.sh}"
INSTALL_CGC_STACK="${INSTALL_CGC_STACK:-1}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ "$INSTALL_CGC_STACK" == "1" ]]; then
  if [[ ! -f "$CGC_BOOTSTRAP_SCRIPT" ]]; then
    echo "ERROR: CGC bootstrap script not found: $CGC_BOOTSTRAP_SCRIPT" >&2
    echo "Set CGC_BOOTSTRAP_SCRIPT=/path/to/cgc.sh or INSTALL_CGC_STACK=0 to skip." >&2
    exit 1
  fi
  log "Installing CodeGraphContext, SCIP indexers, and local Neo4j"
  bash "$CGC_BOOTSTRAP_SCRIPT"
fi

if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "Created config.yaml from config.example.yaml"
fi
mkdir -p repos data

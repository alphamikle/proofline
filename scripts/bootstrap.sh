#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if [[ "${INSTALL_OPTIONAL:-0}" == "1" ]]; then
  pip install -r optional-requirements.txt
fi
if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "Created config.yaml from config.example.yaml"
fi
mkdir -p repos data

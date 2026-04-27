#!/usr/bin/env bash
set -euo pipefail
CONFIG=${CONFIG:-proofline.yaml}
PYTHON=${PYTHON:-}
if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    cat >&2 <<'EOF'
No local Python environment found.

Run:
  ./scripts/bootstrap.sh

Then retry:
  ./run.sh
EOF
    exit 1
  fi
fi
if [[ "${1:-}" == "full" || "${1:-}" == "run" ]]; then
  shift
fi
"$PYTHON" -m proofline run --config "$CONFIG" "$@"

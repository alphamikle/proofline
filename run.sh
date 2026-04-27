#!/usr/bin/env bash
set -euo pipefail
CONFIG=${CONFIG:-config.yaml}
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
if [[ "${1:-}" == "full" ]]; then
  shift
fi
"$PYTHON" -m corp_kb.pipeline.runner full --config "$CONFIG" "$@"

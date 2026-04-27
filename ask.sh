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
  ./ask.sh "your question"
EOF
    exit 1
  fi
fi
cmd=${1:-}
case "$cmd" in
  ask|impact|data-source|dependencies|search)
    shift
    "$PYTHON" -m proofline "$cmd" --config "$CONFIG" "$@"
    ;;
  *)
    "$PYTHON" -m proofline ask --config "$CONFIG" "$@"
    ;;
esac

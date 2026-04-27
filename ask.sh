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
  ./ask.sh "your question"
EOF
    exit 1
  fi
fi
cmd=${1:-}
case "$cmd" in
  ask|impact|data-source|dependency-report|search)
    shift
    "$PYTHON" -m corp_kb.agent.ask "$cmd" --config "$CONFIG" "$@"
    ;;
  *)
    "$PYTHON" -m corp_kb.agent.ask ask --config "$CONFIG" "$@"
    ;;
esac

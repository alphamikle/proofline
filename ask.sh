#!/usr/bin/env bash
set -euo pipefail
CONFIG=${CONFIG:-config.yaml}
cmd=${1:-}
case "$cmd" in
  ask|impact|data-source|dependency-report|search)
    shift
    python3 -m corp_kb.agent.ask "$cmd" --config "$CONFIG" "$@"
    ;;
  *)
    python3 -m corp_kb.agent.ask ask --config "$CONFIG" "$@"
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail
CONFIG=${CONFIG:-config.yaml}
if [[ "${1:-}" == "full" ]]; then
  shift
fi
python3 -m corp_kb.pipeline.runner full --config "$CONFIG" "$@"

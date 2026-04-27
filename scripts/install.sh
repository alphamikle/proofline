#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROOFLINE_LOCAL_SOURCE="${PROOFLINE_LOCAL_SOURCE:-1}"
export PROOFLINE_SOURCE_DIR="${PROOFLINE_SOURCE_DIR:-$ROOT}"
exec "$ROOT/install.sh" "$@"

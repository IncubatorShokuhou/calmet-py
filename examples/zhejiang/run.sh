#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
python_bin=${PYTHON_BIN:-python3}

cd "$repo"
PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" exec "$python_bin" examples/zhejiang/run.py

#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
work_dir=$(mktemp -d /tmp/pycalmet-zhejiang-regress-XXXXXX)
trap 'rm -rf -- "$work_dir"' EXIT
started_at=$SECONDS

cd "$repo_root"

printf 'Running PyCALMET Zhejiang first hour in %s\n' "$work_dir"
"$python_bin" - "$work_dir/python.dat" <<'PY'
import sys

import pycalmet

pycalmet.calmet(
    "calmet_mpi/calmet.inp",
    meteorology="calmet_mpi/calwrfout_2018031800_2018031812.m3d",
    geophysical="MAKEGEO/geo.dat",
    output=sys.argv[1],
    start="2018-03-18_00:00:00",
    end="2018-03-18_01:00:00",
)
PY

printf 'CALMET solve completed in %d s; comparing CALMET.DAT\n' "$((SECONDS-started_at))"
"$python_bin" tools/regress_calmet.py \
    --reference diagnostics/results/reference_outputs/calmetout_2018031800_2018031812_zhejiang_original_wrf.dat \
    --candidate "$work_dir/python.dat" \
    --ending 2018-03-18T01:00:00 \
    --require-bitwise

printf 'CALMET.DAT is bitwise identical; comparing CALMET.AUX\n'
"$python_bin" tools/regress_aux.py \
    --reference calmet_mpi/calmet.dat.aux \
    --candidate "$work_dir/python.dat.aux" \
    --hours 1 \
    --require-bitwise

printf 'Zhejiang first-hour bitwise gate passed in %d s\n' "$((SECONDS-started_at))"

#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
fc=${FC:-gfortran}
common=(-O2 -std=legacy -fallow-argument-mismatch -fno-align-commons -mcmodel=medium)

command -v "$fc" >/dev/null || { echo "Missing compiler: $fc" >&2; exit 1; }
command -v nf-config >/dev/null || { echo "Missing nf-config (NetCDF-Fortran)" >&2; exit 1; }

echo '[1/5] CALWRF'
(cd "$root/CALWRF/code" && "$fc" "${common[@]}" $(nf-config --fflags) \
  calwrf.f -o ../calwrf.exe $(nf-config --flibs))
echo '[2/5] TERREL'
(cd "$root/TERREL" && "$fc" "${common[@]}" TERREL.FOR -o terrel.exe)
echo '[3/5] CTGPROC'
# COORDLIB/CTGPROC predates recursive Fortran defaults and retains local
# projection state between calls.  Without static locals GNU builds can emit
# NaN database windows and a superficially successful, entirely empty LU.DAT.
(cd "$root/CTGPROC" && "$fc" "${common[@]}" -fno-automatic \
  ctgproc.for -o ctgproc.exe)
echo '[4/5] MAKEGEO'
(cd "$root/MAKEGEO" && "$fc" "${common[@]}" makegeo.for -o makegeo.exe)
echo '[5/5] CALMET (OpenMP)'
(cd "$root/calmet_mpi" && "$fc" "${common[@]}" -fno-automatic \
  -fopenmp calmet.for -o calmet.exe)

echo 'GNU build complete.'

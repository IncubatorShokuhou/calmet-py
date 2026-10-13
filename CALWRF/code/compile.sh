#!/bin/sh
# Compile CALWRF on Linux using GNU Fortran and NetCDF-Fortran.

set -eu

command -v gfortran >/dev/null
command -v nf-config >/dev/null

gfortran -O2 -std=legacy -fallow-argument-mismatch -fno-align-commons \
  $(nf-config --fflags) calwrf.f -o ../calwrf.exe $(nf-config --flibs)

ulimit -s unlimited

_calmet_config=${BASH_SOURCE:-$0}
if [ -z "${CALMET_REPO_ROOT:-}" ]; then
    CALMET_REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$_calmet_config")/.." && pwd)
fi
export CALMET_REPO_ROOT

if [ -z "${PGI:-}" ]; then
    echo "ERROR: set PGI to the NVIDIA/PGI compiler installation root." >&2
    return 2 2>/dev/null || exit 2
fi
export PGI
export PATH=$PGI/linux86-64/10.6/bin:$PATH
export MANPATH=$PGI/linux86-64/10.6/man:$MANPATH
export LM_LICENSE_FILE=$PGI/license.dat
export f90=pgf90
export cc=pgcc
export f77=pgf90
export FC=pgf90

if [ -z "${NETCDF:-}" ]; then
    echo "ERROR: set NETCDF to the NetCDF installation root." >&2
    return 2 2>/dev/null || exit 2
fi
export NETCDF
export PATH=$NETCDF/bin:$PATH
export NETCDF_LIB=$NETCDF/lib
export NETCDF_INC=$NETCDF/include

unset _calmet_config


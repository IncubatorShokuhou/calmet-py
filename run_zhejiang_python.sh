#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_zhejiang_python.sh --wrf PATH --start YYYYMMDDHH --end YYYYMMDDHH \
    [--srtm-dir DIR --landuse PATH] [--geo PATH] [--output FILE] \
    [--work-dir DIR] [--skip-static]

PATH may be one WRF NetCDF file or a directory containing wrfout* files.
The WRF input must include one hourly record after --end for time averaging.
--srtm-dir and --landuse are required unless --skip-static reuses work-dir/geo.nc.
With --skip-static, --geo may select an existing GEO.DAT or canonical geo.nc.
This uses only pycalmet NumPy/SciPy; no Fortran executable is required.
Set PYTHON_BIN to select a Python interpreter.
EOF
}

repo=$(cd "$(dirname "$0")" && pwd)
wrf_path= srtm_dir= landuse_path= geo_path= start= end= output= work_dir=
skip_static=0
while (($#)); do
  case "$1" in
    --wrf) wrf_path=${2:?}; shift 2 ;;
    --srtm-dir|--strm-dir) srtm_dir=${2:?}; shift 2 ;;
    --landuse) landuse_path=${2:?}; shift 2 ;;
    --geo|--geophysical) geo_path=${2:?}; shift 2 ;;
    --start) start=${2:?}; shift 2 ;;
    --end) end=${2:?}; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --work-dir) work_dir=${2:?}; shift 2 ;;
    --skip-static) skip_static=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $start =~ ^[0-9]{10}$ && $end =~ ^[0-9]{10}$ ]] || {
  echo '--start and --end must be YYYYMMDDHH.' >&2; exit 2;
}
[[ -e $wrf_path ]] || { echo "WRF input not found: $wrf_path" >&2; exit 2; }
python_bin=${PYTHON_BIN:-python3}
work_dir=${work_dir:-$repo/output/python_work_${start}_${end}}
output=${output:-$repo/output/pycalmet_${start}_${end}.nc}
if (( skip_static )); then
  geo_path=${geo_path:-$work_dir/geo.nc}
  [[ -f $geo_path ]] || {
    echo "--skip-static requested but geophysical input is missing: $geo_path" >&2; exit 2;
  }
else
  [[ -d $srtm_dir ]] || { echo "SRTM directory not found: $srtm_dir" >&2; exit 2; }
  [[ -f $landuse_path ]] || { echo "Land-use input not found: $landuse_path" >&2; exit 2; }
fi

mkdir -p "$work_dir" "$(dirname "$output")"

export PYCALMET_REPO=$repo PYCALMET_WRF=$(realpath "$wrf_path")
export PYCALMET_START=$start PYCALMET_END=$end
export PYCALMET_WORK=$(realpath "$work_dir") PYCALMET_OUTPUT=$(realpath -m "$output")
export PYCALMET_SKIP_STATIC=$skip_static
if (( skip_static )); then
  export PYCALMET_SRTM= PYCALMET_LANDUSE= PYCALMET_GEO=$(realpath "$geo_path")
else
  export PYCALMET_GEO= PYCALMET_SRTM=$(realpath "$srtm_dir")
  export PYCALMET_LANDUSE=$(realpath "$landuse_path")
fi

PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" - <<'PY'
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import os
import pycalmet
from pycalmet.control import ctgproc_config, terrel_config

repo=Path(os.environ['PYCALMET_REPO']); wrf=Path(os.environ['PYCALMET_WRF'])
work=Path(os.environ['PYCALMET_WORK'])
start=os.environ['PYCALMET_START']; end=os.environ['PYCALMET_END']
output=Path(os.environ['PYCALMET_OUTPUT'])
wrf_files=sorted(wrf.glob('wrfout*')) if wrf.is_dir() else [wrf]
if not wrf_files: raise SystemExit(f'No wrfout* files found in {wrf}')

tc=terrel_config(repo/'TERREL/terrel.inp')
met=work/'calwrf.nc'
legacy_met=work/'calwrf.m3d'
calwrf_end=(datetime.strptime(end,'%Y%m%d%H')+timedelta(hours=1)).strftime('%Y%m%d%H')
print(f'[1/5] CALWRF through {calwrf_end} boundary:',met,flush=True)
pycalmet.calwrf(
    wrf_files=wrf_files,start=start,end=calwrf_end,output=met,
    legacy_output=legacy_met,create_surface=False,
)

if os.environ['PYCALMET_SKIP_STATIC']=='1':
    geo=Path(os.environ['PYCALMET_GEO'])
    if not geo.is_file(): raise SystemExit(f'--skip-static requested but {geo} is missing')
    print('[2-4/5] Reusing:',geo,flush=True)
else:
    geo=work/'geo.nc'
    srtm=Path(os.environ['PYCALMET_SRTM'])
    landuse_input=Path(os.environ['PYCALMET_LANDUSE'])
    terrain=work/'terrain.nc'; landuse=work/'landuse.nc'
    tiles=[]; terrain_sources=[]
    for source in tc.terrain_sources:
        configured=Path(source.path); name=configured.name
        candidates=(srtm/name,srtm/name.upper(),srtm/name.lower(),configured)
        found=next((item for item in candidates if item.is_file()),None)
        if found is None: raise SystemExit(f'Missing required SRTM tile: {name}')
        tiles.append(found)
        terrain_sources.append(replace(source,path=found))
    print('[2/5] TERREL:',terrain,flush=True)
    pycalmet.terrel(replace(tc,srtm_dir=srtm,terrain_files=tuple(tiles),
                            terrain_sources=tuple(terrain_sources),output=terrain,
                            save_output=None,plot_output=None,raw_echo_output=None))
    print('[3/5] CTGPROC:',landuse,flush=True)
    cc=ctgproc_config(repo/'CTGPROC/ctgproc.inp')
    pycalmet.ctrproc(replace(cc,landuse_files=(landuse_input,),
                             landuse_sources=(),output=landuse))
    print('[4/5] MAKEGEO:',geo,flush=True)
    pycalmet.makegeo(repo/'MAKEGEO/makegeo.inp',terrain=terrain,landuse=landuse,output=geo)

print('[5/5] CALMET:',output,flush=True)
pycalmet.calmet(
    repo/'calmet_mpi/calmet_zhejiang.inp',meteorology=legacy_met,geophysical=geo,
    output=output,start=start,end=end,
)
print('Complete:',output)
PY

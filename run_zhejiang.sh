#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_zhejiang.sh --wrf PATH --start YYYYMMDDHH \
    --end YYYYMMDDHH [--strm-dir DIR] [--output FILE] [--jobs N] [--skip-static]

PATH may be one WRF NetCDF file or a directory containing wrfout* files.
The WRF input must include one hourly record after --end for time averaging.
--strm-dir defaults to the bundled Zhejiang subset in TERREL/SRTM3.
EOF
}

root=$(cd "$(dirname "$0")" && pwd)
wrf_path=
terrain_root=$root/TERREL/SRTM3
start=
end=
output=
jobs=${OMP_NUM_THREADS:-1}
skip_static=0

while (($#)); do
  case "$1" in
    --wrf) wrf_path=${2:?}; shift 2 ;;
    --strm-dir|--srtm-dir) terrain_root=${2:?}; shift 2 ;;
    --start) start=${2:?}; shift 2 ;;
    --end) end=${2:?}; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --jobs) jobs=${2:?}; shift 2 ;;
    --skip-static) skip_static=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $start =~ ^[0-9]{10}$ && $end =~ ^[0-9]{10}$ ]] || {
  echo '--start and --end must be YYYYMMDDHH.' >&2; exit 2;
}
calwrf_end=$(date -u -d \
  "${end:0:4}-${end:4:2}-${end:6:2} ${end:8:2}:00:00 UTC +1 hour" \
  +%Y%m%d%H) || {
  echo "Invalid --end time: $end" >&2; exit 2;
}
[[ -e $wrf_path ]] || { echo "WRF input not found: $wrf_path" >&2; exit 2; }
[[ -d $terrain_root ]] || { echo "Terrain directory not found: $terrain_root" >&2; exit 2; }
terrain_root=$(cd "$terrain_root" && pwd)
wrf_path=$(realpath "$wrf_path")
output=${output:-$root/output/calmetout_${start}_${end}_zhejiang.dat}
mkdir -p "$(dirname "$output")"

for exe in CALWRF/calwrf.exe TERREL/terrel.exe CTGPROC/ctgproc.exe \
           MAKEGEO/makegeo.exe calmet_mpi/calmet.exe; do
  [[ -x $root/$exe ]] || { echo "Missing $exe; run ./build_gnu.sh first." >&2; exit 2; }
done

if [[ -d $wrf_path ]]; then
  mapfile -d '' wrf_files < <(find "$wrf_path" -maxdepth 1 -type f -name 'wrfout*' -print0 | sort -z)
else
  wrf_files=("$wrf_path")
fi
((${#wrf_files[@]})) || { echo 'No wrfout files found.' >&2; exit 2; }

export work_root=$root COMMAND_ROOT=$root/bin
export wrfout_root=$(dirname "${wrf_files[0]}")
export calwrf_root=$root/CALWRF calwrfout_root=$root/CALWRF/calwrf_out
export terrel_root=$root/TERREL ctgproc_root=$root/CTGPROC
export makegeo_root=$root/MAKEGEO calmet_root=$root/calmet_mpi
export terrain_root
export map_proj=LCC rlat=27N rlon=117.5E rlat1=30N rlat2=60N
export xrefkm=0 yrefkm=0 nx=610 ny=499 dgridkm=1 time_step=3600
export BEG_CYMDH=$start END_CYMDH=$end
export BEG_CCYY=${start:0:4} BEG_YY=${start:2:2} BEG_MM=${start:4:2}
export BEG_DD=${start:6:2} BEG_HH=${start:8:2}
export END_CCYY=${end:0:4} END_MM=${end:4:2} END_DD=${end:6:2} END_HH=${end:8:2}
export OMP_NUM_THREADS=$jobs
export OMP_STACKSIZE=${OMP_STACKSIZE:-512M}
# CALMET's large legacy local arrays otherwise exceed the common 8 MiB stack.
ulimit -s unlimited

calwrf_name=calwrfout_${start}_${calwrf_end}.m3d
echo "[1/5] CALWRF through $calwrf_end boundary: WRF -> m3d"
mkdir -p "$calwrfout_root"
rm -f "$calwrfout_root"/*.m3d "$calwrfout_root"/*.m2d
{
  echo 'Create 3D.DAT file for WRF output'
  echo 'calwrf.lst'
  echo "calwrf_out/$calwrf_name"
  echo '-9,-9,-9,-9,-9,-9'
  echo "$start"
  echo "$calwrf_end"
  echo "${#wrf_files[@]}"
  printf '%s\n' "${wrf_files[@]}"
} > "$calwrf_root/calwrf.inp"
# CALWRF 2.0.2 ends successful runs with legacy `STOP 99999`, which GNU
# Fortran reports as a non-zero process status.  Verify its explicit success
# marker and product rather than treating that historical STOP code as failure.
if ! (cd "$calwrf_root" && ./calwrf.exe > calwrf.log 2>&1); then
  grep -q 'CALWRF succeeded' "$calwrf_root/calwrf.log" || {
    echo "CALWRF failed; see $calwrf_root/calwrf.log" >&2; exit 1;
  }
fi
[[ -s $calwrfout_root/$calwrf_name ]] || {
  echo 'CALWRF did not create the expected m3d file.' >&2; exit 1;
}

if ((skip_static == 0)); then
  echo '[2/5] TERREL: SRTM -> TERR.DAT'
  (cd "$terrel_root" && csh ./terrel_check.csh > terrel_check.log 2>&1)
  mapfile -t tiles < <(sed -nE 's/^[[:space:]]*([nNsS][0-9]{2}[eEwW][0-9]{3})\.HGT or.*$/\1/p' "$terrel_root/TERR_CHECK.LST" | tr '[:lower:]' '[:upper:]' | sort -u)
  ((${#tiles[@]})) || { echo 'TERREL did not report any required SRTM tiles.' >&2; exit 1; }
  : > "$terrel_root/filename.txt"
  find "$terrel_root" -maxdepth 1 -type l -name '*.HGT' -delete
  for tile in "${tiles[@]}"; do
    src="$terrain_root/$tile.HGT"
    [[ -f $src ]] || src="$terrain_root/${tile,,}.hgt"
    if [[ ! -f $src && -f $terrel_root/SRTM3/$tile.hgt ]]; then
      echo "  $tile.HGT is absent from --strm-dir; using bundled TERREL/SRTM3 copy."
      src="$terrel_root/SRTM3/$tile.hgt"
    fi
    [[ -f $src ]] || { echo "Missing SRTM tile: $tile.HGT" >&2; exit 1; }
    ln -s "$src" "$terrel_root/$tile.HGT"
    printf '!SRTM3 = %s.HGT! !END!\n' "$tile" >> "$terrel_root/filename.txt"
  done
  (cd "$terrel_root" && csh ./terrel.csh > terrel.log 2>&1)

  echo '[3/5] CTGPROC: land use -> LULC.DAT'
  (cd "$ctgproc_root" && csh ./ctgproc.csh > ctgproc.log 2>&1)
  echo '[4/5] MAKEGEO: terrain + land use -> geo.dat'
  ln -sfn "$terrel_root/TERR.DAT" "$makegeo_root/terr.dat"
  ln -sfn "$ctgproc_root/LULC.DAT" "$makegeo_root/lulc.dat"
  (cd "$makegeo_root" && csh ./makegeo.csh > makegeo.log 2>&1)
else
  echo '[2-4/5] Reusing MAKEGEO/geo.dat (--skip-static)'
  [[ -f $makegeo_root/geo.dat ]] || { echo 'MAKEGEO/geo.dat is missing.' >&2; exit 1; }
fi

echo '[5/5] CALMET: diagnostic downscaling'
rm -f "$calmet_root/calmet.lst" "$calmet_root/calmet.dat"
(cd "$calmet_root" && csh ./calmet.csh > calmet.log 2>&1)
if [[ ! -f $calmet_root/calmet.lst ]] || \
   ! grep -q 'End of run' "$calmet_root/calmet.lst" || \
   grep -q 'ERROR in subr' "$calmet_root/calmet.lst"; then
  echo "CALMET did not finish normally; see $calmet_root/calmet.log and calmet.lst" >&2
  exit 1
fi
mv -f "$calmet_root/calmet.dat" "$output"
echo "Complete: $output"

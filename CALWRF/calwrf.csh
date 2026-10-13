#!/bin/csh -f

if (! $?wrfout_root) then
   echo "ERROR: set wrfout_root to the directory containing WRF output files."
   exit 2
endif
if (! $?BEG_CYMDH || ! $?END_CYMDH) then
   echo "ERROR: set BEG_CYMDH and END_CYMDH before running CALWRF."
   exit 2
endif

#2010-02-01 00~ 2010-02-28 23 672hrs
cat <<EOF >! calwrf.inp
Create 3D.DAT file for WRF output
calwrf.lst          ! Log file name
calwrf_out/calwrfout_${BEG_CYMDH}_${END_CYMDH}.m3d ! Output file name
-9,-9,-9,-9,-9,-9    ! Beg/End I/J/K ("-" for all)
${BEG_CYMDH}         ! Start datetime (UTC yyyymmddhh, "-" for all)
${END_CYMDH}         ! End   datetime (UTC yyyymmddhh, "-" for all)
1                 ! Number of WRF output files
${wrfout_root}/wrfout_d01_2018-05-17_00_00_00
EOF

./calwrf.exe


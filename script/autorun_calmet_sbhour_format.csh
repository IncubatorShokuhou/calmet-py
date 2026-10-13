#!/bin/csh -f
############root directories setting##########
set script_dir = `cd "$0:h" && pwd`
if (! $?work_root) setenv work_root "$script_dir/.."
if (! $?wrfout_root) then
   if ($?WRFOUT_ROOT) then
      setenv wrfout_root "$WRFOUT_ROOT"
   else
      echo "ERROR: set wrfout_root or WRFOUT_ROOT to the directory containing WRF output files."
      exit 2
   endif
endif
set path =  (. $path ${work_root}/bin)
setenv COMMAND_ROOT ${work_root}/bin
if (! $?calwrf_root) setenv calwrf_root ${work_root}/CALWRF
if (! $?calwrfout_root) setenv calwrfout_root ${calwrf_root}/calwrf_out
if (! $?terrel_root) setenv terrel_root ${work_root}/TERREL
if (! $?ctgproc_root) setenv ctgproc_root ${work_root}/CTGPROC
if (! $?makegeo_root) setenv makegeo_root ${work_root}/MAKEGEO
if (! $?calmet_root) setenv calmet_root ${work_root}/calmet_mpi
if (! $?calmet_output) setenv calmet_output ${work_root}/output
if (! $?terrain_root) setenv terrain_root ${terrel_root}/SRTM3

foreach required_dir ($work_root $wrfout_root $calwrf_root $calwrfout_root $terrel_root $ctgproc_root $makegeo_root $calmet_root $calmet_output $terrain_root)
   if (! -d "$required_dir") then
      echo "ERROR: required directory does not exist: $required_dir"
      exit 2
   endif
end

#############domain setting####################
############changable variables################
setenv input_dom 3             #wrf output domain used for downscaling
setenv input_res 3             #wrf output resolution used for downscaling
setenv map_proj LCC            #map projection LCC
setenv rlat 37.27N              #lower left latitude
setenv rlon 105.34E             #lower left longitude
setenv rlat1 15N               #no need to change unless MAP PROJECTION IS CHANGED
setenv rlat2 45N               #no need to change unless MAP PROJECTION IS CHANGED
setenv xrefkm 0                #no need to chang unless MAP PROJECTION IS CHANGED
setenv yrefkm 0                #no need to change unless MAP PROJECTION IS CHANGED
setenv nx 206                  #x grid cells for calmet
setenv ny 290                  #y grid cells for calmet
setenv dgridkm 0.1               #grid spacing (km) for calmet
setenv FCST_HR 2             #running hours 
setenv time_step 900           #time_step for calmet run
##################### Determine initial time ##################
date -u
##################### Determine initial time ##################
if ($#argv == 1) then
   echo "Usage: $0 [BEGIN_TIME END_TIME]"
   exit 2
endif
if( $#argv >= 2 ) then
   setenv BEG_CYMDH_HM $1
   setenv END_CYMDH_HM $2
   echo 'Initial Time:' $BEG_CYMDH_HM
     echo 'End Time:' $END_CYMDH_HM
   goto override_start
endif
###############################################################
################################################################
#setenv CUR_CYMDH `cur_cymdhms | cut -c 1-10`
setenv CUR_CYMDH `/bin/date +%Y%m%d%H%M%S | cut -c 1-10`
setenv CUR_H `echo $CUR_CYMDH  | cut -c 9-10`

if ($CUR_H >= 8 && $CUR_H <= 19) then
   setenv HR_MINUS $CUR_H
endif

if ($CUR_H >= 20 && $CUR_H <= 23) then
   @ temp_hr = $CUR_H - 20
   @ temp_hr = $temp_hr + 8
   setenv HR_MINUS $temp_hr
endif

if ($CUR_H >= 0 && $CUR_H <= 7) then
   @ temp_hr = $CUR_H + 12
   setenv HR_MINUS $temp_hr
endif

setenv BEG_CYMDH `advance_cymdh $CUR_CYMDH -$HR_MINUS`

echo LOCAL_CYMDH `advance_cymdh $CUR_CYMDH +8`
echo CUR_CYMDH $CUR_CYMDH
echo BEG_CYMDH $BEG_CYMDH
################################################################

override_start:

###################  20170508_08:00
setenv BEG_CCYY `echo $BEG_CYMDH_HM | cut -c 1-4`
setenv BEG_MM `echo $BEG_CYMDH_HM | cut -c 5-6`
setenv BEG_DD `echo $BEG_CYMDH_HM | cut -c 7-8`
setenv BEG_HH `echo $BEG_CYMDH_HM | cut -c 10-11`
setenv BEG_MI `echo $BEG_CYMDH_HM | cut -c 13-14`
@ m =60 * $BEG_MI
setenv BEG_SEC `printf "%04d" "$m"`


################################################################
setenv END_CCYY  `echo $END_CYMDH_HM | cut -c 1-4`
setenv END_MM  `echo $END_CYMDH_HM | cut -c 5-6`
setenv END_DD `echo $END_CYMDH_HM | cut -c 7-8`
setenv END_HH `echo $END_CYMDH_HM | cut -c 10-11`
setenv END_MI `echo $END_CYMDH_HM | cut -c 13-14`
@ n =60 * $END_MI
setenv END_SEC `printf "%04d" "$n"`


setenv BEG_CYMDH `printf "%d" "$BEG_CCYY$BEG_MM$BEG_DD$BEG_HH$BEG_SEC"`
setenv END_CYMDH `printf "%d" "$END_CCYY$END_MM$END_DD$END_HH$END_SEC"`







setenv BEG_CCYYMMDD `echo $BEG_CYMDH | cut -c 1-8`
setenv END_CCYYMMDD `echo $END_CYMDH | cut -c 1-8`

set julday0=`ymd2yj ${BEG_CCYYMMDD}`
set julday1=`yj_next ${julday0}`
set julday2=`yj_next ${julday1}`

############### WRF TIME ################# 
set date_start = $BEG_CYMDH
set date_end = $END_CYMDH

setenv year_start `echo ${date_start} | cut -c 1-4`
setenv month_start `echo ${date_start} | cut -c 5-6`
setenv day_start `echo ${date_start} | cut -c 7-8`
setenv hour_start `echo ${date_start} | cut -c 9-10`
setenv year_end `echo ${date_end} | cut -c 1-4`
setenv month_end `echo ${date_end} | cut -c 5-6`
setenv day_end `echo ${date_end} | cut -c 7-8`
setenv hour_end `echo ${date_end} | cut -c 9-10`
###############################################################

############running CALWRF#################
echo "converting WRFOUT for calemt!"
cd ${calwrf_root}
rm -f ${calwrfout_root}/*
./calwrf.csh > calwrf.log
echo "Finished converting WRFOUT!"
############making geological and landuse data########
echo "making terrain data!"
echo "checking needed terrain file!"
cd ${terrel_root}
./terrel_check.csh > terrel_check.log
echo "these terrain files are needed for calmet run!"
echo "linking terrain files!"
setenv filnum `grep -c 'n*e*HGT or' TERR_CHECK.LST`
setenv filename  `grep 'n*e*HGT or' TERR_CHECK.LST |cut -c1-12 |tr "[a-z]" "[A-Z]"`
rm -f filename.txt
rm -f *.HGT
foreach instr($filename)
        echo ''!'SRTM3 = ' $instr ''!'  '!'END!' >>&! filename.txt
        setenv fname `echo ${instr} |cut -c1-7`
        ln -sf ${terrain_root}/${fname}.hgt ${fname}.HGT
end
awk '{print}' filename.txt 
echo "finished linking the needed terrain files!"
echo "generating terrain data for calmet run!"
./terrel.csh > terrel.log
echo "Finished generating terrain data for calmet run!"

echo "Converting Landuse data for calmet run!"
cd ${ctgproc_root}
./ctgproc.csh > ctgproc.log
echo "finished converting Landuse data for calmet run!"

echo "Merging terrain & land use data for calmet run!"
cd ${makegeo_root}
ln -sf ${terrain_root}/TERR.DAT terr.dat
ln -sf ${ctgproc_root}/LULC.DAT lulc.dat
./makegeo.csh > makegeo.log
echo "Finished merging terrain & land use data for calmet run!"

#################running CALMET######################
echo "running calmet!"
cd ${calmet_root}
./calmet.csh > calmet.log
echo "Finished running calmet!"
#################saving CALMET output###############
echo "saving calmet output!"
mv calmet.dat ${calmet_output}/calmetout_${BEG_CYMDH}_${END_CYMDH}.dat
echo "Finished saving calmet output!"
echo "Finished calmet run"
exit 0

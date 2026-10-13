;*********************************************   sad
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"
begin
;********************************************  
fili = new(5,"string")
fili2 = new(5,"string")
diro = getenv("CALMET_OUTPUT_DIR")
if (ismissing(diro) .or. diro.eq."") then
  print("FATAL: set CALMET_OUTPUT_DIR to the directory containing CALMET output files")
  status_exit(2)
end if
filedate="20161120000000_20161120120000"
filo = "calmetout_15min_"+filedate+".nc"   
	filwrfout=getenv("WRFOUT_FILE")
if (ismissing(filwrfout) .or. filwrfout.eq."") then
  print("FATAL: set WRFOUT_FILE to the WRF NetCDF file used for timestamps")
  status_exit(2)
end if
	fili(0) = diro+"/calmetout_"+filedate+"_u.dat"
	fili(1) = diro+"/calmetout_"+filedate+"_v.dat"
	fili(2) = diro+"/calmetout_"+filedate+"_t.dat"
	fili(3) = diro+"/calmetout_"+filedate+"_s.dat"
	fili(4) = diro+"/calmetout_"+filedate+"_e.dat"

            

setfileoption("bin",	 	"RecordMarkerSize",4)    
setfileoption("bin",	"ReadByteOrder","LittleEndian" )   
	

miss_val = -9999
nx=206
ny=290
nz=5
nt=48
lon_min=105.34  
lon_max=105.5742357091
lat_min=37.27   
lat_max=37.5303885075



fwrfout = addfile(filwrfout,"r")
wrfouttime  = fwrfout->Times(48:nt+47,:)   ;2017-06-24_13:45:00

wrfouttimestr=new(nt,"string",-9999)
do t=0,nt-1
wrfouttimestr(t)=chartostring(wrfouttime(t,:))
end do
wrfouttimestr@long_name = "time"
;wrfouttimestr@units = "YYYY-MM-DD_HH:MM:SS"

;wrftimenum=wrf_times_c(wrfouttime,2)	;opt=2 returns units of "yyyymmddhhmnss" (type "double")
;printVarSummary(wrftimenum)
;print(wrftimenum(0))	;20170621000000 units :	yyyymmddhhmnss
;format="%Y-%N-%D_%H:%M:%S"
;wrftimeformat=cd_string(wrftimenum,format)
;printVarSummary(wrftimeformat)
;print(wrftimeformat(0))



;*********************************************	 
print("----------------------readvar-----------------------")
jultime = new(nt,"integer",-9999)
sec = new(nt,"integer",-9999)
minu = new(nt,"integer",-9999)
yyyymmdd=new(nt,"integer",-9999)
jultimestr=new(nt,"string",-9999)
jultimesplit=new((/nt,2/),"string",-9999)
yyyyjul=new(nt,"string",-9999)
yyyyddd=new(nt,"integer",-9999)
hh=new(nt,"string",-9999)
hhint=new(nt,"integer",-9999)
yyyymmddhh=new(nt,"integer",-9999)
uwnd = new((/nz,ny,nx,nt/),		"float",-9999)
vwnd = new((/nz,ny,nx,nt/),		"float",-9999) 
;ane = new((/nz,ny,nx,nt/),		"float",-9999) 
elev=  new((/ny,nx/),		"float",-9999)   
  
uwnd = 			fbinrecread  (fili(0), 0,   (/nz,ny,nx,nt/), "float") 
vwnd = 			fbinrecread  (fili(1), 0,   (/nz,ny,nx,nt/), "float") 
jultime = 			fbinrecread  (fili(2), 0,   nt, "integer") 
sec = 			fbinrecread  (fili(3), 0,   nt, "integer") 
elev = 			fbinrecread  (fili(4), 0,   (/ny,nx/), "float") 


uwnd1 = new((/nt,nz,ny,nx/),		"float",-9999)
vwnd1 = new((/nt,nz,ny,nx/),		"float",-9999) 
;ane1 = new((/nt,nz,ny,nx/),		"float",-9999) 

do t=0,nt-1

uwnd1(t,:,:,:)=uwnd(:,:,:,t)
vwnd1(t,:,:,:)=vwnd(:,:,:,t)
end do
;ane=sqrt(uwnd^2+vwnd^2)



;*******************yyyyjulhh to yyyymmddhh**************************	
minu=sec/60
jultimestr=tostring(jultime)
dtime_map=(/7,2/)
print(jultime)
print(jultimestr)
jultimesplit=str_split_by_length(jultimestr,dtime_map)
yyyyjul=jultimesplit(:,0)
hh=jultimesplit(:,1)
;print(yyyyjul)
;print(hh)
yyyyddd=toint(yyyyjul)
yyyymmdd=yyyyddd_to_yyyymmdd(yyyyddd)
print(yyyymmdd)
hhint=toint(hh)
yyyymmddhh=yyyymmdd*100+hhint

do t=0,nt-1
yyyymmddhh(t)=t
end do

yyyymmddhh@long_name = "time"
yyyymmddhh@units = "YYYYMMDDHH"

 print(yyyymmddhh(1))
  print(minu(1))
 yyyymmddhhmm=new(nt,"string",-9999)
 do i=0,nt-1
 yyyymmddhhmm(i)=sprinti("%10d",yyyymmddhh(i))+sprinti("%02d",minu(i))
 end do
 yyyymmddhhmm@long_name = "time"
yyyymmddhhmm@units = "YYYYMMDDHHMN"
 print(yyyymmddhhmm(1))

 
 
;*********************************************	           
system("/bin/rm -f " + diro + "/" + filo)
fout  = addfile (diro + "/" + filo, "c")
            print("outputing "+diro + "/" + filo)
fout@title="calmet netcdf"
fout@creation_date=systemfunc("date")
;filedimdef(fout,"time",-1,True)
;*********************************************	 
lon = new(nx,"float",-9999)
lat = new(ny,"float",-9999)
lev10 = new(nz,"float",-9999)
lon = fspan(lon_min,lon_max,nx)
lat = fspan(lat_min,lat_max,ny) 
lev10 = (/ 20.00000, 40.00000, 80.00000, 160.0000, 300.0000 /)
lon@long_name = "longitude"
lon@units = "degree_east"
lon@_FillValue = miss_val
lat@long_name = "latitude"
lat@units = "degree_north"
lat@_FillValue = miss_val
lev10@long_name = "level"
lev10@units = "m"


;*********************************************	 
	uwnd1!1 = "lev" 
	uwnd1!2 = "lat" 
	uwnd1!3 = "lon"
	uwnd1!0 = "time"
	uwnd1@long_name = "uwnd_calmet_out"
	uwnd1@units = "m/s"
	uwnd1@_FillValue = miss_val
	uwnd1&lat = lat
	uwnd1&lon = lon  
	uwnd1&lev = lev10  
	uwnd1&time = yyyymmddhh  
        printVarSummary(uwnd)

;
	vwnd1!1 = "lev" 
	vwnd1!2 = "lat" 
	vwnd1!3 = "lon"
	vwnd1!0 = "time"
	vwnd1@long_name = "vwnd_calmet_out"
	vwnd1@units = "m/s"
	vwnd1@_FillValue = miss_val
	vwnd1&lat = lat
	vwnd1&lon = lon  
	vwnd1&lev = lev10  
	vwnd1&time = yyyymmddhh  
;

dh=new((/nt,nz,ny,nx/),		"float",-9999) 
do z=0,nz-1
dh(:,z,:,:)=lev10(z)
end do 
    dh@description = "Height"
    dh@units       = "m"
printVarSummary(dh)

uwind=new((/nt,ny,nx/),		"float",-9999) 
vwind=new((/nt,ny,nx/),		"float",-9999) 

do t=0,nt-1
uwind(t,:,:)=wrf_interp_3d_z(uwnd1(t,:,:,:),dh(t,:,:,:),70.) 
vwind(t,:,:)=wrf_interp_3d_z(vwnd1(t,:,:,:),dh(t,:,:,:),70.) 
end do

	
	uwind!1 = "lat" 
	uwind!2 = "lon"
	uwind!0 = "time"
	uwind@long_name = "vwnd_calmet_out"
	uwind@units = "m/s"
	uwind@_FillValue = miss_val
	uwind&lat = lat
	uwind&lon = lon  
	uwind&time = yyyymmddhh   

	vwind!1 = "lat" 
	vwind!2 = "lon"
	vwind!0 = "time"
	vwind@long_name = "vwnd_calmet_out"
	vwind@units = "m/s"
	vwind@_FillValue = miss_val
	vwind&lat = lat
	vwind&lon = lon  
	vwind&time = yyyymmddhh   

	;ane!0 = "lev" 
	;ane!1 = "lat" 
	;ane!2 = "lon"
	;ane!3 = "time"
	;ane@long_name = "ane_calmet_out"
	;ane@units = "m/s"
	;ane@_FillValue = miss_val
	;ane&lat = lat
	;ane&lon = lon  
	;ane&lev = lev10  
	;ane&time = yyyymmddhh 

	elev!0 = "lat" 
	elev!1 = "lon"
	elev&lat = lat
	elev&lon = lon 
	

	fout->Elev=elev

	fout->U=uwind 
	fout->V=vwind
	;fout->ane=ane

            print("end output")
end
;*********************************************
;*********************************************
;*********************************************

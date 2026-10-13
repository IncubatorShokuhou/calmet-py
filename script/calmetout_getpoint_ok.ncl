;*********************************************   sad
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/skewt_func.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/wind_rose.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
begin
;********************************************  
filedate=new(1,"string")

filedate="2016012100_2016012400_30km"
nt=72

input_dir = getenv("CALMET_OUTPUT_DIR")
if (ismissing(input_dir) .or. input_dir.eq."") then
  print("FATAL: set CALMET_OUTPUT_DIR to the directory containing extracted CALMET files")
  status_exit(2)
end if
point_output_dir = getenv("CALMET_POINT_OUTPUT_DIR")
if (ismissing(point_output_dir) .or. point_output_dir.eq."") then
  print("FATAL: set CALMET_POINT_OUTPUT_DIR to the directory for point text files")
  status_exit(2)
end if

fili = new(5,"string")
fili(0)  = input_dir+"/calmetout_"+filedate+"_u.dat"
fili(1)  = input_dir+"/calmetout_"+filedate+"_v.dat"
fili(2)  = input_dir+"/calmetout_"+filedate+"_t.dat"
fili(3)  = input_dir+"/calmetout_"+filedate+"_el.dat"
fili(4)  = input_dir+"/calmetout_"+filedate+"_lu.dat"
              

;setfileoption("bin",	 	"RecordMarkerSize",8)    
setfileoption("bin",	"ReadByteOrder","LittleEndian" )   


miss_val = -9999
nx=300
ny=300
nz=5

lon_min=109.2
lon_max=109.4932918428
lat_min=22.0
lat_max=22.2684492572
;*********************************************	 
print("----------------------readvar-----------------------")
jultime = new(nt,"integer",-9999)
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
;elev=  new((/ny,nx/),		"float",-9999)   
;landu=  new((/ny,nx/),		"integer",-9999)   
uwnd = 			fbinrecread  (fili(0), 0,   (/nz,ny,nx,nt/), "float") 
vwnd = 			fbinrecread  (fili(1), 0,   (/nz,ny,nx,nt/), "float") 
jultime = 			fbinrecread  (fili(2), 0,   nt, "integer") 
;elev = 			fbinrecread  (fili(3), 0,  (/ny,nx/), "float") 
;landu = 			fbinrecread  (fili(4), 0,  (/ny,nx/), "integer") 


;*******************yyyyjulhh to yyyymmddhh**************************	

jultimestr=tostring(jultime)
dtime_map=(/7,2/)
print(jultime)
print(jultimestr)
jultimesplit=str_split_by_length(jultimestr,dtime_map)
yyyyjul=jultimesplit(:,0)
hh=jultimesplit(:,1)

yyyyddd=toint(yyyyjul)
yyyymmdd=yyyyddd_to_yyyymmdd(yyyyddd)
print(yyyymmdd)
hhint=toint(hh)
yyyymmddhh=yyyymmdd*100+hhint
yyyymmddhh@long_name = "time"
yyyymmddhh@units = "YYYYMMDDHH"

 
 
 
;*********************************************	           


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

dh=new((/nt,nz,ny,nx/),"float",-9999)
do z=0,nz-1
	dh(:,z,:,:)=lev10(z)
end do
;*********************************************	
uwnd2 = new((/nt,nz,ny,nx/),		"float",-9999)
vwnd2 = new((/nt,nz,ny,nx/),		"float",-9999) 
	do t= 1-1, nt-1
      uwnd2(t,:,:,:) = uwnd(:,:,:,t)
      vwnd2(t,:,:,:) = vwnd(:,:,:,t)
	end do

uwnd70 = new((/nt,ny,nx/),		"float",-9999)
vwnd70 = new((/nt,ny,nx/),		"float",-9999)
	do t= 1-1, nt-1
      uwnd70(t,:,:) = wrf_interp_3d_z(uwnd2(t,:,:,:),dh(t,:,:,:),80.) 
      vwnd70(t,:,:) = wrf_interp_3d_z(vwnd2(t,:,:,:),dh(t,:,:,:),80.)  
	end do

;*********************************************	
lat2d=new((/ny,nx/),		"float",-9999)
lon2d=new((/ny,nx/),		"float",-9999)

do y=0,ny-1
lat2d(y,:)=lat(y)
end do
do x=0,nx-1
lon2d(:,x)=lon(x)
end do

   latlist=(/22.139,22.1/)
   lonlist=(/109.319,109.377/)
   numlist=(/0001,0002/)
   
do i=0,1,1
   u2  = rcm2points (lat2d, lon2d, uwnd70, latlist(i), lonlist(i), 0)   ;opt=0 means use an inverse distance weight interpolation. 
   v2  = rcm2points (lat2d, lon2d, vwnd70, latlist(i), lonlist(i), 0)
   
   printVarSummary(u2)   ;[8784] x [1]
   ;print(aa)

   
;;;==================compute wind direction==============================================  
ane=sqrt(u2^2+v2^2)
r2d = 45.0/atan(1.0)     ; conversion factor (radians to degrees)
dir = atan2(u2, v2) * r2d + 180  
printVarSummary(dir)
printVarSummary(ane)

	data=new((/nt,3/),"float")  
   data(:,0)=yyyymmddhh
   data(:,1)=ane(:,0)
   data(:,2)=dir(:,0)
   

   
   printVarSummary(data)
;;;====================write 70m ane&dir==================add time==========================
 write_table(point_output_dir+"/Mast"+numlist(i)+"#("+latlist(i)+"N,"+lonlist(i)+"E)_Calmetout80m_"+filedate+".txt", "w", [/yyyymmddhh,data(:,1),data(:,2)/],"%d %16.3f %16.3f")
  
  

delete(u2) 
delete(v2) 
delete(ane) 

delete(dir) 
delete(data) 

end do	


            print("end output")
end
;*********************************************
;*********************************************
;*********************************************

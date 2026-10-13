program ctgproc_gselect_reference
  implicit none
  integer :: center_hits
  real :: center_lon,center_lat,north,south,east,west
  center_lon=-100.
  center_lat=8.4999999
  north=90.
  south=8.5
  east=-20.
  west=-180.
  center_hits=0
  if(center_lat.le.north.and.center_lat.ge.south) center_hits=center_hits+1
  if(center_lon.le.east.and.center_lon.ge.west) center_hits=center_hits+1
  write (*,'(I0,1X,Z8.8)') center_hits,transfer(center_lat,0)
end program ctgproc_gselect_reference

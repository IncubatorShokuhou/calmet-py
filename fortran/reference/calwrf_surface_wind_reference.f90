subroutine calwrf_surface_wind(n,u,v,alpha,speed,direction) bind(C)
  use iso_c_binding, only: c_int, c_float
  implicit none
  integer(c_int), value :: n
  real(c_float), intent(in) :: u(n),v(n),alpha(n)
  real(c_float), intent(out) :: speed(n),direction(n)
  real(c_float) :: east(n),north(n),ut(n),vt(n)
  real(c_float), parameter :: pi=3.141592653589793_c_float
  real(c_float), parameter :: rdp=180.0_c_float/pi

  ut=u
  vt=v
  east=vt*sin(alpha)+ut*cos(alpha)
  north=vt*cos(alpha)-ut*sin(alpha)
  speed=sqrt(east*east+north*north)
  direction=0.0_c_float
  where(speed >= 1.0e-9_c_float)
    direction=270.0_c_float-rdp*atan2(north,east)
  end where
  where(direction > 360.0_c_float)
    direction=direction-360.0_c_float
  end where
  where(direction <= 0.0_c_float)
    direction=direction+360.0_c_float
  end where
end subroutine calwrf_surface_wind

subroutine calwrf_rotation(n,map_proj,truelat1,truelat2,stand_lon,lat,lon,alpha) bind(C)
  use iso_c_binding, only: c_int, c_float
  implicit none
  integer(c_int), value :: n,map_proj
  real(c_float), value :: truelat1,truelat2,stand_lon
  real(c_float), intent(in) :: lat(n),lon(n)
  real(c_float), intent(out) :: alpha(n)
  real(c_float), parameter :: pi=3.141592653589793_c_float
  real(c_float), parameter :: rpd=pi/180.0_c_float
  real(c_float) :: cone

  if (map_proj == 2) then
    cone=1.0_c_float
  else if (abs(truelat1-truelat2) <= 0.1_c_float) then
    cone=sin(abs(truelat1)*rpd)
  else
    cone=(log(cos(truelat1*rpd))-log(cos(truelat2*rpd))) / &
         (log(tan((45.0_c_float-abs(truelat1)/2.0_c_float)*rpd))- &
          log(tan((45.0_c_float-abs(truelat2)/2.0_c_float)*rpd)))
  end if
  alpha=(lon-stand_lon)*cone*rpd
  where(lon-stand_lon > 180.0_c_float)
    alpha=(lon-stand_lon-360.0_c_float)*cone*rpd
  elsewhere(lon-stand_lon < -180.0_c_float)
    alpha=(lon-stand_lon+360.0_c_float)*cone*rpd
  end where
  where(lat < 0.0_c_float) alpha=-alpha
end subroutine calwrf_rotation

subroutine calwrf_cone(truelat1,truelat2,cone) bind(C)
  use iso_c_binding, only: c_float
  implicit none
  real(c_float), value :: truelat1,truelat2
  real(c_float), intent(out) :: cone
  real(c_float), parameter :: pi=3.141592653589793_c_float
  real(c_float), parameter :: rpd=pi/180.0_c_float
  if (abs(truelat1-truelat2) <= 0.1_c_float) then
    cone=sin(abs(truelat1)*rpd)
  else
    cone=(log(cos(truelat1*rpd))-log(cos(truelat2*rpd))) / &
         (log(tan((45.0_c_float-abs(truelat1)/2.0_c_float)*rpd))- &
          log(tan((45.0_c_float-abs(truelat2)/2.0_c_float)*rpd)))
  end if
end subroutine calwrf_cone

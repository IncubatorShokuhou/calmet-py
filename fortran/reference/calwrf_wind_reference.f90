program calwrf_wind_reference
  implicit none
  integer :: i
  real, parameter :: pi=3.141592653589793
  real, parameter :: rdp=180.0/pi
  real, dimension(4) :: u, v, alpha, east, north, speed, direction
  real, dimension(4) :: latitude, longitude, difference, lcc_alpha
  real :: truelat1, truelat2, stand_lon, cone, rpd
  u=(/ -3.123456, 0.0, 5.0, 0.0000000001 /)
  v=(/ 4.765432, -2.5, 0.0, -0.0000000002 /)
  alpha=(/ 0.3712345, -0.8123456, 0.0, 1.125 /)
  do i=1,4
    east(i)=v(i)*sin(alpha(i))+u(i)*cos(alpha(i))
    north(i)=v(i)*cos(alpha(i))-u(i)*sin(alpha(i))
    speed(i)=sqrt(east(i)*east(i)+north(i)*north(i))
    direction(i)=0.0
    if (speed(i) >= 1.0e-9) direction(i)=270.0-rdp*atan2(north(i),east(i))
    if (direction(i) > 360.0) direction(i)=direction(i)-360.0
    if (direction(i) <= 0.0) direction(i)=direction(i)+360.0
  end do
  write (*,'(16(Z8.8,1X))') transfer(east,(/0,0,0,0/)), &
    transfer(north,(/0,0,0,0/)),transfer(speed,(/0,0,0,0/)), &
    transfer(direction,(/0,0,0,0/))

  truelat1=30.1234
  truelat2=60.5678
  stand_lon=170.25
  latitude=(/ 60.0, -60.0, 30.0, -15.0 /)
  longitude=(/ -170.1, -170.1, 160.123, 350.5 /)
  rpd=pi/180.0
  cone=(alog(cos(truelat1*rpd))-alog(cos(truelat2*rpd)))/ &
    (alog(tan((90.0-abs(truelat1))*rpd*0.5))- &
     alog(tan((90.0-abs(truelat2))*rpd*0.5)))
  difference=longitude-stand_lon
  where (difference > 180.0) difference=difference-360.0
  where (difference < -180.0) difference=difference+360.0
  lcc_alpha=difference*cone*rpd
  where (latitude < 0.0) lcc_alpha=-1.0*lcc_alpha
  write (*,'(5(Z8.8,1X))') transfer(cone,0),transfer(lcc_alpha,(/0,0,0,0/))
end program calwrf_wind_reference

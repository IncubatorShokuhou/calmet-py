program rdup_wind_reference
  implicit none
  integer :: i
  real, dimension(4) :: direction, speed, u, v
  real :: convergence, wdmap, radians
  direction=(/ 217.3, 0.0, 10.0, 999.0 /)
  speed=(/ 5.718242, 5.0, 0.01, 5.0 /)
  convergence=1.125
  do i=1,4
    if (direction(i) < 998.99 .and. speed(i) < 998.99) then
      if (direction(i) > 0.01 .and. speed(i) > 0.01) then
        wdmap=direction(i)-convergence
        if (wdmap < 0.0) wdmap=wdmap+360.0
        if (wdmap > 360.0) wdmap=wdmap-360.0
      else
        wdmap=direction(i)
      end if
      radians=0.0174533*wdmap
      u(i)=-speed(i)*sin(radians)
      v(i)=-speed(i)*cos(radians)
    else
      u(i)=999.0
      v(i)=999.0
    end if
  end do
  write (*,'(8(Z8.8,1X))') transfer(u,(/0,0,0,0/)),transfer(v,(/0,0,0,0/))
end program rdup_wind_reference

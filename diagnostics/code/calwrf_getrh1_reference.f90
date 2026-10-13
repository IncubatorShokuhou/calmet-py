program calwrf_getrh1_reference
  implicit none
  integer :: i
  real, dimension(6) :: p,tk,qq,rh
  p=(/ 1000.125, 850.375, 500.625, 250.875, 75.125, 1013.25 /)
  tk=(/ 300.125, 273.175, 250.625, 220.375, 195.125, 310.75 /)
  qq=(/ 12.345, 4.567, 0.8125, 0.03125, 0.00075, 35.5 /)
  do i=1,6
    call getrh1(p(i),tk(i),qq(i),rh(i))
  end do
  write (*,'(6(Z8.8,1X))') transfer(rh,(/0,0,0,0,0,0/))
contains
  subroutine getrh1(p,tk,qq,rh)
    real, intent(in) :: p,tk,qq
    real, intent(out) :: rh
    real :: qq2,qsat
    qq2=qq/1000.
    qsat=0.622*6.112/p*exp(17.67*(tk-273.15)/(tk-29.65))
    rh=max(100.*qq2/qsat,0.)
    if(rh.gt.100.) rh=100.
  end subroutine getrh1
end program calwrf_getrh1_reference

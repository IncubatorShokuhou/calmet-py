program makegeo_snow_albedo_reference
  implicit none
  integer :: i,nd,nds
  integer, dimension(5) :: age
  real :: normal,snowy,ralb,dd,rdc,ratio,ww
  real, dimension(5) :: result
  normal=0.1234567
  snowy=0.8123456
  ralb=0.1376543
  nds=13
  age=(/ 0, 1, 7, 13, 25 /)
  dd=snowy-normal
  rdc=dd*ralb
  do i=1,5
    nd=age(i)
    if(nd.gt.nds) nd=nds
    ratio=float(nd)/nds
    ww=1.0-rdc*ratio
    result(i)=snowy*ww
  end do
  write (*,'(7(Z8.8,1X))') transfer((/dd,rdc/),(/0,0/)), &
    transfer(result,(/0,0,0,0,0/))
end program makegeo_snow_albedo_reference

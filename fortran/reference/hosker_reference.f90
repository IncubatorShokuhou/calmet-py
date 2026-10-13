program hosker_reference
  implicit none
  real, dimension(4) :: u, v, speed, roughness
  integer :: i
  u=(/ -3.123456, 0.0, 10.125, -0.00003 /)
  v=(/ 4.765432, 0.0, -2.375, 0.00004 /)
  do i=1,4
    speed(i)=sqrt(u(i)**2+v(i)**2)
    roughness(i)=amax1(2.0e-6,2.0e-6*speed(i)**2.5)
  end do
  write (*,'(8(Z8.8,1X))') transfer(speed,(/0,0,0,0/)), &
    transfer(roughness,(/0,0,0,0/))
end program hosker_reference

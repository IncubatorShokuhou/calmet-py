program calwrf_temperature_reference
  implicit none
  integer :: i
  real, dimension(4) :: p, pb, perturbation_theta, pressure, temperature
  p=(/ 1234.567, -4321.125, 0.125, 18765.5 /)
  pb=(/ 91345.25, 100123.75, 54321.875, 70111.125 /)
  perturbation_theta=(/ 2.3456, -12.75, 18.125, -0.03125 /)
  do i=1,4
    pressure(i)=(p(i)+pb(i))*0.01
    temperature(i)=(perturbation_theta(i)+300.0)* &
      (pressure(i)/1000.0)**(287.04/1004.0)
  end do
  write (*,'(8(Z8.8,1X))') transfer(pressure,(/0,0,0,0/)), &
    transfer(temperature,(/0,0,0,0/))
end program calwrf_temperature_reference

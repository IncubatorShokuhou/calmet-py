program heatfx_power_reference
  implicit none
  integer :: i
  real :: value, fourth, sixth
  integer, dimension(5) :: bits = [ &
       int(z'3f8ccccd'), int(z'40490fdb'), int(z'42f6e979'), &
       int(z'438fe818'), int(z'447a199a') ]

  do i = 1, size(bits)
     value = transfer(bits(i), value)
     fourth = value**4
     sixth = value**6
     write(*,'(3(z8.8,1x))') transfer(value,0), transfer(fourth,0), &
          transfer(sixth,0)
  end do
end program heatfx_power_reference

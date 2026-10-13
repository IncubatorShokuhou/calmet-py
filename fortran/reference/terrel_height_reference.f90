program terrel_height_reference
  implicit none
  real :: total(3),height(3)
  integer :: count(3),i
  total=(/ 12345678.0, 98765.43, 0.1234567 /)
  count=(/ 7, 13, 3 /)
  do i=1,3
    height(i)=total(i)/float(count(i))
  end do
  write (*,'(3(Z8.8,1X))') transfer(height,(/0,0,0/))
end program terrel_height_reference

program makegeo_split_reference
  implicit none
  integer :: kk
  real :: pland(3),prec(2),sum
  pland=(/ 1.0, 0.125, 0.25 /)
  prec=(/ 25.0, 75.0 /)
  sum=0.0
  do kk=1,2
    pland(kk+1)=pland(kk+1)+prec(kk)*pland(1)
    sum=sum+prec(kk)
  end do
  pland(1)=pland(1)-sum*pland(1)
  write (*,'(3(Z8.8,1X))') transfer(pland,(/0,0,0/))
end program makegeo_split_reference

program makegeo_snow_fill_reference
  implicit none
  integer :: i
  real :: values(3),total,temporal,spatial
  values=(/ 0.13080607, 0.14924558, 0.40711287 /)
  total=0.0
  do i=1,3
    total=total+values(i)
  end do
  temporal=total/3
  total=0.0
  ! FILLS encounters these three valid cells in this JJ/II order.
  do i=1,3
    total=total+values(i)
  end do
  spatial=total/3
  write (*,'(2(Z8.8,1X))') transfer((/temporal,spatial/),(/0,0/))
end program makegeo_snow_fill_reference

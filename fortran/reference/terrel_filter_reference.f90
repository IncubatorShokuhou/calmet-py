program terrel_filter_reference
  implicit none
  integer :: i
  real :: z(4),znoise,terdef,void
  z=(/1.00000002,.9999999,-32000.,-32000.1/)
  znoise=1.00000003
  terdef=12.3456789
  void=-32000.
  do i=1,4
    if(z(i).lt.znoise.and.z(i).gt.void) z(i)=znoise
    if(z(i).lt.void) z(i)=terdef
  enddo
  write (*,'(6(Z8.8,1X))') transfer(znoise,0),transfer(terdef,0), &
    (transfer(z(i),0),i=1,4)
end program terrel_filter_reference

program terrel_height_type_coordinates_reference
  implicit none
  integer, parameter :: nx=2,ny=4
  integer :: i,ixxr,iyyr
  real :: disk(nx),cbdisk(nx),xorgk,yorgk,sizek,xllk,yllk,xxr,yyr
  disk=(/1.123456,2.876543/)
  xorgk=10.12345
  yorgk=20.54321
  sizek=.731245
  cbdisk(1)=.5*(disk(1)+disk(2))
  cbdisk(nx)=2.*disk(nx)-cbdisk(nx-1)
  xllk=xorgk-cbdisk(nx)
  yllk=yorgk-cbdisk(nx)
  do i=1,nx*ny
    ixxr=mod(i,nx)
    if(ixxr.eq.0) ixxr=nx
    iyyr=i/nx+1
    if(ixxr.eq.nx) iyyr=iyyr-1
    xxr=(float(ixxr-1)+.5)*sizek+xllk
    yyr=(float(iyyr-1)+.5)*sizek+yllk
    write (*,'(2(Z8.8,1X))') transfer(xxr,0),transfer(yyr,0)
  enddo
end program terrel_height_type_coordinates_reference

program terrel_polar_cellind_reference
  implicit none
  integer,parameter :: nx=3,ny=4,np=5
  integer :: p,i,ix,iy,ixp,iyp,iym1,iyp1
  real :: disk(nx),ang(ny),cbdism(nx),cbang(ny)
  real :: xm(np),ym(np),xi,yi,dels,upedge,dnedge,dels2
  real,parameter :: rad2deg=57.29578
  disk=(/1.123456,2.876543,4.222222/)
  ang=(/10.12345,77.77777,190.5432,299.8765/)
  xm=(/1234.567,-3456.789,0.,2500.,-1000./)
  ym=(/2345.678,123.456,0.,-1000.,-2500./)
  dels=250.123
  do i=1,nx-1
    cbdism(i)=.5*(disk(i)+disk(i+1))
    cbdism(nx)=2.*disk(nx)-cbdism(nx-1)
  enddo
  cbdism=cbdism*1000.
  do i=1,ny-1
    cbang(i)=.5*(ang(i)+ang(i+1))-ang(1)
  enddo
  cbang(ny)=.5*(ang(ny)+ang(1)+360.)-ang(1)
  do p=1,np
    ix=1; iy=1
    xi=sqrt(xm(p)**2+ym(p)**2)
    do i=1,nx
      if(xi.ge.cbdism(i)) ix=i+1
    enddo
    if(ix.gt.nx) then
      iy=ny+1
    else
      yi=atan2(xm(p),ym(p))*rad2deg
      yi=yi-ang(1)
      if(yi.lt.0.) yi=yi+360.
      do i=1,ny
        if(yi.ge.cbang(i)) iy=i+1
      enddo
      if(iy.gt.ny) iy=1
    endif
    ixp=ix; iyp=iy
    if(ix.le.nx) then
      if(xi+dels.gt.cbdism(ix)) ixp=ix+1
      if(ix.ge.2) then
        if(xi-dels.lt.cbdism(ix-1)) ixp=ix-1
      endif
      if(iy.eq.1) then; iym1=ny; else; iym1=iy-1; endif
      if(iy.eq.ny) then; iyp1=1; else; iyp1=iy+1; endif
      upedge=cbang(iy)-yi
      if(upedge.lt.0.) upedge=upedge+360.
      dnedge=yi-cbang(iym1)
      if(dnedge.lt.0.) dnedge=dnedge+360.
      if(xi.eq.0.) xi=.001
      dels2=dels*rad2deg/xi
      if(upedge.lt.dels2) then; iyp=iyp1
      elseif(dnedge.lt.dels2) then; iyp=iym1; endif
    endif
    write (*,'(4(Z8.8,1X),4(I0,1X))') transfer(xi,0),transfer(yi,0), &
      transfer(upedge,0),transfer(dels2,0),ix,iy,ixp,iyp
  enddo
end program terrel_polar_cellind_reference

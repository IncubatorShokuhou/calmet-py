program terrel_xyelev_reference
  implicit none
  integer,parameter :: nr=2,np=6
  integer :: p,i,q
  real :: xr(nr),yr(nr),xm(np),ym(np),z(np),radius
  real :: el(nr),xx(4,nr),yy(4,nr),zz(4,nr),dist(4,nr)
  xr=(/1.234567,-3.210987/)
  yr=(/-2.345678,4.567891/)
  xm=(/1100.,1400.,1200.,-3200.,-3000.,-3400./)
  ym=(/-2500.,-2500.,-2200.,4500.,4700.,4800./)
  z=(/10.123,20.456,30.789,40.12,50.34,60.56/)
  radius=.3
  el=-999.
  xx(:,1)=(/-1.e15,1.e15,1.e15,-1.e15/)
  xx(:,2)=xx(:,1)
  yy(:,1)=(/-1.e15,-1.e15,1.e15,1.e15/)
  yy(:,2)=yy(:,1)
  zz=-999.
  dist=1.e15
  do p=1,np
    call add_point(xm(p),ym(p),z(p),.false.)
  enddo
  write (*,'(2(Z8.8,1X))') (transfer(el(i),0),i=1,nr)
  do p=1,np
    call add_point(xm(p),ym(p),z(p),.true.)
  enddo
  do i=1,nr
    do q=1,4
      write (*,'(4(Z8.8,1X))') transfer(xx(q,i),0),transfer(yy(q,i),0), &
        transfer(zz(q,i),0),transfer(dist(q,i),0)
    enddo
  enddo
contains
  subroutine add_point(xm0,ym0,elev,lint)
    implicit none
    real,intent(in) :: xm0,ym0,elev
    logical,intent(in) :: lint
    integer :: j,k
    real :: xkm,ykm,diff
    xkm=.001*xm0
    ykm=.001*ym0
    do j=1,nr
      diff=sqrt((xkm-xr(j))**2+(ykm-yr(j))**2)
      if(.not.lint) then
        if(diff.le.radius.and.elev.gt.el(j)) el(j)=elev
      else
        if(xkm.le.xr(j)) then
          if(ykm.le.yr(j)) then; k=1; else; k=4; endif
        else
          if(ykm.le.yr(j)) then; k=2; else; k=3; endif
        endif
        if(diff.lt.dist(k,j)) then
          xx(k,j)=xkm; yy(k,j)=ykm; zz(k,j)=elev; dist(k,j)=diff
        endif
      endif
    enddo
  end subroutine add_point
end program terrel_xyelev_reference

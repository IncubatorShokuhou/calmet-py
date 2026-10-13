program terrel_interp2d_reference
  implicit none
  integer :: i
  real :: xo(4),yo(4),zo(4),dist(4),x,y,z,zerr
  xo=(/ 0.125, 3.375, 3.625, -0.25 /)
  yo=(/ -0.375, 0.25, 4.125, 3.75 /)
  zo=(/ 123.456, -45.625, 301.875, 78.125 /)
  x=1.4321
  y=2.3456
  do i=1,4
    dist(i)=sqrt((x-xo(i))**2+(y-yo(i))**2)
  end do
  call interp2d(xo,yo,zo,dist,x,y,100.0,z,zerr)
  write (*,'(2(Z8.8,1X))') transfer((/z,zerr/),(/0,0/))
contains
  subroutine interp2d(xo,yo,zo,dist,x,y,distmax,z,zerr)
    implicit none
    integer :: i,imn
    integer :: indx(5)
    real :: xo(4),yo(4),zo(4),dist(4),x,y,distmax,z,zerr
    real :: side(4),ango(4),angp(4),pdist(4),ztemp(4)
    real :: pi,dx,dy,ymn,xmn,angomn,angomx,dxs,dys,angr,ds,z1,z2
    pi=abs(atan2(0.,-1.))
    do i=1,4
      dx=x-xo(i); dy=y-yo(i); angp(i)=0.
      if(dx.ne.0..or.dy.ne.0.) angp(i)=atan2(dy,dx)
    end do
    if(minval(dist).le.1.e-3) then
      do i=1,4
        if(dist(i).eq.minval(dist)) then
          z=zo(i); zerr=0.; return
        end if
      end do
    end if
    if(minval(zo).lt.-998..or.maxval(dist).gt.distmax) then
      z=-999.; zerr=0.; return
    end if
    ymn=minval(yo); xmn=1.e15; imn=1
    do i=1,4
      if(yo(i).eq.ymn.and.xo(i).lt.xmn) then
        xmn=xo(i); imn=i
      end if
    end do
    indx(1)=imn; angomn=1.e15; angomx=-1.e15
    do i=1,4
      if(i.eq.indx(1)) cycle
      ango(i)=atan2(yo(i)-yo(indx(1)),xo(i)-xo(indx(1))+1.e-30)
      if(ango(i).lt.(-pi/2.)) ango(i)=ango(i)+(2.*pi)
      if(ango(i).lt.angomn) then; angomn=ango(i); indx(2)=i; end if
      if(ango(i).gt.angomx) then; angomx=ango(i); indx(4)=i; end if
    end do
    indx(3)=10-(indx(1)+indx(2)+indx(4)); indx(5)=indx(1)
    do i=1,4
      dxs=xo(indx(i))-xo(indx(i+1)); dys=yo(indx(i))-yo(indx(i+1))
      side(i)=sqrt(dxs**2+dys**2)
      if(dys.ne.0..or.dxs.ne.0.) then; ango(i)=atan2(dys,dxs)
      else; ango(i)=0.; end if
    end do
    do i=1,4
      angr=angp(indx(i))-ango(i)
      ds=abs(dist(indx(i))*cos(angr))
      pdist(i)=abs(dist(indx(i))*sin(angr))
      ztemp(i)=(ds*zo(indx(i+1))+(side(i)-ds)*zo(indx(i)))/side(i)
      if(pdist(i).lt.1.e-3) then; z=ztemp(i); zerr=0.; return; end if
    end do
    z1=(ztemp(1)*pdist(3)+ztemp(3)*pdist(1))/(pdist(1)+pdist(3))
    z2=(ztemp(2)*pdist(4)+ztemp(4)*pdist(2))/(pdist(2)+pdist(4))
    z=(z1*(pdist(2)+pdist(4))+z2*(pdist(1)+pdist(3)))/ &
      (pdist(1)+pdist(2)+pdist(3)+pdist(4))
    zerr=abs(z1-z2)/2.
  end subroutine interp2d
end program terrel_interp2d_reference

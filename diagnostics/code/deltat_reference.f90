program deltat_reference
  implicit none
  real :: wind, height, difference, difference10, theta10, theta, z0
  real :: length, ustar, flux
  integer :: i

  do i = 1, 3
    select case (i)
    case (1)
      wind=5.; height=10.; difference=-2.; difference10=-2.; theta10=290.; theta=290.; z0=1.11803399e-4
    case (2)
      wind=5.; height=4.; difference=2.; difference10=2.3153384; theta10=292.; theta=292.; z0=1.11803399e-4
    case default
      wind=0.; height=10.; difference=0.; difference10=0.; theta10=290.; theta=290.; z0=2.e-6
    end select
    call deltat(wind,height,difference,difference10,theta10,theta,z0,length,ustar,flux)
    write(*,'(3(ES24.16,1X))') length,ustar,flux
  end do
end program

subroutine deltat(ws10,zanem,dthz,dth10,th10,theta,z02,el2,ustar2,qb)
  implicit none
  real, intent(in) :: ws10,zanem,dthz,dth10,th10,theta,z02
  real, intent(out) :: el2,ustar2,qb
  real :: vkg,cun,xln10z0,xlnzz0,z10l,ztl,psiu,psit,x,y,thstar,elold
  integer :: niter
  vkg=.4*9.81
  cun=.75e-3+.067e-3*ws10
  xln10z0=log(10./z02); xlnzz0=log(zanem/z02)
  if (abs(dth10).le.1.e-6) goto 1200
  el2=th10*cun**1.5*ws10**2/(5.096e-3*dth10)
  if (abs(el2).gt.9999.) goto 1200
  if (abs(el2).lt.5.) el2=sign(5.,el2)
  elold=el2
  do niter=1,3
    z10l=10./el2; ztl=zanem/el2
    if (el2.gt.0.) then
      psiu=-4.7*z10l; psit=-4.7*ztl
    else
      x=(1.-15.*z10l)**.25
      psiu=2.*log(.5*(1.+x))+log(.5*(1.+x*x))-2.*atan(x)+1.5707963
      y=sqrt(1.-9.*ztl); psit=1.48*log(.5*(1.+y))
    end if
    ustar2=max(.4*ws10/(xln10z0-psiu),1.e-9)
    thstar=.4*dthz/(.74*xlnzz0-psit)
    if (abs(thstar).le.1.e-6) goto 1200
    el2=theta*ustar2**2/(vkg*thstar)
    if (abs(el2).gt.9999.) goto 1200
    if (abs(el2).lt.5.) el2=sign(5.,el2)
    if (abs((el2-elold)/elold).lt..05) goto 400
    elold=el2
  end do
  goto 400
1200 continue
  ustar2=max(.4*ws10/log(10./z02),1.e-9)
  el2=9.9e9; thstar=0.
400 continue
  qb=-1195.2*theta*ustar2**3/(vkg*el2)
end subroutine

program elustr_daytime_reference
  implicit none
  real :: u,v,z0,zm,rho,temp,qh,ws,xlnzz0,xl,xlold,xustar
  real :: psizl,psiz0l,ratio
  real(kind=8) :: r8
  integer :: niter
  u=-1.012058973312378
  v=-2.737722635269165
  z0=.25
  zm=10.
  rho=1.235607385635376
  temp=285.5619201660156
  qh=25.949554443359375
  ws=sqrt(u**2+v**2)
  xlnzz0=alog(zm/z0)
  xl=-9.9e9
  xustar=max(.4*ws/xlnzz0,.05)
  xlold=xl
  do niter=1,5
    xl=-253.8226*rho*temp*xustar**3/qh
    r8=zm/xl
    psizl=psiud(r8)
    r8=z0/xl
    psiz0l=psiud(r8)
    xustar=max(.4*ws/(xlnzz0-psizl+psiz0l),.05)
    ratio=abs((xl-xlold)/xl)
    print '(i1,3(1x,z8.8))',niter,transfer(xl,0),transfer(xustar,0),transfer(ratio,0)
    if(ratio.lt..005)exit
    xlold=xl
  enddo
contains
  real(kind=8) function psiud(zl)
    implicit none
    real(kind=8),intent(in) :: zl
    real(kind=8) :: x,y,psik,psic,f,c
    if(zl.lt.0)then
      x=(1-16.*zl)**.25
      psik=2.*log((1.+x)/2.)+log((1.+x*x)/2.)-2.*atan(x)+2.*atan(1.)
      y=(1.-10.15*zl)**.3333
      psic=1.5*log((1.+y+y*y)/3.)-sqrt(3.)*atan((1.+2.*y)/sqrt(3.))+4.*atan(1.)/sqrt(3.)
      f=zl*zl/(1.+zl*zl)
      psiud=(1.-f)*psik+f*psic
    else
      c=min(50.d0,.35d0*zl)
      psiud=-((1.+zl)+.6667*(zl-14.28)/exp(c)+8.525)
    endif
  end function psiud
end program elustr_daytime_reference

program elustr_stable_reference
  implicit none
  real :: ws,z0,zm,tsf,rho,cloud,xlnzz0,theta1,theta2,thetas
  real :: cdn,u02,cu2,ustar,qh,el,xcrit,toler
  integer :: jcc,k
  ws=20.0
  z0=0.25
  zm=10.0
  tsf=290.0
  rho=1.2
  cloud=0.0
  xcrit=0.05
  xlnzz0=alog(zm/z0)
  ws=max(ws,0.5)
  jcc=10.0*cloud+0.5
  theta1=0.09*(1.0-5.e-3*jcc**2)
  cdn=0.4/xlnzz0
  theta2=tsf*cdn*ws**2/(184.428*zm)
  thetas=max(min(theta1,theta2),1.e-9)
  u02=46.107*zm*thetas/tsf
  cu2=max(0.0,ws*ws-4.0*u02/cdn)
  ustar=max(0.5*cdn*(ws+sqrt(cu2)),0.05)
  if(ustar*thetas.gt.xcrit)then
    toler=1.05*xcrit
    do k=1,3
      thetas=xcrit/ustar
      u02=46.107*zm*thetas/tsf
      cu2=max(0.0,ws*ws-4.0*u02/cdn)
      ustar=0.5*cdn*(ws+sqrt(cu2))
      if(ustar*thetas.le.toler)exit
    enddo
  endif
  qh=-996.0*rho*ustar*thetas
  el=-253.8226*rho*tsf*ustar**3/qh
  print '(3(z8.8,1x))',transfer(ustar,0),transfer(qh,0),transfer(el,0)
end program elustr_stable_reference

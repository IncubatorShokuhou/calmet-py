program elustr2_surface_reference
  implicit none
  integer,parameter :: n=4
  integer :: i,ifcn(n)
  real :: u(n),v(n),z0(n),za(n),rho(n),temp(n),cloud(n),qh(n),ustar(n),ell(n)
  u=(/-3.,4.,-.2,7./); v=(/4.,-3.,.1,1./); z0=(/.1,.25,.05,.4/)
  za=(/10.,12.,8.,15./); rho=(/1.2,1.1,1.25,1.0/); temp=(/295.,300.,280.,305./)
  cloud=(/.2,.7,.9,.1/); qh=(/120.,80.,-.1,-.1/); ifcn=(/0,1,0,1/)
  do i=1,n
    call state(u(i),v(i),z0(i),za(i),10.,rho(i),temp(i),cloud(i),qh(i),ifcn(i),ustar(i),ell(i))
    write(*,'(3(Z8.8,1X))') transfer(ustar(i),0),transfer(ell(i),0),transfer(qh(i),0)
  enddo
contains
  double precision function psiud(zl)
    double precision zl,x,y,psik,psic,f,c
    if(zl.lt.0) then
      x=(1-16.*zl)**.25; psik=2.*log((1.+x)/2.)+log((1.+x*x)/2.)-2.*atan(x)+2.*atan(1.)
      y=(1.-10.15*zl)**.3333
      psic=1.5*log((1.+y+y*y)/3.)-sqrt(3.)*atan((1.+2.*y)/sqrt(3.))+4.*atan(1.)/sqrt(3.)
      f=zl*zl/(1.+zl*zl); psiud=(1.-f)*psik+f*psic
    else
      c=min(50.d0,.35d0*zl); psiud=-((1.+zl)+.6667*(zl-14.28)/exp(c)+8.525)
    endif
  end function
  real function psiuc(z,l)
    real z,l,x
    if(l.gt.0.) then; psiuc=-17.*(1.-exp(-.29*z/l)); else
      x=(1.-15.*z/l)**.25; psiuc=2.*log(.5*(1.+x))+log(.5*(1.+x*x))-2.*atan(x)+1.5707963
    endif
  end function
  subroutine state(u,v,z0,za,zmid,rho,temp,cloud,qh,ifcn,ustar,ell)
    real u,v,z0,za,zmid,rho,temp,cloud,qh,ustar,ell,ws,xlog,trial,old,p1,p0
    real theta1,theta2,theta,cdn,u02,cu2,toler
    integer ifcn,jcc,it
    ws=sqrt(u*u+v*v); xlog=log(zmid/z0)
    if(qh.ge.0.) then
      ell=-9.9e9; trial=max(.4*ws/xlog,.05)
      if(qh.ne.0.) then
        old=ell
        do it=1,5
          ell=-253.8226*rho*temp*trial**3/qh
          if(ifcn.eq.0) then; p1=psiud(dble(za/ell)); p0=psiud(dble(z0/ell)); else; p1=psiuc(za,ell); p0=psiuc(z0,ell); endif
          trial=max(.4*ws/(xlog-p1+p0),.05)
          if(abs((ell-old)/ell).lt..005) exit
          old=ell
        enddo
      endif
      ustar=trial
    else
      jcc=10.*cloud+.5; theta1=.09*(1.-5.e-3*jcc**2); cdn=.4/xlog
      theta2=temp*cdn*ws**2/(184.428*zmid); theta=max(min(theta1,theta2),1.e-9)
      u02=46.107*zmid*theta/temp; cu2=max(0.,ws*ws-4.*u02/cdn)
      ustar=max(.5*cdn*(ws+sqrt(cu2)),.05)
      if(ustar*theta.gt..05) then
        toler=1.05*.05
        do it=1,3
          theta=.05/ustar; u02=46.107*zmid*theta/temp; cu2=max(0.,ws*ws-4.*u02/cdn)
          ustar=.5*cdn*(ws+sqrt(cu2)); if(ustar*theta.le.toler) exit
        enddo
      endif
      qh=-996.*rho*ustar*theta
      ell=-253.8226*rho*temp*ustar**3/qh
    endif
  end subroutine
end program

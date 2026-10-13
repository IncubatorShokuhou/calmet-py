program similt_reference
  implicit none
  integer, parameter :: ncase=4,nz=3
  integer :: n,k
  real :: u(ncase),v(ncase),zanem(ncase),el(ncase),z0(ncase),zi(ncase),lat(ncase)
  real :: zmid(nz),zface(nz+1),uo(nz,ncase),vo(nz,ncase)
  integer :: ipsi(ncase)
  zmid=(/10.,40.,100./); zface=(/0.,20.,60.,140./)
  u=(/-3.,4.,-5.,2.5/); v=(/4.,-3.,2.,-6./)
  zanem=(/10.,12.,8.,15./); el=(/10000.,-45.,75.,-12./)
  z0=(/.1,.25,.05,.4/); zi=(/500.,90.,45.,130./)
  lat=(/30.,30.,-25.,-25./); ipsi=(/0,0,0,1/)
  uo=9999.; vo=9999.
  do n=1,ncase
    uo(1,n)=u(n); vo(1,n)=v(n)
    call profile(zanem(n),el(n),z0(n),zi(n),lat(n),ipsi(n),zmid,zface,nz,uo(:,n),vo(:,n))
  enddo
  do n=1,ncase
    do k=1,nz
      write(*,'(2(Z8.8,1X))') transfer(uo(k,n),0),transfer(vo(k,n),0)
    enddo
  enddo
contains
  double precision function psiud(zl)
    double precision zl,x,y,psik,psic,f,c
    if(zl.lt.0) then
      x=(1-16.*zl)**.25
      psik=2.*log((1.+x)/2.)+log((1.+x*x)/2.)-2.*atan(x)+2.*atan(1.)
      y=(1.-10.15*zl)**.3333
      psic=1.5*log((1.+y+y*y)/3.)-sqrt(3.)*atan((1.+2.*y)/sqrt(3.))+4.*atan(1.)/sqrt(3.)
      f=zl*zl/(1.+zl*zl); psiud=(1.-f)*psik+f*psic
    else
      c=min(50.d0,.35d0*zl); psiud=-((1.+zl)+.6667*(zl-14.28)/exp(c)+8.525)
    endif
  end function
  real function psiuc(z,l,ifit)
    real z,l,x,rfit
    integer ifit
    rfit=real(ifit)
    if(l.gt.0.) then
      psiuc=-17.*(1.-exp(-.29*z/l))
    else
      x=(1.-rfit*z/l)**.25
      psiuc=2.*log(.5*(1.+x))+log(.5*(1.+x*x))-2.*atan(x)+1.5707963
    endif
  end function
  subroutine profile(za,l,zrough,zmix,stnlat,ifcn,zc,zf,nlev,uu,vv)
    integer nlev,ifcn,k,i
    real za,l,zrough,zmix,stnlat,zc(nlev),zf(nlev+1),uu(nlev),vv(nlev)
    real eli,elinv,ws1,angle,wd1,xln1,p1,dh,dzan,xln,p,ws,dwd,wd,rad
    real, parameter :: inv(9)=(/-.03333333,-.01,-.0027027,0.,.002857,.007692,.0166667,.05,.1111111/)
    real, parameter :: turn(9)=(/12.,10.,9.,12.,18.,28.,35.,38.,39./)
    eli=l
    if(abs(eli).lt.5.*zrough) eli=sign(5.*zrough,eli)
    if(abs(eli).ge.10000.) then; elinv=0.; else; elinv=1./eli; endif
    ws1=sqrt(uu(1)**2+vv(1)**2); angle=270.-atan2(vv(1),uu(1))*57.2957795
    wd1=amod(angle,360.); if(wd1.eq.0.) wd1=360.
    xln1=log(za/zrough)
    if(abs(elinv).lt..00005) then; p1=0.; else if(ifcn.eq.0) then; p1=psiud(dble(za/eli)); else; p1=psiuc(za,eli,16); endif
    if(abs(elinv).lt..00005) then
      dh=turn(4)
    else
      dh=turn(9)
      do i=1,9
        if(inv(i).ge.elinv) then
          if(i.eq.1) then; dh=turn(1); else; dh=turn(i-1)+(turn(i)-turn(i-1))*(elinv-inv(i-1))/(inv(i)-inv(i-1)); endif
          exit
        endif
      enddo
    endif
    dzan=dh*1.58*(1.-exp(-za/200.))
    do k=1,nlev
      if(zf(k+1).gt.max(zmix+.001,50.)) exit
      xln=log(zc(k)/zrough)
      if(abs(elinv).lt..00005) then; p=0.; else if(ifcn.eq.0) then; p=psiud(dble(zc(k)/eli)); else; p=psiuc(zc(k),eli,16); endif
      ws=ws1*(xln-p)/(xln1-p1); dwd=dh*1.58*(1.-exp(-zc(k)/200.))
      if(stnlat.gt.0.) then; wd=wd1+dwd-dzan; if(wd.gt.360.) wd=wd-360.; else; wd=wd1-dwd+dzan; if(wd.le.0.) wd=360.+wd; endif
      rad=.0174533*wd; uu(k)=-ws*sin(rad); vv(k)=-ws*cos(rad)
    enddo
  end subroutine
end program

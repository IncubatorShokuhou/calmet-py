! Multi-timestep oracle for CALMET's stateful COARE warm-layer branch.
program coare_warm_reference
  implicit none
  integer :: step, ix, iy, jwave, nhrz, iwarm, icool
  real(kind=8) :: zu, zt, depth, wind, tair, tsea, zi, rain, humidity
  real(kind=8) :: shortwave, longwave, longitude, period, height
  real(kind=8) :: z0, ustar, tstar, qstar, qh, qe
  real(kind=4) :: shelf, coast, el

  ix=1; iy=1; jwave=0; iwarm=1; icool=0
  zu=10.d0; zt=10.d0; depth=.6d0; wind=3.d0
  tair=24.85d0; tsea=26.85d0; zi=600.d0; rain=0.d0; humidity=.8d0
  shortwave=700.d0; longwave=350.d0; longitude=120.d0
  period=-999.d0; height=-999.d0; shelf=0.; coast=0.
  do step=1,4
    nhrz=step-1
    call coare(step,ix,iy,zu,zt,depth,wind,tair,tsea,zi,rain,humidity, &
               shortwave,longwave,longitude,jwave,period,height,nhrz, &
               shelf,coast,iwarm,icool,z0,el,ustar,tstar,qstar,qh,qe)
    write(*,'(7(ES24.16,1X))') z0,el,ustar,tstar,qstar,qh,qe
  end do
end program coare_warm_reference

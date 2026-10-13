! One-point oracle for comparing the NumPy COARE port with CALMET 6.5.
program coare_reference
  implicit none
  integer :: itimstep, ix, iy, jwave, nhrz, iwarm, icool
  real(kind=8) :: zu, zt, sst_depth, wind, tair, tsea, zi, rain, humidity
  real(kind=8) :: shortwave, longwave, longitude, wave_period, wave_height
  real(kind=8) :: z0, ustar, tstar, qstar, qh, qe
  real(kind=4) :: shelf, coast, el

  itimstep=1; ix=1; iy=1; jwave=0; nhrz=12; iwarm=0; icool=0
  zu=10.d0; zt=10.d0; sst_depth=.6d0; wind=5.d0
  tair=15.d0; tsea=17.d0; zi=600.d0; rain=0.d0; humidity=.8d0
  shortwave=300.d0; longwave=350.d0; longitude=120.d0
  wave_period=-999.d0; wave_height=-999.d0; shelf=0.; coast=0.
  call coare(itimstep,ix,iy,zu,zt,sst_depth,wind,tair,tsea,zi,rain,humidity, &
             shortwave,longwave,longitude,jwave,wave_period,wave_height,nhrz, &
             shelf,coast,iwarm,icool,z0,el,ustar,tstar,qstar,qh,qe)
  write(*,'(8(ES24.16,1X))') z0,el,ustar,tstar,qstar,qh,qe, &
       -ustar*(tstar+.61d0*qstar*(tair+273.15d0))
end program coare_reference

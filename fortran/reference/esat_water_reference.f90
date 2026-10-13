! GNU Fortran oracle for CALMET ESAT and WATER's DELTAT input preparation.
program esat_water_reference
  implicit none
  integer :: i
  real :: tair(3), twater(3), rh(3), zanem(3)
  real :: tairc, es, e, w, tv, th, twaterc, ws, thsurf, dthz, dth10, th10

  tair=(/288.15,273.15,303.15/)
  twater=(/290.15,275.15,298.15/)
  rh=(/80.,100.,35./)
  zanem=(/10.,4.,17.5/)
  do i=1,3
     tairc=tair(i)-273.15
     es=esat(tairc)
     e=.01*rh(i)*es
     w=.622*e/(1000.-e)
     tv=tair(i)*(1.+.61*w)
     th=tv+.01*zanem(i)
     twaterc=twater(i)-273.15
     es=esat(twaterc)
     ws=.622*es/(1000.-es)
     thsurf=twater(i)*(1.+.61*ws)
     dthz=th-thsurf
     dth10=dthz*11.512925/(alog(zanem(i))+9.2103404)
     th10=thsurf+dth10
     write(*,'(5(Z8.8,1X))') transfer(es,0),transfer(th,0), &
          transfer(dthz,0),transfer(dth10,0),transfer(th10,0)
  enddo
contains
  real function esat(tdegc)
    real, intent(in) :: tdegc
    real :: a0,a1,a2,a3,a4,a5,a6
    data a0/6.107799961/,a1/4.436518521e-1/,a2/1.428945805e-2/
    data a3/2.650648471e-4/,a4/3.031240396e-6/,a5/2.034080948e-8/
    data a6/6.136820929e-11/
    esat=a0+tdegc*(a1+tdegc*(a2+tdegc*(a3+tdegc*(a4+tdegc* &
         (a5+tdegc*a6)))))
  end function esat
end program esat_water_reference

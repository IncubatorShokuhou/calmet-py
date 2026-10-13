program mixht_real_intrinsics_reference
  implicit none
  real :: gamma, tave, fcori, ustar, el, bvf, cmech, hmech
  real :: zi1, zi2, radicand, jump_argument, convective

  gamma = 0.00137
  tave = 287.35
  fcori = 8.7e-5
  ustar = 0.17776217
  el = 81.25
  radicand = 456789.125
  jump_argument = 0.0234567

  bvf = (9.81*gamma/tave)**0.25
  cmech = 1.41/sqrt(fcori)
  hmech = cmech*ustar/bvf
  zi1 = 2400.*ustar**1.5
  zi2 = 0.4*sqrt(ustar*el/fcori)
  convective = sqrt(radicand)+sqrt(jump_argument)/gamma

  write(*,'(6(z8.8,1x))') transfer(bvf,0), transfer(cmech,0), &
       transfer(hmech,0), transfer(zi1,0), transfer(zi2,0), &
       transfer(convective,0)
end program mixht_real_intrinsics_reference

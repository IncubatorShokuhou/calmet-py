program water_humidity_reference
  implicit none
  real :: tair, rh, pres, tairc, es, e, w, q

  tair = 288.15
  rh = 80.0
  pres = 1000.0
  tairc = tair - 273.15
  es = 6.107799961 + tairc * (0.4436518521 + tairc * &
       (0.01428945805 + tairc * (0.0002650648471 + tairc * &
       (0.000003031240396 + tairc * (0.00000002034080948 + &
       tairc * 0.00000000006136820929)))))
  e = 0.01 * rh * es
  w = 0.622 * e / (pres - e)
  q = w / (1.0 + w) * 1000.0
  write(*,'(Z8.8,1X,G15.8)') transfer(q, 0), q
end program water_humidity_reference

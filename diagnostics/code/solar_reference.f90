program solar_reference
  implicit none
  integer :: i
  integer, dimension(5) :: lat_bits = [ &
       int(z'41d945d7'), int(z'41dadb1f'), int(z'41e1f61c'), &
       int(z'41e2b377'), int(z'41e3f0c7') ]
  integer, dimension(5) :: lon_bits = [ &
       int(z'42efcf31'), int(z'42f2c293'), int(z'42f48490'), &
       int(z'42f513f0'), int(z'42efad53') ]
  real :: lat, lon, d, radd, sd, cd, rad2d, s2, c2
  real :: em, sigma, sincd, capd, coscd, radlat, sinlat, coslat
  real :: solha, sinalp, qsw

  d = (77.0 - 1.0) * 0.9856479
  radd = 0.0174533 * d
  sd = sin(radd)
  cd = cos(radd)
  rad2d = 2.0 * radd
  s2 = sin(rad2d)
  c2 = cos(rad2d)
  em = 12.0 + 0.12357*sd - 0.004289*cd + 0.153809*s2 + 0.060783*c2
  sigma = 279.9348 + d + 1.914827*sd - 0.079525*cd + &
       0.019938*s2 - 0.00162*c2
  sincd = 0.39784989 * sin(0.0174533*sigma)
  capd = asin(sincd)
  coscd = cos(capd)

  do i = 1, size(lat_bits)
     lat = transfer(lat_bits(i),lat)
     lon = transfer(lon_bits(i),lon)
     radlat = 0.0174533*lat
     sinlat = sin(radlat)
     coslat = cos(radlat)
     solha = 15.0*(0.5-em) + lon
     sinalp = sinlat*sincd + coslat*coscd*cos(0.0174533*solha)
     qsw = 990.0*sinalp - 30.0
     write(*,'(4(z8.8,1x))') lat_bits(i),lon_bits(i), &
          transfer(sinalp,0),transfer(qsw,0)
  end do
end program solar_reference

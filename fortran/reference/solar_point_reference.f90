program solar_point_reference
  implicit none
  real :: d,radd,xsind,xcosd,rad2d,sin2d,cos2d,em,sigma
  real :: sincd,capd,coscd,radlat,sinlat,coslat,gmt,solha,sinalp
  d=(77.-1.)*.9856479
  radd=.0174533*d
  xsind=sin(radd)
  xcosd=cos(radd)
  rad2d=2.*radd
  sin2d=sin(rad2d)
  cos2d=cos(rad2d)
  em=12.+.12357*xsind-.004289*xcosd+.153809*sin2d+.060783*cos2d
  sigma=279.9348+d+1.914827*xsind-.079525*xcosd+.019938*sin2d-.00162*cos2d
  sincd=.39784989*sin(.0174533*sigma)
  capd=asin(sincd)
  coscd=cos(capd)
  radlat=.0174533*31.11811065673828
  sinlat=sin(radlat)
  coslat=cos(radlat)
  gmt=float(2-1)-.5
  solha=15.*(gmt-em)+117.81071472167969
  sinalp=sinlat*sincd+coslat*coscd*cos(.0174533*solha)
  print '(9(z8.8,1x))',transfer(em,0),transfer(sigma,0),transfer(sincd,0), &
    transfer(coscd,0),transfer(sinlat,0),transfer(coslat,0), &
    transfer(solha,0),transfer(sinalp,0), &
    transfer((990.*sinalp-30.)*(1.-.75*1.**3.4),0)
end program solar_point_reference

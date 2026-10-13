program similt_neutral_reference
  implicit none
  integer :: hemisphere, k
  real :: speed, direction, anemometer, roughness, factor
  real :: angle, u, v, ws1, wd1, xlnz1z0
  real :: dh200, d1, d2, ztable, dzan, xlnzz0
  real :: ws, dwd, wd, wdrad, latitude
  real, dimension(3) :: level, output_u, output_v

  speed = 5.718242
  direction = 217.3
  anemometer = 13.7
  roughness = 0.073
  level = (/ 8.25, 45.5, 200.0 /)

  angle = 0.0174533 * direction
  u = -speed * sin(angle)
  v = -speed * cos(angle)

  factor = 57.2957795
  d1 = 1.58
  d2 = 1.0
  ztable = 200.0
  dh200 = 12.0

  do hemisphere = 1, 2
    if (hemisphere == 1) then
      latitude = 30.0
    else
      latitude = -30.0
    end if

    ws1 = sqrt(u**2 + v**2)
    angle = 270.0 - atan2(v, u) * factor
    wd1 = amod(angle, 360.0)
    if (wd1 == 0.0) wd1 = 360.0
    xlnz1z0 = alog(anemometer / roughness)
    dzan = dh200 * d1 * (1.0 - exp(-d2 * anemometer / ztable))

    do k = 1, 3
      xlnzz0 = alog(level(k) / roughness)
      ws = ws1 * xlnzz0 / xlnz1z0
      dwd = dh200 * d1 * (1.0 - exp(-d2 * level(k) / ztable))
      if (latitude > 0.0) then
        wd = wd1 + dwd - dzan
        if (wd > 360.0) wd = wd - 360.0
      else
        wd = wd1 - dwd + dzan
        if (wd <= 0.0) wd = 360.0 + wd
      end if
      wdrad = 0.0174533 * wd
      output_u(k) = -ws * sin(wdrad)
      output_v(k) = -ws * cos(wdrad)
    end do

    write (*,'(6(Z8.8,1X))') transfer(output_u, (/ 0, 0, 0 /)), &
      transfer(output_v, (/ 0, 0, 0 /))
  end do
end program similt_neutral_reference

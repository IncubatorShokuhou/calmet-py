program diagno_component_profiles_reference
  implicit none
  integer :: k
  real :: u, v, anemometer, roughness, coefficient, exponent
  real :: factor, speed, angle, wd1, denominator, magnitude
  real :: dh_d1, at_anem, at_level, direction, radians
  real, dimension(3) :: level, output_u, output_v, multiplier

  u = -3.123456
  v = 4.765432
  anemometer = 13.7
  roughness = 0.073
  level = (/ 8.25, 45.5, 200.0 /)

  output_u = 9999.0
  output_v = 9999.0
  coefficient = (alog(level(1)) - alog(roughness)) / &
    (alog(anemometer) - alog(roughness))
  output_u(1) = u * coefficient
  output_v(1) = v * coefficient
  call show(output_u, output_v)

  exponent = 0.143
  output_u = 9999.0
  output_v = 9999.0
  output_u(2) = u * (level(2) / anemometer)**exponent
  output_v(2) = v * (level(2) / anemometer)**exponent
  do k = 3, 3
    output_u(k) = output_u(k-1) * (level(k) / level(k-1))**exponent
    output_v(k) = output_v(k-1) * (level(k) / level(k-1))**exponent
  end do
  coefficient = (anemometer / level(1))**exponent
  output_u(1) = u / coefficient
  output_v(1) = v / coefficient
  call show(output_u, output_v)

  multiplier = (/ 0.3333, 1.2345, -0.75 /)
  do k = 3, 1, -1
    output_u(k) = u * multiplier(k)
    output_v(k) = v * multiplier(k)
  end do
  call show(output_u, output_v)

  speed = sqrt(u**2 + v**2)
  factor = 57.2957795
  angle = 270.0 - atan2(v, u) * factor
  wd1 = amod(angle, 360.0)
  if (wd1 == 0.0) wd1 = 360.0
  denominator = alog(anemometer / roughness)
  dh_d1 = 12.0 * 1.58
  at_anem = dh_d1 * (1.0 - exp(-anemometer / 200.0))
  do k = 1, 3
    magnitude = speed * alog(level(k) / roughness) / denominator
    at_level = dh_d1 * (1.0 - exp(-level(k) / 200.0))
    direction = wd1 + at_level - at_anem
    if (direction > 360.0) direction = direction - 360.0
    radians = 0.0174533 * direction
    output_u(k) = -magnitude * sin(radians)
    output_v(k) = -magnitude * cos(radians)
  end do
  call show(output_u, output_v)

contains
  subroutine show(first, second)
    real, dimension(3), intent(in) :: first, second
    write (*,'(6(Z8.8,1X))') transfer(first, (/ 0, 0, 0 /)), &
      transfer(second, (/ 0, 0, 0 /))
  end subroutine show
end program diagno_component_profiles_reference

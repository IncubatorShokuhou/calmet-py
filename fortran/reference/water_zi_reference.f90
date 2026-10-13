program water_zi_reference
  implicit none
  integer :: i, j, k
  real :: xkm(3), ykm(3), ziow(3), xowsta(3), yowsta(3)
  real :: xor2, yor2, dgrid, xgdpt, ygdpt, r2, xsum, ysum, zi

  xor2 = 100.0
  yor2 = 0.0
  dgrid = 700.0
  xkm = (/100.13, 101.02, 100.61/)
  ykm = (/0.21, 0.48, -0.17/)
  ziow = (/123.4567, 987.6543, 345.6789/)
  do k = 1, 3
    xowsta(k) = 1000.0 * (xkm(k) - xor2)
    yowsta(k) = 1000.0 * (ykm(k) - yor2)
  end do

  j = 1
  do i = 1, 2
    xsum = 0.0
    ysum = 0.0
    xgdpt = (float(i) - 0.5) * dgrid
    ygdpt = (float(j) - 0.5) * dgrid
    do k = 1, 3
      r2 = (xowsta(k) - xgdpt)**2 + (yowsta(k) - ygdpt)**2
      r2 = amax1(r2, 1.e-30)
      xsum = xsum + ziow(k) / r2
      ysum = ysum + 1.0 / r2
    end do
    zi = xsum / ysum
    write(*,'(Z8.8,1X,G15.8)') transfer(zi, 0), zi
  end do
end program water_zi_reference

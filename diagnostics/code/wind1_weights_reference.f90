program wind1_weights_reference
  implicit none
  integer, parameter :: nstat=3, nsurface=2, nz=2
  integer :: l, k
  real :: dgrid, dgridi, xc, yc, xr2, wgt, sumwt
  real :: sumwsfold, sumwupold, beta, gamma, unext, vnext
  real, dimension(nstat) :: x, y, rsqwt
  real, dimension(nz) :: bias
  real, dimension(nz,nstat) :: u, v, weight
  real, dimension(nz) :: output_u, output_v

  dgrid=3700.0
  x=(/ 413.25, 8912.75, -1725.5 /)
  y=(/ 744.5, -3811.125, 6200.875 /)
  u=reshape((/ 1.2345, -2.7182, 8.125, 2.3456, -3.1415, 9.875 /),shape(u))
  v=reshape((/ -4.25, 5.5, 0.75, -5.125, 6.25, 1.5 /),shape(v))
  bias=(/ 0.23, -0.37 /)
  xc=1.0-0.5
  yc=1.0-0.5
  dgridi=1.0/dgrid
  sumwt=0.0
  do l=1,nstat
    xr2=(x(l)*dgridi-xc)**2+(y(l)*dgridi-yc)**2
    if (xr2 < 1.0) then
      wgt=1.0
    else
      wgt=1.0/xr2
    end if
    sumwt=sumwt+wgt
    rsqwt(l)=wgt
  end do
  sumwt=1.0/sumwt
  do l=1,nstat
    rsqwt(l)=rsqwt(l)*sumwt
  end do
  sumwsfold=0.0
  sumwupold=0.0
  do l=1,nsurface
    sumwsfold=sumwsfold+rsqwt(l)
  end do
  do l=nsurface+1,nstat
    sumwupold=sumwupold+rsqwt(l)
  end do
  do k=1,nz
    if (bias(k) >= 0.0) then
      beta=1.0-bias(k)
      gamma=(bias(k)+(1.0-bias(k))*sumwupold)/sumwupold
    else
      beta=(-bias(k)+(1.0+bias(k))*sumwsfold)/sumwsfold
      gamma=1.0+bias(k)
    end if
    do l=1,nsurface
      weight(k,l)=beta*rsqwt(l)
    end do
    do l=nsurface+1,nstat
      weight(k,l)=gamma*rsqwt(l)
    end do
    unext=0.0
    vnext=0.0
    do l=1,nstat
      unext=unext+weight(k,l)*u(k,l)
      vnext=vnext+weight(k,l)*v(k,l)
    end do
    output_u(k)=unext
    output_v(k)=vnext
  end do
  write (*,'(4(Z8.8,1X))') transfer(output_u,(/0,0/)),transfer(output_v,(/0,0/))

  call removal_case

contains
  subroutine removal_case
    integer, parameter :: count=5, surface_count=2
    integer :: station, other
    real :: inverse_grid, xcell, ycell, distance2, total, value
    real :: surface_total, upper_total, local_beta, local_gamma, removed
    real, dimension(count) :: sx, sy, base, weights, component_u, component_v

    sx=(/ 413.25, 8912.75, -1725.5, 4550.25, 12020.5 /)
    sy=(/ 744.5, -3811.125, 6200.875, 1510.75, -7200.25 /)
    component_u=(/ 0.0, 2.25, -3.5, 999.0, 7.75 /)
    component_v=(/ 0.0, -1.5, 4.25, 999.0, -2.75 /)
    inverse_grid=1.0/3700.0
    xcell=0.5
    ycell=0.5
    total=0.0
    do station=1,count
      distance2=(sx(station)*inverse_grid-xcell)**2+ &
        (sy(station)*inverse_grid-ycell)**2
      if (distance2 < 1.0) then
        base(station)=1.0
      else
        base(station)=1.0/distance2
      end if
      total=total+base(station)
    end do
    total=1.0/total
    do station=1,count
      base(station)=base(station)*total
    end do
    surface_total=0.0
    upper_total=0.0
    do station=1,surface_count
      surface_total=surface_total+base(station)
    end do
    do station=surface_count+1,count
      upper_total=upper_total+base(station)
    end do
    value=0.17
    local_beta=1.0-value
    local_gamma=(value+(1.0-value)*upper_total)/upper_total
    do station=1,surface_count
      weights(station)=local_beta*base(station)
    end do
    do station=surface_count+1,count
      weights(station)=local_gamma*base(station)
    end do

    ! ICALM=0 removes calm surface station 1 first.
    removed=weights(1)
    do other=1,count
      if (other /= 1) weights(other)=weights(other)/(1.0-removed)
    end do
    weights(1)=0.0
    ! The fourth (upper-air) station is then removed as missing.
    removed=weights(4)
    do other=1,count
      if (other /= 4) weights(other)=weights(other)/(1.0-removed)
    end do
    weights(4)=0.0

    output_u(1)=0.0
    output_v(1)=0.0
    do station=1,count
      output_u(1)=output_u(1)+weights(station)*component_u(station)
      output_v(1)=output_v(1)+weights(station)*component_v(station)
    end do
    write (*,'(2(Z8.8,1X))') transfer(output_u(1),0),transfer(output_v(1),0)
  end subroutine removal_case
end program wind1_weights_reference

program coord_transform_reference
  implicit none
  character(len=8) :: input_map, output_map, ll_map, input_datum, output_datum
  character(len=12) :: action, laza_to_ll, ll_to_ll, ll_to_lcc
  character(len=4) :: hemisphere
  integer :: zone
  real :: x, y, output_x, output_y, direct_x, direct_y
  real :: longitude, latitude, shifted_lon, shifted_lat
  real(8) :: input_vector(9), output_vector(9), laza_in(9), ll_out(9)
  real(8) :: ll_in(9), shifted_out(9), projected_in(9), lcc_out(9)

  input_map = 'LAZA'
  output_map = 'LCC'
  ll_map = 'LL'
  input_datum = 'ESR-S'
  output_datum = 'WGS-84'
  call globe1(input_map, 0, 1.0, 0.0, 0.0, 45.0, 100.0, 0.0, 0.0, &
              output_map, 0, 1.0, 30.0, 60.0, 27.0, 117.5, 0.0, 0.0, &
              action, input_vector, output_vector)
  call globe1(input_map, 0, 1.0, 0.0, 0.0, 45.0, 100.0, 0.0, 0.0, &
              ll_map, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              laza_to_ll, laza_in, ll_out)
  call globe1(ll_map, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              ll_map, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              ll_to_ll, ll_in, shifted_out)
  call globe1(ll_map, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              output_map, 0, 1.0, 30.0, 60.0, 27.0, 117.5, 0.0, 0.0, &
              ll_to_lcc, projected_in, lcc_out)
  do
    read (*, *, end=100) x, y
    call globe(6, action, input_datum, input_vector, output_datum, output_vector, &
               x, y, direct_x, direct_y, zone, hemisphere)
    call globe(6, laza_to_ll, input_datum, laza_in, input_datum, ll_out, &
               x, y, longitude, latitude, zone, hemisphere)
    call globe(6, ll_to_ll, input_datum, ll_in, output_datum, shifted_out, &
               longitude, latitude, shifted_lon, shifted_lat, zone, hemisphere)
    call globe(6, ll_to_lcc, output_datum, projected_in, output_datum, lcc_out, &
               shifted_lon, shifted_lat, output_x, output_y, zone, hemisphere)
    write (*, '(10f20.10)') x, y, longitude, latitude, shifted_lon, shifted_lat, &
                            direct_x, direct_y, output_x, output_y
  end do
100 continue
end program coord_transform_reference

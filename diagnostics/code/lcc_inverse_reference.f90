program lcc_inverse_reference
  implicit none
  character(len=8) :: lcc,ll,datum
  character(len=12) :: action
  character(len=4) :: hemisphere
  real(kind=8) :: input_vector(9),output_vector(9)
  real :: x,y,longitude,latitude
  integer :: zone
  lcc='LCC'
  ll='LL'
  datum='WGS-84'
  call globe1(lcc,50,1.0,30.0,60.0,27.0,117.5,0.0,0.0, &
              ll,0,1.0,0.0,0.0,0.0,0.0,0.0,0.0, &
              action,input_vector,output_vector)
  do
    read(*,*,end=100)x,y
    call globe(6,action,datum,input_vector,datum,output_vector, &
               x,y,longitude,latitude,zone,hemisphere)
    write(*,'(2(f20.10,1x),2(z8.8,1x))')longitude,latitude, &
      transfer(longitude,0),transfer(latitude,0)
  enddo
100 continue
end program lcc_inverse_reference

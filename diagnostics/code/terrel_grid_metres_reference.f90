program terrel_grid_metres_reference
  implicit none
  integer,parameter :: nx=3,ny=2
  real :: xllk,yllk,sizekx,sizeky,xurk,yurk,xllm,xurm,yllm,yurm
  real :: sizem,dels
  xllk=10.1234567
  yllk=-20.7654321
  sizekx=18.790188546553008
  sizeky=7.345678
  xurk=xllk+float(nx)*sizekx
  yurk=yllk+float(ny)*sizeky
  xllm=xllk*1000.
  xurm=xurk*1000.
  yllm=yllk*1000.
  yurm=yurk*1000.
  sizem=sizekx*1000.
  dels=1234.567/sizem
  write (*,'(5(Z8.8,1X))') transfer(xllm,0),transfer(xurm,0), &
    transfer(yllm,0),transfer(yurm,0),transfer(dels,0)
end program terrel_grid_metres_reference

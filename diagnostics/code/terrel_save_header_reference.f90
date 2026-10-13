program terrel_save_header_reference
  implicit none
  integer,parameter :: nx=3,ny=2,nxyrec=7,izone=50
  integer :: ihdr(7)
  real :: xllk,yllk,sizek,xllm,yllm,sizem
  xllk=97.57499973130791
  yllk=-66.642994236278
  sizek=18.790188546553008
  xllm=xllk*1000.
  yllm=yllk*1000.
  sizem=sizek*1000.
  ihdr(1)=int(xllm)
  ihdr(2)=int(yllm)
  ihdr(3)=izone
  ihdr(4)=int(sizem)
  ihdr(5)=nx
  ihdr(6)=ny
  ihdr(7)=nxyrec
  write (*,'(7(I0,1X))') ihdr
end program terrel_save_header_reference

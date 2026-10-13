program elustr_speed_reference
  implicit none
  real :: u,v,ws,z0,zm,xlnzz0,xustar
  u=8.861122
  v=-9.195294
  z0=.25
  zm=10.
  ws=sqrt(u**2+v**2)
  ws=amax1(ws,.001)
  xlnzz0=alog(zm/z0)
  xustar=amax1(.4*ws/xlnzz0,.05)
  write (*,'(2(Z8.8,1X))') transfer((/ws,xustar/),(/0,0/))
end program elustr_speed_reference

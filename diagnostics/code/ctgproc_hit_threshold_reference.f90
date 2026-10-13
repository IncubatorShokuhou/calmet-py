program ctgproc_hit_threshold_reference
  implicit none
  integer :: iavg,ithres,ifract
  iavg=16777299
  ithres=1
  ifract=max(1,int(real(iavg*ithres)/100.))
  write (*,'(I0,1X,Z8.8)') ifract,transfer(real(iavg*ithres)/100.,0)
end program ctgproc_hit_threshold_reference

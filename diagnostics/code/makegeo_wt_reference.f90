program makegeo_wt_reference
  implicit none
  integer :: k
  real :: fraction(4),roughness(4),xref(4),xnum,xden,result,probe_log,probe_exp
  fraction=(/ 0.12345, 0.23456, 0.34567, 0.29632 /)
  roughness=(/ 0.1234567, 1.234567, 0.0034567, 3.45678 /)
  do k=1,4
    xref(k)=alog(roughness(k))
  end do
  xnum=0.0
  xden=0.0
  do k=1,4
    xnum=xnum+fraction(k)*xref(k)
    xden=xden+fraction(k)
  end do
  result=xnum/xden
  result=exp(result)
  probe_log=alog(0.8083694577217102)
  probe_exp=exp(0.047530174255371094)
  write (*,'(9(Z8.8,1X))') transfer(xref,(/0,0,0,0/)), &
    transfer((/xnum,xden,result,probe_log,probe_exp/),(/0,0,0,0,0/))
end program makegeo_wt_reference

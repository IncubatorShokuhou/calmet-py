program ctgproc_get4cnr_window_reference
  implicit none
  integer,parameter :: nxi=100,nyi=100
  integer :: i,ix,iy,nbegx,nendx,nbegy,nendy,nbuffx,nbuffy,nbuff
  real :: xkm(4),ykm(4),xorg,yorg,dxi,dyi
  xkm=(/10.9999999,20.2,20.2,10.9999999/)
  ykm=(/30.1,30.1,35.2,35.2/)
  xorg=0.; yorg=0.; dxi=1.; dyi=1.
  nbegx=0; nendx=0; nbegy=0; nendy=0
  do i=1,4
    ix=int((xkm(i)-xorg)/dxi)+1
    iy=int((ykm(i)-yorg)/dyi)+1
    if(nbegx.eq.0.or.nbegx.gt.ix) nbegx=ix
    if(nendx.eq.0.or.nendx.lt.ix) nendx=ix
    if(nbegy.eq.0.or.nbegy.gt.iy) nbegy=iy
    if(nendy.eq.0.or.nendy.lt.iy) nendy=iy
  enddo
  nbuffx=(nendx-nbegx)/10+1
  nbuffy=(nendy-nbegy)/10+1
  nbuff=max(nbuffx,nbuffy)
  nbegx=max(1,nbegx-nbuff); nendx=min(nxi,nendx+nbuff)
  nbegy=max(1,nbegy-nbuff); nendy=min(nyi,nendy+nbuff)
  write (*,'(4(I0,1X))') nbegx,nendx,nbegy,nendy
end program ctgproc_get4cnr_window_reference

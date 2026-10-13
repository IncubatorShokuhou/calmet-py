program ctgproc_mesh_reference
  implicit none
  integer,parameter :: mesh=12
  integer :: i
  real :: dxi,dyi,step,half,offset(mesh),center,ctg(mesh),origin,pixels(3)
  real :: xorg,yorg,xmul
  real*8 :: scale8(2),tie8(4)
  dxi=1.
  step=dxi/float(mesh)
  half=dxi/2.
  do i=1,mesh
    offset(i)=-half+(float(i)-.5)*step
  enddo
  write (*,'(12(Z8.8,1X))') transfer(offset,(/(0,i=1,mesh)/))
  dxi=.2
  center=500.1
  step=dxi/float(mesh)
  half=dxi/2.
  do i=1,mesh
    ctg(i)=(center-half)+(float(i)-.5)*step
  enddo
  write (*,'(12(Z8.8,1X))') transfer(ctg,(/(0,i=1,mesh)/))
  dxi=1./120.
  origin=-180.+dxi/2.
  pixels(1)=origin+dxi*0.
  pixels(2)=origin+dxi*12345.
  pixels(3)=origin+dxi*43199.
  write (*,'(3(Z8.8,1X))') transfer(pixels,(/0,0,0/))
  scale8=(/30.123456d0,30.654321d0/)
  tie8=(/7.d0,9.d0,500000.123d0,4000000.456d0/)
  dxi=scale8(1)
  dyi=scale8(2)
  xorg=sngl(tie8(3))
  yorg=sngl(tie8(4))
  xmul=.001
  dxi=dxi*xmul
  dyi=dyi*xmul
  xorg=xorg*xmul
  yorg=yorg*xmul
  dyi=-abs(dyi)
  xorg=xorg+dxi/2.
  yorg=yorg+dyi/2.
  xorg=xorg-dxi*7.
  yorg=yorg-dyi*9.
  write (*,'(4(Z8.8,1X))') transfer((/xorg,yorg,dxi,dyi/),(/0,0,0,0/))
end program ctgproc_mesh_reference

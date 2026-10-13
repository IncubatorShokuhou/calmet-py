! GNU Fortran oracle for RDCALMET wind-direction rotation.
program rdcalmet_rotation_reference
  implicit none
  integer :: i
  real :: u(6),v(6),dwdi(6),ws,wd,dir
  u=(/3.,-4.,0.,1.e-12,7.25,-2.75/)
  v=(/4.,3.,-5.,-1.e-12,-1.5,-8.125/)
  dwdi=(/0.,12.5,-37.25,90.,179.9,-181.2/)
  do i=1,6
     ws=sqrt(u(i)**2+v(i)**2)
     if(ws.gt.1.e-9) then
        wd=270.-57.295778*atan2(v(i),u(i))
        wd=amod(wd,360.)
        wd=wd+dwdi(i)
        dir=.0174533*wd
        u(i)=-ws*sin(dir)
        v(i)=-ws*cos(dir)
     endif
     write(*,'(2(Z8.8,1X))') transfer(u(i),0),transfer(v(i),0)
  enddo
end program rdcalmet_rotation_reference

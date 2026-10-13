! GNU Fortran oracle for RDCALMET's vertical wind interpolation.
program rdcalmet_vertical_reference
  implicit none
  integer :: i,k,kp
  real :: source(3),target(7),z0(2),field(2,3),answer,ratio
  source=(/10.,40.,100./); target=(/5.,10.,25.,40.,80.,100.,150./)
  z0=(/.1,.35/)
  field(1,:)=(/1.25,7.75,-2.5/)
  field(2,:)=(/-3.5,4.125,12.25/)
  do k=1,7
     do i=1,2
        if(target(k).lt.source(1)) then
           answer=field(i,1)
        else if(target(k).ge.source(3)) then
           answer=field(i,3)
        else
           do kp=2,3
              if(source(kp-1).le.target(k).and.source(kp).gt.target(k)) then
                 ratio=(target(k)-source(kp-1))/(source(kp)-source(kp-1))
                 answer=field(i,kp-1)+(field(i,kp)-field(i,kp-1))*ratio
                 exit
              endif
           enddo
        endif
        if(target(k).lt.source(1)) then
           ratio=(alog(target(k))-alog(z0(i)))/(alog(source(1))-alog(z0(i)))
           answer=ratio*field(i,1)
        endif
        write(*,'(Z8.8)',advance='no') transfer(answer,0)
        if(i.eq.2) write(*,*)
     enddo
  enddo
end program rdcalmet_vertical_reference

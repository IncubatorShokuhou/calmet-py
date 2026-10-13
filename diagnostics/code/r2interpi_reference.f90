! GNU Fortran oracle for RDHDMET four-neighbor selection and R2INTERPI.
program r2interpi_reference
  implicit none
  integer :: n,ii,jj,k,m,ni(4),nj(4)
  real :: x(3,2),y(3,2),z(3,2),tx(3),ty(3),dnear(4),pdist,total,weight,answer
  x(:,1)=(/0.,2.,5./); x(:,2)=(/.2,2.4,5.5/)
  y(:,1)=(/0.,.1,-.2/); y(:,2)=(/3.,3.2,2.8/)
  z(:,1)=(/1.25,-7.5,20.125/); z(:,2)=(/3.75,11.5,-2.25/)
  tx=(/.01,1.7,4.1/); ty=(/.02,2.2,.7/)
  do n=1,3
     dnear=9.9e19; ni=0; nj=0
     do ii=1,3
        do jj=1,2
           pdist=sqrt((x(ii,jj)-tx(n))**2+(y(ii,jj)-ty(n))**2)
           do k=1,4
              if(pdist.lt.dnear(k)) then
                 do m=4,k+1,-1
                    dnear(m)=dnear(m-1); ni(m)=ni(m-1); nj(m)=nj(m-1)
                 enddo
                 dnear(k)=pdist; ni(k)=ii; nj(k)=jj
                 exit
              endif
           enddo
        enddo
     enddo
     total=0.; weight=0.
     do m=1,4
        pdist=(x(ni(m),nj(m))-tx(n))**2+(y(ni(m),nj(m))-ty(n))**2
        if(pdist.lt..001) pdist=.001
        pdist=1./pdist
        total=total+z(ni(m),nj(m))*pdist
        weight=weight+pdist
     enddo
     answer=total/weight
     write(*,'(Z8.8)') transfer(answer,0)
  enddo
end program r2interpi_reference

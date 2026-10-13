program terrel_fillang_reference
  implicit none
  integer, parameter :: nx=2,ny=6
  integer :: i
  real :: ht(nx*ny),void
  void=-32000.
  ht=(/ void,void, void,20., 30.,void, 40.,void, void,50., void,void /)
  call fillang(ht,nx,ny)
  write (*,'(12(Z8.8,1X))') transfer(ht,(/(0,i=1,nx*ny)/))
contains
  subroutine fillang(ht,nx,ny)
    implicit none
    integer, intent(in) :: nx,ny
    real, intent(inout) :: ht(nx*ny)
    integer :: ix,iy,i,ilast,iyp1,inext
    real :: rlast,rnext
    logical :: lagain
    do ix=1,nx
      ilast=ix+(ny-1)*nx
      rlast=ht(ilast)
1     lagain=.false.
      do iy=1,ny
        i=ix+(iy-1)*nx
        if(ht(i).le.-9999.) then
          iyp1=iy+1
          if(iyp1.gt.ny) iyp1=1
          inext=ix+(iyp1-1)*nx
          rnext=ht(inext)
          if(rlast.le.-9999.) then
            if(rnext.le.-9999.) then
              lagain=.true.
            else
              ht(i)=rnext
            endif
          elseif(rnext.le.-9999.) then
            ht(i)=rlast
          else
            ht(i)=(rlast+rnext)/2.
          endif
        endif
        rlast=ht(i)
      enddo
      if(lagain) goto 1
    enddo
  end subroutine fillang
end program terrel_fillang_reference

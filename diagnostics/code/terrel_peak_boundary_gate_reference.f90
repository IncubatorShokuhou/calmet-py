program terrel_peak_boundary_gate_reference
  implicit none
  integer,parameter :: nx=2,ny=2
  integer :: ix,iy,ixp,iyp,knt(nx*ny)
  knt=0
  ix=2
  iy=1
  ixp=3
  iyp=2
  call save_peak_indices(ix,iy,ixp,iyp,knt,nx,ny)
  write (*,'(4(I0,1X))') knt
contains
  subroutine save_peak_indices(ix,iy,ixp,iyp,knt,nx,ny)
    implicit none
    integer,intent(in) :: ix,iy,ixp,iyp,nx,ny
    integer,intent(inout) :: knt(nx*ny)
    integer :: i
    i=ix+(iy-1)*nx
    knt(i)=knt(i)+1
    if(ixp.ne.ix.or.iyp.ne.iy) then
      if(ixp.ge.1.and.ixp.le.nx) then
        if(iyp.ge.1.and.iyp.le.ny) then
          i=ixp+(iyp-1)*nx
          knt(i)=knt(i)+1
          if(ix.ge.1.and.ix.le.nx) then
            i=ix+(iyp-1)*nx
            knt(i)=knt(i)+1
          endif
          if(iy.ge.1.and.iy.le.ny) then
            i=ixp+(iy-1)*nx
            knt(i)=knt(i)+1
          endif
        endif
      endif
    endif
  end subroutine save_peak_indices
end program terrel_peak_boundary_gate_reference

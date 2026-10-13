program calwrf_destagger_reference
  implicit none
  integer :: k
  real, dimension(5) :: ph, phb, gh
  real, dimension(4) :: height
  ph=(/ 1234.567, -4321.125, 0.125, 18765.5, -234.75 /)
  phb=(/ 91345.25, 100123.75, 54321.875, 70111.125, 88765.375 /)
  gh=(ph+phb)/9.81
  do k=1,4
    height(k)=(gh(k)+gh(k+1))/2.0
  end do
  write (*,'(4(Z8.8,1X))') transfer(height,(/0,0,0,0/))
end program calwrf_destagger_reference

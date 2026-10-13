Program main
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! edit filedate & nt !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	
		Character(len=100) ::filedate="20161120000000_20161120120000"
	Character(len=512) :: fname_in1,fname_out1,fname_out2,fname_out3,fname_out4,fname_out5
	Character(len=512) :: output_dir
	Integer :: env_status,env_length
	
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	
      parameter(mxnx=206,mxny=290,mxnz=10,nt=4*12)
	  character*16 dataset,dataver 
      character*64 datamod
	  integer::io7,io99,ncom 
	  !real ::dummy 
	  integer ::jbyr,jbmo,jbdy,jbhr,jbsec,jeyr,jemo,jedy, jehr,jesec
	  character*8 axtz
	  integer::irlg,irtype,nx, ny, nz
	  integer::counttimes
	  real::  dgrid, xorigr, yorigr 
	  integer:: iwfcod, nssta
	  
	  integer:: nusta, npsta, nowsta, nlu, iwat1, iwat2
	  logical ::lcalgrd
	  character*8 pmap,datum
	  character*12 daten
	  
	  real::  feast,fnorth
	  character*4 utmhem
	  integer::iutmzn
	  real:: rnlat0,relon0,xlat1,xlat2
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!
		  integer:: i,j,k,t,iz,z
      character*132 comment1 
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!
		  integer, parameter :: nzp1=11
	  real :: nface(nzp1)
	  character*8 clabel
	  integer	nsecb1,nsece,ndathrb1,ndathre1
	  integer ::ndathrb(nt,13),ndathre(nt,13),nsecb(nt,13)
	  !real :: data_real(mxnx,mxny)
	  !integer :: data_int(mxnx,mxny)
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  real :: z0(mxnx,mxny)
	  integer :: ilandu(mxnx,mxny)
	  real :: ilandu_real(mxnx,mxny)
	  real :: elev(mxnx,mxny)
	  real :: xlai(mxnx,mxny)
	  real :: data_uwnd_xyz(:,:,:,:),data_vwnd_xyz(:,:,:,:)
	  	  real :: data_uwnd_5l(:,:,:,:),data_vwnd_5l(:,:,:,:)
	  ALLOCATABLE data_uwnd_xyz,data_vwnd_xyz
		ALLOCATABLE data_uwnd_5l,data_vwnd_5l
	  real :: ipgt(mxnx,mxny),ustar(mxnx,mxny),zi(mxnx,mxny)
	  real :: el(mxnx,mxny),wstar(mxnx,mxny),rmm(mxnx,mxny)
	  real :: temp2d(mxnx,mxny),rho(mxnx,mxny),qsw(mxnx,mxny)
	  real :: irh2d(mxnx,mxny),ipcode2d(mxnx,mxny)
	  character(len=9) ::datejul	 
	  integer jultime(nt),second(nt)
	  	  ALLOCATE(data_uwnd_xyz(nt,mxnx,mxny,mxnz),data_vwnd_xyz(nt,mxnx,mxny,mxnz))
	      ALLOCATE(data_uwnd_5l(nt,mxnx,mxny,5),data_vwnd_5l(nt,mxnx,mxny,5))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!OUTHD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	call get_environment_variable("CALMET_OUTPUT_DIR",output_dir,length=env_length,status=env_status)
	if (env_status /= 0 .or. env_length == 0) then
	  write(*,*) "FATAL: set CALMET_OUTPUT_DIR to the directory containing CALMET output files"
	  stop 2
	endif
	output_dir = Trim(AdjustL(output_dir))
	if (len_trim(output_dir) == 0) then
	  write(*,*) "FATAL: CALMET_OUTPUT_DIR must not contain only whitespace"
	  stop 2
	endif
	if (output_dir(len_trim(output_dir):len_trim(output_dir)) /= "/") then
	  output_dir = Trim(output_dir)//"/"
	endif
	fname_in1 = Trim(output_dir)//"calmetout_"// Trim(AdjustL(filedate))//".dat"
	fname_out1 = Trim(output_dir)//"calmetout_"// Trim(AdjustL(filedate))//"_u.dat"
	fname_out2 = Trim(output_dir)//"calmetout_"// Trim(AdjustL(filedate))//"_v.dat"
	fname_out3 = Trim(output_dir)//"calmetout_"// Trim(AdjustL(filedate))//"_t.dat"
	fname_out4 = Trim(output_dir)//"calmetout_"// Trim(AdjustL(filedate))//"_s.dat"
	fname_out5 = Trim(output_dir)//"calmetout_"// Trim(AdjustL(filedate))//"_e.dat"
	   io7=7
	open(io7,file=fname_in1,status='old',form='unformatted' ) 
	print*,'---------- '  
		read(io7) dataset,dataver,datamod
        read(io7) ncom
		 
        do i=1,ncom 
          read(io7) comment1 
        enddo
		 
           read(io7) jbyr,jbmo,jbdy,jbhr,jbsec,jeyr,jemo,jedy,jehr,jesec,	&
     &     axtz,irlg,irtype,												&
     &     nx, ny, nz, dgrid, xorigr, yorigr, iwfcod, nssta,				&
     &     nusta, npsta, nowsta, nlu, iwat1, iwat2, lcalgrd,				&
     &     pmap,datum,daten,												&
     &     feast,fnorth,		utmhem,iutmzn,								&
     &     rnlat0,relon0,xlat1,xlat2 										
	 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		 write(*,*)' dataset ', dataset 
		 write(*,*)' dataver ', dataver 
		 write(*,*)' datamod ', datamod
		 write(*,*)' real ',  SIZEOF(dgrid)  
		 write(*,*)' integer ',  SIZEOF(irlg)  
		 write(*,*)' logical ',  SIZEOF(lcalgrd)  
 
	print*,' ---------- '
		write(*,*)'ncom 1118 ',ncom, SIZEOF(ncom)  
	print*,'---------- '  
		   write(*,*)'   jbyr  ',jbyr
		   write(*,*)'   jbmo  ',jbmo
		   write(*,*)'   jbdy  ',jbdy
		   write(*,*)'   jbhr  ',jbhr
		   write(*,*)'   jbsec  ',jbsec
		   	print*,' ---------- '
		   write(*,*)'   jeyr  ',jeyr
		   write(*,*)'   jemo  ',jemo
		   write(*,*)'   jedy  ',jedy		   
		   write(*,*)'   jehr  ',jehr			   
		   write(*,*)'   jesec  ',jesec			   
		   
		   
		   
		   
		   
		   write(*,*)'   axtz  ',axtz
		   write(*,*)'   nx, ny, nz  ', nx,' ', ny, ' ',nz 
		   write(*,*)'   lcalgrd  ',lcalgrd 
		   write(*,*)'   xlat1   ',xlat1,'  xlat2  ',xlat2  
		   write(*,*)'   rnlat0   ',rnlat0 
	print*,'---------- '   !必须整体读取
	 
	  !!!!!	 record #4 - cell face heights (NZ + 1 words)
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  !!!!!       CLABEL - character*8 - Variable name
	  !!!!!       NDATHRB - integer     - Beginning date and time (YYYYJJJHH)
	  !!!!!         NSECB - integer     - Beginning seconds (SSSS)
	  !!!!!       NDATHRE - integer     - Ending date and time (YYYYJJJHH)
	  !!!!!         NSECE - integer     - Ending seconds (SSSS)
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!nface
         read(io7)		clabel,ndathrb1,nsecb1,ndathre1,nsece,	nface
		   write(*,*)'   clabel   ',clabel
		   write(*,*)'   ndathrb   ',ndathrb1
		   write(*,*)'   nsecb   ',nsecb(t,1)
		   write(*,*)'   ndathre   ',ndathre1
		   write(*,*)'   nsece   ',nsece
		   write(*,*)'   nface   ',nface
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!z0		surface roughness lengths
            read(io7)	clabel,ndathrb1,nsecb1,ndathre1,nsece,z0
		   write(*,*)'   clabel   ',clabel
		   write(*,*)'   ndathrb   ',ndathrb1
		   write(*,*)'   ndathre   ',ndathre1
		   write(*,*)'   z0   ',z0(1,1)
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!ilandu	land use categories			int
            read(io7)	clabel,ndathrb1,nsecb1,ndathre1,nsece,ilandu
		   write(*,*)'   clabel   ',clabel
		   write(*,*)'   ndathrb   ',ndathrb1
		   write(*,*)'   ndathre   ',ndathre1
		   write(*,*)'   ilandu   ',ilandu(1,1)
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!elev
            read(io7)	clabel,ndathrb1,nsecb1,ndathre1,nsece,elev
		   write(*,*)'   clabel   ',clabel
		   write(*,*)'   ndathrb   ',ndathrb1
		   write(*,*)'   ndathre   ',ndathre1 
		   write(*,*)'   elev   ',elev(140,140)
		   
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!xlai		leaf area index
            read(io7)	clabel,ndathrb1,nsecb1,ndathre1,nsece,xlai
		   write(*,*)'   clabel   ',clabel
		   write(*,*)'   ndathrb   ',ndathrb1
		   write(*,*)'   ndathre   ',ndathre1 
		   write(*,*)'   xlai   ',xlai(1,1)
		   
		   counttimes=0
	  !!!!!!!!!!!!!!!!!!!!!!!!!!!!OUTHR
		   write(*,*)'  ---------------------------  '  
		   do t=1,nt
      do   iz=1,mxnz !!!nzm 
            read(io7)	clabel,ndathrb(t,1),nsecb(t,1),ndathre(t,1),nsece,data_uwnd_xyz(t,:,:,iz)
		   write(*,*)'   clabel   ',clabel
		   write(*,*)'   ndathrb   ',ndathrb(t,1)
		   write(*,*)'   nsecb   ',nsecb(t,1)
		   write(*,*)'   ndathre(t,1)   ',ndathre(t,1)
		   write(*,*)'   nsece   ',nsece
		   write(*,*)'   data_uwnd_xyz   ',data_uwnd_xyz(1,140,140,iz)
		   
		   !write(*,*)'   --  '  
            read(io7)	clabel,ndathrb(t,2),nsecb(t,2),ndathre(t,2),nsece,data_vwnd_xyz(t,:,:,iz)
		   !write(*,*)'   clabel   ',clabel
		   !write(*,*)'   ndathrb   ',ndathrb(t,2)
		   !write(*,*)'   ndathre   ',ndathre(t,2) 
		   write(*,*)'   data_uwnd_xyz   ',data_vwnd_xyz(1,140,140,iz)
		!   do a=1,250
		!   do b=1,250
		!   vavg=abs(data_vwnd_xyz(t,a,b,iz))+vavg
		   
		!   		   uavg=abs(data_uwnd_xyz(t,a,b,iz))+uavg
		  
		!   end do
		!   end do
		!   vavg=vavg/250/250
		!   uavg=uavg/250/250
		 !   write(*,*)'   vavg   ',vavg
		!			    write(*,*)'   uavg   ',uavg
		   write(*,*)'  ---------------------------  '  
		  
	      end do  
	   
	  !!!!!!!!!!!!!!!2-D meteorological fields
	   !!!!!!!!!!!!!!!!!!!!!!!!!!!!ipgt
            read(io7)	clabel,ndathrb(t,3),nsecb(t,3),ndathre(t,3),iesec,ipgt(:,:)
		  ! write(*,*)'   clabel   ',clabel
		  ! write(*,*)'   ndathrb   ',ndathrb(t,3)
		  ! write(*,*)'   ndathre   ',ndathre(t,3) 
		  ! write(*,*)'   ipgt   ',ipgt(t,1,1)
		   
	   !!!!!!!!!!!!!!!!!!!!!!!!!!!!ustar
            read(io7)	clabel,ndathrb(t,4),nsecb(t,4),ndathre(t,4),iesec,ustar(:,:)
		  ! write(*,*)'   clabel   ',clabel
		  ! write(*,*)'   ndathrb   ',ndathrb(t,4)
		  ! write(*,*)'   ndathre   ',ndathre(t,4) 
		  ! write(*,*)'   ustar   ',ustar(t,1,1)
	   !!!!!!!!!!!!!!!!!!!!!!!!!!!!zi
            read(io7)	clabel,ndathrb(t,5),nsecb(t,5),ndathre(t,5),iesec,zi(:,:)
		 !  write(*,*)'   clabel   ',clabel
		 !  write(*,*)'   ndathrb   ',ndathrb(t,5)
		 !  write(*,*)'   ndathre   ',ndathre(t,5) 
		  ! write(*,*)'   zi   ',zi(t,1,1)
	   !!!!!!!!!!!!!!!!!!!!!!!!!!!!el
            read(io7)	clabel,ndathrb(t,6),nsecb(t,6),ndathre(t,6),iesec,el(:,:)
		 !  write(*,*)'   clabel   ',clabel
		 !  write(*,*)'   ndathrb   ',ndathrb(t,6)
		 !  write(*,*)'   ndathre   ',ndathre(t,6) 
		  ! write(*,*)'   el   ',el(t,1,1)
	   !!!!!!!!!!!!!!!!!!!!!!!!!!!!wstar
            read(io7)	clabel,ndathrb(t,7),nsecb(t,7),ndathre(t,7),iesec,wstar(:,:)
		  ! write(*,*)'   clabel   ',clabel
		  ! write(*,*)'   ndathrb   ',ndathrb(t,7)
		  ! write(*,*)'   ndathre   ',ndathre(t,7) 
		  ! write(*,*)'   wstar   ',wstar(t,1,1)
	   !!!!!!!!!!!!!!!!!!!!!!!!!!!!rmm
            read(io7)	clabel,ndathrb(t,8),nsecb(t,8),ndathre(t,8),iesec,rmm(:,:)
		  ! write(*,*)'   clabel   ',clabel
		  ! write(*,*)'   ndathrb   ',ndathrb(t,8)
		 !  write(*,*)'   ndathre   ',ndathre(t,8) 
		  ! write(*,*)'   rmm   ',rmm(t,1,1)
	   !!!!!!!!!!!!!!!!!!!!!!!!!!!!temp2d
            read(io7)	clabel,ndathrb(t,9),nsecb(t,9),ndathre(t,9),iesec,temp2d(:,:)
		  ! write(*,*)'   clabel   ',clabel
		 !  write(*,*)'   ndathrb   ',ndathrb(t,9)
		  ! write(*,*)'   ndathre   ',ndathre(t,9) 
		  ! write(*,*)'   temp2d   ',temp2d(t,1,1)
	   !!!!!!!!!!!!!!!!!!!!!!!!!!!!rho
            read(io7)	clabel,ndathrb(t,10),nsecb(t,10),ndathre(t,10),iesec,rho(:,:)
		  ! write(*,*)'   clabel   ',clabel
		  ! write(*,*)'   ndathrb   ',ndathrb(t,10)
		  ! write(*,*)'   ndathre   ',ndathre(t,10) 
		  ! write(*,*)'   rho   ',rho(t,1,1)	   !!!!!!!!!!!!!!!!!!!!!!!!!!!!qsw
            read(io7)	clabel,ndathrb(t,11),nsecb(t,11),ndathre(t,11),iesec,qsw(:,:)
		 !  write(*,*)'   clabel   ',clabel
		  ! write(*,*)'   ndathrb   ',ndathrb(t,11)
		  ! write(*,*)'   ndathre   ',ndathre(t,11) 
		  ! write(*,*)'   qsw   ',qsw(t,1,1)
		   !!!!!!!!!!!!!!!!!!!!!!!!!!!!irh2d
            read(io7)	clabel,ndathrb(t,12),nsecb(t,12),ndathre(t,12),iesec,irh2d(:,:)
		  ! write(*,*)'   clabel   ',clabel
		  ! write(*,*)'   ndathrb   ',ndathrb(t,12)
		  ! write(*,*)'   ndathre   ',ndathre(t,12) 
		  ! write(*,*)'   irh2d   ',irh2d(t,1,1)
		   !!!!!!!!!!!!!!!!!!!!!!!!!!!!ipcode2d
            read(io7)	clabel,ndathrb(t,13),nsecb(t,13),ndathre(t,13),iesec,ipcode2d(:,:)
		  ! write(*,*)'   clabel   ',clabel
		  ! write(*,*)'   ndathrb   ',ndathrb(t,13)
		  ! write(*,*)'   ndathre   ',ndathre(t,13) 
		  ! write(*,*)'   ipcode2d   ',ipcode2d(t,1,1)
		  
	  counttimes=counttimes+1
	  end do !nt
	  !read(io7)	dummy
	  !write(*,*)'   dummy   ',dummy         !attempt to read past end of file.
	print*,'----Closing file------ '  
	close(io7)
	!!!!!!!!!!!check time!!!!!!!!!!!!!!!!!!!
	!do var=1,12
	!if (ndathrb(nt,var).ne.ndathrb(nt,var+1) .or. ndathrb(nt,var).ne.ndathre(nt,var))
	!	print*,'----time not fit, var =  ' var
	!	stop
	!endif
	!end do
	
	!end do
	!!!!!!!!!!!!!!!!!!!output!!!!!!!!!!!!!!!!!!!!!
	jultime=ndathrb(:,1)
	second=nsecb(:,1)
	write(*,*) counttimes

	write(*,*) size(data_uwnd_xyz)
	do z=1,5
	data_uwnd_5l(:,:,:,z)=data_uwnd_xyz(:,:,:,z)
	data_vwnd_5l(:,:,:,z)=data_vwnd_xyz(:,:,:,z)
	
	end do
	write(*,*) size(data_uwnd_5l)	
	
		open(91,file=fname_out1,form='unformatted',access='sequential') 
		write(91)data_uwnd_5l
		close(91)

		open(92,file=fname_out2,form='unformatted',access='sequential') 
		write(92)data_vwnd_5l
		close(92)		

		open(io93,file=fname_out3,form='unformatted',access='sequential') 
		write(io93)jultime
		close(io93)
		
		open(io94,file=fname_out4,form='unformatted',access='sequential') 
		write(io94)second
		close(io94)
		
		open(io95,file=fname_out5,form='unformatted',access='sequential') 
		write(io95)elev
		close(io95)
		
	print*,'----end------ '  

end


     

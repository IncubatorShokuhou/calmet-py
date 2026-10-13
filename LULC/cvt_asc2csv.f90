			program main
			implicit none 	
			integer nr, nc, i, j				
			integer a(3,7)
			integer b(3*7)		
			nr=3
			nc=7

			open(1,file='GLC2000CN.txt')
			do i=1,nr
			read(1,99) a(i,1)
			end do 
			b=reshape(a,(/nr*nc/))
		print*, b
99		    format(I2)
			
			
			
			
			
			end
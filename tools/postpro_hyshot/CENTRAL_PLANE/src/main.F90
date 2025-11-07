program central_plane
        use parameters 
        implicit none
        integer :: N_S, nxsmax, nysmax, nvar, m, num_xstations, num_ystations, i, ii, j, jj, k
        integer, dimension(:), allocatable :: x_stations, y_stations
        real(rkind), dimension(:,:,:), allocatable :: w_slice_sym
        real(rkind), dimension(:), allocatable :: xg,yg 
        character(6)  :: chmean

        print *, 'Starting central plane postpro...'
        call execute_command_line('mkdir -p PROFILES/')
        print *, 'Reading central_plane_infty_norm.bin...'
        open(10, file='central_plane_infty_norm.bin', &
                 action='read', status='old', form='unformatted')
        ! reading header
        read(10) nxsmax, nysmax, nvar
        ! nvar is 6+N_S
        N_S = 3

        allocate(w_slice_sym(nvar,nxsmax,nysmax))
        allocate(xg(nxsmax),yg(nysmax))

        read(10) xg
        read(10) yg
 
        do m=1,nvar
         read(10) w_slice_sym(m,:,:) 
        enddo
        close(10)

        print *, 'End reading central_plane_infty_norm.bin...'
        
        open(12, file='./input_central_plane.dat', action='read', status='old')
        read(12,*) 
        read(12,*) num_xstations,num_ystations
        read(12,*) 
        allocate(x_stations(num_xstations),y_stations(num_ystations))
        do k=1,num_xstations
         read(12,*) x_stations(k)
        enddo
        read(12,*)
        do k=1,num_ystations
         read(12,*) y_stations(k)
        enddo

        do k=1,num_xstations
         write(chmean,1006) x_stations(k)
         write(*,*) 'writing mean profile,',k,chmean 
         open(unit=15,file='PROFILES/stat_i_'//chmean//'.prof')
         ii = x_stations(k)
         do j=1,nysmax
          write(15,100) yg(j), (w_slice_sym(m,ii,j),m=1,nvar)
         enddo
         close(15)
        enddo

        do k=1,num_ystations
         write(chmean,1006) y_stations(k)
         write(*,*) 'writing mean profile,',k,chmean
         open(unit=15,file='PROFILES/stat_j_'//chmean//'.prof')
         jj = y_stations(k)
         do i=1,nxsmax
          write(15,100) xg(i), (w_slice_sym(m,i,jj),m=1,nvar)
         enddo
         close(15)
        enddo

  100  format(200ES20.10) 
  1006 format(I6.6)
        
endprogram

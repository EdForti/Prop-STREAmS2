PROGRAM postpro_vel_tau_serialstats
! 
! This program reads stat3d_*_*_*.bin, evaluates average wrt sym plane and 
! write sub1 and sub2 vtks.
! 
! ATTENTION: All the variables are normalised by U_infty and U_infty^2
!
! INPUT FILES
! ./input_subvolume_serial.dat
!   nxmax, nymax, nzmax      Streamwise/Wall-normal/Spanwise total number of points
!   nprocx, nprocy, nprocz        "          "         "     number of blocks
!   re_delta                 Reynolds number from simulation "flow_params.dat"
!   mach                     Mach number       "       "            " 
!   ig_start                 Min global streamwise index considered
!   jg_end                   Max   "    wall-normal  "      " 
!   nzsmax                   Spanwise number of points considered in symmetric average
! ./dxg, ./dyg, ./dzg        Coords file from simulation
! ./stat3d0_'//chx//'_0000_'//chz//'.bin  Time-averaged 3D stats for controlled case (block chx, chz)
! ../stat.bin                Time- and span-wise averaged 2D stats for uncontrolled case
!
! OUTPUT FILES
!   sub1_0001.vtk (rho_bar, (u_i)_tilde, p_bar)
!   sub2_0001.vtk (rho_bar, (tau_ij)_tilde (11,22,33,12,13,23))
!   central_plane_infty_norm.bin  Binary file containing mean rho, u, v, w, p at symmetry plane
!
! 11/01/2023 G. Della Posta 
!
 USE modpostpro
 IMPLICIT NONE
!
!------------------------------------------------------------------------------
! Subvolume

 INTEGER :: i,j,k,l,ks,kk,m,jj,ii,lsp
 INTEGER :: ig_start, ig_end, num_blocks, nv_sub, jg_start, jg_end
 INTEGER :: kproc, ksym_start, ksym_end
 INTEGER :: nprocx, nprocy, nprocz, nxserial, nyserial, nzserial
 INTEGER :: nxsmax, nysmax, nzsmax, nxs, nys, nzs
 INTEGER :: k_start, i_l, i_r, j_l, j_r, k_l, k_r
 INTEGER :: idestin, isource
 INTEGER :: mp_cart_xz

 INTEGER, DIMENSION(:), ALLOCATABLE :: i_stress
 INTEGER, DIMENSION(:), ALLOCATABLE :: i_primitive
 
 INTEGER, DIMENSION(3,3) :: it 

 REAL(rkind) :: duiidx, duiidy, duiidz, pp, dxm1, dym1, dzm1

 REAL(rkind), DIMENSION(:,:,:), ALLOCATABLE :: w_slice_sym

 REAL(rkind), DIMENSION(:,:,:,:), ALLOCATABLE :: w_sym
 REAL(rkind), DIMENSION(:,:,:,:), ALLOCATABLE :: production
 
 CHARACTER(len = 4) :: chx, chz
 CHARACTER(len = 12), DIMENSION(:), ALLOCATABLE :: names1
 CHARACTER(len = 12), DIMENSION(:), ALLOCATABLE :: names2

!------------------------------------------------------------------------------
! MPI

 LOGICAL :: reord
 
 INTEGER, DIMENSION(3) :: sizes    ! Dimensions of the total grid
 INTEGER, DIMENSION(3) :: subsizes ! Dimensions of grid local to a procs
 INTEGER, DIMENSION(3) :: starts   ! Starting coordinates
 INTEGER :: ntot3d
 INTEGER :: mpi_io_file
 INTEGER :: filetype
 INTEGER :: size_real
 INTEGER (kind=mpi_offset_kind) :: offset
 INTEGER :: N_S

!------------------------------------------------------------------------------

 call mpi_init(iermpi)
 call mpi_comm_rank(mpi_comm_world,nrank,iermpi)
 call mpi_comm_size(mpi_comm_world,nproc,iermpi)

 masterproc = .false.
 if (nrank==0) masterproc = .true.

 IF (masterproc) WRITE(*,*) '!--------START PROGRAM POSTPRO_VEL_TAU-------'

 OPEN(12, FILE='./input_subvolume_serial.dat', ACTION='read', STATUS='old')
 READ(12,*) 
 READ(12,*) N_S, nxmax, nymax, nzmax, nprocx, nprocy, nprocz
 READ(12,*) 
 READ(12,*) 
 READ(12,*) re_delta, u0, ig_start, jg_end, nzsmax
 CLOSE(12)

 nv_stat_3d = 16+2*N_S
 ! 1  => rho 
 ! 2  => rho*u 
 ! 3  => rho*v 
 ! 4  => rho*w 
 ! 5  => rho*T
 ! 6  => rho**2
 ! 7  => (rho*u)**2/rho
 ! 8  => (rho*v)**2/rho
 ! 9  => (rho*w)**2/rho
 ! 10 => (rho*u)*(rho*v)/rho
 ! 11 => (rho*u)*(rho*w)/rho
 ! 12 => (rho*v)*(rho*w)/rho
 ! 13 => rho*T**2
 ! 14 => (rho*T)**2
 ! 15 => p
 ! 16 => p**2
 ! 16 + lsp => rho_lsp
 ! 16 + N_S + lsp => rho_lsp**2
 nv_sub     = nv_stat_3d

! u0         = SQRT(gamma)*mach

 nblocks(1)  = nproc
 nblocks(2)  = 1
 nblocks(3)  = 1

 pbc(1) = .false.
 pbc(2) = .false.
 pbc(3) = .true.

! Create 3D topology

 reord = .false.
 call mpi_cart_create(mpi_comm_world,ndims,nblocks,pbc,reord,mp_cart,iermpi)
 call mpi_cart_coords(mp_cart,nrank,ndims,ncoords,iermpi)

! Sizes of stats files and procs data

 nxserial = nxmax/nprocx
 nyserial = nymax/nprocy
 nzserial = nzmax/nprocz

 nx       = nxserial ! Parallel in x
 ny       = nymax/1
 nz       = nzmax/1

 i_block_start = ig_start/nxserial
 i_block_end   = nprocx - 1
 num_blocks    = i_block_end - i_block_start + 1

 IF (num_blocks /= nproc) THEN
  IF (masterproc) WRITE(*,*) 'num_blocks, nproc = ', num_blocks, nproc
  IF (masterproc) WRITE(*,*) 'i_block_start, i_block_end = ', i_block_start, i_block_end
  RETURN
 ENDIF

 ig_start = nxserial*i_block_start + 1
 ig_end   = nxmax
 jg_start = 1

!------------------------------------------------------------------------------
! Subvolume inputs
!

 IF (MOD(nzsmax,2) /= 0) THEN
  nzsmax = CEILING(REAL(nzsmax)/2)*2
  IF (masterproc) WRITE(*,*) 'nzsmax (multiple of 2!) = ', nzsmax
 ENDIF

 k_start = (nzmax - nzsmax)/2 ! First point before central subvolume

 nxsmax  = ig_end - ig_start + 1
 nysmax  = jg_end - jg_start + 1
 nzsmax  = nzsmax/2

 nxs     = nx
 nys     = nysmax
 nzs     = nzsmax

 allocate(xg(nxmax))
 allocate(yg(nymax), dyg(nymax-1))
 allocate(zg(nzmax))

 open(18,file='../dxg.dat',action='read',status='old')
 do i=1,nxmax
  read(18,*) xg(i)
 enddo
 close(18)
 open(18,file='../dyg.dat',action='read',status='old')
 do j=1,nymax
  read(18,*) yg(j)
 enddo
 close(18)
 open(18,file='../dzg.dat',action='read',status='old')
 do k=1,nzmax
  read(18,*) zg(k)
 enddo
 close(18)

 call mpi_barrier(mpi_comm_world,iermpi)

! Read stat3d.bin

 IF (masterproc) WRITE(*,*) 'Start read stat3d and average subvolume.'

 ALLOCATE(w_stat_3d(nv_stat_3d,nxserial,nyserial,nzserial))
 ALLOCATE(w_sym(nv_sub, nxs, nys, nzs+1))

 DO kproc = 0, nprocz-1
  WRITE(chx,'(I4.4)') nrank + i_block_start
  WRITE(chz,'(I4.4)') kproc
  IF (masterproc) WRITE(*,*) 'Reading kproc = ', kproc
  w_stat_3d = 0._rkind
  OPEN(11,file='../stat3d0_'//chx//'_0000_'//chz//'.bin',form='unformatted', &
      & STATUS='old', ACTION='read')
  READ(11) w_stat_3d(1:nv_stat_3d,1:nxserial,1:nyserial,1:nzserial)
  CLOSE(11)
  IF (kproc < nprocz/2) THEN
    ksym_start = kproc*nzserial + 1
    ksym_end   = ksym_start + nzserial - 1
    w_sym(:,:,:,ksym_start:ksym_end) = w_stat_3d(1:nv_sub,1:nxs,1:nys,1:nzserial)
  ELSE
    ksym_start = (nprocz - kproc - 1)*nzserial + 1
    ksym_end   = ksym_start + nzserial - 1
    DO m = 1, nv_sub
      IF (kproc == nprocz/2) THEN
        w_sym(m,:,:,ksym_end+1) = w_stat_3d(m, 1:nxs, 1:nys, 1)
      ENDIF
      IF ((m /= 4) .AND. (m /= 11) .AND. (m /= 12) ) THEN
        w_sym(m,:,:,ksym_start+1:ksym_end) = &
             & 0.5_rkind * (w_sym(m,:,:,ksym_start+1:ksym_end) + w_stat_3d(m, 1:nxs, 1:nys, nzserial:2:-1))
        w_sym(m,:,:,ksym_end+1) = 0.5_rkind * (w_sym(m,:,:,ksym_end+1) + w_stat_3d(m, 1:nxs, 1:nys, 1))
      ELSE ! w is the only antisymmetric primitive variable
        w_sym(m,:,:,ksym_start+1:ksym_end) = &
             & 0.5_rkind * (w_sym(m,:,:,ksym_start+1:ksym_end) - w_stat_3d(m, 1:nxs, 1:nys, nzserial:2:-1))
        w_sym(m,:,:,ksym_end+1) = 0.5_rkind * (w_sym(m,:,:,ksym_end+1) - w_stat_3d(m, 1:nxs, 1:nys, 1))
      ENDIF
    ENDDO
  ENDIF
 ENDDO
 DEALLOCATE(w_stat_3d)

 IF (masterproc) WRITE(*,*) 'End read stat3d.bin.'

! Define output quantities

 ! 1: rho_bar
 w_sym( 2,:,:,:) = w_sym(2,:,:,:)/w_sym(1,:,:,:) ! u_tilde
 w_sym( 3,:,:,:) = w_sym(3,:,:,:)/w_sym(1,:,:,:) ! v_tilde
 w_sym( 4,:,:,:) = w_sym(4,:,:,:)/w_sym(1,:,:,:) ! w_tilde
 w_sym( 5,:,:,:) = w_sym(5,:,:,:)/w_sym(1,:,:,:) ! T_tilde
 ! 6: (rho^2)_bar
 w_sym( 7,:,:,:) = w_sym(1,:,:,:)*w_sym(2,:,:,:)*w_sym(2,:,:,:) - w_sym( 7,:,:,:) ! tau_11_tilde (P)
 w_sym( 8,:,:,:) = w_sym(1,:,:,:)*w_sym(3,:,:,:)*w_sym(3,:,:,:) - w_sym( 8,:,:,:) ! tau_22_tilde (P)
 w_sym( 9,:,:,:) = w_sym(1,:,:,:)*w_sym(4,:,:,:)*w_sym(4,:,:,:) - w_sym( 9,:,:,:) ! tau_33_tilde (P)
 w_sym(10,:,:,:) = w_sym(1,:,:,:)*w_sym(2,:,:,:)*w_sym(3,:,:,:) - w_sym(10,:,:,:) ! tau_12_tilde (P)
 w_sym(11,:,:,:) = w_sym(1,:,:,:)*w_sym(2,:,:,:)*w_sym(4,:,:,:) - w_sym(11,:,:,:) ! tau_13_tilde (D)
 w_sym(12,:,:,:) = w_sym(1,:,:,:)*w_sym(3,:,:,:)*w_sym(4,:,:,:) - w_sym(12,:,:,:) ! tau_23_tilde (D)
 w_sym(13,:,:,:) = sqrt(w_sym(13,:,:,:)/w_sym(1,:,:,:)-w_sym(5,:,:,:)*w_sym(5,:,:,:)) ! Trms 
 ! 14: (rho*T)**2 useless...
 ! 15: p_bar
 w_sym(16,:,:,:) = sqrt(w_sym(16,:,:,:)-w_sym(15,:,:,:)*w_sym(15,:,:,:)) ! Prms 

 do lsp=1,N_S
  w_sym(16+lsp,:,:,:) = w_sym(16+lsp,:,:,:)/w_sym(1,:,:,:)
  w_sym(16+N_S+lsp,:,:,:) = sqrt(w_sym(16+N_S+lsp,:,:,:)/w_sym(1,:,:,:)-w_sym(16+lsp,:,:,:)**2)
 enddo

 !
 ! Central slice .dat
 !
 ALLOCATE(w_slice_sym(nv_stat_3d, nxsmax, nysmax))
 DO j = 1, nysmax
   DO i = 1, nv_stat_3d
   CALL MPI_GATHER(w_sym(i,:,j,nzs+1), nxs, mpi_prec, w_slice_sym(i,:,j), nxs, &
        & mpi_prec, 0, mpi_comm_world, iermpi)
   ENDDO
 ENDDO

 !------------------------------------------------------------------------------
 ! WRITE OUTPUT
 !------------------------------------------------------------------------------

 IF (masterproc) WRITE(*,*) '!------------START WRITING OUTPUT------------'

 IF (masterproc) THEN
   WRITE(*,*) 'Writing primitive variables on symmetry plane.'
   OPEN(666,FILE='central_plane_infty_norm.bin', &
       & ACTION='write', STATUS='new', FORM='unformatted')
   WRITE(666) nxsmax, nysmax, 15+2*N_S
   WRITE(666) xg(ig_start:ig_start+nxsmax-1)
   WRITE(666) yg(1:nysmax)
   DO j = 1, 16
     if (j.ne.14) WRITE(666) w_slice_sym(j,:,:)
   ENDDO
   DO j = 1,2*N_S
    WRITE(666) w_slice_sym(16+j,:,:)
   enddo
   CLOSE(666)
 ENDIF

 !
 ! Lateral slice .dat
 !
 DEALLOCATE(w_slice_sym)
 ALLOCATE(w_slice_sym(nv_stat_3d, nxsmax, nysmax))
 DO j = 1, nysmax
   DO i = 1, nv_stat_3d
   CALL MPI_GATHER(w_sym(i,:,j,1), nxs, mpi_prec, w_slice_sym(i,:,j), nxs, &
        & mpi_prec, 0, mpi_comm_world, iermpi)
   ENDDO
 ENDDO

 !------------------------------------------------------------------------------
 ! WRITE OUTPUT
 !------------------------------------------------------------------------------

 IF (masterproc) WRITE(*,*) '!------------START WRITING OUTPUT------------'

 IF (masterproc) THEN
   WRITE(*,*) 'Writing primitive variables on symmetry plane.'
   OPEN(666,FILE='lateral_plane_infty_norm.bin', &
       & ACTION='write', STATUS='new', FORM='unformatted')
   WRITE(666) nxsmax, nysmax, 15+2*N_S
   WRITE(666) xg(ig_start:ig_start+nxsmax-1)
   WRITE(666) yg(1:nysmax)
   DO j = 1, 16
     if (j.ne.14) WRITE(666) w_slice_sym(j,:,:)
   ENDDO
   DO j = 1,2*N_S
    WRITE(666) w_slice_sym(16+j,:,:)
   enddo
   CLOSE(666)
 ENDIF

 CALL MPI_BARRIER(mpi_comm_world,iermpi)

 ! Redefine grid => same formal operations
 nxmax       = nxsmax
 nymax       = nysmax
 nzmax       = nzsmax+1
 nx          = nxs
 ny          = nys
 nz          = nzs+1
 i_l         = ig_start
 i_r         = ig_end
 j_l         = 1
 j_r         = nymax
 k_l         = k_start + 1
 k_r         = k_start + nzsmax + 1
 xg(1:nxmax) = xg(i_l:i_r)
 yg(1:nymax) = yg(j_l:j_r)
 zg(1:nzmax) = zg(k_l:k_r)
 nblocks(1)  = num_blocks

 IF (masterproc) THEN
   WRITE(*,*) 'nxsmax, nysmax, nzsmax, ig_start, ig_end, jg_start, jg_end, kg_start, kg_end'
   WRITE(*,*)  nxmax, nymax, nzmax, i_l, i_r, j_l, j_r, k_l, k_r
   WRITE(*,*) 'nblocks = ', nblocks
 ENDIF
 
 allocate(i_primitive(6+N_S))
 i_primitive = 0
 do l=1,5
  i_primitive(l) = l
 enddo
 i_primitive(6) = 15
 do l=1,N_S
  i_primitive(6+l) = 16+l
 enddo
 allocate(names1(6+N_S))
 names1 = ["Reyn_Density", "FaVelocity_x", "FaVelocity_y", & 
         & "FaVelocity_z", "FTemperature", "ReynPressure", &
         & "Y_H2        ", "Y_O2        ", "Y_N2        "]
 allocate(i_stress(9+N_S))
 i_stress(1:9) = (/ 1, 7, 8, 9, 10, 11, 12, 13, 16/)
 do l=1,N_S
  i_stress(9+l) = 16+N_S+l
 enddo
 allocate(names2(9+N_S))
 names2 = ["rho_reynolds", "tau_11_favre", "tau_22_favre", "tau_33_favre", &
         & "tau_12_favre", "tau_13_favre", "tau_23_favre", "T_rms       ", "P_rms       ",&
         & "Y_H2_rms    ", "Y_O2_rms    ", "Y_N2_rms    "]

 IF (masterproc) WRITE(*,*) 'Writing primitive variables.'
 CALL write_vtk_reduced_extended_x(6+N_S, w_sym(i_primitive,:,:,:), 'sub1', names1)

 IF (masterproc) WRITE(*,*) 'Writing Favre-averaged Reynolds stress and rms of T, P, Yn.'
 CALL write_vtk_reduced_extended_x(9+N_S, w_sym(i_stress   ,:,:,:), 'sub2', names2)

 IF (masterproc) WRITE(*,*) '!-------------END WRITING OUTPUT-------------'

 IF (masterproc) WRITE(*,*) '!---------END PROGRAM POSTPRO_VEL_TAU--------'

 call mpi_finalize(iermpi)

END PROGRAM postpro_vel_tau_serialstats

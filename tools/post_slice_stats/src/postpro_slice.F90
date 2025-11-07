#include "index_define.h"
module postpro_slice 
      use parameters
      use global_variables
      use utils
contains
      subroutine set_var_names
          allocate(var_names(nvstats))
          var_names(1:3)   = ['Y_H2Fav', 'Y_O2Fav', 'Y_N2Fav'] 
          var_names(4:10)  = ['UFav', 'VFav', 'WFav', 'TFav', 'P   ', 'Rho ', 'ZFav']
          var_names(11:13) = ['Y_H2Fav_rms', 'Y_O2Fav_rms', 'Y_N2Fav_rms'] 
          var_names(14:20) = ['UFav_rms', 'VFav_rms', 'WFav_rms', 'TFav_rms', 'P_rms   ', 'Rho_rms ', 'ZFav_rms']

      endsubroutine set_var_names

      subroutine postpro_slice_xy
          implicit none
          integer :: i,j,k,ii,ib,kb,istatus,kk,m
          integer :: num_records,k_slice
          integer :: nimax,njmax,nkmax,ni,nj,nk,i1,i2,j1,j2,k1,k2
          character(3) :: chx,chz
          character(30) :: filename
          character(len=20) :: file_prefix
          logical :: my_master
          real(rkind), dimension(:,:,:,:), allocatable :: wstat

          my_master = .false.
          if (ncoords(1)==0) my_master = .true.

          allocate(wstat(1:nx,1:ny,knum,1:nvstats))
          wstat(:,:,:,:) = 0._rkind
          write(chx,1003) ncoords(1)
          write(chz,1003) ncoords(3)
          filename = 'slicexy_'//chx//'_000_'//chz//'.bin'
          if (my_master) write(*,*) '----------------reading slice xy----------------'
          open(14,file=filename, action='read', form='unformatted')
          num_records = 0
          do 
           read(14,iostat=istatus) 
           if (istatus.ne.0) exit
           read(14,iostat=istatus) 
           read(14,iostat=istatus) 
           read(14,iostat=istatus) 
           read(14,iostat=istatus) slicexy_aux
           if (istatus.ne.0) exit
           do kk=1,knum
            do j=1,ny
             do i=1,nx 
              wstat(i,j,kk,1)  = num_records*wstat(i,j,kk,1) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,H2) ! YH2
              wstat(i,j,kk,2)  = num_records*wstat(i,j,kk,2) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,O2) ! YO2
              wstat(i,j,kk,3)  = num_records*wstat(i,j,kk,3) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,N2) ! YN2
              wstat(i,j,kk,4)  = num_records*wstat(i,j,kk,4) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,4) ! u 
              wstat(i,j,kk,5)  = num_records*wstat(i,j,kk,5) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,5) ! v
              wstat(i,j,kk,6)  = num_records*wstat(i,j,kk,6) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,6) ! w
              wstat(i,j,kk,7)  = num_records*wstat(i,j,kk,7) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,7) ! T
              wstat(i,j,kk,8)  = num_records*wstat(i,j,kk,8) + slicexy_aux(i,j,kk,8)                       ! p
              wstat(i,j,kk,9)  = num_records*wstat(i,j,kk,9) + slicexy_aux(i,j,kk,12)                      ! rho
              wstat(i,j,kk,10) = num_records*wstat(i,j,kk,10) + slicexy_aux(i,j,kk,12)*&
                                 ( (0.5_rkind*(slicexy_aux(i,j,kk,H2) - slicexy_aux(i,j,kk,O2))&
                                 +0.1165_rkind)/0.6165_rkind )                                            ! Z

              wstat(i,j,kk,11) = num_records*wstat(i,j,kk,11) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,H2)**2 ! YH2**2
              wstat(i,j,kk,12) = num_records*wstat(i,j,kk,12) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,O2)**2 ! YO2**2
              wstat(i,j,kk,13) = num_records*wstat(i,j,kk,13) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,N2)**2 ! YN2**2
              wstat(i,j,kk,14) = num_records*wstat(i,j,kk,14) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,4)**2 ! u
              wstat(i,j,kk,15) = num_records*wstat(i,j,kk,15) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,5)**2 ! v
              wstat(i,j,kk,16) = num_records*wstat(i,j,kk,16) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,6)**2 ! w
              wstat(i,j,kk,17) = num_records*wstat(i,j,kk,17) + slicexy_aux(i,j,kk,12)*slicexy_aux(i,j,kk,7)**2 ! T
              wstat(i,j,kk,18) = num_records*wstat(i,j,kk,18) + slicexy_aux(i,j,kk,8)**2                       ! p
              wstat(i,j,kk,19) = num_records*wstat(i,j,kk,19) + slicexy_aux(i,j,kk,12)**2                      ! rho**2
              wstat(i,j,kk,20) = num_records*wstat(i,j,kk,20) + slicexy_aux(i,j,kk,12)*&
                                 ( (0.5_rkind*(slicexy_aux(i,j,kk,H2) - slicexy_aux(i,j,kk,O2))&
                                 +0.1165_rkind)/0.6165_rkind )**2                                             ! Z**2 

             enddo
            enddo
           enddo
           wstat = wstat/(num_records+1)
           num_records = num_records + 1 
          enddo
          do kk=1,knum
           do j=1,ny
            do i=1,nx 
             wstat(i,j,kk,1)  = wstat(i,j,kk,1)/wstat(i,j,kk,9)
             wstat(i,j,kk,2)  = wstat(i,j,kk,2)/wstat(i,j,kk,9) 
             wstat(i,j,kk,3)  = wstat(i,j,kk,3)/wstat(i,j,kk,9) 
             wstat(i,j,kk,4)  = wstat(i,j,kk,4)/wstat(i,j,kk,9) 
             wstat(i,j,kk,5)  = wstat(i,j,kk,5)/wstat(i,j,kk,9) 
             wstat(i,j,kk,6)  = wstat(i,j,kk,6)/wstat(i,j,kk,9) 
             wstat(i,j,kk,7)  = wstat(i,j,kk,7)/wstat(i,j,kk,9) 
             wstat(i,j,kk,8)  = wstat(i,j,kk,8) 
             wstat(i,j,kk,9)  = wstat(i,j,kk,9)
             wstat(i,j,kk,10) = wstat(i,j,kk,10)/wstat(i,j,kk,9)

             wstat(i,j,kk,11) = sqrt(wstat(i,j,kk,11)/wstat(i,j,kk,9)-wstat(i,j,kk,1)**2)
             wstat(i,j,kk,12) = sqrt(wstat(i,j,kk,12)/wstat(i,j,kk,9)-wstat(i,j,kk,2)**2)
             wstat(i,j,kk,13) = sqrt(wstat(i,j,kk,13)/wstat(i,j,kk,9)-wstat(i,j,kk,3)**2)
             wstat(i,j,kk,14) = sqrt(wstat(i,j,kk,14)/wstat(i,j,kk,9)-wstat(i,j,kk,4)**2)
             wstat(i,j,kk,15) = sqrt(wstat(i,j,kk,15)/wstat(i,j,kk,9)-wstat(i,j,kk,5)**2)
             wstat(i,j,kk,16) = sqrt(wstat(i,j,kk,16)/wstat(i,j,kk,9)-wstat(i,j,kk,6)**2)
             wstat(i,j,kk,17) = sqrt(wstat(i,j,kk,17)/wstat(i,j,kk,9)-wstat(i,j,kk,7)**2)
             wstat(i,j,kk,18) = sqrt(wstat(i,j,kk,18)-wstat(i,j,kk,8)**2)
             wstat(i,j,kk,19) = sqrt(wstat(i,j,kk,19)-wstat(i,j,kk,9)**2)
             wstat(i,j,kk,20) = sqrt(wstat(i,j,kk,20)/wstat(i,j,kk,9)-wstat(i,j,kk,10)**2)
            enddo
           enddo
          enddo

          nimax = nxmax
          njmax = nymax
          nkmax = 1 
          ni    = nx
          nj    = ny
          nk    = 1
          file_prefix = "stats_slicexy_??????"
          do kk=1,knum
           k_slice = kslice(kk) 
           write(file_prefix(15:20),'(I6.6)') k_slice + ncoords(3)*nz
           call write_vtk_general(file_prefix,nimax,njmax,nkmax,ni,nj,nk,1,ni,1,ny,kk,kk,mp_cartx,my_master,&
                                  wstat(1:nx,1:ny,kk:kk,1:nvstats),nvstats)
          enddo
          deallocate(wstat)

!
!            endif
!           enddo
!           ii = ii+nxb
!          enddo
1003 format(I3.3)
     endsubroutine postpro_slice_xy

      subroutine postpro_slice_xz
          implicit none
          integer :: i,j,k,ii,ib,kb,istatus,jj,m
          integer :: num_records,j_slice
          integer :: nimax,njmax,nkmax,ni,nj,nk,i1,i2,j1,j2,k1,k2
          character(3) :: chx,chz
          character(30) :: filename
          character(len=20) :: file_prefix
          real(rkind), dimension(:,:,:,:), allocatable :: wstat
          integer :: icyc
          real(rkind) :: time

          allocate(wstat(1:nx,jnum,1:nz,1:nvstats))
          wstat(:,:,:,:) = 0._rkind
          write(chx,1003) ncoords(1)
          write(chz,1003) ncoords(3)
          filename = 'slicexz_'//chx//'_000_'//chz//'.bin'
          if (masterproc) write(*,*) '----------------reading slice xz----------------'
          open(15,file=filename, action='read', form='unformatted')
          num_records = 0
          do 
           read(15,iostat=istatus) icyc,time 
           if (istatus.ne.0) exit
           read(15,iostat=istatus) 
           read(15,iostat=istatus) 
           read(15,iostat=istatus) 
           read(15,iostat=istatus) slicexz_aux
           if (istatus.ne.0) exit
           do k=1,nz
            do jj=1,jnum
             do i=1,nx 
              wstat(i,jj,k,1)  = num_records*wstat(i,jj,k,1) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,H2) ! YH2
              wstat(i,jj,k,2)  = num_records*wstat(i,jj,k,2) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,O2) ! YO2
              wstat(i,jj,k,3)  = num_records*wstat(i,jj,k,3) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,N2) ! YN2
              wstat(i,jj,k,4)  = num_records*wstat(i,jj,k,4) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,4) ! u 
              wstat(i,jj,k,5)  = num_records*wstat(i,jj,k,5) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,5) ! v
              wstat(i,jj,k,6)  = num_records*wstat(i,jj,k,6) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,6) ! w
              wstat(i,jj,k,7)  = num_records*wstat(i,jj,k,7) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,7) ! T
              wstat(i,jj,k,8)  = num_records*wstat(i,jj,k,8) + slicexz_aux(i,jj,k,8)                       ! p
              wstat(i,jj,k,9)  = num_records*wstat(i,jj,k,9) + slicexz_aux(i,jj,k,12)                      ! rho
              wstat(i,jj,k,10) = num_records*wstat(i,jj,k,10) + slicexz_aux(i,jj,k,12)*&
                                 ( (0.5_rkind*(slicexz_aux(i,jj,k,H2) - slicexz_aux(i,jj,k,O2))&
                                 +0.1165_rkind)/0.6165_rkind )                                            ! Z

              wstat(i,jj,k,11) = num_records*wstat(i,jj,k,11) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,H2)**2 ! YH2**2
              wstat(i,jj,k,12) = num_records*wstat(i,jj,k,12) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,O2)**2 ! YO2**2
              wstat(i,jj,k,13) = num_records*wstat(i,jj,k,13) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,N2)**2 ! YN2**2
              wstat(i,jj,k,14) = num_records*wstat(i,jj,k,14) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,4)**2 ! u
              wstat(i,jj,k,15) = num_records*wstat(i,jj,k,15) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,5)**2 ! v
              wstat(i,jj,k,16) = num_records*wstat(i,jj,k,16) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,6)**2 ! w
              wstat(i,jj,k,17) = num_records*wstat(i,jj,k,17) + slicexz_aux(i,jj,k,12)*slicexz_aux(i,jj,k,7)**2 ! T
              wstat(i,jj,k,18) = num_records*wstat(i,jj,k,18) + slicexz_aux(i,jj,k,8)**2                       ! p
              wstat(i,jj,k,19) = num_records*wstat(i,jj,k,19) + slicexz_aux(i,jj,k,12)**2                      ! rho**2
              wstat(i,jj,k,20) = num_records*wstat(i,jj,k,20) + slicexz_aux(i,jj,k,12)*&
                                 ( (0.5_rkind*(slicexz_aux(i,jj,k,H2) - slicexz_aux(i,jj,k,O2))&
                                 +0.1165_rkind)/0.6165_rkind )**2                                             ! Z**2 

             enddo
            enddo
           enddo
           wstat = wstat/(num_records+1)
           num_records = num_records + 1 
          enddo

          do k=1,nz
           do jj=1,jnum
            do i=1,nx 
             wstat(i,jj,k,1)  = wstat(i,jj,k,1)/wstat(i,jj,k,9)
             wstat(i,jj,k,2)  = wstat(i,jj,k,2)/wstat(i,jj,k,9) 
             wstat(i,jj,k,3)  = wstat(i,jj,k,3)/wstat(i,jj,k,9) 
             wstat(i,jj,k,4)  = wstat(i,jj,k,4)/wstat(i,jj,k,9) 
             wstat(i,jj,k,5)  = wstat(i,jj,k,5)/wstat(i,jj,k,9) 
             wstat(i,jj,k,6)  = wstat(i,jj,k,6)/wstat(i,jj,k,9) 
             wstat(i,jj,k,7)  = wstat(i,jj,k,7)/wstat(i,jj,k,9) 
             wstat(i,jj,k,8)  = wstat(i,jj,k,8) 
             wstat(i,jj,k,9)  = wstat(i,jj,k,9)
             wstat(i,jj,k,10) = wstat(i,jj,k,10)/wstat(i,jj,k,9)

             wstat(i,jj,k,11) = sqrt(wstat(i,jj,k,11)/wstat(i,jj,k,9)-wstat(i,jj,k,1)**2)
             wstat(i,jj,k,12) = sqrt(wstat(i,jj,k,12)/wstat(i,jj,k,9)-wstat(i,jj,k,2)**2)
             wstat(i,jj,k,13) = sqrt(wstat(i,jj,k,13)/wstat(i,jj,k,9)-wstat(i,jj,k,3)**2)
             wstat(i,jj,k,14) = sqrt(wstat(i,jj,k,14)/wstat(i,jj,k,9)-wstat(i,jj,k,4)**2)
             wstat(i,jj,k,15) = sqrt(wstat(i,jj,k,15)/wstat(i,jj,k,9)-wstat(i,jj,k,5)**2)
             wstat(i,jj,k,16) = sqrt(wstat(i,jj,k,16)/wstat(i,jj,k,9)-wstat(i,jj,k,6)**2)
             wstat(i,jj,k,17) = sqrt(wstat(i,jj,k,17)/wstat(i,jj,k,9)-wstat(i,jj,k,7)**2)
             wstat(i,jj,k,18) = sqrt(wstat(i,jj,k,18)-wstat(i,jj,k,8)**2)
             wstat(i,jj,k,19) = sqrt(wstat(i,jj,k,19)-wstat(i,jj,k,9)**2)
             wstat(i,jj,k,20) = sqrt(wstat(i,jj,k,20)/wstat(i,jj,k,9)-wstat(i,jj,k,10)**2)
            enddo
           enddo
          enddo

          nimax = nxmax
          njmax = 1 
          nkmax = nzmax 
          ni    = nx
          nj    = 1
          nk    = nz
          file_prefix = "stats_slicexz_??????"
          do jj=1,jnum
           j_slice = jslice(jj) 
           write(file_prefix(15:20),'(I6.6)') j_slice + ncoords(2)*ny
           call write_vtk_general(file_prefix,nimax,njmax,nkmax,ni,nj,nk,1,ni,jj,jj,1,nk,mp_cart,masterproc,&
                                 wstat(1:nx,jj:jj,1:nz,1:nvstats),nvstats)
          enddo
          deallocate(wstat)

1003 format(I3.3)
     endsubroutine postpro_slice_xz


     subroutine write_vtk_general(file_prefix,nimax, njmax, nkmax, ni, nj, nk, i1, i2, j1, j2, k1, k2, &
                                  mp_cart, my_master, w_slice_io, nv_io)
             integer none
             integer, parameter :: int64_kind = selected_int_kind(2*range(1))
             integer, intent(in) :: nimax, njmax, nkmax, ni, nj, nk, i1, i2, j1, j2, k1, k2, mp_cart, nv_io
             logical, intent(in) :: my_master
             real(rkind), dimension(i1:i2,j1:j2,k1:k2,1:nv_io), intent(in) :: w_slice_io
             character(len=20), intent(in) :: file_prefix

             integer :: ntot,mpi_err,mpi_io_file,filetype,size_real,l
             integer :: ig1, jg1, kg1, ig2, jg2, kg2
             integer(int64_kind) :: gridsize_64
             integer,dimension(3) :: sizes     ! Dimensions of the total grid
             integer,dimension(3) :: subsizes  ! Dimensions of grid local to a procs
             integer,dimension(3) :: starts    ! Starting coordinates
             integer (kind=mpi_offset_kind) :: offset
             integer (kind=mpi_offset_kind) :: offset_x,offset_y,offset_z,delta_offset_w
             integer, dimension(mpi_status_size) :: istatus
             character(len=    7) :: vtk_float
             character(len=  100) :: namefile
             character(len=65536) :: xml_part

             namefile = trim(file_prefix)//'.vtr'
             if (my_master) write(*,*) 'writing ', namefile
               
             ig1 = 1
             jg1 = 1
             kg1 = 1
             ig2 = nimax
             jg2 = njmax
             kg2 = nkmax
             if (i1==i2) then
              ig1 = islice(i1) + ncoords(1)*nx
              ig2 = ig1
             endif
             if (j1==j2) then
              jg1 = jslice(j1) + ncoords(2)*ny
              jg2 = jg1
             endif
             if (k1==k2) then
              kg1 = kslice(k1) + ncoords(3)*nz
              kg2 = kg1
             endif
             sizes(1)    = ig2-ig1+1
             sizes(2)    = jg2-jg1+1
             sizes(3)    = kg2-kg1+1
             subsizes(1) = i2-i1+1
             subsizes(2) = j2-j1+1
             subsizes(3) = k2-k1+1
             starts(1)   = 0 + ncoords(1)*subsizes(1)
             starts(2)   = 0 + ncoords(2)*subsizes(2)
             starts(3)   = 0 + ncoords(3)*subsizes(3)
             ntot        = subsizes(1)*subsizes(2)*subsizes(3)
             if (i1==i2) starts(1) = 0
             if (j1==j2) starts(2) = 0
             if (k1==k2) starts(3) = 0
 
             call MPI_TYPE_SIZE(mpi_prec,size_real,mpi_err)
             if (size_real == 4) then
              vtk_float = "Float32"
             elseif (size_real == 8) then
              vtk_float = "Float64"
             else
              if (my_master) write(*,*) "Error on VTK write! size_real must be either 4 or 8"
               call MPI_ABORT(MPI_COMM_WORLD,mpi_err,mpi_err)
             endif

             gridsize_64 = int(size_real,int64_kind)*int(nimax,int64_kind)*int(njmax,int64_kind)*&
                           int(nkmax,int64_kind)
             if(storage_size(gridsize_64) /= 64) then
              if(my_master) write(*,*) "Error on VTK write! Size of int64_kind integers is not 8 bytes!"
              call MPI_ABORT(MPI_COMM_WORLD, mpi_err, mpi_err)
             endif
               
             if (my_master) then
              offset_x = 0
              offset_y = offset_x + size_real*nimax + storage_size(gridsize_64)/8
              offset_z = offset_y + size_real*njmax + storage_size(gridsize_64)/8
              delta_offset_w = gridsize_64 + storage_size(gridsize_64)/8 ! the second part is because of the header of bytes before data

               open(unit=123, file=trim(namefile), access="stream", form="unformatted", status="replace")
               xml_part = ' &
               & <?xml version="1.0"?> &
               & <VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64"> &
               &  <RectilinearGrid WholeExtent="+1 +'&
               &   //int2str(nimax)//' +1 +'//int2str(njmax)//' +1 +'//int2str(nkmax)//'"> &
               &   <Piece Extent="+1 +'//int2str(nimax)//&
               &    ' +1 +'//int2str(njmax)//' +1 +'//int2str(nkmax)//'"> &
               &    <Coordinates> &
               &     <DataArray type="'//vtk_float//'" NumberOfComponents="1" Name="X" format="appended" offset="'//&
               &       int2str_o(offset_x)//'"/> &
               &     <DataArray type="'//vtk_float//'" NumberOfComponents="1" Name="Y" format="appended" offset="'//&
               &       int2str_o(offset_y)//'"/> &
               &     <DataArray type="'//vtk_float//'" NumberOfComponents="1" Name="Z" format="appended" offset="'//&
               &       int2str_o(offset_z)//'"/> &
               &    </Coordinates> &
               &   <PointData> '
                offset = offset_z + size_real*nkmax + storage_size(gridsize_64)/8
                do l=1,nv_io
                 !ll = list_aux_slice(l)
                 xml_part = trim(adjustl(xml_part)) // ' &
                & <DataArray type="'//vtk_float//'" NumberOfComponents="1" Name="'//trim(var_names(l))//'" format="appended" &
                &  offset="'//int2str_o(offset)//'"/>'
                 offset = offset + delta_offset_w
                enddo
                xml_part = trim(adjustl(xml_part)) // ' &
                &       </PointData> &
                &     </Piece> &
                &   </RectilinearGrid> &
                &   <AppendedData encoding="raw"> '
                write(123) trim(adjustl(xml_part))

                write(123) "_"

                write(123) size_real*int(nimax,int64_kind) , xg(ig1:ig2)
                write(123) size_real*int(njmax,int64_kind) , yg(jg1:jg2)
                write(123) size_real*int(nkmax,int64_kind) , zg(kg1:kg2)
                flush(123)
               close(123)
              endif 

              call MPI_TYPE_CREATE_SUBARRAY(3,sizes,subsizes,starts,MPI_ORDER_FORTRAN,mpi_prec,filetype,mpi_err)
              call MPI_TYPE_COMMIT(filetype,mpi_err)

              do l=1,nv_io
               call MPI_BARRIER(mp_cart,mpi_err)
               if (my_master) then
                open(unit=123, file=trim(namefile), access="stream", form="unformatted", position="append")
                write(123) gridsize_64
                flush(123)
                close(123)
               endif
               call MPI_BARRIER(mp_cart, mpi_err)
               call MPI_FILE_OPEN(mp_cart,trim(namefile),MPI_MODE_RDWR,MPI_INFO_NULL,mpi_io_file,mpi_err)
               call MPI_FILE_GET_SIZE(mpi_io_file, offset, mpi_err)
               call MPI_BARRIER(mp_cart, mpi_err)
               call MPI_FILE_SET_VIEW(mpi_io_file,offset,mpi_prec,filetype,"native",MPI_INFO_NULL,mpi_err)
               call MPI_FILE_WRITE_ALL(mpi_io_file,w_slice_io(i1:i2,j1:j2,k1:k2,l),ntot,mpi_prec,istatus,mpi_err)
              enddo

              call MPI_FILE_CLOSE(mpi_io_file,mpi_err)
              call MPI_TYPE_FREE(filetype,mpi_err)
              if (my_master) then
               open(unit=123, file=trim(namefile), access="stream", position="append", form="unformatted")
               write(123) ' &
                 &    </AppendedData> &
                 &  </VTKFile>'
               close(123)
              endif

     endsubroutine write_vtk_general


endmodule postpro_slice

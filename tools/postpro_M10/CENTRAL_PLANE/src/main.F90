program central_plane
        use parameters 
        implicit none
        integer :: N_S, nxsmax, nysmax, nvar, m, num_xstations, num_ystations, i, ii, j, jj, k
        integer, dimension(:), allocatable :: x_stations, y_stations
        real(rkind), dimension(:,:,:,:), allocatable :: w_slice_sym
        real(rkind), dimension(:), allocatable :: xg,yg,zg
        character(6)  :: chmean
        character(32)  :: file_prefix

        print *, 'Starting central plane postpro...'
        call execute_command_line('mkdir -p PROFILES/')
        print *, 'Reading central_plane_infty_norm.bin...'
        open(10, file='central_plane_infty_norm.bin', &
                 action='read', status='old', form='unformatted')
        ! reading header
        read(10) nxsmax, nysmax, nvar
        ! nvar is 6+N_S
        N_S = 5

        allocate(w_slice_sym(nvar,nxsmax,nysmax,1))
        allocate(xg(nxsmax),yg(nysmax),zg(1))

        read(10) xg
        read(10) yg
        zg = 0._rkind
 
        do m=1,nvar
         read(10) w_slice_sym(m,:,:,1) 
        enddo
        close(10)

        print *, 'End reading central_plane_infty_norm.bin...'

        file_prefix = 'central_slice'
        
        call write_vtk(nxsmax, nysmax, 1 , nvar, w_slice_sym, xg, yg, zg, file_prefix)
!        open(12, file='./input_central_plane.dat', action='read', status='old')
!        read(12,*) 
!        read(12,*) num_xstations,num_ystations
!        read(12,*) 
!        allocate(x_stations(num_xstations),y_stations(num_ystations))
!        do k=1,num_xstations
!         read(12,*) x_stations(k)
!        enddo
!        read(12,*)
!        do k=1,num_ystations
!         read(12,*) y_stations(k)
!        enddo
!
!        do k=1,num_xstations
!         write(chmean,1006) x_stations(k)
!         write(*,*) 'writing mean profile,',k,chmean 
!         open(unit=15,file='PROFILES/stat_i_'//chmean//'.prof')
!         ii = x_stations(k)
!         do j=1,nysmax
!          write(15,100) yg(j), (w_slice_sym(m,ii,j),m=1,nvar)
!         enddo
!         close(15)
!        enddo
!
!        do k=1,num_ystations
!         write(chmean,1006) y_stations(k)
!         write(*,*) 'writing mean profile,',k,chmean
!         open(unit=15,file='PROFILES/stat_j_'//chmean//'.prof')
!         jj = y_stations(k)
!         do i=1,nxsmax
!          write(15,100) xg(i), (w_slice_sym(m,i,jj),m=1,nvar)
!         enddo
!         close(15)
!        enddo
  100  format(200ES20.10) 
  1006 format(I6.6)

  contains
    subroutine write_vtk(nxmax, nymax, nzmax, nv_io, w_aux_io, xg, yg, zg, file_prefix)
!   Writing field (MPI I/O)
    integer, intent(in) :: nv_io
    character(32) :: file_prefix
    real(rkind), dimension(:,:,:,:), allocatable :: w_io_
    real(rkind), dimension(:,:,:,:) :: w_aux_io
    real(rkind), dimension(:) :: xg, yg, zg
    integer :: l
    integer :: mpi_io_file
    integer :: filetype
    integer :: size_real
    integer :: ntot
    integer (kind=8) :: offset
    integer (kind=8) :: offset_x,offset_y,offset_z,delta_offset_w
    character(4) :: chstore
    character(2) :: chvar
!   integer, dimension(mpi_status_size) :: istatus
    integer, parameter :: int64_kind = selected_int_kind(2*range(1))
    integer(int64_kind) :: gridsize_64
!   character(len=16) :: int2str
!   character(len=32) :: int2str_o
    character(len=65536) :: xml_part
    character(len=7) :: vtk_float
    character(len=12), dimension(:), allocatable :: names
    integer :: mpi_err
    integer :: nxmax, nymax, nzmax
!
    nxmax = size(w_aux_io, 2)
    nymax = size(w_aux_io, 3)
    nzmax = size(w_aux_io, 4)

    print *, 'Storing VTK sol'
!
!   call MPI_TYPE_SIZE(mpi_prec,size_real,self%mpi_err)
    size_real = 8
!
    if(size_real == 4) then
      vtk_float = "Float32"
    elseif(size_real == 8) then
      vtk_float  = "Float64"
    else
      write(*,*) "Error on VTK write! size_real must be either 4 or 8"
      STOP
    endif
    gridsize_64 = int(size_real,int64_kind)*int(nxmax,int64_kind)*int(nymax,int64_kind)*&
    int(nzmax,int64_kind)
!
    if(storage_size(gridsize_64) /= 64) then
      write(*,*) "Error on VTK write! Size of int64_kind integers is not 8 bytes!"
      STOP
    endif
!
    allocate(w_io_(nxmax, nymax, nzmax, nv_io))
    do l = 1, nv_io
      w_io_(:,:,:,l) = w_aux_io(l,:,:,:)
    enddo
!
!    names_dns = [character(12) :: "density", "velocity_x", "velocity_y", "velocity_z", &
!            "enthalpy", "temperature", "visc", "cond", "sensor", "div3"]
!    names_les = [character(12) :: "density", "velocity_x", "velocity_y", "velocity_z", &
!            "enthalpy", "temperature", "visc_tot", "cond_tot", "sensor", "div3_rhokmu", &
!            "visc_sgs", "rhokmu"]

    allocate(names(nv_io))
    names = ["Density     ", "Velocity_x  ", "Velocity_y  ", &
           & "Velocity_z  ", "Temperature ", "Density2    ", &
           & "tau11       ", "tau22       ", "tau33       ", "tau12       ", "tau13       ", "tau23       ", &
           & "Trms        ", "Pressure    ", "Prms        ", &
           & "Y1          ", "Y2          ", "Y3          ", "Y4          ", "Y5          ", &
           & "Y1_rms      ", "Y2_rms      ", "Y3_rms      ", "Y4_rms      ", "Y5_rms      "]
!    do l = 1, nv_io
!      if (enable_les > 0) then
!        names(l) = names_les(list_aux_slice(l))
!      else
!        names(l) = names_dns(list_aux_slice(l))
!      endif
!       write(chvar, '(I2.2)') l
!       names(l) = trim('var_'//chvar)
!    enddo
!
!
    offset_x = 0
    offset_y = size_real*nxmax + storage_size(gridsize_64)/8
    offset_z = offset_y + size_real*nymax + storage_size(gridsize_64)/8
    delta_offset_w = gridsize_64 + storage_size(gridsize_64)/8 ! the second part is because of the header of bytes before data
!
    open(unit=123, file=trim(file_prefix)//'.vtr', access="stream", form="unformatted", status="replace")
    xml_part = ' &
    & <?xml version="1.0"?> &
    & <VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64"> &
    &  <RectilinearGrid WholeExtent="+1 +'&
    &   //int2str(nxmax)//' +1 +'//int2str(nymax)//' +1 +'//int2str(nzmax)//'"> &
    &   <Piece Extent="+1 +'//int2str(nxmax)//&
    &    ' +1 +'//int2str(nymax)//' +1 +'//int2str(nzmax)//'"> &
    &    <Coordinates> &
    &     <DataArray type="'//vtk_float//'" NumberOfComponents="1" Name="X" format="appended" offset="'//&
    &       int2str_o(offset_x)//'"/> &
    &     <DataArray type="'//vtk_float//'" NumberOfComponents="1" Name="Y" format="appended" offset="'//&
    &       int2str_o(offset_y)//'"/> &
    &     <DataArray type="'//vtk_float//'" NumberOfComponents="1" Name="Z" format="appended" offset="'//&
    &       int2str_o(offset_z)//'"/> &
    &    </Coordinates> &
    &   <PointData> '
    offset = offset_z + size_real*nzmax + storage_size(gridsize_64)/8
    do l=1,nv_io
      xml_part = trim(adjustl(xml_part)) // ' &
      & <DataArray type="'//vtk_float//'" NumberOfComponents="1" Name="'//trim(names(l))//'" format="appended" &
      &  offset="'//int2str_o(offset)//'"/>'
      offset = offset + delta_offset_w
    enddo
    xml_part = trim(adjustl(xml_part)) // ' &
    &       </PointData> &
    &     </Piece> &
    &   </RectilinearGrid> &
    &   <AppendedData encoding="raw"> '
    write(123) trim(adjustl(xml_part))
!
    write(123) "_"
!   write(123) storage_size(gridsize_64)/8*int(nxmax,int64_kind) , xg(1:nxmax)
!   write(123) storage_size(gridsize_64)/8*int(nymax,int64_kind) , yg(1:nymax)
!   write(123) storage_size(gridsize_64)/8*int(nzmax,int64_kind) , zg(1:nzmax)
    write(123) size_real*int(nxmax,int64_kind) , xg(1:nxmax)
    write(123) size_real*int(nymax,int64_kind) , yg(1:nymax)
    write(123) size_real*int(nzmax,int64_kind) , zg(1:nzmax)
    flush(123)
    close(123)
!
!   -----------------------------------------------------------------
!   SERIAL WRITE
!   -----------------------------------------------------------------
    open(unit=123, file=trim(file_prefix)//'.vtr', access="stream", form="unformatted", position="append")
    do l=1,nv_io
      write(123) gridsize_64
      write(123) w_io_(:,:,:, l)
    enddo
    close(123)
    open(unit=123, file=trim(file_prefix)//'.vtr', access="stream", position="append", form="unformatted")
    write(123) ' &
      &    </AppendedData> &
      &  </VTKFile>'
    close(123)
  end subroutine write_vtk

    function int2str(int_num)
    implicit none
    integer :: int_num
    character(len=16) :: int2str, ret_value
    write(ret_value, "(I0)") int_num
    int2str = ret_value
  endfunction int2str
!
  function int2str_o(int_num)
    implicit none
    integer(KIND=8) :: int_num
    character(len=32) :: int2str_o, ret_value
    write(ret_value, "(I0)") int_num
    int2str_o = ret_value
  endfunction int2str_o
        
endprogram

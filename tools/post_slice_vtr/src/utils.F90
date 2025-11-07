module join_utils
  use parameters
contains
  subroutine write_vtk(time, istore, w_aux_io, xg, yg, zg, file_prefix, enable_les, list_aux_slice)
!   Writing field (MPI I/O)
    real(rkind), optional :: time
    real(rkind) ::  time_
    integer :: istore
    character(32) :: file_prefix
    real(rkind), dimension(:,:,:,:), allocatable :: w_io_
    real(rkind), dimension(:,:,:,:) :: w_aux_io
    real(rkind), dimension(:) :: xg, yg, zg
    integer :: nv_io
    integer :: l
    integer :: mpi_io_file
    integer :: filetype
    integer :: enable_les
    integer, dimension(:), allocatable :: list_aux_slice
    integer,dimension(3) :: sizes     ! Dimensions of the total grid
    integer,dimension(3) :: subsizes  ! Dimensions of grid local to a procs
    integer,dimension(3) :: starts    ! Starting coordinates
    integer :: size_real
    integer :: ntot
    integer (kind=8) :: offset
    integer (kind=8) :: offset_x,offset_y,offset_z,delta_offset_w
    character(4) :: chstore
!   integer, dimension(mpi_status_size) :: istatus
    integer, parameter :: int64_kind = selected_int_kind(2*range(1))
    integer(int64_kind) :: gridsize_64
!   character(len=16) :: int2str
!   character(len=32) :: int2str_o
    character(len=65536) :: xml_part
    character(len=7) :: vtk_float
    character(len=12), dimension(10) :: names_dns
    character(len=12), dimension(12) :: names_les
    character(len=12), dimension(:), allocatable :: names
    integer :: mpi_err
    integer :: nxmax, nymax, nzmax
!
    nxmax = size(w_aux_io, 1)
    nymax = size(w_aux_io, 2)
    nzmax = size(w_aux_io, 3)
!
    write(chstore(1:4), '(I4.4)') istore
!
    time_        = 1._rkind ; if(present(time))        time_        = time
    print *, 'Storing VTK sol', istore,'at time', time
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
    nv_io = size(list_aux_slice)
!   w_io_ = self%w(:,1:self%nx,1:self%ny,1:self%nz)
!   allocate(w_io_, mold=w_aux_io)
    allocate(w_io_(nxmax, nymax, nzmax, nv_io))
    do l = 1, nv_io
      w_io_(:,:,:,l) = w_aux_io(:,:,:,l)
    enddo
!
    names_dns = [character(12) :: "density", "velocity_x", "velocity_y", "velocity_z", & 
            "enthalpy", "temperature", "visc", "cond", "sensor", "div3"]
    names_les = [character(12) :: "density", "velocity_x", "velocity_y", "velocity_z", & 
            "enthalpy", "temperature", "visc_tot", "cond_tot", "sensor", "div3_rhokmu", & 
            "visc_sgs", "rhokmu"]

    allocate(names(nv_io))
    do l = 1, nv_io
      if (enable_les > 0) then
        names(l) = names_les(list_aux_slice(l))
      else
        names(l) = names_dns(list_aux_slice(l))
      endif
    enddo
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
!   -----------------------------------------------------------------
!
!   !! !    -----------------------------------------------------------------
!   !! !    MPI WRITE
!   !! !    -----------------------------------------------------------------
!   !!      sizes(1) = self%nblocks(1)*self%nx
!   !!      sizes(2) = self%nblocks(2)*self%ny
!   !!      sizes(3) = self%nblocks(3)*self%nz
!   !!      subsizes(1) = self%nx
!   !!      subsizes(2) = self%ny
!   !!      subsizes(3) = self%nz
!   !!      starts(1) = 0 + self%ncoords(1)*subsizes(1)
!   !!      starts(2) = 0 + self%ncoords(2)*subsizes(2)
!   !!      starts(3) = 0 + self%ncoords(3)*subsizes(3)
!   !!      ntot = self%nx*self%ny*self%nz
!   !!         !
!   !!      call MPI_TYPE_CREATE_SUBARRAY(3,sizes,subsizes,starts,MPI_ORDER_FORTRAN,mpi_prec,filetype,self%mpi_err)
!   !!      call MPI_TYPE_COMMIT(filetype,self%mpi_err)
!   !!      !
!   !!      do l=1,nv_io
!   !!       call MPI_BARRIER(self%mp_cart,self%mpi_err)
!   !!       if (self%masterproc) then
!   !!        open(unit=123, file=trim(file_prefix)//'.vtr', access="stream", form="unformatted", position="append")
!   !!         write(123) gridsize_64
!   !!         flush(123)
!   !!        close(123)
!   !!       endif
!   !!       call MPI_BARRIER(self%mp_cart, self%mpi_err)
!   !!       call MPI_FILE_OPEN(self%mp_cart,trim(file_prefix_)//'.vtr',MPI_MODE_RDWR,MPI_INFO_NULL,mpi_io_file,self%mpi_err)
!   !!       call MPI_FILE_GET_SIZE(mpi_io_file, offset, self%mpi_err)
!   !!       call MPI_BARRIER(self%mp_cart, self%mpi_err)
!   !!       call MPI_FILE_SET_VIEW(mpi_io_file,offset,mpi_prec,filetype,"native",MPI_INFO_NULL,self%mpi_err)
!   !!       call MPI_FILE_WRITE_ALL(mpi_io_file,w_io_(l,1:self%nx,1:self%ny,1:self%nz),ntot,mpi_prec,istatus,self%mpi_err)
!   !!       call MPI_FILE_CLOSE(mpi_io_file,self%mpi_err)
!   !!      enddo
!   !!
!   !!      call MPI_TYPE_FREE(filetype,self%mpi_err)
!   !! !    -----------------------------------------------------------------
!
    open(unit=123, file=trim(file_prefix)//'.vtr', access="stream", position="append", form="unformatted")
    write(123) ' &
      &    </AppendedData> &
      &  </VTKFile>'
    close(123)
  end subroutine write_vtk
!
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
endmodule join_utils

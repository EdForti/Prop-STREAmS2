!
! Interpolation tool for the compressible channel case in the x and z directions. Assume periodicity in x and z.
! Read rst.bin from a previous grid (dxg.dat, dyg.dat and dzg,dat) and write rstnew.bin.
! The code is parallelized only in the y direction (number of partitions must divide the total number of points in y)
!
! Input parameters in the file:
!
! nblocks(1) = 1   only 1 supported
! nblocks(2) = 80  number of partitions in y
! nblocks(3) = 1   only 1 supported
! nxmaxnew = 9216  number of x points for the new mesh
! nzmaxnew = 6750  number of z points for the new mesh
! mord = 2         order of accuracy for interpolation
!
module uty
 use, intrinsic :: iso_fortran_env
 use, intrinsic :: ieee_arithmetic
 implicit none
 integer, parameter :: rkind = REAL64
 contains
!
 subroutine locateval(xx,n,x,ii)
! 
  integer, intent(in) :: n
  integer, intent(out) :: ii
  real(rkind), intent(in) :: x
  real(rkind), dimension(1:n), intent(in) :: xx
  integer :: il,jm,juu
!
  il=0
  juu=n+1
  do while (juu-il.gt.1)
   jm=(juu+il)/2
   if ((xx(n).gt.xx(1)).eqv.(x.gt.xx(jm))) then
    il=jm
   else
    juu=jm
   endif
  end do
  ii=il
  return
!
 end subroutine locateval
!
 subroutine pol_int(x,y,n,xs,ys)
!
! Polynomial interpolation using Neville's algorithm
! Order of accuracy of the interpolation is n-1
!
  integer, intent(in) :: n
  real(rkind), dimension(n), intent(in) :: x,y
  real(rkind), intent(in) :: xs
  real(rkind), intent(out) :: ys
  integer :: i,m
  real(rkind), dimension(n)  :: v,vold
!
  v = y
!
  do m=2,n ! Tableu columns
   vold = v
   do i=1,n+1-m
    v(i) = (xs-x(m+i-1))*vold(i)+(x(i)-xs)*vold(i+1)
    v(i) = v(i)/(x(i)-x(i+m-1))
   enddo
  enddo
  ys = v(1)
!
  return
 end subroutine pol_int
!
end module uty

program xyzinterpolation
! use, intrinsic :: iso_fortran_env
! use, intrinsic :: ieee_arithmetic
  use uty
!
! Interpolation of restart file for STREAmS-2
!
  implicit none
!
! integer, parameter :: rkind = REAL64
  integer, parameter :: N_S = 5 ! Number of conservative variables
  integer, parameter :: nv = 9 ! Number of conservative variables
  integer, parameter :: ng = 3 ! Number of ghost nodes for interpolation
!
! mpi-related parameters
  real(rkind), allocatable, dimension(:,:,:,:) :: w,wx,wxz
  real(rkind), allocatable, dimension(:) :: xg,yg,zg
  real(rkind), allocatable, dimension(:) :: xgnew,zgnew
  real(rkind) :: rho,uu,vv,ww,ee,ri,rhou,rhov,rhow,rhoe,qq
  real(rkind) :: eps_shift
  real(rkind) :: xxnew,zznew,wint
  integer :: mord
!
  integer, parameter :: ndims=3
  logical :: reord
!
  integer :: ireadstat
  integer :: nxmax,nymax,nzmax,nx,ny,nz
  integer :: nxmaxnew,nzmaxnew,nxnew,nznew
  integer :: istat,i,j,k,ii,kk,ip,kp,l,lsp
  real(rkind) :: lx,lz,dx,dz,dxnew,dznew
!
  nxmaxnew = 14800
  nzmaxnew = 640
  mord = 1
!
  open(12,file='dxg.dat',form='formatted')
  nxmax = 0
  do
   read(12,*,iostat=ireadstat) 
   if (ireadstat/=0) exit
   nxmax = nxmax+1
  enddo
  close(12)
  open(12,file='dyg.dat',form='formatted')
  nymax = 0
  do
   read(12,*,iostat=ireadstat) 
   if (ireadstat/=0) exit
   nymax = nymax+1
  enddo
  close(12)
  open(12,file='dzg.dat',form='formatted')
  nzmax = 0
  do
   read(12,*,iostat=ireadstat) 
   if (ireadstat/=0) exit
   nzmax = nzmax+1
  enddo
  close(12)
!
  allocate(xg(nxmax),yg(nymax),zg(nzmax))
!
  open(12,file='dxg.dat',form='formatted')
  do i=1,nxmax
   read(12,*) xg(i)
  enddo
  close(12)
  open(12,file='dyg.dat',form='formatted')
  do j=1,nymax
   read(12,*) yg(j)
  enddo
  close(12)
  open(12,file='dzg.dat',form='formatted')
  do k=1,nzmax
   read(12,*) zg(k)
  enddo
  close(12)
!
  dx = xg(2)-xg(1)
  dz = zg(2)-zg(1)
  lx = xg(nxmax)-xg(1)+dx
  lz = zg(nzmax)-zg(1)+dz
  write(*,*) 'nxmax,nymax,nzmax:', nxmax,nymax,nzmax
  write(*,*) 'lx,lz:', lx,lz
  dxnew = lx/nxmaxnew
  dznew = lz/nzmaxnew
  allocate(xgnew(nxmaxnew),zgnew(nzmaxnew))
  eps_shift = tiny(1._rkind)
  do i=1,nxmaxnew
   xgnew(i) = eps_shift+(i-1)*dxnew
  enddo
  do k=1,nzmaxnew
   zgnew(k) = eps_shift+(k-1)*dznew
  enddo
  write(*,*) 'nxmaxnew,nzmaxnew:', nxmaxnew,nzmaxnew
  write(*,*) xg(1), lx
  write(*,*) xgnew(1), xgnew(nxmaxnew)
  write(*,*) zg(1), lz
  write(*,*) zgnew(1), zgnew(nzmaxnew)
!
  nx = nxmax
  ny = nymax
  nz = nzmax
  nxnew = nxmaxnew
  nznew = nzmaxnew
!
  allocate(w(nv,nxmax,nymax,nzmax))
!
  write(*,*) 'Reading rst.bin'
!
  open(10, file='rst.bin', access='stream', form='unformatted', status='old', iostat=istat)

  if (istat /= 0) then
   print *, "File open error! iostat =", istat
   stop
  end if

  do l=1,nv
   write(*,*) l
   read(10, iostat=istat) w(l,1:nxmax,1:nymax,1:nzmax)
   if (istat /= 0) then
        print *, "Read error at variable", l, "iostat =", istat
        stop
    end if
  enddo
!
  close(10)
  write(*,*) 'Done'
!
!  if (masterproc) write(*,*) 'Compute primitive variables'
!!
!  do k=1,nz
!  if (masterproc) write(*,*) 'k', k
!   do j=1,ny
!    do i=1,nx!+1
!     ii = i
!     !if (i==nx+1) ii = 1 ! x periodicity for w
!     rho  = 0._rkind
!     do lsp=1,N_S
!      rho = rho + w(lsp,ii,j,k)
!     enddo
!     rhou = w(N_S+1,ii,j,k)
!     rhov = w(N_S+2,ii,j,k)
!     rhow = w(N_S+3,ii,j,k)
!     rhoe = w(N_S+4,ii,j,k)
!     ri   = 1._rkind/rho
!     uu   = rhou*ri
!     vv   = rhov*ri
!     ww   = rhow*ri
!     qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
!     ee   = rhoe/rho-qq
!     w(N_S+1,i,j,k) = uu
!     w(N_S+2,i,j,k) = vv
!     w(N_S+3,i,j,k) = ww
!     w(N_S+4,i,j,k) = ee
!    enddo
!   enddo
!  enddo
!

  allocate(wx(nv,nxnew,ny,nz+1))
  write(*,*) 'Interpolate along x'
!
  do k=1,nz+1
   kk = k
   if (k==nz+1) kk = 1
   do j=1,ny
    do i=1,nxnew
     xxnew = xgnew(i)
     call locateval(xg,nx,xxnew,ii) ! xxnew è traxgcii) e xg(ii+1)
     if (ii==0) write(*,*) 'Warning ii = 0'
     do l=1,nv
      call pol_int(xg(ii:ii+mord-1),w(l,ii:ii+mord-1,j,kk),mord,xxnew,wint)
      wx(l,i,j,k) = wint
     enddo
    enddo
   enddo
  enddo
!
  allocate(wxz(nv,nxnew,ny,nznew))
  write(*,*) 'Interpolate along z'
! 
  do k=1,nznew
   zznew = zgnew(k)
   call locateval(zg,nz,zznew,kk)
   do j=1,ny
    do i=1,nxnew
     do l=1,nv
      call pol_int(zg(kk:kk+mord-1),wx(l,i,j,kk:kk+mord-1),mord,zznew,wint)
      wxz(l,i,j,k) = wint
     enddo
    enddo
   enddo
  enddo
!
 ! if (masterproc) write(*,*) 'Return to conservative variables'
!!
 ! do k=1,nznew
 !  do j=1,ny
 !   do i=1,nxnew
 !    rho  = rho
 !    do lsp=1,N_S
 !     rho = rho + wxz(lsp,i,j,k)
 !    enddo
 !    uu   = wxz(N_S+1,i,j,k)
 !    vv   = wxz(N_S+2,i,j,k)
 !    ww   = wxz(N_S+3,i,j,k)
 !    ee   = wxz(N_S+4,i,j,k)
 !    qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
 !    rhou = rho*uu
 !    rhov = rho*vv
 !    rhow = rho*ww
 !    rhoe = rho*(ee+qq)
 !    wxz(N_S+1,i,j,k) = rhou
 !    wxz(N_S+2,i,j,k) = rhov
 !    wxz(N_S+3,i,j,k) = rhow
 !    wxz(N_S+4,i,j,k) = rhoe
 !   enddo
 !  enddo
 ! enddo
!
!  if (masterproc) write(*,*) 'Writing rstnew.bin'
!!  
!  sizes(1) = nblocks(1)*nxnew
!  sizes(2) = nblocks(2)*ny
!  sizes(3) = nblocks(3)*nznew
!  subsizes(1) = nxnew
!  subsizes(2) = ny
!  subsizes(3) = nznew
!  starts(1) = 0 + ncoords(1)*subsizes(1)
!  starts(2) = 0 + ncoords(2)*subsizes(2)
!  starts(3) = 0 + ncoords(3)*subsizes(3)
!  ntot = nxnew*ny*nznew
!!
!  call mpi_type_create_subarray(3,sizes,subsizes,starts,mpi_order_fortran,mpi_prec,filetype,ierr)
!  call mpi_type_commit(filetype,ierr)
!  call mpi_file_open(mp_cart,'rstnew.bin',mpi_mode_create+mpi_mode_wronly,mpi_info_null,mpi_io_file,ierr)
!  offset = 0
!  do l=1,nv
!   if (masterproc) write(*,*) l
!   call mpi_file_set_view(mpi_io_file,offset,mpi_prec,filetype,"native",mpi_info_null,ierr)
!   call mpi_file_write(mpi_io_file,wxz(l,1:nxnew,1:ny,1:nznew),ntot,mpi_prec,istatus,ierr)
!   call mpi_type_size(mpi_prec,size_real,ierr)
!   do m=1,nblocks(1)*nblocks(2)*nblocks(3)
!    offset = offset+size_real*ntot
!   enddo
!  enddo
!!
!  call mpi_file_close(mpi_io_file,ierr)
!  call mpi_type_free(filetype,ierr)
!!
!if (masterproc) write(*,*) 'Done'
!!
!  deallocate(w,wx,wxz)
!!
!  call mpi_finalize(ierr)
 end program xyzinterpolation

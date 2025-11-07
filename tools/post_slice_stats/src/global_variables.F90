#include "index_define.h"
module global_variables

use parameters
!
implicit none
integer                                    :: nv,nv_slice,nvstats 
!
integer                                    :: nxmax,nymax,nzmax,nx,ny,nz,ng,ystag
integer                                    :: grid_dim
integer                                    :: flow_init
integer                                    :: enable_les,enable_chemistry
integer                                    :: nstatloc
integer                                    :: stat_0_1
!
real(rkind)                                :: Reynolds,Prandtl,rfac,Mach,theta_wall,Reynolds_friction,Mach_input
real(rkind)                                :: u0,rho0,p0,t0,Twall,mu0,k0,cp0,cv0,l0,c0,gam
real(rkind), dimension(:), allocatable     :: xg,yg,zg
real(rkind), dimension(:,:,:), allocatable :: wstat,wstatb
real(rkind), dimension(:,:), allocatable   :: wstattemp,vtemp
integer, dimension(:), allocatable         :: ixstat
!
real(rkind)                                :: R_curv
real(rkind), dimension(:,:), allocatable   :: jac,csimod,etamod,mcsi,meta
real(rkind), dimension(:,:), allocatable   :: dxdcsi ,dydcsi ,dxdeta ,dydeta
real(rkind), dimension(:,:), allocatable   :: dxdcsin,dydcsin,dxdetan,dydetan
real(rkind), dimension(:,:), allocatable   :: dcsidx ,dcsidy ,detadx ,detady
real(rkind), dimension(:,:), allocatable   :: dcsidxn,dcsidyn,detadxn,detadyn
real(rkind), dimension(:,:,:), allocatable :: wstatz
integer                                    :: ile,ite,itu
real(rkind)                                :: alpha_airfoil

integer                                    :: npoints_bl, i_loc, j_loc, iv, i_bl, i_stat_bl
integer                                    :: istart,iend,ii,jj,ierr,iprint,i1,i2,j1,j2
integer                                    :: ix_ramp_skip, ix_out
real(rkind), dimension(:), allocatable     :: delta_bl
real(rkind)                                :: x_bl, y_bl, p_int
real(rkind), dimension(:,:), allocatable   :: points_bl, wstat_bl
real(rkind)                                :: p_loc(4),v_loc(4,2), xst(2)
real(rkind), dimension(:,:,:), allocatable :: xnf

integer                                    :: recompute_avg
integer                                    :: save_plot3d
integer                                    :: io_type_w
integer                                    :: mpi_split_x,mpi_split_z
integer                                    :: it_start, it_end, it_out
integer, dimension(:), allocatable         :: plot3d_vars 
character(len=30), dimension(N_S)          :: species_names
character(30), dimension(:), allocatable   :: var_names 
logical                                    :: slicexy_exists, slicexz_exists, sliceyz_exists
integer(ikind), dimension(:), allocatable  :: igslice, jgslice, kgslice
integer(ikind), dimension(:), allocatable  :: islice, jslice, kslice
real(rkind), dimension(:,:,:,:), allocatable :: slicexy_aux, slicexz_aux, sliceyz_aux
integer :: inum,jnum,knum,icord,jcord,kcord
integer :: H2,O2,N2

! MPI
integer(ikind)              :: myrank
integer(ikind)              :: nprocs
logical :: masterproc
integer, dimension(3)       :: nblocks, ncoords
integer :: mp_cart,mp_cartx,mp_carty,mp_cartz
integer :: nproc,nrank_x, nrank_y, nrank_z
integer :: ileftx,irightx,ilefty,irighty,ileftz,irightz
integer :: ileftbottom, irightbottom, ilefttop, irighttop
end module

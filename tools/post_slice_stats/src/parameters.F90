module parameters
!
use, intrinsic :: iso_fortran_env
use mpi
implicit none

#ifdef SINGLE_PRECISION
integer, parameter :: rkind = REAL32
integer, parameter :: mpi_prec = mpi_real4
#else
integer, parameter :: rkind = REAL64
integer, parameter :: mpi_prec = mpi_real8
#endif
integer, parameter :: ikind = INT32

real(rkind) :: pi=acos(-1.d0)
!
end module parameters

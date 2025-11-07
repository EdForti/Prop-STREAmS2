module parameters
!
use, intrinsic :: iso_fortran_env
implicit none

integer, parameter :: rkind = REAL64

real(rkind), parameter :: pi=acos(-1.d0)
real(rkind), parameter :: R_univ = 8314.46261815324_rkind
real(rkind), parameter :: tol_iter_nr = 0.000000000001_rkind

!
end module parameters


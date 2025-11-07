program test_cantera_utils
    use cantera
    use cantera_utils
    use iso_fortran_env
 
    implicit none

    !integer, parameter :: rkind = REAL64
    type(phase_t) :: mixture_yaml
    integer :: nSp, nRx
    integer :: i_specie, ir, ic, i_reaction
    real(8), allocatable, dimension(:) :: tem_ranges
    real(8), allocatable, dimension(:,:) :: cp_coeff_matrix ! 1st=coeffs, 2nd=ranges
    real(8), allocatable, dimension(:,:) :: arr_a, arr_b, arr_ea, falloffCoeff
    real(8), allocatable, dimension(:,:) :: tb_eff
    real(8), allocatable, dimension(:)   :: sigma, epsK, dipole, polariz
    integer, allocatable, dimension(:)   :: geometry 
    real(8) :: TminTab, TmaxTab, T, dtTab, Tloc
    integer :: i,j,k,NumTab
    integer :: ispec, it, lsp
    character(len=20) :: filename
    character(len=64) :: spName
    integer :: unit


    mixture_yaml = importphase('input_cantera.yaml')
    nSp = nSpecies(mixture_yaml)
    nRx = nReactions(mixture_yaml)
    call get_transport_data_from_cantera(sigma, epsK, dipole, polariz, geometry)

    print *, '-------------------------------------------------------------'
    print *, ' Transport properties for each species'
    print *, '-------------------------------------------------------------'
    print '(A20,5A15)', 'Species', 'Sigma [m]', 'Eps/k [K]', 'Dipole [C*m]', 'Polariz [m^3]', 'Geometry'

    do i_specie = 1, nSp
        call getSpeciesName(mixture_yaml, i_specie, spName) 
        print '(A20,4E15.6,I15)', trim(spName), sigma(i_specie), epsK(i_specie), &
                                 dipole(i_specie), polariz(i_specie), geometry(i_specie)
    end do

!    do i_specie = 1,nSp
!        call get_cp_from_cantera(i_specie, tem_ranges, cp_coeff_matrix)
!        print*,'i_specie: ',i_specie
!        do ir=1,size(tem_ranges)-1
!            print*,'range = ',ir,' - MIN/MAX = ',tem_ranges(ir), tem_ranges(ir+1)
!            do ic=1,size(cp_coeff_matrix,1)
!                print*,'cp_coeff[',ic,'] = ',cp_coeff_matrix(ic,ir)
!            enddo
!        enddo
!    enddo
!
!call get_arrhenius_from_cantera(arr_a, arr_b, arr_ea,tb_eff, falloffCoeff)
!
!print *, arr_a(1,1), arr_a(1,2)
!print *, arr_a(2,1), arr_a(2,2)
!print *, arr_a(3,1), arr_a(3,2)
!do i = 1,nRx
!        print '(A,I2,A,5(g0.3,1x),/)', "nRx: ", i, " coeff:", falloffCoeff(i,1:5)
!end do

endprogram test_cantera_utils


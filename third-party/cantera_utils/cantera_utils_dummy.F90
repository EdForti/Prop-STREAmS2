module cantera_utils

    use iso_fortran_env
    use iso_c_binding
    integer, parameter :: rckind = REAL64

    contains

    subroutine get_cp_from_cantera(i_specie, tem_ranges, cp_coeff_matrix)
        integer, intent(in) :: i_specie
        real(rckind), allocatable, dimension(:), intent(out) :: tem_ranges
        real(rckind), allocatable, dimension(:,:), intent(out) :: cp_coeff_matrix ! 1st=coeffs, 2nd=ranges
        print *, 'Warning calling get_cp_from_cantera_dummy: this must not happen!'
    endsubroutine get_cp_from_cantera


    subroutine get_transport_data_from_cantera(sigma, epsK, dipole, polariz, geometry)
        integer :: nSpecies
        integer, allocatable, dimension(:), intent(out) :: geometry
        real(rckind), allocatable, dimension(:), intent(out) :: sigma, epsK, dipole, polariz

        print *, 'Warning calling get_transport_data_from_cantera_dummy: this must not happen!'

    end subroutine get_transport_data_from_cantera

    subroutine get_arrhenius_from_cantera(Arrhenius_A, Arrhenius_b, Arrhenius_Ea, thirdbody_eff,falloffCoeff)
        integer :: nSpecies, nReactions,i
        real(rckind), allocatable, dimension(:,:), intent(out)   :: Arrhenius_A, Arrhenius_b, Arrhenius_Ea, falloffCoeff
        real(rckind), allocatable, dimension(:,:), intent(out) :: thirdbody_eff

        print *, 'Warning calling get_arrhenius_from_cantera_dummy: this must not happen!'
 
    endsubroutine get_arrhenius_from_cantera


endmodule cantera_utils

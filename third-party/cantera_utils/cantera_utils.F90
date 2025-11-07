module cantera_utils

    use iso_fortran_env
    use iso_c_binding
    integer, parameter :: rckind = REAL64

    interface
        subroutine get_transport_data_from_cantera_c(nSpecies, sigma, epsK, dipole, polariz, geometry) bind(C)
            import :: c_int, c_double, c_ptr
            integer(c_int) ::  nSpecies
            type(c_ptr) :: sigma, epsK, dipole, polariz, geometry
        endsubroutine get_transport_data_from_cantera_c

        subroutine get_cp_from_cantera_c(i_specie, n_ranges, n_coeffs, tem_ranges, cp_coeff_matrix) bind(C)
            import :: c_int, c_double, c_ptr
            integer(c_int) :: n_ranges, n_coeffs
            integer(c_int), value :: i_specie
            type(c_ptr) :: tem_ranges
            type(c_ptr) :: cp_coeff_matrix
        endsubroutine get_cp_from_cantera_c

        subroutine get_arrhenius_from_cantera_c(nSpecies, nReactions, Arrhenius_A, Arrhenius_b, Arrhenius_Ea, &
                                                thirdbody_eff, falloffCoeff) bind(C)
            import :: c_int, c_double, c_ptr
            integer(c_int) :: nSpecies, nReactions
            type(c_ptr) :: Arrhenius_A, Arrhenius_b, Arrhenius_Ea, thirdbody_eff, falloffCoeff
        endsubroutine get_arrhenius_from_cantera_c

    endinterface

    contains

    subroutine get_transport_data_from_cantera(sigma, epsK, dipole, polariz, geometry)
        integer :: nSpecies
        integer, allocatable, dimension(:), intent(out) :: geometry 
        integer, pointer, dimension(:) :: geometry_p 
        real(rckind), allocatable, dimension(:), intent(out) :: sigma, epsK, dipole, polariz
        real(rckind), pointer, dimension(:) :: sigma_p, epsK_p, dipole_p, polariz_p
        type(c_ptr) :: sigma_c, epsK_c, dipole_c, polariz_c, geometry_c

        call get_transport_data_from_cantera_c(nSpecies, sigma_c, epsK_c, dipole_c, polariz_c, geometry_c) 
        call c_f_pointer(sigma_c   , sigma_p   , [nSpecies])
        call c_f_pointer(epsK_c    , epsK_p    , [nSpecies])
        call c_f_pointer(dipole_c  , dipole_p  , [nSpecies])
        call c_f_pointer(polariz_c , polariz_p , [nSpecies])
        call c_f_pointer(geometry_c, geometry_p, [nSpecies])
        allocate(sigma(nSpecies),epsK(nSpecies),dipole(nSpecies),polariz(nSpecies),geometry(nSpecies))
        sigma    = sigma_p  
        epsK     = epsK_p   
        dipole   = dipole_p 
        polariz  = polariz_p
        geometry = geometry_p

    end subroutine get_transport_data_from_cantera


    subroutine get_cp_from_cantera(i_specie, tem_ranges, cp_coeff_matrix)
        integer, intent(in) :: i_specie
        real(rckind), allocatable, dimension(:), intent(out) :: tem_ranges
        real(rckind), allocatable, dimension(:,:), intent(out) :: cp_coeff_matrix ! 1st=coeffs, 2nd=ranges
        real(rckind), pointer, dimension(:) :: tem_ranges_p
        real(rckind), pointer, dimension(:,:) :: cp_coeff_matrix_p ! 1st=coeffs, 2nd=ranges
        type(c_ptr) :: tem_ranges_c
        type(c_ptr) :: cp_coeff_matrix_c ! 1st=coeffs, 2nd=ranges
        integer :: n_ranges, n_coeffs
        call get_cp_from_cantera_c(i_specie, n_ranges, n_coeffs, tem_ranges_c, cp_coeff_matrix_c)
        call c_f_pointer(tem_ranges_c, tem_ranges_p, [n_ranges+1])
        call c_f_pointer(cp_coeff_matrix_c, cp_coeff_matrix_p, [n_coeffs, n_ranges])
        !print*,'n_ranges, n_coeffs: ',n_ranges, n_coeffs
        !print*,'tem_ranges_p: ',tem_ranges_p
        !print*,'cp_coeff_matrix_p: ',cp_coeff_matrix_p
        allocate(tem_ranges(n_ranges+1))
        tem_ranges = tem_ranges_p
        allocate(cp_coeff_matrix(n_coeffs,n_ranges))
        cp_coeff_matrix = cp_coeff_matrix_p
    endsubroutine get_cp_from_cantera

    subroutine get_arrhenius_from_cantera(Arrhenius_A, Arrhenius_b, Arrhenius_Ea, thirdbody_eff,falloffCoeff)
        integer :: nSpecies, nReactions,i
        real(rckind), allocatable, dimension(:,:), intent(out)   :: Arrhenius_A, Arrhenius_b, Arrhenius_Ea, falloffCoeff
        real(rckind), allocatable, dimension(:,:), intent(out) :: thirdbody_eff
        real(rckind), allocatable, dimension(:,:)              :: thirdbody_eff_temp
        real(rckind), pointer, dimension(:,:)                    :: Arrhenius_A_p, Arrhenius_b_p, Arrhenius_Ea_p, falloffCoeff_p
        real(rckind), pointer, dimension(:,:)                  :: thirdbody_eff_p
        type(c_ptr) :: Arrhenius_A_c, Arrhenius_b_c, Arrhenius_Ea_c, thirdbody_eff_c, falloffCoeff_c
        call get_arrhenius_from_cantera_c(nSpecies, nReactions, Arrhenius_A_c, Arrhenius_b_c,&
Arrhenius_Ea_c,thirdbody_eff_c, falloffCoeff_c)
        call c_f_pointer(Arrhenius_A_c   , Arrhenius_A_p, [2,nReactions])
        call c_f_pointer(Arrhenius_b_c   , Arrhenius_b_p   , [2,nReactions])
        call c_f_pointer(Arrhenius_Ea_c  , Arrhenius_Ea_p  , [2,nReactions])
        call c_f_pointer(thirdbody_eff_c , thirdbody_eff_p , [nSpecies,nReactions])
        call c_f_pointer(falloffCoeff_c  , falloffCoeff_p  , [5,nReactions])

        allocate(Arrhenius_A(nReactions,2), Arrhenius_b(nReactions,2), Arrhenius_Ea(nReactions,2))
        Arrhenius_A  = transpose(Arrhenius_A_p)
        Arrhenius_b  = transpose(Arrhenius_b_p)
        Arrhenius_Ea = transpose(Arrhenius_Ea_p)

        allocate(thirdbody_eff_temp(nSpecies , nReactions))
        allocate(thirdbody_eff(nReactions, nSpecies))
        thirdbody_eff_temp = thirdbody_eff_p
        thirdbody_eff = transpose(thirdbody_eff_temp)

        allocate(falloffCoeff(nReactions,5))
        falloffCoeff = transpose(falloffCoeff_p)
    endsubroutine get_arrhenius_from_cantera

endmodule cantera_utils


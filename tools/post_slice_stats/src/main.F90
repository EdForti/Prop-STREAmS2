program main

    use mpi
    use utils 
    use postpro_slice
    use global_variables
    implicit none
    integer :: mpi_err

    nvstats = 20

    call read_input
    call initialize_mpi

    call read_grid_bl
    call find_slices
    call slice_prepare

    call set_var_names
    if (slicexy_exists) call postpro_slice_xy
    call mpi_barrier(mp_cart,mpi_err)
    if (slicexz_exists) call postpro_slice_xz
    !if (sliceyz_exists) call postpro_slice_xy
    call mpi_finalize(mpi_err)

endprogram main

#include 'index_define.h'
module streams_equation_multideal_gpu_object
    !< STREAmS, Navier-Stokes equations, GPU backend.

    use streams_base_gpu_object
    use streams_field_object
    use streams_grid_object
    use streams_kernels_gpu
    use streams_parameters
    use crandom_f_mod
    use streams_equation_multideal_object
    use MPI
    use CUDAFOR
    use ISO_C_BINDING
    use, intrinsic :: iso_fortran_env
    use tcp

!   INSITU CATALYST2
    use catalyst_api
    use catalyst_conduit

    implicit none
    private
    public :: equation_multideal_gpu_object

    integer(kind=cuda_stream_kind) :: stream1

    integer, parameter :: EULERCENTRAL_THREADS_X=128,EULERCENTRAL_THREADS_Y=3
    integer, parameter :: EULERWENO_THREADS_X=128   ,EULERWENO_THREADS_Y=2

    type :: equation_multideal_gpu_object
        !<
        !< w(1): rho
        !< w(2): rho * u
        !< w(3): rho * v
        !< w(4): rho * w
        !< w(5): rho * E
        !<```
        !< w_aux(1) : rho
        !< w_aux(2) : u
        !< w_aux(3) : v
        !< w_aux(4) : w
        !< w_aux(5) : h
        !< w_aux(6) : T
        !< w_aux(7) : viscosity
        !< w_aux(8) : ducros
        !< w_aux(9) : |omega|
        !< w_aux(10): div
        !<```
        type(base_gpu_object)       :: base_gpu               !< The base GPU handler.
        type(equation_multideal_object)    :: equation_base  !< The equation base.
        type(field_object), pointer :: field=>null()          !< The field.
        type(grid_object),  pointer :: grid=>null()           !< The grid.
        integer(ikind)              :: ng                     !< Number of ghost cells.
        integer(ikind)              :: nx                     !< Number of cells in i direction.
        integer(ikind)              :: ny                     !< Number of cells in j direction.
        integer(ikind)              :: nz                     !< Number of cells in k direction.
        integer(ikind)              :: nv                     !< Number of variables.
        integer(ikind)              :: nv_aux                 !< Number of auxiliary variables.
        integer(ikind)              :: nprocs                 !< Number of auxiliary variables.
        integer(ikind)              :: myrank                 !< Number of auxiliary variables.
        integer(ikind)              :: error                  !< Number of auxiliary variables.
        real(rkind)                 :: time0
        real(rkind), allocatable, dimension(:,:), device :: coeff_deriv1_gpu
        real(rkind), allocatable, dimension(:,:), device :: coeff_deriv2_gpu
        
        integer :: ierr
        integer :: icyc0, num_iter
        logical :: masterproc
        integer :: mpi_err
        real(rkind), allocatable, dimension(:), device :: winf_gpu
        real(rkind), allocatable, dimension(:,:,:,:), device :: w_aux_gpu
        real(rkind), allocatable, dimension(:,:,:,:), device :: fhat_gpu
        real(rkind), allocatable, dimension(:,:,:,:), device :: w_aux_trans_gpu, fl_trans_gpu, fhat_trans_gpu
        real(rkind), allocatable, dimension(:,:,:,:), device :: fl_gpu, fln_gpu, fl_sav_gpu
        real(rkind), allocatable, dimension(:,:), device :: dcoe_gpu
        real(rkind), allocatable, dimension(:,:,:,:) :: w_var, w_var_t
!       real(rkind), allocatable, dimension(:,:,:,:), device :: gplus_x_gpu, gminus_x_gpu
!       real(rkind), allocatable, dimension(:,:,:,:), device :: gplus_y_gpu, gminus_y_gpu
!       real(rkind), allocatable, dimension(:,:,:,:), device :: gplus_z_gpu, gminus_z_gpu
        integer, allocatable, dimension(:,:,:), device :: fluid_mask_gpu
        integer, allocatable, dimension(:,:,:), device :: fluid_mask_ini_gpu
        integer, allocatable, dimension(:,:,:,:), device :: ep_ord_change_gpu

        real(rkind), allocatable, dimension(:,:,:,:), device :: wrecyc_gpu
        real(rkind), allocatable, dimension(:,:,:), device :: wrecycav_gpu
        real(rkind), dimension(:,:,:), allocatable, device   :: wmean_gpu

        real(rkind), dimension(:,:,:), allocatable, device :: inflow_random_plane_gpu
        real(rkind), dimension(:), allocatable, device :: weta_inflow_gpu
        real(rkind), dimension(:), allocatable, device :: yplus_inflow_gpu, eta_inflow_gpu
        real(rkind), dimension(:), allocatable, device :: yplus_recyc_gpu, eta_recyc_gpu, eta_recyc_blend_gpu
        integer, dimension(:), allocatable, device     :: map_j_inn_gpu, map_j_out_gpu, map_j_out_blend_gpu
        real(rkind), dimension(:), allocatable, device :: weta_inflow2_gpu
        real(rkind), dimension(:), allocatable, device :: yplus_inflow2_gpu, eta_inflow2_gpu
        real(rkind), dimension(:), allocatable, device :: yplus_recyc2_gpu, eta_recyc2_gpu, eta_recyc_blend2_gpu
        integer, dimension(:), allocatable, device     :: map_j_inn2_gpu, map_j_out2_gpu, map_j_out_blend2_gpu


        real(rkind), dimension(:,:,:), allocatable, device :: cv_coeff_gpu, cp_coeff_gpu
        real(rkind), dimension(:,:), allocatable, device :: trange_gpu
        real(rkind), dimension(:), allocatable, device :: mw_gpu, mwinv_gpu, rgas_gpu, h298_gpu, init_mf_gpu 
        real(rkind), dimension(:,:), allocatable, device :: visc_species_gpu, lambda_species_gpu
        real(rkind), dimension(:,:,:), allocatable, device :: diffbin_species_gpu

        real(rkind), dimension(:), allocatable, device :: endepo_param_gpu
        integer, dimension(:), allocatable, device :: reac_ty_gpu,isRev_gpu
        real(rkind), dimension(:,:), allocatable, device :: arr_a_gpu, arr_b_gpu, arr_ea_gpu, falloff_coeffs_gpu
        real(rkind), dimension(:,:), allocatable, device :: tb_eff_gpu

        real(rkind), dimension(:,:), allocatable, device :: r_coeffs_gpu, p_coeffs_gpu
        real(rkind), dimension(:,:), allocatable, device :: kc_tab_gpu

        integer :: N_EoI_gpu
        real(rkind), allocatable, dimension(:), device :: aw_EoI_gpu,Beta0_gpu,coeff_EoI_gpu
        integer, allocatable, dimension(:,:), device :: NainSp_gpu

        real(rkind), dimension(:,:), allocatable, device :: w_aux_probe_gpu
        real(rkind), dimension(:,:,:,:), allocatable, device :: probe_coeff_gpu
        integer, dimension(:,:), allocatable, device :: ijk_probe_gpu

        ! ibm_var_start
        integer,     dimension(:,:,:),   allocatable, device :: ibm_sbody_gpu
        integer,     dimension(:,:,:),   allocatable, device :: ibm_is_interface_node_gpu
        integer,     dimension(:,:  ),   allocatable, device :: ibm_ijk_interface_gpu
        integer,     dimension(:,:  ),   allocatable, device :: ibm_ijk_hwm_gpu
        real(rkind), dimension(:,:),     allocatable, device :: ibm_nxyz_interface_gpu
        integer,     dimension(:,:),     allocatable, device :: ibm_bc_gpu
        real(rkind), dimension(:,:,:,:), allocatable, device :: ibm_coeff_hwm_gpu
        real(rkind), dimension(:,:    ), allocatable, device :: ibm_w_refl_gpu
        real(rkind), dimension(:,:    ), allocatable, device :: ibm_w_hwm_gpu
        real(rkind), dimension(:      ), allocatable, device :: ibm_dist_hwm_gpu
        real(rkind), dimension(:,:    ), allocatable, device :: ibm_wm_correction_gpu,ibm_wm_wallprop_gpu
        real(rkind), dimension(:,:    ), allocatable, device :: ibm_parbc_gpu
        real(rkind), dimension(:), allocatable, device :: randvar_a_gpu,randvar_p_gpu
        real(rkind), dimension(:,:,:),   allocatable, device :: ibm_body_dist_gpu
        real(rkind), dimension(:,:,:),   allocatable, device :: ibm_reflection_coeff_gpu
        real(rkind), allocatable, dimension(:,:,:,:), device :: ibm_dw_aux_eikonal_gpu

        ! ibm_var_old
        integer,     dimension(:,:,:),   allocatable, device :: ibm_inside_moving_gpu
        integer,     dimension(:,:  ),   allocatable, device :: ibm_ijk_refl_gpu
        integer,     dimension(:,:  ),   allocatable, device :: ibm_ijk_wall_gpu
        real(rkind), dimension(:,:),     allocatable, device :: ibm_dist_gpu
        real(rkind), dimension(:,:,:,:), allocatable, device :: ibm_coeff_d_gpu
        real(rkind), dimension(:,:,:,:), allocatable, device :: ibm_coeff_n_gpu
        integer,     dimension(:      ), allocatable, device :: ibm_refl_type_gpu
        real(rkind), dimension(:,:,:),   allocatable, device :: ibm_vega_dist_gpu, ibm_vega_distini_gpu   
        real(rkind), allocatable, dimension(:,:,:,:), device :: ibm_dw_aux_vega_gpu     
        ! ibm_var_end

        ! insitu_var_start
        !real(rkind), dimension(:,:,:,:), allocatable, managed :: psi_pv_managed
        real(rkind), dimension(:,:,:,:), allocatable, device :: psi_gpu
        integer, dimension(:), allocatable, device :: aux_list_gpu, add_list_gpu
        ! insitu_var_end

        ! jcf_var_start
        real(rkind), allocatable, dimension(:,:), device :: jcf_parbc_gpu, jcf_coords_gpu
        ! jcf_var_end
!
    contains
        ! public methods
        procedure, pass(self) :: compute_dt       
        procedure, pass(self) :: initialize       
        procedure, pass(self) :: compute_residual
        procedure, pass(self) :: print_progress  
        procedure, pass(self) :: run              
        procedure, pass(self) :: rk_sync
        procedure, pass(self) :: sav_flx
!       procedure, pass(self) :: rk_async               
        procedure, pass(self) :: rosenbrock 
        procedure, pass(self) :: rosenbrock_krylov
        procedure, pass(self) :: compute_chemistry 
        procedure, pass(self) :: energy_deposition 
        procedure, pass(self) :: point_to_field   
        procedure, pass(self) :: point_to_grid   
        procedure, pass(self) :: alloc            
        procedure, pass(self) :: update_ghost     
        procedure, pass(self) :: euler_x          
        procedure, pass(self) :: euler_y          
        procedure, pass(self) :: euler_z          
        procedure, pass(self) :: visflx          
        procedure, pass(self) :: compute_aux
        procedure, pass(self) :: compute_aux_les
        procedure, pass(self) :: compute_chem_aux
        procedure, pass(self) :: recyc_exchange
        procedure, pass(self) :: bcrecyc
        procedure, pass(self) :: bc_nr
        procedure, pass(self) :: manage_output
        procedure, pass(self) :: tripping 
        procedure, pass(self) :: limiter 
        procedure, pass(self) :: advance_solution
        !ibm
!       ibm_proc_start
        procedure, pass(self) :: ibm_apply
        procedure, pass(self) :: ibm_compute_force !Not changed
        procedure, pass(self) :: ibm_alloc_gpu
        procedure, pass(self) :: ibm_apply_wm

        procedure, pass(self) :: ibm_apply_old
        procedure, pass(self) :: ibm_inside_old
        procedure, pass(self) :: ibm_alloc_gpu_old

!       ibm_proc_end
        ! insitu
!       insitu_proc_start
        procedure, pass(self) :: insitu_alloc_gpu
        procedure, pass(self) :: insitu_coprocess
        procedure, pass(self) :: insitu_compute_psi
        procedure, pass(self) :: insitu_do_catalyst_execute
!       insitu_proc_end

    endtype equation_multideal_gpu_object

contains

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!   Utilities
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    subroutine point_to_field(self, field)
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        class(field_object), target :: field              !< The equation.
        self%field => field
    endsubroutine point_to_field

    subroutine point_to_grid(self, grid)
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        class(grid_object), target :: grid              !< The equation.
        self%grid => grid
    endsubroutine point_to_grid

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
    subroutine ibm_alloc_gpu(self)
        class(equation_multideal_gpu_object), intent(inout) :: self
        associate(ibm_num_interface => self%equation_base%ibm_num_interface, &
                  ibm_num_bc => self%equation_base%ibm_num_bc, &
                  nx => self%nx, ny => self%ny, nz => self%nz,  &
                  ng => self%ng, nv => self%nv)
        allocate(self%ibm_sbody_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
        allocate(self%ibm_is_interface_node_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
        allocate(self%ibm_body_dist_gpu(0-ng:nx+ng+1,0-ng:ny+ng+1,0-ng:nz+ng+1))
        self%ibm_body_dist_gpu = self%equation_base%ibm_body_dist
        allocate(self%ibm_reflection_coeff_gpu(nx,ny,nz))
        self%ibm_reflection_coeff_gpu = self%equation_base%ibm_reflection_coeff
        allocate(self%ibm_dw_aux_eikonal_gpu(1-ng:nx+ng, 1-ng:ny+ng,1-ng:nz+ng,nv))
        allocate(self%ibm_parbc_gpu(ibm_num_bc,IBM_MAX_PARBC))
        if (ibm_num_interface>0) then
         allocate(self%ibm_ijk_interface_gpu (3,ibm_num_interface))      ! Local values of i,j,k for the interface node
         allocate(self%ibm_nxyz_interface_gpu(3,ibm_num_interface))      ! Wall-normal components
         allocate(self%ibm_bc_gpu            (2,ibm_num_interface))      ! Bc tag for interface nodes
         allocate(self%ibm_ijk_hwm_gpu       (3,ibm_num_interface))
         allocate(self%ibm_coeff_hwm_gpu     (2,2,2,ibm_num_interface))
         allocate(self%ibm_w_refl_gpu(ibm_num_interface,nv))
         allocate(self%ibm_dist_hwm_gpu(ibm_num_interface))
         allocate(self%ibm_wm_correction_gpu(2,ibm_num_interface)) ! mu_t, k_cond_t
         allocate(self%ibm_wm_wallprop_gpu(2,ibm_num_interface))
         allocate(self%ibm_w_hwm_gpu(ibm_num_interface,3))
        endif
        self%ibm_sbody_gpu             = self%equation_base%ibm_sbody
        self%ibm_is_interface_node_gpu = self%equation_base%ibm_is_interface_node
        self%ibm_parbc_gpu             = self%equation_base%ibm_parbc
        if (self%equation_base%turinf .eq. 1) then
         allocate(self%randvar_a_gpu(8),self%randvar_p_gpu(8))
         self%randvar_a_gpu = self%equation_base%randvar_a
         self%randvar_p_gpu = self%equation_base%randvar_p
        endif 
        if (ibm_num_interface>0) then
         self%ibm_ijk_interface_gpu     = self%equation_base%ibm_ijk_interface
         self%ibm_nxyz_interface_gpu    = self%equation_base%ibm_nxyz_interface
         self%ibm_bc_gpu                = self%equation_base%ibm_bc
         self%ibm_ijk_hwm_gpu           = self%equation_base%ibm_ijk_hwm
         self%ibm_coeff_hwm_gpu         = self%equation_base%ibm_coeff_hwm
         self%ibm_dist_hwm_gpu          = self%equation_base%ibm_dist_hwm
        endif
        endassociate
    endsubroutine ibm_alloc_gpu
!
    subroutine insitu_alloc_gpu(self)
        class(equation_multideal_gpu_object), intent(inout) :: self      
        associate( nxsl_ins => self%equation_base%nxsl_ins, nxel_ins => self%equation_base%nxel_ins, &
          nysl_ins => self%equation_base%nysl_ins, nyel_ins => self%equation_base%nyel_ins, &
          nzsl_ins => self%equation_base%nzsl_ins, nzel_ins => self%equation_base%nzel_ins, &
          npsi => self%equation_base%npsi, npsi_pv => self%equation_base%npsi_pv, &
          nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, &
          ng => self%grid%ng, n_aux_list => self%equation_base%n_aux_list, n_add_list => self%equation_base%n_add_list )
        !NOMANAGED if (npsi_pv > 0) allocate(self%psi_pv_managed(nxsl_ins:nxel_ins,nysl_ins:nyel_ins,nzsl_ins:nzel_ins,npsi_pv))
        if (npsi > 0) allocate(self%psi_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,npsi))
        if (n_aux_list > 0) then
         allocate(self%aux_list_gpu(1:n_aux_list))
         self%aux_list_gpu = self%equation_base%aux_list
        endif
        if (n_add_list > 0) then
         allocate(self%add_list_gpu(1:n_add_list))
         self%add_list_gpu = self%equation_base%add_list
        endif
        endassociate
    endsubroutine insitu_alloc_gpu

    subroutine visflx(self, mode)
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        integer, intent(in) :: mode
        type(dim3) :: grid, tBlock
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  nv => self%nv,  nv_aux => self%nv_aux, &
                  dt => self%equation_base%dt, &
                  visc_order => self%equation_base%visc_order, &
                  coeff_deriv1_gpu => self%coeff_deriv1_gpu, coeff_deriv2_gpu => self%coeff_deriv2_gpu, &
                  fhat_trans_gpu => self%fhat_trans_gpu, fl_trans_gpu => self%fl_trans_gpu, fl_gpu => self%fl_gpu, &
                  w_aux_gpu  => self%w_aux_gpu, w_aux_trans_gpu => self%w_aux_trans_gpu, &
                  dcsidx_gpu => self%base_gpu%dcsidx_gpu,   &  
                  detady_gpu => self%base_gpu%detady_gpu,   &
                  dzitdz_gpu => self%base_gpu%dzitdz_gpu,   &
                  dcsidxs_gpu => self%base_gpu%dcsidxs_gpu, &
                  detadys_gpu => self%base_gpu%detadys_gpu, &
                  dzitdzs_gpu => self%base_gpu%dzitdzs_gpu, &
                  dcsidx2_gpu => self%base_gpu%dcsidx2_gpu, &
                  detady2_gpu => self%base_gpu%detady2_gpu, &
                  dzitdz2_gpu => self%base_gpu%dzitdz2_gpu, &
                  x_gpu => self%base_gpu%x_gpu, &
                  y_gpu => self%base_gpu%y_gpu, &
                  z_gpu => self%base_gpu%z_gpu, &
                  eul_imin => self%equation_base%eul_imin, eul_imax => self%equation_base%eul_imax, &
                  eul_jmin => self%equation_base%eul_jmin, eul_jmax => self%equation_base%eul_jmax, &
                  eul_kmin => self%equation_base%eul_kmin, eul_kmax => self%equation_base%eul_kmax, &
                  indx_cp_l => self%equation_base%indx_cp_l, &
                  indx_cp_r => self%equation_base%indx_cp_r, &
                  nsetcv    => self%equation_base%nsetcv, &
                  cp_coeff_gpu => self%cp_coeff_gpu, &
                  trange_gpu => self%trange_gpu, &
                  rgas_gpu => self%rgas_gpu, &
                  enable_ibm => self%equation_base%enable_ibm)

        if (mode == 0) then ! laplacian
            call visflx_cuf(nx, ny, nz, ng, visc_order, &
                self%w_aux_gpu, self%fl_gpu, &
                coeff_deriv1_gpu, coeff_deriv2_gpu, &
                dcsidx_gpu, detady_gpu, dzitdz_gpu,  &
                dcsidxs_gpu, detadys_gpu, dzitdzs_gpu,  &
                dcsidx2_gpu, detady2_gpu, dzitdz2_gpu)
        elseif (mode == 1) then ! staggered
            if (self%grid%nxmax>1) call visflx_x_cuf(nx, ny, nz, nv, nv_aux, ng, &
                self%base_gpu%x_gpu, self%w_aux_gpu, self%fl_gpu, self%fhat_gpu, &
                indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,rgas_gpu,R_univ)
            if (self%grid%nymax>1) call visflx_y_cuf(nx, ny, nz, nv, nv_aux, ng, &
                self%base_gpu%y_gpu, self%w_aux_gpu, self%fl_gpu, self%fhat_gpu, &
                indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,rgas_gpu,R_univ)
            if (self%grid%nzmax>1) call visflx_z_cuf(nx, ny, nz, nv, nv_aux, ng, &
                self%base_gpu%z_gpu, self%w_aux_gpu, self%fl_gpu, self%fhat_gpu, &
                indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,rgas_gpu,R_univ)
        elseif (mode == 2) then ! reduced
            call visflx_reduced_cuf(nx, ny, nz, ng, visc_order, &
                 self%w_aux_gpu, self%fl_gpu, coeff_deriv1_gpu, &
                 dcsidx_gpu, detady_gpu, dzitdz_gpu)
        endif
        endassociate
    endsubroutine visflx

    subroutine compute_aux(self, central, ghost)
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        integer, intent(in), optional :: central, ghost
        integer :: central_, ghost_
        central_ = 1 ; if(present(central)) central_ = central
        ghost_ = 1   ; if(present(ghost))   ghost_ = ghost
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, nv => self%nv, &
                  nv_aux => self%nv_aux, &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  cp_coeff_gpu => self%cp_coeff_gpu, &
                  trange_gpu => self%trange_gpu, &
                  nsetcv => self%equation_base%nsetcv, &
                  indx_cp_l => self%equation_base%indx_cp_l, &
                  indx_cp_r => self%equation_base%indx_cp_r, &
                  num_t_tab => self%equation_base%num_t_tab, &
                  dt_tab => self%equation_base%dt_tab, &
                  t_min_tab => self%equation_base%t_min_tab, &
                  p0 => self%equation_base%p0, &
                  mw_gpu => self%mw_gpu, &
                  mwinv_gpu => self%mwinv_gpu, &
                  visc_species_gpu => self%visc_species_gpu, &
                  lambda_species_gpu => self%lambda_species_gpu, &
                  rgas_gpu => self%rgas_gpu, &
                  coeff_deriv1_gpu => self%coeff_deriv1_gpu, &
                  dcsidx_gpu => self%base_gpu%dcsidx_gpu,   &
                  detady_gpu => self%base_gpu%detady_gpu,   &
                  dzitdz_gpu => self%base_gpu%dzitdz_gpu,   &
                  diffbin_species_gpu => self%diffbin_species_gpu, &
                  visc_order => self%equation_base%visc_order, &
                  eps_sensor => self%equation_base%eps_sensor, &
                  sensor_type => self%equation_base%sensor_type)

        if (central_ == 1 .and. ghost_ == 1) then
            call eval_aux_cuf(nx, ny, nz, ng, nv, nv_aux, 1-ng, nx+ng, 1-ng, ny+ng, 1-ng, nz+ng, self%base_gpu%w_gpu, self%w_aux_gpu, &
                p0, t_min_tab,dt_tab, R_univ,&
                rgas_gpu, cv_coeff_gpu, cp_coeff_gpu,  nsetcv, trange_gpu, &
                indx_cp_l, indx_cp_r,tol_iter_nr,stream1,mw_gpu,mwinv_gpu,visc_species_gpu,lambda_species_gpu,diffbin_species_gpu,num_t_tab,&
                self%N_EoI_gpu,self%aw_EoI_gpu,self%NainSp_gpu,self%Beta0_gpu,self%coeff_EoI_gpu)
        elseif (central_ == 1 .and. ghost_ == 0) then
            call eval_aux_cuf(nx, ny, nz, ng, nv, nv_aux, 1, nx, 1, ny, 1, nz, self%base_gpu%w_gpu, self%w_aux_gpu, &
                p0, t_min_tab,dt_tab, R_univ, &
                rgas_gpu, cv_coeff_gpu, cp_coeff_gpu,  nsetcv, trange_gpu, &
                indx_cp_l, indx_cp_r,tol_iter_nr,stream1,mw_gpu,mwinv_gpu,visc_species_gpu,lambda_species_gpu,diffbin_species_gpu,num_t_tab,&
                self%N_EoI_gpu,self%aw_EoI_gpu,self%NainSp_gpu,self%Beta0_gpu,self%coeff_EoI_gpu)             
        elseif (central_ == 0 .and. ghost_ == 1) then
            call eval_aux_cuf(nx, ny, nz, ng, nv, nv_aux, 1-ng, 0, 1-ng, ny+ng, 1-ng, nz+ng, self%base_gpu%w_gpu, self%w_aux_gpu, &
                p0, t_min_tab,dt_tab, R_univ, &
                rgas_gpu, cv_coeff_gpu, cp_coeff_gpu,  nsetcv, trange_gpu, &
                indx_cp_l, indx_cp_r,tol_iter_nr,stream1,mw_gpu,mwinv_gpu,visc_species_gpu,lambda_species_gpu,diffbin_species_gpu,num_t_tab,&
                self%N_EoI_gpu,self%aw_EoI_gpu,self%NainSp_gpu,self%Beta0_gpu,self%coeff_EoI_gpu)
            call eval_aux_cuf(nx, ny, nz, ng, nv, nv_aux, nx+1, nx+ng, 1-ng, ny+ng, 1-ng, nz+ng, self%base_gpu%w_gpu, self%w_aux_gpu, &
                p0, t_min_tab,dt_tab, R_univ, &
                rgas_gpu, cv_coeff_gpu, cp_coeff_gpu,  nsetcv, trange_gpu, &
                indx_cp_l, indx_cp_r,tol_iter_nr,stream1,mw_gpu,mwinv_gpu,visc_species_gpu,lambda_species_gpu,diffbin_species_gpu,num_t_tab,&
                self%N_EoI_gpu,self%aw_EoI_gpu,self%NainSp_gpu,self%Beta0_gpu,self%coeff_EoI_gpu)
            call eval_aux_cuf(nx, ny, nz, ng, nv, nv_aux, 1-ng, nx+ng, 1-ng, 0, 1-ng, nz+ng, self%base_gpu%w_gpu, self%w_aux_gpu, &
                p0, t_min_tab,dt_tab, R_univ, &
                rgas_gpu, cv_coeff_gpu, cp_coeff_gpu,  nsetcv, trange_gpu, &
                indx_cp_l, indx_cp_r,tol_iter_nr,stream1,mw_gpu,mwinv_gpu,visc_species_gpu,lambda_species_gpu,diffbin_species_gpu,num_t_tab, &
                self%N_EoI_gpu,self%aw_EoI_gpu,self%NainSp_gpu,self%Beta0_gpu,self%coeff_EoI_gpu)
            call eval_aux_cuf(nx, ny, nz, ng, nv, nv_aux, 1-ng, nx+ng, ny+1, ny+ng, 1-ng, nz+ng, self%base_gpu%w_gpu, self%w_aux_gpu, &
                p0, t_min_tab,dt_tab, R_univ, &
                rgas_gpu, cv_coeff_gpu, cp_coeff_gpu,  nsetcv, trange_gpu, &
                indx_cp_l, indx_cp_r,tol_iter_nr,stream1,mw_gpu,mwinv_gpu,visc_species_gpu,lambda_species_gpu,diffbin_species_gpu,num_t_tab,&
                self%N_EoI_gpu,self%aw_EoI_gpu,self%NainSp_gpu,self%Beta0_gpu,self%coeff_EoI_gpu)
            call eval_aux_cuf(nx, ny, nz, ng, nv, nv_aux, 1-ng, nx+ng, 1-ng, ny+ng, 1-ng, 0, self%base_gpu%w_gpu, self%w_aux_gpu, &
                p0, t_min_tab,dt_tab, R_univ, &
                rgas_gpu, cv_coeff_gpu, cp_coeff_gpu,  nsetcv, trange_gpu, &
                indx_cp_l, indx_cp_r,tol_iter_nr,stream1,mw_gpu,mwinv_gpu,visc_species_gpu,lambda_species_gpu,diffbin_species_gpu,num_t_tab,&
        self%N_EoI_gpu,self%aw_EoI_gpu,self%NainSp_gpu,self%Beta0_gpu,self%coeff_EoI_gpu)
            call eval_aux_cuf(nx, ny, nz, ng, nv, nv_aux, 1-ng, nx+ng, 1-ng, ny+ng, nz+1, nz+ng, self%base_gpu%w_gpu, self%w_aux_gpu, &
                p0, t_min_tab,dt_tab, R_univ, &
                rgas_gpu, cv_coeff_gpu, cp_coeff_gpu,  nsetcv, trange_gpu, &
                indx_cp_l, indx_cp_r,tol_iter_nr,stream1,mw_gpu,mwinv_gpu,visc_species_gpu,lambda_species_gpu,diffbin_species_gpu,num_t_tab,&
                self%N_EoI_gpu,self%aw_EoI_gpu,self%NainSp_gpu,self%Beta0_gpu,self%coeff_EoI_gpu)
        endif
        if (ghost_ == 1) then
            call eval_aux2_cuf(nx, ny, nz, ng, visc_order, self%w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, detady_gpu, dzitdz_gpu, &
                               eps_sensor, sensor_type)
        endif

        endassociate
    endsubroutine compute_aux

    subroutine compute_aux_les(self)
        class(equation_multideal_gpu_object), intent(inout) :: self
        integer :: iercuda

        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, nv_aux => self%nv_aux, nv => self%nv, &
                  ep_order => self%equation_base%ep_order, les_model => self%equation_base%les_model, &
                  w_gpu => self%base_gpu%w_gpu, &
                  w_aux_gpu => self%w_aux_gpu, &
                  coeff_deriv1_gpu => self%coeff_deriv1_gpu, &
                  dcsidx_gpu => self%base_gpu%dcsidx_gpu,   &
                  detady_gpu => self%base_gpu%detady_gpu,   &
                  dzitdz_gpu => self%base_gpu%dzitdz_gpu,   &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  cp_coeff_gpu => self%cp_coeff_gpu, &
                  trange_gpu => self%trange_gpu, &
                  nsetcv    => self%equation_base%nsetcv, &
                  indx_cp_l => self%equation_base%indx_cp_l, &
                  indx_cp_r => self%equation_base%indx_cp_r, &
                  t0 => self%equation_base%t0, &
                  p0 => self%equation_base%p0, &
                  num_t_tab => self%equation_base%num_t_tab, &
                  dt_tab => self%equation_base%dt_tab, &
                  t_min_tab => self%equation_base%t_min_tab, &
                  rgas_gpu => self%rgas_gpu,& 
                  mw_gpu => self%mw_gpu, &
                  mwinv_gpu => self%mwinv_gpu, &
                  visc_species_gpu => self%visc_species_gpu, &
                  lambda_species_gpu => self%lambda_species_gpu, &
                  diffbin_species_gpu => self%diffbin_species_gpu, &
                  les_c_wale => self%equation_base%les_c_wale, &
                  les_pr => self%equation_base%les_pr, &
                  les_sc => self%equation_base%les_sc, &
                  eps_sensor => self%equation_base%eps_sensor, &
                  sensor_type => self%equation_base%sensor_type)
!
        call eval_velaux_cuf(nx,ny,nz,ng,nv,w_gpu,w_aux_gpu)
!
        select case (les_model)
         case(1,2)
          call les_wale_mut_cuf(nx,ny,nz,ng,ep_order,w_aux_gpu,coeff_deriv1_gpu,dcsidx_gpu, &
                                detady_gpu,dzitdz_gpu,les_c_wale,eps_sensor,sensor_type)
        end select

        call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DUC:J_DUC)) ! ducros
        call bcextr_var_cuf(nx, ny, nz, ng, self%w_aux_gpu(:,:,:,J_DIV:J_DIV))
        call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DIV:J_DIV)) ! div/3
        call bcextr_var_cuf(nx, ny, nz, ng, self%w_aux_gpu(:,:,:,J_LES1:J_LES1))
        call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_LES1:J_LES1))
!
        call eval_aux_les_cuf(nx, ny, nz, ng, nv, nv_aux, self%base_gpu%w_gpu, self%w_aux_gpu, &
            p0, t_min_tab, dt_tab, R_univ, rgas_gpu, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
            tol_iter_nr, mw_gpu, mwinv_gpu, visc_species_gpu, lambda_species_gpu, diffbin_species_gpu, num_t_tab, les_pr, les_sc)
        endassociate
    endsubroutine compute_aux_les

    subroutine compute_chem_aux(self)
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  nv_aux => self%nv_aux, &
                  num_t_tab => self%equation_base%num_t_tab, &
                  dt_tab => self%equation_base%dt_tab, &
                  t_min_tab => self%equation_base%t_min_tab, &
                  mw_gpu => self%mw_gpu, &
                  w_aux_gpu => self%w_aux_gpu, &
                  dcsidx_gpu => self%base_gpu%dcsidx_gpu,   &
                  detady_gpu => self%base_gpu%detady_gpu,   &
                  dzitdz_gpu => self%base_gpu%dzitdz_gpu,   &
                  nreactions   => self%equation_base%nreactions, &
                  arr_a_gpu    => self%arr_a_gpu, &
                  arr_b_gpu    => self%arr_b_gpu, &
                  arr_ea_gpu   => self%arr_ea_gpu, &
                  tb_eff_gpu   => self%tb_eff_gpu, &
                  falloff_coeffs_gpu => self%falloff_coeffs_gpu, &
                  reac_ty_gpu    => self%reac_ty_gpu, &
                  isRev_gpu    => self%isRev_gpu, &
                  r_coeffs_gpu => self%r_coeffs_gpu, &
                  p_coeffs_gpu => self%p_coeffs_gpu, &
                  kc_tab_gpu   => self%kc_tab_gpu, &
                  h298_gpu => self%h298_gpu,&
                  enable_pasr => self%equation_base%enable_pasr, &
                  les_c_yoshi=> self%equation_base%les_c_yoshi, &
                  les_c_mix=> self%equation_base%les_c_mix, &
                  les_c_eps=> self%equation_base%les_c_eps)

         call eval_chem_aux_cuf(nx,ny,nz,nv_aux,ng,nreactions,w_aux_gpu,mw_gpu,arr_a_gpu,arr_b_gpu,arr_ea_gpu,&
                                tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,kc_tab_gpu,num_t_tab,&
                                t_min_tab,dt_tab,R_univ,dcsidx_gpu,detady_gpu,dzitdz_gpu,h298_gpu,enable_pasr,les_c_yoshi,&
                                les_c_mix,les_c_eps)
        endassociate
    endsubroutine compute_chem_aux

    subroutine rk_sync(self,simpler_splitting)
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        integer, intent(in), optional :: simpler_splitting
        integer :: simpler_splitting_
        integer :: istep, lmax, iercuda
        real(rkind) :: rhodt, gamdt, alpdt
        simpler_splitting_ = 0; if (present(simpler_splitting)) simpler_splitting_ = simpler_splitting
!
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, nv => self%nv, &
                  dt => self%equation_base%dt, ep_order => self%equation_base%ep_order, &
                  weno_scheme => self%equation_base%weno_scheme, &
                  conservative_viscous => self%equation_base%conservative_viscous, &
                  eul_imin => self%equation_base%eul_imin, eul_imax => self%equation_base%eul_imax, &
                  eul_jmin => self%equation_base%eul_jmin, eul_jmax => self%equation_base%eul_jmax, &
                  eul_kmin => self%equation_base%eul_kmin, eul_kmax => self%equation_base%eul_kmax, &
                  nv_aux => self%nv_aux, flow_init => self%equation_base%flow_init, a_tr => self%equation_base%a_tr, &
                  enable_chemistry => self%equation_base%enable_chemistry, &
                  enable_ibm => self%equation_base%enable_ibm, &
                  enable_limiter => self%equation_base%enable_limiter)
!
            if (enable_ibm>0) then
             self%equation_base%ibm_force_x = 0._rkind
             self%equation_base%ibm_force_y = 0._rkind
             self%equation_base%ibm_force_z = 0._rkind
            endif
!
            do istep=1,self%equation_base%nrk
                rhodt = self%equation_base%rhork(istep)*dt
                gamdt = self%equation_base%gamrk(istep)*dt
                alpdt = self%equation_base%alprk(istep)*dt

                if (simpler_splitting_ == 1) then
                 call init_flux_simpler_cuf(nx, ny, nz, nv, self%fl_gpu, self%fln_gpu, self%fl_sav_gpu, rhodt)
                else
                 call init_flux_cuf(nx, ny, nz, nv, self%fl_gpu, self%fln_gpu, rhodt)
                endif
!
                call self%base_gpu%bcswap()
                if (enable_ibm>0) then
                 !call self%base_gpu%bcswap_corner()
                 call self%base_gpu%bcswap_edges_corners()
                 !call self%ibm_apply()
                 call self%ibm_apply_old()
                 call self%update_ghost() ! needed after application of ibm
                endif
                if (self%equation_base%enable_les>0) then
                 call self%compute_aux_les()
                else
                 call self%compute_aux()
                 call bcextr_var_cuf(nx, ny, nz, ng, self%w_aux_gpu(:,:,:,J_DIV:J_DIV))
                 call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DUC:J_DUC)) ! ducros
                 call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DIV:J_DIV)) ! div/3
                endif
                call self%euler_x(eul_imin, eul_imax)
                !@cuf iercuda=cudaDeviceSynchronize()
                if (self%grid%nymax>1) call self%euler_y(eul_jmin,eul_jmax) 
                !@cuf iercuda=cudaDeviceSynchronize()
                if (self%grid%nzmax>1) call self%euler_z(eul_kmin,eul_kmax) 
                if (self%equation_base%ibm_wm==1) call self%ibm_apply_wm ! modify viscosity at the first interface node
                if (conservative_viscous==1) then
                 call self%visflx(mode=1)  ! 0=all, 1=stag, 2=reduced
                 call self%visflx(mode=2)  ! 0=all, 1=stag, 2=reduced
                else
                 call self%visflx(mode=0)
                endif
                !@cuf iercuda=cudaDeviceSynchronize()
                call self%bc_nr() 
                if (flow_init == 1 .and. a_tr>0.) call self%tripping()
                if (enable_chemistry == 1) call self%compute_chemistry()

                if (simpler_splitting_ == 1) then
                 call update_simpler_flux_cuf(nx, ny, nz, nv, self%fl_gpu, self%fln_gpu, self%fl_sav_gpu, gamdt)
                else
                 call update_flux_cuf(nx, ny, nz, nv, self%fl_gpu, self%fln_gpu, gamdt)
                endif

                call update_field_cuf(nx, ny, nz, ng, nv, self%base_gpu%w_gpu, self%fln_gpu, self%fluid_mask_gpu)
                if (enable_limiter > 0) call self%limiter()
                call self%update_ghost(do_swap=0)
                if (enable_ibm>0) then
                  call self%ibm_compute_force(istep)
                endif
            enddo

        endassociate
    endsubroutine rk_sync

    subroutine sav_flx(self)
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        integer :: istep, lmax, iercuda
        real(rkind) :: rhodt, gamdt, alpdt
!       
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, nv => self%nv, &
                  dt => self%equation_base%dt, ep_order => self%equation_base%ep_order, &
                  weno_scheme => self%equation_base%weno_scheme, &
                  conservative_viscous => self%equation_base%conservative_viscous, &
                  eul_imin => self%equation_base%eul_imin, eul_imax => self%equation_base%eul_imax, &
                  eul_jmin => self%equation_base%eul_jmin, eul_jmax => self%equation_base%eul_jmax, &
                  eul_kmin => self%equation_base%eul_kmin, eul_kmax => self%equation_base%eul_kmax, &
                  nv_aux => self%nv_aux, flow_init => self%equation_base%flow_init, a_tr => self%equation_base%a_tr, &
                  enable_chemistry => self%equation_base%enable_chemistry, &
                  enable_ibm => self%equation_base%enable_ibm, &

                  enable_limiter => self%equation_base%enable_limiter)
        if (enable_ibm>0) then
         self%equation_base%ibm_force_x = 0._rkind
         self%equation_base%ibm_force_y = 0._rkind
         self%equation_base%ibm_force_z = 0._rkind
        endif
        istep = 1
        call init_flux_cuf(nx, ny, nz, nv, self%fl_gpu, self%fln_gpu, rhodt)

        call self%base_gpu%bcswap()
        if (enable_ibm>0) then
         !call self%base_gpu%bcswap_corner()
         call self%base_gpu%bcswap_edges_corners()
         !call self%ibm_apply()
         call self%ibm_apply_old()
         call self%update_ghost() ! needed after application of ibm
        endif

        if (self%equation_base%enable_les>0) then
         call self%compute_aux_les()
        else
         call self%compute_aux()
         call bcextr_var_cuf(nx, ny, nz, ng, self%w_aux_gpu(:,:,:,J_DIV:J_DIV))
         call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DUC:J_DUC)) ! ducros
         call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DIV:J_DIV)) ! div/3
        endif

        call self%euler_x(eul_imin, eul_imax)
        !@cuf iercuda=cudaDeviceSynchronize()
        if (self%grid%nymax>1) call self%euler_y(eul_jmin,eul_jmax)
        !@cuf iercuda=cudaDeviceSynchronize()
        if (self%grid%nzmax>1) call self%euler_z(eul_kmin,eul_kmax)
        if (self%equation_base%ibm_wm==1) call self%ibm_apply_wm ! modify viscosity at the first interface node
        if (conservative_viscous==1) then
         call self%visflx(mode=1)  ! 0=all, 1=stag, 2=reduced
         call self%visflx(mode=2)  ! 0=all, 1=stag, 2=reduced
        else
         call self%visflx(mode=0)
        endif

        !@cuf iercuda=cudaDeviceSynchronize()
        call self%bc_nr()
        if (flow_init == 1 .and. a_tr>0.) call self%tripping()
        if (enable_chemistry == 1) call self%compute_chemistry()
        if (enable_limiter > 0) call self%limiter()
        call self%update_ghost(do_swap=0)
        if (enable_ibm>0) then
         call self%ibm_compute_force(istep)
        endif

        call sav_flx_cuf(nx, ny, nz, nv, self%fl_gpu, self%fl_sav_gpu)

        endassociate
    endsubroutine sav_flx

    subroutine rosenbrock(self,simpler_splitting)
        class(equation_multideal_gpu_object), intent(inout) :: self
        integer, intent(in), optional :: simpler_splitting
        integer :: simpler_splitting_
        integer :: iercuda, ierror, step, i, j, k, lsp, istat, update_hrr
        real(rkind) :: time_start, time_end, dttry
        type(dim3) :: grid, tBlock
        simpler_splitting_ = 0 ; if(present(simpler_splitting)) simpler_splitting_ = simpler_splitting

        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  nv => self%nv,  nv_aux => self%nv_aux, &
                  nreactions => self%equation_base%nreactions, &
                  w_gpu => self%base_gpu%w_gpu, w_aux_gpu => self%w_aux_gpu, &
                  fl_gpu => self%fl_gpu, &
                  mw_gpu => self%mw_gpu, &
                  mwinv_gpu => self%mwinv_gpu, &
                  arr_a_gpu => self%arr_a_gpu, &
                  arr_b_gpu => self%arr_b_gpu, &
                  arr_ea_gpu => self%arr_ea_gpu, &
                  tb_eff_gpu => self%tb_eff_gpu, &
                  falloff_coeffs_gpu => self%falloff_coeffs_gpu, &
                  reac_ty_gpu => self%reac_ty_gpu, &
                  isRev_gpu => self%isRev_gpu, &
                  r_coeffs_gpu => self%r_coeffs_gpu, &
                  p_coeffs_gpu => self%p_coeffs_gpu, &
                  kc_tab_gpu => self%kc_tab_gpu, &
                  num_t_tab => self%equation_base%num_t_tab ,&
                  t_min_tab => self%equation_base%t_min_tab, &
                  dt_tab => self%equation_base%dt_tab, &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  trange_gpu => self%trange_gpu, &
                  nsetcv => self%equation_base%nsetcv, &
                  indx_cp_l => self%equation_base%indx_cp_l, indx_cp_r => self%equation_base%indx_cp_r, &
                  enable_pasr => self%equation_base%enable_pasr , &
                  les_c_yoshi => self%equation_base%les_c_yoshi , &
                  les_c_mix => self%equation_base%les_c_mix , &
                  les_c_eps => self%equation_base%les_c_eps , &
                  dcsidx_gpu => self%base_gpu%dcsidx_gpu, &
                  detady_gpu => self%base_gpu%detady_gpu, &
                  dzitdz_gpu => self%base_gpu%dzitdz_gpu, &
                  h298_gpu => self%h298_gpu, &
                  dtmax => self%equation_base%ros_dtmax, eps => self%equation_base%ros_eps, &
                  maxsteps => self%equation_base%ros_maxsteps, maxtry => self%equation_base%ros_maxtry)

        update_hrr = 0
        if ((self%equation_base%time_from_last_stat      + self%equation_base%dt >= self%equation_base%dtstat     ) .or. &
            (self%equation_base%time_from_last_slice     + self%equation_base%dt >= self%equation_base%dtslice    ) .or. &
            (self%equation_base%time_from_last_slice_vtr + self%equation_base%dt >= self%equation_base%dtslice_vtr) .or. &
            (self%equation_base%time_from_last_write     + self%equation_base%dt >= self%equation_base%dtsave)) then
         update_hrr = 1
        endif

!       dttry = self%equation_base%ros_dttry
!       dttry = min(dttry,dtmax) RIMOSSO
!       dttry = min(dttry,self%equation_base%dt_chem)
        dttry = self%equation_base%dt_chem

        time_start = self%equation_base%time
        time_end   = self%equation_base%time + self%equation_base%dt_chem

        !tBlock = dim3(EULERWENO_THREADS_X,EULERWENO_THREADS_Y,1)
        tBlock = dim3(8,8,8)
        grid = dim3(ceiling(real(nx)/tBlock%x),ceiling(real(ny)/tBlock%y),ceiling(real(nz)/tBlock%z))

        call rosenbrock_kernel<<<grid, tBlock>>>(nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,&
                                                 mw_gpu,nreactions,arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,&
                                                 isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,&
                                                 cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l,indx_cp_r,dttry,time_start,time_end,&
                                                 maxsteps,maxtry,eps,simpler_splitting_,enable_pasr,les_c_yoshi,les_c_mix,les_c_eps,&
                                                 dcsidx_gpu,detady_gpu,dzitdz_gpu)
         !@cuf iercuda=cudaDeviceSynchronize()

       endassociate
    endsubroutine rosenbrock

    subroutine rosenbrock_krylov(self,simpler_splitting)
        class(equation_multideal_gpu_object), intent(inout) :: self
        integer, intent(in), optional :: simpler_splitting
        integer :: simpler_splitting_
        integer :: iercuda, ierror, step, i, j, k, lsp, istat, update_hrr
        real(rkind) :: time_start, time_end, dttry
        type(dim3) :: grid, tBlock
        simpler_splitting_ = 0 ; if(present(simpler_splitting)) simpler_splitting_ = simpler_splitting

        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  nv => self%nv,  nv_aux => self%nv_aux, &
                  nreactions => self%equation_base%nreactions, &
                  w_gpu => self%base_gpu%w_gpu, w_aux_gpu => self%w_aux_gpu, &
                  fl_gpu => self%fl_gpu, &
                  fl_sav_gpu => self%fl_sav_gpu, &
                  mw_gpu => self%mw_gpu, &
                  mwinv_gpu => self%mwinv_gpu, &
                  arr_a_gpu => self%arr_a_gpu, &
                  arr_b_gpu => self%arr_b_gpu, &
                  arr_ea_gpu => self%arr_ea_gpu, &
                  tb_eff_gpu => self%tb_eff_gpu, &
                  falloff_coeffs_gpu => self%falloff_coeffs_gpu, &
                  reac_ty_gpu => self%reac_ty_gpu, &
                  isRev_gpu => self%isRev_gpu, &
                  r_coeffs_gpu => self%r_coeffs_gpu, &
                  p_coeffs_gpu => self%p_coeffs_gpu, &
                  kc_tab_gpu => self%kc_tab_gpu, &
                  num_t_tab => self%equation_base%num_t_tab ,&
                  t_min_tab => self%equation_base%t_min_tab, &
                  dt_tab => self%equation_base%dt_tab, &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  trange_gpu => self%trange_gpu, &
                  nsetcv => self%equation_base%nsetcv, &
                  indx_cp_l => self%equation_base%indx_cp_l, indx_cp_r => self%equation_base%indx_cp_r, &
                  enable_pasr => self%equation_base%enable_pasr , &
                  les_c_yoshi => self%equation_base%les_c_yoshi , &
                  les_c_mix => self%equation_base%les_c_mix , &
                  les_c_eps => self%equation_base%les_c_eps , &
                  dcsidx_gpu => self%base_gpu%dcsidx_gpu, &
                  detady_gpu => self%base_gpu%detady_gpu, &
                  dzitdz_gpu => self%base_gpu%dzitdz_gpu, &
                  h298_gpu => self%h298_gpu, &
                  dtmax => self%equation_base%ros_dtmax, eps => self%equation_base%ros_eps, &
                  maxsteps => self%equation_base%ros_maxsteps, maxtry => self%equation_base%ros_maxtry, &
                  ncoords => self%field%ncoords)

        update_hrr = 0
        if ((self%equation_base%time_from_last_stat      + self%equation_base%dt >= self%equation_base%dtstat     ) .or. &
            (self%equation_base%time_from_last_slice     + self%equation_base%dt >= self%equation_base%dtslice    ) .or. &
            (self%equation_base%time_from_last_slice_vtr + self%equation_base%dt >= self%equation_base%dtslice_vtr) .or. &
            (self%equation_base%time_from_last_write     + self%equation_base%dt >= self%equation_base%dtsave)) then
         update_hrr = 1
        endif

        dttry = self%equation_base%dt_chem
        time_start = self%equation_base%time
        time_end   = self%equation_base%time + self%equation_base%dt_chem
        tBlock = dim3(8,8,8)
        grid = dim3(ceiling(real(nx)/tBlock%x),ceiling(real(ny)/tBlock%y),ceiling(real(nz)/tBlock%z))

        call rosenbrock_krylov_kernel<<<grid, tBlock>>>(nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,&
                                                        mw_gpu,nreactions,arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,&
                                                        r_coeffs_gpu,p_coeffs_gpu,kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,&
                                                        cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l,indx_cp_r,dttry,time_start,time_end,&
                                                        simpler_splitting_,fl_sav_gpu,maxsteps,enable_pasr,les_c_yoshi,les_c_mix,les_c_eps,&
                                                        dcsidx_gpu,detady_gpu,dzitdz_gpu)

        !@cuf iercuda=cudaDeviceSynchronize()
        endassociate
    endsubroutine rosenbrock_krylov

    subroutine compute_chemistry(self)
       class(equation_multideal_gpu_object), intent(inout) :: self
!
       associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                 nv => self%nv, nv_aux => self%nv_aux, &
                 nreactions   => self%equation_base%nreactions, &
                 w_gpu        => self%base_gpu%w_gpu, &
                 w_aux_gpu    => self%w_aux_gpu, &
                 fl_gpu       => self%fl_gpu, &
                 mw_gpu       => self%mw_gpu, &
                 arr_a_gpu    => self%arr_a_gpu, &
                 arr_b_gpu    => self%arr_b_gpu, &
                 arr_ea_gpu   => self%arr_ea_gpu, &
                 tb_eff_gpu   => self%tb_eff_gpu, &
                 falloff_coeffs_gpu => self%falloff_coeffs_gpu, &
                 reac_ty_gpu    => self%reac_ty_gpu, &
                 isRev_gpu    => self%isRev_gpu, &
                 r_coeffs_gpu => self%r_coeffs_gpu, &
                 p_coeffs_gpu => self%p_coeffs_gpu, &
                 kc_tab_gpu   => self%kc_tab_gpu, &
                 num_t_tab    => self%equation_base%num_t_tab, &
                 dt_tab       => self%equation_base%dt_tab, &
                 t_min_tab    => self%equation_base%t_min_tab, &
                 cv_coeff_gpu => self%cv_coeff_gpu, &
                 trange_gpu => self%trange_gpu, &
                 nsetcv => self%equation_base%nsetcv, &
                 indx_cp_l => self%equation_base%indx_cp_l, &
                 indx_cp_r => self%equation_base%indx_cp_r)


       call compute_chemistry_cuf(nx,ny,nz,nv,nv_aux,ng,nreactions,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,&
                                  falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,&
                                  indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,tol_iter_nr)

        end associate
    endsubroutine compute_chemistry

    subroutine euler_x(self, istart, iend, istart_face, iend_face, do_update)
        class(equation_multideal_gpu_object), intent(inout) :: self       
        integer, intent(in) :: istart, iend
        integer :: lmax, weno_size, iercuda, ierror
        logical, optional :: do_update
        integer, optional :: istart_face, iend_face
        integer :: istart_face_, iend_face_
        logical :: do_update_
        type(dim3) :: grid, tBlock
        integer :: force_zero_flux_min,force_zero_flux_max

        do_update_   = .true.   ; if(present(do_update))   do_update_   = do_update
        istart_face_ = istart-1 ; if(present(istart_face)) istart_face_ = istart_face
        iend_face_   = iend     ; if(present(iend_face))   iend_face_   = iend_face

        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  nv => self%nv,  nv_aux => self%nv_aux, &
                  ep_order => self%equation_base%ep_order, &
                  force_zero_flux => self%equation_base%force_zero_flux, &
                  coeff_deriv1_gpu => self%coeff_deriv1_gpu,  dcsidx_gpu => self%base_gpu%dcsidx_gpu, &
                  fhat_gpu => self%fhat_gpu, w_gpu => self%base_gpu%w_gpu, w_aux_gpu => self%w_aux_gpu, &
                  fl_gpu => self%fl_gpu, sensor_threshold => self%equation_base%sensor_threshold, &
                  weno_scheme => self%equation_base%weno_scheme, &
                  weno_version => self%equation_base%weno_version, &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  cp_coeff_gpu => self%cp_coeff_gpu, &
                  trange_gpu => self%trange_gpu, &
                  nsetcv => self%equation_base%nsetcv, &
                  indx_cp_l => self%equation_base%indx_cp_l, indx_cp_r => self%equation_base%indx_cp_r, &
                  ep_ord_change_gpu => self%ep_ord_change_gpu, nkeep => self%equation_base%nkeep, &
                  flux_splitting => self%equation_base%flux_splitting, &
                  eul_imin => self%equation_base%eul_imin, eul_imax => self%equation_base%eul_imax, &
                  rgas_gpu => self%rgas_gpu, &
                  rho0 => self%equation_base%rho0, u0 => self%equation_base%u0)

        if(iend - istart >= 0) then  
         weno_size  = 2*weno_scheme
         lmax = ep_order/2 ! max stencil width
         force_zero_flux_min = force_zero_flux(1)
         force_zero_flux_max = force_zero_flux(2)
         tBlock = dim3(EULERWENO_THREADS_X,EULERWENO_THREADS_Y,1)

         grid = dim3(ceiling(real(iend-istart+2)/tBlock%x),ceiling(real(ny)/tBlock%y),1)

         if (flux_splitting==1) then

          call euler_x_fluxes_hybrid_rusanov_kernel<<<grid, tBlock, 0, stream1>>>(nv, nv_aux, nx, ny, nz, ng, &
              istart_face_, iend_face_, lmax, nkeep, w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, fhat_gpu, &
              force_zero_flux_min, force_zero_flux_max, weno_scheme, weno_version, &
              sensor_threshold, weno_size, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
              ep_ord_change_gpu, tol_iter_nr, rho0, u0)
         else

          call euler_x_fluxes_hybrid_kernel<<<grid, tBlock, 0, stream1>>>(nv, nv_aux, nx, ny, nz, ng, &
              istart_face_, iend_face_, lmax, nkeep, rgas_gpu, w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, fhat_gpu, &
              force_zero_flux_min, force_zero_flux_max, weno_scheme, weno_version, &
              sensor_threshold, weno_size, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
              ep_ord_change_gpu, tol_iter_nr, rho0, u0)
         endif
        endif

        if (do_update_) then
         call euler_x_update_cuf(nx, ny, nz, ng, nv, eul_imin, eul_imax, fhat_gpu, fl_gpu, dcsidx_gpu, stream1)
        endif

        endassociate
    endsubroutine euler_x

    subroutine euler_y(self, eul_jmin, eul_jmax)
        class(equation_multideal_gpu_object), intent(inout) :: self       
        integer, intent(in) :: eul_jmin, eul_jmax
        integer :: lmax, weno_size
        type(dim3) :: grid, tBlock
        integer :: force_zero_flux_min,force_zero_flux_max
       
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  nv => self%nv,  nv_aux => self%nv_aux, &
                  ep_order => self%equation_base%ep_order, &
                  force_zero_flux => self%equation_base%force_zero_flux, &
                  coeff_deriv1_gpu => self%coeff_deriv1_gpu, detady_gpu => self%base_gpu%detady_gpu, &
                  fhat_gpu => self%fhat_gpu, w_aux_gpu => self%w_aux_gpu, &
                  fl_gpu => self%fl_gpu, sensor_threshold => self%equation_base%sensor_threshold, &
                  weno_scheme => self%equation_base%weno_scheme, &
                  weno_version => self%equation_base%weno_version, &
                  indx_cp_l => self%equation_base%indx_cp_l, indx_cp_r => self%equation_base%indx_cp_r, &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  cp_coeff_gpu => self%cp_coeff_gpu, &
                  trange_gpu => self%trange_gpu, &
                  nsetcv => self%equation_base%nsetcv, &
                  ep_ord_change_gpu => self%ep_ord_change_gpu, nkeep => self%equation_base%nkeep, &
                  flux_splitting => self%equation_base%flux_splitting, &
                  rgas_gpu => self%rgas_gpu, &
                  rho0 => self%equation_base%rho0, u0 => self%equation_base%u0)

        weno_size = 2*weno_scheme
        lmax  = ep_order/2 ! max stencil width
        force_zero_flux_min = force_zero_flux(3)
        force_zero_flux_max = force_zero_flux(4)

        tBlock = dim3(EULERWENO_THREADS_X,EULERWENO_THREADS_Y,1)
        grid = dim3(ceiling(real(nx)/tBlock%x),ceiling(real(nz)/tBlock%y),1)
        if (flux_splitting==1) then
         call euler_y_hybrid_rusanov_kernel<<<grid, tBlock, 0, stream1>>>(nv, nv_aux, nx, ny, nz, ng, &
             eul_jmin, eul_jmax, lmax, nkeep, w_aux_gpu, fl_gpu, coeff_deriv1_gpu, detady_gpu, fhat_gpu, &
             force_zero_flux_min, force_zero_flux_max, weno_scheme, weno_version, &
             sensor_threshold, weno_size, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
             ep_ord_change_gpu, tol_iter_nr, rho0, u0)
        else
         call euler_y_hybrid_kernel<<<grid, tBlock, 0, stream1>>>(nv, nv_aux, nx, ny, nz, ng, &
             eul_jmin, eul_jmax, lmax, nkeep, rgas_gpu, w_aux_gpu, fl_gpu, coeff_deriv1_gpu, detady_gpu, fhat_gpu, &
             force_zero_flux_min, force_zero_flux_max, weno_scheme, weno_version, &
             sensor_threshold, weno_size, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
             ep_ord_change_gpu, tol_iter_nr, rho0, u0)
        endif
        call euler_y_update_cuf(nx, ny, nz, ng, nv, eul_jmin, eul_jmax, fhat_gpu, fl_gpu, detady_gpu, stream1)
        endassociate
    endsubroutine euler_y

    subroutine euler_z(self, eul_kmin, eul_kmax)
        class(equation_multideal_gpu_object), intent(inout) :: self       
        integer, intent(in) :: eul_kmin, eul_kmax
        integer :: lmax, weno_size
        type(dim3) :: grid, tBlock
        integer :: force_zero_flux_min, force_zero_flux_max
       
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  nv => self%nv,  nv_aux => self%nv_aux, &
                  ep_order => self%equation_base%ep_order, &
                  force_zero_flux => self%equation_base%force_zero_flux, &
                  coeff_deriv1_gpu => self%coeff_deriv1_gpu, dzitdz_gpu => self%base_gpu%dzitdz_gpu, &
                  fhat_gpu => self%fhat_gpu, w_aux_gpu => self%w_aux_gpu, &
                  fl_gpu => self%fl_gpu, sensor_threshold => self%equation_base%sensor_threshold, &
                  weno_scheme => self%equation_base%weno_scheme, &
                  weno_version => self%equation_base%weno_version, &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  cp_coeff_gpu => self%cp_coeff_gpu, &
                  trange_gpu => self%trange_gpu, &
                  nsetcv => self%equation_base%nsetcv, &
                  indx_cp_l => self%equation_base%indx_cp_l, indx_cp_r => self%equation_base%indx_cp_r, &
                  ep_ord_change_gpu => self%ep_ord_change_gpu, nkeep => self%equation_base%nkeep, &
                  flux_splitting => self%equation_base%flux_splitting, &
                  rgas_gpu => self%rgas_gpu,&
                  rho0 => self%equation_base%rho0, u0 => self%equation_base%u0, &
                  inflow_random_plane => self%equation_base%inflow_random_plane, &
                  inflow_random_plane_gpu => self%inflow_random_plane_gpu)

        weno_size = 2*weno_scheme
        lmax = ep_order/2 ! max stencil width
        force_zero_flux_min = force_zero_flux(5)
        force_zero_flux_max = force_zero_flux(6)
        tBlock = dim3(EULERWENO_THREADS_X,EULERWENO_THREADS_Y,1)
        grid = dim3(ceiling(real(nx)/tBlock%x),ceiling(real(ny)/tBlock%y),1)
        if (flux_splitting==1) then
         call euler_z_hybrid_rusanov_kernel<<<grid, tBlock, 0, 0>>>(nv, nv_aux, nx, ny, nz, ng, &
             eul_kmin, eul_kmax, lmax, nkeep, w_aux_gpu, fl_gpu, coeff_deriv1_gpu, dzitdz_gpu, fhat_gpu, &
             force_zero_flux_min, force_zero_flux_max, weno_scheme, weno_version, &
             sensor_threshold, weno_size, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
             ep_ord_change_gpu, tol_iter_nr, rho0, u0)
        else
         call euler_z_hybrid_kernel<<<grid, tBlock, 0, 0>>>(nv, nv_aux, nx, ny, nz, ng, &
             eul_kmin, eul_kmax, lmax, nkeep, rgas_gpu, w_aux_gpu, fl_gpu, coeff_deriv1_gpu, dzitdz_gpu, fhat_gpu, &
             force_zero_flux_min, force_zero_flux_max, weno_scheme, weno_version, &
             sensor_threshold, weno_size, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
             ep_ord_change_gpu, tol_iter_nr, rho0, u0)
        endif
        call euler_z_update_cuf(nx, ny, nz, ng, nv, eul_kmin, eul_kmax, fhat_gpu, fl_gpu, dzitdz_gpu, 0_cuda_stream_kind)
        if (self%equation_base%recyc) call get_crandom_f(inflow_random_plane(2:self%equation_base%jbl_inflow,1:nz,1:3))
        endassociate
    endsubroutine euler_z

    subroutine recyc_exchange(self)
     class(equation_multideal_gpu_object), intent(inout) :: self
     integer :: indx
     integer, dimension(4) :: req
     integer :: kcoordsendto1,kcoordsendto2
     integer :: kcoordrecvfrom1,kcoordrecvfrom2
     integer :: sendto1,sendto2
     integer :: recvfrom1,recvfrom2
     integer :: kshiftglob
     integer :: n1_start_send,n1_end_send,n2_start_send,n2_end_send
     integer :: n1_start_recv,n1_end_recv,n2_start_recv,n2_end_recv
     integer :: iercuda

     associate(nzmax => self%equation_base%grid%nzmax, ng => self%equation_base%grid%ng, &
               nx => self%equation_base%field%nx, ny => self%equation_base%field%ny, nz => self%equation_base%field%nz, &
               ncoords => self%equation_base%field%ncoords, mp_cart => self%equation_base%field%mp_cart, &
               nblocks => self%equation_base%field%nblocks, &
               iermpi => self%mpi_err, w_gpu => self%base_gpu%w_gpu, nv => self%equation_base%nv, &
               wbuf1s_gpu => self%base_gpu%wbuf1s_gpu, wbuf2r_gpu => self%base_gpu%wbuf2r_gpu, &
               wbuf1r_gpu => self%base_gpu%wbuf1r_gpu, wrecyc_gpu => self%wrecyc_gpu, &
               ibrecyc => self%equation_base%ib_recyc, irecyc => self%equation_base%i_recyc, &
               nv_recyc => self%equation_base%nv_recyc)


     kshiftglob    = nzmax/2 ! global shift in the spanwise direction (between 0 and nzmax-1)
     n1_start_send = 1
     n1_end_send   = nz-mod(kshiftglob,nz)
     n2_start_send = n1_end_send+1
     n2_end_send   = nz
     n1_start_recv = 1+mod(kshiftglob,nz)
     n1_end_recv   = nz
     n2_start_recv = 1
     n2_end_recv   = mod(kshiftglob,nz)

     req = mpi_request_null

     if (ncoords(1)==ibrecyc) then ! Send data
      kcoordsendto1 = ncoords(3)+kshiftglob/nz
      kcoordsendto2 = kcoordsendto1+1
      kcoordsendto1 = mod(kcoordsendto1,nblocks(3))
      kcoordsendto2 = mod(kcoordsendto2,nblocks(3))
      call mpi_cart_rank(mp_cart,[0,0,kcoordsendto1],sendto1,iermpi)
      call mpi_cart_rank(mp_cart,[0,0,kcoordsendto2],sendto2,iermpi)
      call recyc_exchange_cuf_1(irecyc, w_gpu, wbuf1s_gpu, nx, ny, nz, ng)
      indx = nv_recyc*ng*ny*nz
      call mpi_isend(wbuf1s_gpu,indx,mpi_prec,sendto1,2000,mp_cart,req(1),iermpi)
      call mpi_isend(wbuf1s_gpu,indx,mpi_prec,sendto2,3000,mp_cart,req(2),iermpi)
    !! call mpi_ssend(wbuf1s_gpu,indx,mpi_prec,0,2000,mp_cartx,iermpi)
     endif
     if (ncoords(1)==0) then ! Receive data
      kcoordrecvfrom1 = ncoords(3)-kshiftglob/nz+nblocks(3)
      kcoordrecvfrom2 = kcoordrecvfrom1-1
      kcoordrecvfrom1 = mod(kcoordrecvfrom1,nblocks(3))
      kcoordrecvfrom2 = mod(kcoordrecvfrom2,nblocks(3))
      call mpi_cart_rank(mp_cart,[ibrecyc,0,kcoordrecvfrom1],recvfrom1,iermpi)
      call mpi_cart_rank(mp_cart,[ibrecyc,0,kcoordrecvfrom2],recvfrom2,iermpi)
      indx = nv_recyc*ng*ny*nz
      call mpi_irecv(wbuf1r_gpu,indx,mpi_prec,recvfrom1,2000,mp_cart,req(3),iermpi)
      call mpi_irecv(wbuf2r_gpu,indx,mpi_prec,recvfrom2,3000,mp_cart,req(4),iermpi)
     endif
     call mpi_waitall(4,req,mpi_statuses_ignore,iermpi)
     if (ncoords(1)==0) then
      call recyc_exchange_cuf_2(n1_start_recv, n1_start_send, n1_end_recv, wrecyc_gpu, wbuf1r_gpu, nx, ny, nz, ng, nv_recyc)
      call recyc_exchange_cuf_3(n2_start_recv, n2_start_send, n2_end_recv, wrecyc_gpu, wbuf2r_gpu, nx, ny, nz, ng, nv_recyc)
     endif
     endassociate

    end subroutine recyc_exchange

    subroutine bc_nr(self)
        class(equation_multideal_gpu_object), intent(inout) :: self      
        integer :: ilat, dir, start_or_end
        type(dim3) :: grid, tBlock

        associate(bctags_nr => self%equation_base%bctags_nr, nx => self%nx, ny => self%ny, nz => self%nz, nv => self%nv, &
                  ng => self%ng, w_aux_gpu => self%w_aux_gpu, w_gpu => self%base_gpu%w_gpu, &
                  fl_gpu => self%fl_gpu, dcsidx_gpu => self%base_gpu%dcsidx_gpu,  detady_gpu => self%base_gpu%detady_gpu, &
                  dzitdz_gpu => self%base_gpu%dzitdz_gpu, &
                  indx_cp_l => self%equation_base%indx_cp_l, indx_cp_r => self%equation_base%indx_cp_r, &
                  cp_coeff_gpu => self%cp_coeff_gpu, winf_gpu => self%winf_gpu, &
                  trange_gpu => self%trange_gpu, &
                  nsetcv => self%equation_base%nsetcv, &
                  cv_coeff_gpu => self%cv_coeff_gpu, rgas_gpu => self%rgas_gpu)

        do ilat=1,6! loop on all sides of the boundary (3D -> 6)
            if(bctags_nr(ilat) > 0) then
                dir          = (ilat-1)/2   +1
                start_or_end = mod(ilat-1,2)+1
                ! 1 - NR
                ! 2 - relax
                ! 6 - reflective wall
                if(dir == 1) then
                    tBlock = dim3(EULERWENO_THREADS_X,EULERWENO_THREADS_Y,1)
                    grid = dim3(ceiling(real(ny)/tBlock%x),ceiling(real(nz)/tBlock%y),1)
                    call bc_nr_lat_x_kernel<<<grid, tBlock, 0, 0>>>(start_or_end, bctags_nr(ilat), &
                         nx, ny, nz, ng, nv, w_aux_gpu, w_gpu, fl_gpu, dcsidx_gpu, &
                         indx_cp_l, indx_cp_r, rgas_gpu, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, winf_gpu)
                endif
                if(dir == 2) then
                    tBlock = dim3(EULERWENO_THREADS_X,EULERWENO_THREADS_Y,1)
                    grid = dim3(ceiling(real(nx)/tBlock%x),ceiling(real(nz)/tBlock%y),1)
                    call bc_nr_lat_y_kernel<<<grid, tBlock, 0, 0>>>(start_or_end, bctags_nr(ilat), &
                         nx, ny, nz, ng, nv, w_aux_gpu, w_gpu, fl_gpu, detady_gpu, &
                         indx_cp_l, indx_cp_r, rgas_gpu, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, winf_gpu)
                endif
                if(dir == 3) then
                    tBlock = dim3(EULERWENO_THREADS_X,EULERWENO_THREADS_Y,1)
                    grid = dim3(ceiling(real(nx)/tBlock%x),ceiling(real(ny)/tBlock%y),1)
                    call bc_nr_lat_z_kernel<<<grid, tBlock, 0, 0>>>(start_or_end, bctags_nr(ilat), &
                         nx, ny, nz, ng, nv, w_aux_gpu, w_gpu, fl_gpu, dzitdz_gpu, &
                         indx_cp_l, indx_cp_r, rgas_gpu, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, winf_gpu)
                endif
            endif
        enddo
        endassociate

    endsubroutine bc_nr

    subroutine update_ghost(self, do_swap)
        class(equation_multideal_gpu_object), intent(inout) :: self      
        integer, intent(in), optional :: do_swap
        integer :: do_swap_
        integer :: iercuda, ilat, i, j, k, l, m
        real(rkind) :: app

        do_swap_ = 1 ; if (present(do_swap)) do_swap_ = do_swap

        if (self%equation_base%recyc) call self%recyc_exchange()

        do ilat=1,6! loop on all sides of the boundary (3D -> 6)
          select case(self%equation_base%bctags(ilat))
            case(0)
            case(1)
                call bcfree_cuf(ilat, self%nx, self%ny, self%nz, self%ng, self%nv, self%winf_gpu, self%base_gpu%w_gpu)
            case(2)
                call bcextr_cuf(ilat, self%nx, self%ny, self%nz, self%ng, self%nv, self%base_gpu%w_gpu)
            case(4)
                call bcextr_sub_cuf(ilat, self%nx, self%ny, self%nz, self%ng, self%nv, self%nv_aux, &
                     self%equation_base%p0, self%rgas_gpu, self%base_gpu%w_gpu, self%w_aux_gpu, &
                     self%equation_base%indx_cp_l, self%equation_base%indx_cp_r, self%cv_coeff_gpu, &
                     self%equation_base%nsetcv, self%trange_gpu)
            case(5)
                call bcsym_cuf(ilat, self%nx, self%ny, self%nz, self%ng, self%base_gpu%w_gpu)
            case(6)
                call bcwall_cuf(ilat,self%nx,self%ny,self%nz,self%ng,self%nv,self%nv_aux,self%base_gpu%w_gpu,&
                     self%w_aux_gpu,self%rgas_gpu,self%equation_base%indx_cp_l,self%equation_base%indx_cp_r,&
                     self%cv_coeff_gpu,self%equation_base%nsetcv,self%trange_gpu,tol_iter_nr,self%equation_base%T_wall)
            case(7)
                call bcoutopenfoam_cuf(ilat, self%nx, self%ny, self%nz, self%ng, self%nv, self%base_gpu%w_gpu)
            case(8)
                call bcwall_ad_cuf(ilat, self%nx, self%ny, self%nz, self%ng, self%base_gpu%w_gpu)
            case(9)
                call bclam_cuf(ilat, self%nx, self%ny, self%nz, self%ng, self%nv, self%nv_aux, &
                     self%base_gpu%w_gpu, self%w_aux_gpu, self%wmean_gpu, self%equation_base%p0, self%equation_base%rmixt0, &
                     self%equation_base%indx_cp_l, self%equation_base%indx_cp_r, self%cv_coeff_gpu, &
                     self%equation_base%nsetcv, self%trange_gpu)
            case(10)
                call self%bcrecyc(ilat)
            case(26)
                call bcwall_jcf_cuf(ilat,self%nx,self%ny,self%nz,self%ng,self%nv,self%nv_aux,self%base_gpu%w_gpu,&
                     self%w_aux_gpu,self%cv_coeff_gpu,self%equation_base%nsetcv,self%trange_gpu,self%rgas_gpu,&
                     self%equation_base%indx_cp_l,self%equation_base%indx_cp_r,self%equation_base%T_wall,&
                     tol_iter_nr,self%base_gpu%x_gpu,self%base_gpu%z_gpu,self%equation_base%jcf_jet_num, &
                     self%jcf_parbc_gpu,self%jcf_coords_gpu,self%equation_base%jcf_jet_rad,self%equation_base%jcf_relax_factor)
            case(28)
                call bcwall_ad_jcf_cuf(ilat,self%nx,self%ny,self%nz,self%ng,self%base_gpu%w_gpu,&
                     self%base_gpu%x_gpu,self%base_gpu%z_gpu,&
                     self%equation_base%jcf_jet_num,self%jcf_parbc_gpu,self%jcf_coords_gpu,self%equation_base%jcf_jet_rad,&
                     self%equation_base%jcf_relax_factor)
            case(46)
                call bcwall_LE_cuf(ilat,self%nx,self%ny,self%nz,self%ng,self%nv,self%nv_aux,self%base_gpu%w_gpu,&
                     self%w_aux_gpu,self%rgas_gpu,self%equation_base%indx_cp_l,self%equation_base%indx_cp_r,&
                     self%cv_coeff_gpu,self%equation_base%nsetcv,self%trange_gpu,tol_iter_nr,self%equation_base%T_wall,self%init_mf_gpu)
            case(56)
                if (self%equation_base%enable_les>0) then
                 call self%compute_aux_les()
                else
                 call self%compute_aux()
                 !@cuf iercuda=cudaDeviceSynchronize()
                endif

                call bcwall_FC_cuf(ilat,self%nx,self%ny,self%nz,self%ng,self%nv,self%nv_aux,self%base_gpu%w_gpu,&
                     self%w_aux_gpu,self%rgas_gpu,self%mw_gpu,self%equation_base%indx_cp_l,self%equation_base%indx_cp_r,&
                     self%cv_coeff_gpu,self%equation_base%nsetcv,self%trange_gpu,tol_iter_nr,self%equation_base%T_wall,&
                     self%base_gpu%y_gpu,self%equation_base%idx_O,self%equation_base%idx_O2,self%equation_base%idx_N,&
                     self%equation_base%idx_N2,self%equation_base%idx_NO)
          endselect
        enddo

        if (do_swap_ == 1) call self%base_gpu%bcswap()

    endsubroutine update_ghost

    subroutine bcrecyc(self, ilat)
    !
     class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
    ! Apply recycling-rescaling boundary condition
    !
     integer, intent(in) :: ilat
     integer :: ntot,jmin,jmax
    !
     if (ilat == 1) then
      associate(nx => self%nx, ny => self%ny, nz => self%nz, nzmax => self%grid%nzmax, ng => self%ng, nv => self%nv, &
               wrecycav_gpu => self%wrecycav_gpu, wrecyc_gpu => self%wrecyc_gpu, &
               mp_cartz => self%field%mp_cartz, iermpi => self%mpi_err, &
               p0 => self%equation_base%p0, w_gpu => self%base_gpu%w_gpu, &
               nv_aux => self%nv_aux, w_aux_gpu => self%w_aux_gpu, &
               wmean_gpu => self%wmean_gpu, weta_inflow_gpu  => self%weta_inflow_gpu, &
               map_j_inn_gpu => self%map_j_inn_gpu, map_j_out_gpu => self%map_j_out_gpu, &
               map_j_out_blend_gpu => self%map_j_out_blend_gpu, &
               eta_recyc_blend_gpu => self%eta_recyc_blend_gpu, &
               yplus_inflow_gpu => self%yplus_inflow_gpu, eta_inflow_gpu => self%eta_inflow_gpu, &
               yplus_recyc_gpu => self%yplus_recyc_gpu, eta_recyc_gpu => self%eta_recyc_gpu, &
               betarecyc => self%equation_base%betarecyc, i_recyc => self%equation_base%i_recyc, &
               glund1 => self%equation_base%glund1, &
               weta_inflow2_gpu  => self%weta_inflow2_gpu, map_j_inn2_gpu => self%map_j_inn2_gpu, &
               map_j_out2_gpu  => self%map_j_out2_gpu, map_j_out_blend2_gpu => self%map_j_out_blend2_gpu, &
               eta_recyc_blend2_gpu => self%eta_recyc_blend2_gpu, yplus_inflow2_gpu => self%yplus_inflow2_gpu, &
               eta_inflow2_gpu => self%eta_inflow2_gpu, yplus_recyc2_gpu => self%yplus_recyc2_gpu, &
               eta_recyc2_gpu => self%eta_recyc2_gpu, betarecyc2 => self%equation_base%betarecyc2, &
               glund12 => self%equation_base%glund12, &
               indx_cp_l    => self%equation_base%indx_cp_l, &
               indx_cp_r    => self%equation_base%indx_cp_r, &
               cp_coeff_gpu => self%cp_coeff_gpu, &
               trange_gpu => self%trange_gpu, &
               nsetcv => self%equation_base%nsetcv, &
               rmixt0 => self%equation_base%rmixt0, &
               rand_type => self%equation_base%rand_type, &
               t0 => self%equation_base%t0, &
               u0 => self%equation_base%u0, &
               delta0 => self%equation_base%delta0, &
               cv_coeff_gpu => self%cv_coeff_gpu, &
               inflow_random_plane => self%equation_base%inflow_random_plane, &
               inflow_random_plane_gpu => self%inflow_random_plane_gpu, & 
               nv_recyc => self%equation_base%nv_recyc)
      ! Compute spanwise averages at the recycling station
      call bcrecyc_cuf_1(nx, ny, nz, ng, nv_recyc, wrecycav_gpu, wrecyc_gpu)

      !     call get_crandom_f(inflow_random_plane(2:ny,1:nz,1:3)) ! this is done in euler z to overlap
      inflow_random_plane_gpu = inflow_random_plane

      ntot = ng*ny*nv_recyc
      call mpi_allreduce(MPI_IN_PLACE,wrecycav_gpu,ntot,mpi_prec,mpi_sum,mp_cartz,iermpi)

      ! Remove average
      call bcrecyc_cuf_2(nx, ny, nz, nzmax, ng, wrecycav_gpu, wrecyc_gpu)

      ! Apply bc recycling
      if (.not. self%equation_base%double_bl_case) then
       jmin = 1 ; jmax = ny
       call bcrecyc_cuf_3(nx,ny,nz,nv,nv_aux,ng,p0,u0,rmixt0,w_gpu,w_aux_gpu,wmean_gpu,wrecyc_gpu, &
                          weta_inflow_gpu,map_j_inn_gpu,map_j_out_gpu,map_j_out_blend_gpu, &
                          yplus_inflow_gpu,eta_inflow_gpu,yplus_recyc_gpu,eta_recyc_gpu, &
                          eta_recyc_blend_gpu,betarecyc,glund1,inflow_random_plane_gpu, &
                          indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,rand_type)
      else
       jmin = 1 ; jmax = ny/2
       call bcrecyc_doublebl_cuf_3(jmin,jmax,nx,ny,nz,nv,nv_aux,ng,p0,u0,rmixt0,w_gpu,w_aux_gpu,wmean_gpu,wrecyc_gpu, &
                                   weta_inflow_gpu,map_j_inn_gpu,map_j_out_gpu,map_j_out_blend_gpu, &
                                   yplus_inflow_gpu,eta_inflow_gpu,yplus_recyc_gpu,eta_recyc_gpu, &
                                   eta_recyc_blend_gpu,betarecyc,glund1,inflow_random_plane_gpu, &
                                   indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,rand_type)

       jmin = ny/2+1 ; jmax = ny
       call bcrecyc_doublebl_cuf_3(jmin,jmax,nx,ny,nz,nv,nv_aux,ng,p0,u0,rmixt0,w_gpu,w_aux_gpu,wmean_gpu,wrecyc_gpu, &
                                   weta_inflow2_gpu,map_j_inn2_gpu,map_j_out2_gpu,map_j_out_blend2_gpu, &
                                   yplus_inflow2_gpu,eta_inflow2_gpu,yplus_recyc2_gpu,eta_recyc2_gpu, &
                                   eta_recyc_blend2_gpu,betarecyc2,glund12,inflow_random_plane_gpu, &
                                   indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,rand_type)
      endif
      endassociate
     endif

    end subroutine bcrecyc

    subroutine initialize(self, filename)
        !< Initialize the equation.
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        character(*) , intent(in) :: filename !< Input file name.
        integer :: lsp

        call get_mpi_basic_info(self%nprocs, self%myrank, self%masterproc, self%mpi_err)

        call self%equation_base%initialize(filename)
        self%nx = self%equation_base%field%nx
        self%ny = self%equation_base%field%ny
        self%nz = self%equation_base%field%nz
        self%ng = self%equation_base%grid%ng
        self%nv = self%equation_base%nv
        self%nv_aux = self%equation_base%nv_aux

        call self%point_to_field(self%equation_base%field)
        call self%point_to_grid(self%equation_base%grid)
        self%num_iter = self%equation_base%num_iter
        self%time0    = self%equation_base%time0
        self%icyc0    = self%equation_base%icyc0

        call self%base_gpu%initialize(self%equation_base%field)

        if(self%equation_base%debug_memory>0) then
            call self%field%check_cpu_mem(description="--Initialize-start--")
            call self%base_gpu%check_gpu_mem(description="--Initialize-start--")
        endif

        call self%base_gpu%copy_cpu_gpu()

        if(self%equation_base%debug_memory>0) then
            call self%field%check_cpu_mem(description="--Initialize-first-GPU-usage--")
            call self%base_gpu%check_gpu_mem(description="--Initialize-first-GPU-usage--")
        endif

        !self%nv = self%equation_base%field%nv
        !self%nv_aux        =  N_S+9

        call self%alloc()

        self%w_aux_gpu = self%equation_base%w_aux
        self%fluid_mask_gpu = self%equation_base%fluid_mask
        self%fluid_mask_ini_gpu = self%fluid_mask_gpu
        self%ep_ord_change_gpu = self%equation_base%ep_ord_change
        self%winf_gpu       = self%equation_base%winf

        self%cp_coeff_gpu = self%equation_base%cp_coeff
        self%cv_coeff_gpu = self%equation_base%cv_coeff
        self%trange_gpu   = self%equation_base%trange
        self%mw_gpu   = self%equation_base%mw
        self%rgas_gpu = self%equation_base%rgas
        self%h298_gpu = self%equation_base%h298
        self%init_mf_gpu = self%equation_base%init_mf
        self%visc_species_gpu = self%equation_base%visc_species
        self%lambda_species_gpu = self%equation_base%lambda_species
        self%diffbin_species_gpu = self%equation_base%diffbin_species
!
        do lsp=1,N_S
         self%mwinv_gpu(lsp) = 1._rkind/self%equation_base%mw(lsp)
        enddo
!        
        if (self%equation_base%enable_chemistry > 0) then
         self%arr_a_gpu = self%equation_base%arr_a
         self%arr_b_gpu = self%equation_base%arr_b
         self%arr_ea_gpu = self%equation_base%arr_ea
         self%tb_eff_gpu = self%equation_base%tb_eff
         self%falloff_coeffs_gpu = self%equation_base%falloff_coeffs
         self%reac_ty_gpu = self%equation_base%reac_ty
         self%isRev_gpu = self%equation_base%isRev

         self%r_coeffs_gpu = self%equation_base%r_coeffs
         self%p_coeffs_gpu = self%equation_base%p_coeffs

         self%kc_tab_gpu = self%equation_base%kc_tab
        endif

        !Mixture Fraction
        if (self%equation_base%enable_Zbil>0) then
         self%NainSp_gpu = self%equation_base%NainSp
         self%coeff_EoI_gpu = self%equation_base%coeff_EoI
         self%Beta0_gpu = self%equation_base%Beta0
         self%aw_EoI_gpu = self%equation_base%aw_EoI
        endif
        self%N_EoI_gpu = self%equation_base%N_EoI
        
        ! jcf
        if (self%equation_base%enable_jcf > 0) then
         self%jcf_parbc_gpu  = self%equation_base%jcf_parbc
         self%jcf_coords_gpu = self%equation_base%jcf_coords
        endif
!
        if (self%equation_base%num_probe>0) then
            self%probe_coeff_gpu = self%equation_base%probe_coeff
            self%ijk_probe_gpu   = self%equation_base%ijk_probe
        endif
!
        self%wmean_gpu = self%equation_base%wmean
!
        allocate(self%coeff_deriv1_gpu(1:4,4))
        allocate(self%coeff_deriv2_gpu(0:4,4))
        self%coeff_deriv1_gpu = self%equation_base%coeff_deriv1
        self%coeff_deriv2_gpu = self%equation_base%coeff_deriv2
!
        if (self%equation_base%recyc) then
            self%yplus_inflow_gpu    = self%equation_base%yplus_inflow
            self%eta_inflow_gpu      = self%equation_base%eta_inflow
            self%yplus_recyc_gpu     = self%equation_base%yplus_recyc
            self%eta_recyc_gpu       = self%equation_base%eta_recyc
            self%eta_recyc_blend_gpu = self%equation_base%eta_recyc_blend
            self%map_j_inn_gpu       = self%equation_base%map_j_inn
            self%map_j_out_gpu       = self%equation_base%map_j_out
            self%map_j_out_blend_gpu = self%equation_base%map_j_out_blend
            self%weta_inflow_gpu     = self%equation_base%weta_inflow
         if (self%equation_base%double_bl_case) then
            self%yplus_inflow2_gpu    = self%equation_base%yplus_inflow2
            self%eta_inflow2_gpu      = self%equation_base%eta_inflow2
            self%yplus_recyc2_gpu     = self%equation_base%yplus_recyc2
            self%eta_recyc2_gpu       = self%equation_base%eta_recyc2
            self%eta_recyc_blend2_gpu = self%equation_base%eta_recyc_blend2
            self%map_j_inn2_gpu       = self%equation_base%map_j_inn2
            self%map_j_out2_gpu       = self%equation_base%map_j_out2
            self%map_j_out_blend2_gpu = self%equation_base%map_j_out_blend2
            self%weta_inflow2_gpu     = self%equation_base%weta_inflow2
         endif
        endif
!
!        if (self%equation_base%enable_ibm > 0) call self%ibm_alloc_gpu()
        if (self%equation_base%enable_ibm > 0) call self%ibm_alloc_gpu_old()
!
        if (self%equation_base%enable_insitu > 0) then
            call self%insitu_alloc_gpu()
        endif
!
        if (self%equation_base%debug_memory>0) then
            call self%field%check_cpu_mem(description="Initialize-completed")
            call self%base_gpu%check_gpu_mem(description="Initialize-completed")
        endif
!
        ! Allocate field_gpu variables
        !call self%base_gpu%alloc(field=self%field, nv_aux=self%nv_aux)

        !! Use base_gpu as pointee
        !self%field         => self%base_gpu%field
        !self%grid          => self%base_gpu%field%grid
!
        self%mpi_err = cudaStreamCreate(stream1)
    endsubroutine initialize

    subroutine alloc(self)
        class(equation_multideal_gpu_object), intent(inout) :: self 
        associate(nx => self%nx, ny => self%ny, nz => self%nz,  &
                  ng => self%ng, nv => self%nv, nv_aux => self%nv_aux, &
                  weno_scheme => self%equation_base%weno_scheme,&
                  nsetcv => self%equation_base%nsetcv ,&
                  num_t_tab => self%equation_base%num_t_tab ,&
                  indx_cp_l => self%equation_base%indx_cp_l ,&
                  indx_cp_r => self%equation_base%indx_cp_r)

        allocate(self%winf_gpu(nv))
        allocate(self%w_aux_gpu(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng, nv_aux))
        allocate(self%fl_gpu(1:nx, 1:ny, 1:nz, nv))
        if (self%equation_base%operator_splitting == 1) then 
         allocate(self%fl_sav_gpu(1:nx, 1:ny, 1:nz, nv))
        endif
        allocate(self%fln_gpu(1:nx, 1:ny, 1:nz, nv))
        allocate(self%w_var(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng, 1))
        allocate(self%w_var_t(1, 1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng))
        allocate(self%fhat_gpu(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng, nv))
        !allocate(self%w_aux_trans_gpu(1-ng:ny+ng, 1-ng:nx+ng, 1-ng:nz+ng, 8))
        !allocate(self%fhat_trans_gpu(1-ng:ny+ng, 1-ng:nx+ng, 1-ng:nz+ng, nv))
        !allocate(self%fl_trans_gpu(1:ny, 1:nx, 1:nz, nv))
!       allocate(self%gplus_x_gpu (0:nx,ny,nv,2*weno_scheme))
!       allocate(self%gminus_x_gpu(0:nx,ny,nv,2*weno_scheme))
!       allocate(self%gplus_y_gpu (nx,nz,nv,2*weno_scheme))
!       allocate(self%gminus_y_gpu(nx,nz,nv,2*weno_scheme))
!       allocate(self%gplus_z_gpu (nx,ny,nv,2*weno_scheme))
!       allocate(self%gminus_z_gpu(nx,ny,nv,2*weno_scheme))
        allocate(self%fluid_mask_gpu(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng))
        allocate(self%fluid_mask_ini_gpu(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng))
        allocate(self%ep_ord_change_gpu(0:nx, 0:ny, 0:nz, 1:3))

        allocate(self%wmean_gpu(1-ng:nx+ng+1,1:ny,nv))

        if (self%equation_base%recyc) then
         allocate(self%wrecycav_gpu(ng,ny,self%equation_base%nv_recyc))
         allocate(self%wrecyc_gpu(ng,ny,nz,self%equation_base%nv_recyc))
         allocate(self%yplus_inflow_gpu(1-ng:ny+ng))
         allocate(self%eta_inflow_gpu(1-ng:ny+ng))
         allocate(self%yplus_recyc_gpu(1-ng:ny+ng))
         allocate(self%eta_recyc_gpu(1-ng:ny+ng))
         allocate(self%eta_recyc_blend_gpu(1-ng:ny+ng))
         allocate(self%map_j_inn_gpu(1:ny))
         allocate(self%map_j_out_gpu(1:ny))
         allocate(self%map_j_out_blend_gpu(1:ny))
         allocate(self%weta_inflow_gpu(1:ny))
         allocate(self%inflow_random_plane_gpu(1:ny,1:nz,3))
         if (self%equation_base%double_bl_case) then
          allocate(self%yplus_inflow2_gpu(1-ng:ny+ng))
          allocate(self%eta_inflow2_gpu(1-ng:ny+ng))
          allocate(self%yplus_recyc2_gpu(1-ng:ny+ng))
          allocate(self%eta_recyc2_gpu(1-ng:ny+ng))
          allocate(self%eta_recyc_blend2_gpu(1-ng:ny+ng))
          allocate(self%map_j_inn2_gpu(1:ny))
          allocate(self%map_j_out2_gpu(1:ny))
          allocate(self%map_j_out_blend2_gpu(1:ny))
          allocate(self%weta_inflow2_gpu(1:ny))
         endif
        endif

        allocate(self%cv_coeff_gpu(indx_cp_l:indx_cp_r+2,N_S,nsetcv))
        allocate(self%cp_coeff_gpu(indx_cp_l:indx_cp_r+2,N_S,nsetcv))
        allocate(self%trange_gpu(N_S,nsetcv+1))
        allocate(self%mw_gpu(N_S),self%mwinv_gpu(N_S))
        allocate(self%rgas_gpu(N_S))
        allocate(self%h298_gpu(N_S))
        allocate(self%init_mf_gpu(N_S))
        allocate(self%visc_species_gpu(num_t_tab+1,N_S))
        allocate(self%lambda_species_gpu(num_t_tab+1,N_S))
        allocate(self%diffbin_species_gpu(num_t_tab+1,N_S,N_S))

        if (self%equation_base%enable_chemistry > 0) then
         allocate(self%arr_a_gpu(self%equation_base%nreactions,2))
         allocate(self%arr_b_gpu(self%equation_base%nreactions,2))
         allocate(self%arr_ea_gpu(self%equation_base%nreactions,2))
         allocate(self%falloff_coeffs_gpu(self%equation_base%nreactions,5))
         allocate(self%tb_eff_gpu(self%equation_base%nreactions,N_S))
         allocate(self%reac_ty_gpu(self%equation_base%nreactions))
         allocate(self%isRev_gpu(self%equation_base%nreactions))

         allocate(self%r_coeffs_gpu(self%equation_base%nreactions,N_S))
         allocate(self%p_coeffs_gpu(self%equation_base%nreactions,N_S))

         allocate(self%kc_tab_gpu(num_t_tab+1,self%equation_base%nreactions))
        endif

        ! Mixture Fraction
        if (self%equation_base%enable_Zbil>0) then
         allocate(self%NainSp_gpu(N_S,self%equation_base%N_EoI))
         allocate(self%coeff_EoI_gpu(self%equation_base%N_EoI))
         allocate(self%aw_EoI_gpu(self%equation_base%N_EoI))
         allocate(self%Beta0_gpu(2))
        endif

        ! Energy deposition parameters
        if (self%equation_base%enable_endepo == 1) then
         allocate(self%endepo_param_gpu(11))
         self%endepo_param_gpu = self%equation_base%endepo_param
        endif

        if (self%equation_base%num_probe>0) then
         allocate(self%w_aux_probe_gpu(6,self%equation_base%num_probe))
         allocate(self%ijk_probe_gpu(3,self%equation_base%num_probe))
         allocate(self%probe_coeff_gpu(2,2,2,self%equation_base%num_probe))
        endif

        if (self%equation_base%enable_jcf > 0) then
         allocate(self%jcf_parbc_gpu(self%equation_base%jcf_jet_num,nv))
         allocate(self%jcf_coords_gpu(self%equation_base%jcf_jet_num,3))
        endif

        endassociate
    endsubroutine alloc

    subroutine compute_residual(self)
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        real(rkind) :: restemp
        integer :: i_tmin, j_tmin, k_tmin
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, nv => self%nv, &
                  dt => self%equation_base%dt, fln_gpu => self%fln_gpu, residual_rhou => self%equation_base%residual_rhou, &
                  fluid_mask_gpu => self%fluid_mask_gpu, vmax => self%equation_base%vmax, w_aux_gpu => self%w_aux_gpu, &
                  rhomin => self%equation_base%rhomin, rhomax => self%equation_base%rhomax, &
                  pmin => self%equation_base%pmin, pmax => self%equation_base%pmax, &
                  tmin => self%equation_base%tmin, tmax => self%equation_base%tmax, &
                  ncoords => self%field%ncoords)

        call compute_residual_cuf(nx, ny, nz, ng, nv, fln_gpu, dt, residual_rhou, fluid_mask_gpu)
        call mpi_allreduce(MPI_IN_PLACE,residual_rhou,1,mpi_prec,mpi_sum,self%equation_base%field%mp_cart,self%mpi_err)

        ! Find BOOM
        if (ieee_is_nan(residual_rhou)) then
         call find_boom_cuf(nx, ny, nz, ng, nv, fln_gpu, dt, fluid_mask_gpu, ncoords(1), ncoords(3))
        endif

        restemp = real(nx,rkind)*real(ny,rkind)*real(nz,rkind)*real(self%nprocs,rkind)
        residual_rhou = sqrt(residual_rhou/restemp)

        call compute_rho_t_p_minmax_cuf(nx, ny, nz, ng, w_aux_gpu,rhomin,rhomax, tmin, tmax, pmin, pmax, fluid_mask_gpu)
        call mpi_allreduce(MPI_IN_PLACE,rhomin,1,mpi_prec,mpi_min,self%equation_base%field%mp_cart,self%mpi_err)
        call mpi_allreduce(MPI_IN_PLACE,rhomax,1,mpi_prec,mpi_max,self%equation_base%field%mp_cart,self%mpi_err)
        call mpi_allreduce(MPI_IN_PLACE,tmin,1,mpi_prec,mpi_min,self%equation_base%field%mp_cart,self%mpi_err)
        call mpi_allreduce(MPI_IN_PLACE,tmax,1,mpi_prec,mpi_max,self%equation_base%field%mp_cart,self%mpi_err)
        call mpi_allreduce(MPI_IN_PLACE,pmin,1,mpi_prec,mpi_min,self%equation_base%field%mp_cart,self%mpi_err)
        call mpi_allreduce(MPI_IN_PLACE,pmax,1,mpi_prec,mpi_max,self%equation_base%field%mp_cart,self%mpi_err)

        endassociate
    endsubroutine compute_residual

    subroutine compute_dt(self)
        !< Initialize the equation.
        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        real(rkind) :: dt_min, dtinv_max
        real(rkind) :: dtxi_max, dtyi_max, dtzi_max, dtxv_max, dtyv_max, dtzv_max, dtxk_max, dtyk_max, dtzk_max
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, nv => self%nv, &
                  dt => self%equation_base%dt, CFL => self%equation_base%CFL, nv_aux => self%nv_aux, &
                  w_gpu => self%base_gpu%w_gpu, &
                  w_aux_gpu  => self%w_aux_gpu, &
                  dcsidx_gpu => self%base_gpu%dcsidx_gpu,   &  
                  detady_gpu => self%base_gpu%detady_gpu,   &
                  dzitdz_gpu => self%base_gpu%dzitdz_gpu,   &
                  dcsidxs_gpu  => self%base_gpu%dcsidxs_gpu, &
                  detadys_gpu  => self%base_gpu%detadys_gpu, &
                  dzitdzs_gpu  => self%base_gpu%dzitdzs_gpu, &
                  indx_cp_l    => self%equation_base%indx_cp_l, &
                  indx_cp_r    => self%equation_base%indx_cp_r, &
                  cp_coeff_gpu => self%cp_coeff_gpu, &
                  trange_gpu => self%trange_gpu, &
                  nsetcv => self%equation_base%nsetcv, &
                  fluid_mask_gpu => self%fluid_mask_gpu)
        if (CFL < 0) then
          dt = -CFL
        else
          call compute_dt_cuf(nx, ny, nz, ng, nv, nv_aux, &
                              dcsidx_gpu, detady_gpu, dzitdz_gpu, dcsidxs_gpu, detadys_gpu, dzitdzs_gpu, w_gpu, w_aux_gpu, &
                              dtxi_max, dtyi_max, dtzi_max, dtxv_max, dtyv_max, dtzv_max, dtxk_max, dtyk_max, dtzk_max,    &
                              indx_cp_l, indx_cp_r, cp_coeff_gpu, nsetcv, trange_gpu,fluid_mask_gpu)
          !open(unit=116, file="dt_values.dat", position="append")
          !write(116,'(100(f16.8,2x))') dtxi_max, dtyi_max, dtzi_max, dtxv_max, dtyv_max, dtzv_max, dtxk_max, dtyk_max, dtzk_max
          !close(116)
          dtinv_max = maxval([dtxi_max, dtyi_max, dtzi_max, dtxv_max, dtyv_max, dtzv_max, dtxk_max, dtyk_max, dtzk_max])
          call mpi_allreduce(MPI_IN_PLACE,dtinv_max,1,mpi_prec,mpi_max,self%equation_base%field%mp_cart,self%mpi_err)
          dt_min = 1._rkind/dtinv_max
          dt = self%equation_base%CFL*dt_min
        endif
        endassociate
    endsubroutine compute_dt

    subroutine run(self, filename)

        class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
        character(*)                    , intent(in)          :: filename          !< Input file name.
        real(rkind)                                           :: timing(1:2)       !< Tic toc timing.
        real(rkind)                                           :: timing_step(1:2)  !< Tic toc timing.
        real(rkind) :: step_time
        integer :: icyc_loop, iercuda

        call self%initialize(filename=filename)

        associate(icyc0 => self%equation_base%icyc0, icyc => self%equation_base%icyc, &
                  time => self%equation_base%time, iter_dt_recompute => self%equation_base%iter_dt_recompute, &
                  residual_rhou => self%equation_base%residual_rhou, &
                  rhobulk => self%equation_base%rhobulk, ubulk => self%equation_base%ubulk, &
                  tbulk => self%equation_base%tbulk, nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  indx_cp_l    => self%equation_base%indx_cp_l, &
                  indx_cp_r    => self%equation_base%indx_cp_r, &
                  enable_ibm => self%equation_base%enable_ibm, &
                  mode_async => self%equation_base%mode_async, &
                  time_from_last_rst    => self%equation_base%time_from_last_rst,    &
                  time_from_last_write  => self%equation_base%time_from_last_write,  &
                  time_from_last_stat   => self%equation_base%time_from_last_stat,   &
                  time_from_last_slice  => self%equation_base%time_from_last_slice,  &
                  time_from_last_insitu => self%equation_base%time_from_last_insitu, &
                  time_is_freezed => self%equation_base%time_is_freezed)

        call self%update_ghost()
        if (enable_ibm>0) then
         !call self%base_gpu%bcswap_corner()
         call self%base_gpu%bcswap_edges_corners()
         if (self%equation_base%enable_les>0) then
          call self%compute_aux_les()
         else
          call self%compute_aux()
         endif
         !call self%ibm_apply()
         call self%ibm_apply_old()
         call self%update_ghost() ! needed after application of ibm
        endif

        if (self%equation_base%enable_les>0) then
         call self%compute_aux_les()
        else
         call self%compute_aux()
         call bcextr_var_cuf(nx, ny, nz, ng, self%w_aux_gpu(:,:,:,J_DIV:J_DIV))
         call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DIV:J_DIV)) ! div/3
         call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DUC:J_DUC )) ! ducros
        endif

        call zero_flux_cuf(self%nx,self%ny,self%nz,self%nv,self%fl_gpu)

        if (self%equation_base%restart_type==0) then
         self%equation_base%w_aux = self%w_aux_gpu
         if (self%equation_base%enable_plot3d>0) then
          call self%field%write_plot3d(mach=self%equation_base%Mach, reynolds=self%equation_base%Reynolds, &
                     time=0._rkind, istore=0, plot3dgrid=.true., plot3dfield=.true., &
                     w_aux_io=self%equation_base%w_aux(1:self%nx,1:self%ny,1:self%nz,1:self%nv_aux), &
                     l1=J_R,l2=J_U,l3=J_V,l4=J_W,l5=J_T)
         endif
         if (self%equation_base%enable_vtk>0) then
          call self%field%write_vtk(N_S,self%equation_base%species_names,time=0._rkind, istore=0, &
                     w_aux_io=self%equation_base%w_aux(1:self%nx,1:self%ny,1:self%nz,1:self%nv_aux), &
                     l1=J_R,l2=J_U,l3=J_V,l4=J_W,l5=J_T,l6=J_P)
         endif
        endif
        call self%compute_dt()
        if (self%masterproc) write(*,*) 'dt =', self%equation_base%dt

        call MPI_BARRIER(MPI_COMM_WORLD, self%error) ; timing(1) = mpi_wtime()

        ! Reinit random to avoid missing reproducibility with inflow random numbers
        if(self%equation_base%rand_type == 0) then
            call init_crandom_f(0,reproducible=.true.)
            if (self%masterproc) write(*,*) 'Random numbers disabled'
        elseif(self%equation_base%rand_type < 0) then
            call init_crandom_f(self%myrank+1,reproducible=.false.)
            if (self%masterproc) write(*,*) 'Random numbers NOT reproducible'
        else
            call init_crandom_f(self%myrank+1,reproducible=.true.)
            if (self%masterproc) write(*,*) 'Random numbers reproducible'
        endif

        !
        icyc_loop = icyc
        integration: do

            icyc_loop = icyc_loop + 1
            time_is_freezed = self%equation_base%time_is_freezed_fun()

            if ( time_is_freezed ) then
             self%equation_base%dt = 0._rkind
            else
             call MPI_BARRIER(MPI_COMM_WORLD, self%error) ; timing_step(1) = mpi_wtime()
             icyc = icyc + 1

             if(mod(icyc-icyc0, iter_dt_recompute)==0) then
                 if (self%equation_base%enable_les>0) then
                  call self%compute_aux_les()
                 else
                  call self%compute_aux(central=1, ghost=0)
                  !@cuf iercuda=cudaDeviceSynchronize()
                 endif
                 call self%compute_dt()
             endif
!
             call self%advance_solution()
             if (self%equation_base%enable_les>0) then
              call self%compute_aux_les()
             else
              call self%compute_aux(central=1, ghost=0)
              !@cuf iercuda=cudaDeviceSynchronize()
             endif

             if (self%equation_base%enable_chemistry > 0) then
              if ((self%equation_base%time_from_last_stat      + self%equation_base%dt >= self%equation_base%dtstat     ) .or. &
                  (self%equation_base%time_from_last_slice     + self%equation_base%dt >= self%equation_base%dtslice    ) .or. &
                  (self%equation_base%time_from_last_slice_vtr + self%equation_base%dt >= self%equation_base%dtslice_vtr) .or. &
                  (self%equation_base%time_from_last_insitu    + self%equation_base%dt >= self%equation_base%dt_insitu  ) .or. &
                  (self%equation_base%time_from_last_write     + self%equation_base%dt >= self%equation_base%dtsave)) then
                  call self%compute_chem_aux()
              endif
             endif
!
             if (mod(icyc-icyc0, self%equation_base%print_control)==0) call self%compute_residual()
             if (ieee_is_nan(self%equation_base%residual_rhou)) then
              if (self%masterproc) write(*,*) 'BOOM!!!'
              self%equation_base%w_aux = self%w_aux_gpu
              call self%field%write_vtk(N_S,self%equation_base%species_names,time=self%equation_base%time, istore=6666, &
                   w_aux_io=self%equation_base%w_aux(1:self%nx,1:self%ny,1:self%nz,1:self%nv_aux), &
                   l1=J_R,l2=J_U,l3=J_V,l4=J_W,l5=J_T,l6=J_P)
              call mpi_barrier(mpi_comm_world,self%mpi_err)
              call mpi_abort(mpi_comm_world,99,self%mpi_err)
             endif 
            endif
!
            self%equation_base%time = self%equation_base%time + self%equation_base%dt
            call self%manage_output()

            call MPI_BARRIER(MPI_COMM_WORLD, self%error) ; timing_step(2) = mpi_wtime()
            step_time = timing_step(2)-timing_step(1)
            if(self%masterproc.and.mod(icyc-icyc0, self%equation_base%print_control)==0) then
                call self%print_progress(step_time)
            endif
            if ((self%equation_base%icyc-self%equation_base%icyc0) >= self%num_iter) exit integration
            !print '(A, F18.10)', 'step timing: ', timing_step(2) - timing_step(1)
        enddo integration

        if (allocated(self%equation_base%islice)) then
         wait(133)
         close(133)
        endif
        if (allocated(self%equation_base%jslice)) then
         wait(134)
         close(134)
        endif
        if (allocated(self%equation_base%kslice)) then
         wait(135)
         close(135)
        endif
        if (self%equation_base%num_probe>0) close(136)

        call MPI_BARRIER(MPI_COMM_WORLD, self%error) ; timing(2) = mpi_wtime()
        if(self%num_iter > 0) then
          if (self%masterproc) then
              write(*,'(A, F18.10)') 'averaged timing: ', &
             (timing(2) - timing(1))/(self%equation_base%icyc-self%equation_base%icyc0)
          endif
        endif

        call self%base_gpu%copy_gpu_cpu()
        if (self%equation_base%io_type_w==1) then
         call self%field%write_field_serial()
         call self%equation_base%write_stats_serial()
         if (self%equation_base%enable_stat_3d>0) call self%equation_base%write_stats_3d_serial()
        endif
        if (self%equation_base%io_type_w==2) then
         call self%field%write_field()
         call self%equation_base%write_stats()
         if (self%equation_base%enable_stat_3d>0) call self%equation_base%write_stats_3d()
        endif

        if (self%equation_base%enable_ibm>0) then
         if (self%equation_base%ibm_wm>0) then
          call self%equation_base%ibm_write_wm_stat()
         endif
        endif

        call self%equation_base%write_field_info()

        endassociate
    endsubroutine run

    subroutine advance_solution(self)
        class(equation_multideal_gpu_object), intent(inout) :: self             
        integer :: iercuda

!       enable chemistry   : 0 chemistry disabled
!                            1 explicit integration
!                            2 implicit integration
!       operator_splitting : 1 ihme's simpler splitting (only with krilov's rosenbrok version) 
!                            2 strang's splitting 
!       rosenbrock_version : 1 full jacobian
!                            2 krylov subspace

! Energy deposition
        if (self%equation_base%enable_endepo == 1) then
         call self%energy_deposition()
         call self%compute_aux(central=1, ghost=0)
         !@cuf iercuda=cudaDeviceSynchronize()
        endif

        if (self%equation_base%enable_chemistry == 2) then
         select case (self%equation_base%operator_splitting)
         case (1)
          self%equation_base%dt_chem = self%equation_base%dt
          self%equation_base%dt = 0.5_rkind*self%equation_base%dt

          call self%sav_flx() ! computes fluxes to be added at the rhs

          if (self%equation_base%enable_les>0) then
           call self%compute_aux_les()
          else
           call self%compute_aux(central=1, ghost=0)
           !@cuf iercuda=cudaDeviceSynchronize()
          endif
        
          call self%rosenbrock_krylov(simpler_splitting=1)

          call self%rk_sync(simpler_splitting=1)

          self%equation_base%dt = 2._rkind*self%equation_base%dt
         case(2)
          self%equation_base%dt_chem = 0.5_rkind*self%equation_base%dt  

          select case(self%equation_base%rosenbrock_version)
          case(1)
           call self%rosenbrock()
          case(2)
           call self%rosenbrock_krylov()
          endselect

          call self%rk_sync()
          if (self%equation_base%enable_les>0) then
           call self%compute_aux_les()
          else
           call self%compute_aux(central=1, ghost=0)
           !@cuf iercuda=cudaDeviceSynchronize()
          endif

          select case(self%equation_base%rosenbrock_version)
          case(1)
           call self%rosenbrock()
          case(2)
           call self%rosenbrock_krylov()
          endselect

        endselect
!
        else

!       select case(self%equation_base%rk_type)
!       case(RK_WRAY,RK_JAMESON)
!       if (mode_async ==  0) call self%rk_sync() 
         call self%rk_sync() 
!       if(mode_async ==  1) call self%rk_async() 
!       case(RK_SHU)
!        call self%rk() 
!       end select
!
        endif
    endsubroutine advance_solution

    subroutine print_progress(self,step_time)
        class(equation_multideal_gpu_object), intent(inout) :: self             
        character(6) :: pos_io
        real(rkind) :: step_time
        associate(icyc => self%equation_base%icyc, time => self%equation_base%time, dt => self%equation_base%dt, &
                  residual_rhou => self%equation_base%residual_rhou, &
                  vmax => self%equation_base%vmax, &
                  rhobulk => self%equation_base%rhobulk, ubulk => self%equation_base%ubulk, &
                  tbulk => self%equation_base%tbulk, &
                  w_aux => self%equation_base%w_aux, &
                  w_aux_gpu => self%w_aux_gpu, &
                  ibm_force_x => self%equation_base%ibm_force_x, &
                  ibm_force_y => self%equation_base%ibm_force_y, &
                  ibm_force_z => self%equation_base%ibm_force_z, &
                  rhomin => self%equation_base%rhomin, rhomax => self%equation_base%rhomax, &
                  pmin => self%equation_base%pmin, pmax => self%equation_base%pmax, &
                  tmin => self%equation_base%tmin, tmax => self%equation_base%tmax)
!
        residual_rhou = residual_rhou
        pos_io = 'append'
        if (self%equation_base%icyc==1) pos_io = 'rewind'
        if (self%masterproc) then
         open(unit=15,file='progress.out',position=pos_io)
         if (self%equation_base%enable_ibm>0) then
          write(* ,'(1I10,F18.10,20ES20.10)') icyc, step_time, dt, time, residual_rhou, ibm_force_x, ibm_force_y, ibm_force_z, &
                                      rhomin, rhomax, tmin, tmax, pmin, pmax
          write(15,'(1I10,F18.10,20ES20.10)') icyc, step_time, dt, time, residual_rhou, ibm_force_x, ibm_force_y, ibm_force_z, &
                                      rhomin, rhomax, tmin, tmax, pmin, pmax
         else
!         write(* ,'(1I10,20ES20.10)') icyc, dt, time, residual_rhou,vmax
          write(* ,'(1I10,F18.10,20ES20.10)') icyc, step_time, dt, time, residual_rhou, rhomin, rhomax, tmin, tmax, pmin, pmax
          write(15,'(1I10,F18.10,20ES20.10)') icyc, step_time, dt, time, residual_rhou, rhomin, rhomax, tmin, tmax, pmin, pmax
!         w_aux(1,1,1,:) = w_aux_gpu(1,1,1,:)
!         write(* ,'(1I10,20ES20.10)') icyc, dt, time, w_aux(1,1,1,J_U),w_aux(1,1,1,J_R),w_aux(1,1,1,J_P),w_aux(1,1,1,J_T)
!         write(15 ,'(1I10,20ES20.10)') icyc, dt, time, w_aux(1,1,1,J_U),w_aux(1,1,1,J_R),w_aux(1,1,1,J_P),w_aux(1,1,1,J_T)
         endif
         close(15)
        endif
        endassociate
    endsubroutine print_progress
!
    subroutine ibm_apply_wm(self)
        class(equation_multideal_gpu_object), intent(inout) :: self
!
        associate(ibm_num_interface => self%equation_base%ibm_num_interface, w_aux_gpu => self%w_aux_gpu, &
                 ng => self%ng, nx => self%nx, ny => self%ny, nz => self%nz, &
                 ibm_ijk_interface_gpu => self%ibm_ijk_interface_gpu, &
                 ibm_bc_gpu => self%ibm_bc_gpu, &
                 ibm_ijk_hwm_gpu => self%ibm_ijk_hwm_gpu, &
                 ibm_coeff_hwm_gpu => self%ibm_coeff_hwm_gpu, &
                 ibm_w_hwm_gpu => self%ibm_w_hwm_gpu, &
                 ibm_nxyz_interface_gpu => self%ibm_nxyz_interface_gpu, &
                 ibm_dist_hwm_gpu => self%ibm_dist_hwm_gpu, &
                 ibm_wm_correction_gpu => self%ibm_wm_correction_gpu,&
                 ibm_wm_wallprop_gpu => self%ibm_wm_wallprop_gpu, &
                 nv_aux => self%equation_base%nv_aux, les_pr => self%equation_base%les_pr, &
                 ibm_parbc_gpu => self%ibm_parbc_gpu, u0 => self%equation_base%u0, &
                 indx_cp_l => self%equation_base%indx_cp_l, indx_cp_r => self%equation_base%indx_cp_r, &
                 nsetcv => self%equation_base%nsetcv, trange_gpu => self%trange_gpu, &
                 cp_coeff_gpu => self%cp_coeff_gpu)
!
         if (ibm_num_interface>0) then 
          call ibm_interpolate_hwm_cuf(nx,ny,nz,ng,w_aux_gpu,ibm_num_interface,ibm_ijk_hwm_gpu, &
                                       ibm_coeff_hwm_gpu,ibm_w_hwm_gpu,ibm_nxyz_interface_gpu,ibm_bc_gpu)
          call ibm_solve_wm_cuf(nx,ny,nz,ng,nv_aux,w_aux_gpu,ibm_num_interface,ibm_ijk_interface_gpu,ibm_w_hwm_gpu, &
                                ibm_dist_hwm_gpu,ibm_bc_gpu,ibm_wm_correction_gpu,ibm_wm_wallprop_gpu,&
                                indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,ibm_parbc_gpu,u0)
          call ibm_apply_wm_cuf(nx,ny,nz,ng,w_aux_gpu,ibm_num_interface,ibm_ijk_interface_gpu,ibm_bc_gpu,ibm_wm_correction_gpu)
         endif
!
        endassociate
!
    end subroutine ibm_apply_wm
!
    subroutine ibm_apply(self)
        class(equation_multideal_gpu_object), intent(inout) :: self
!
        associate(ibm_num_interface => self%equation_base%ibm_num_interface, &
                 ibm_is_interface_node_gpu => self%ibm_is_interface_node_gpu, ibm_bc_gpu => self%ibm_bc_gpu,  &
                 w_gpu => self%base_gpu%w_gpu, w_aux_gpu => self%w_aux_gpu, cv_coeff_gpu => self%cv_coeff_gpu, &
                 cp_coeff_gpu => self%cp_coeff_gpu, &
                 indx_cp_l => self%equation_base%indx_cp_l, ibm_w_refl_gpu => self%ibm_w_refl_gpu, ng => self%ng, &
                 ibm_w_hwm_gpu => self%ibm_w_hwm_gpu, nx => self%nx, ny => self%ny, nz => self%nz, &
                 indx_cp_r => self%equation_base%indx_cp_r, &
                 t0 => self%equation_base%t0, nv => self%nv, &
                 ibm_ijk_interface_gpu => self%ibm_ijk_interface_gpu, ibm_nxyz_interface_gpu => self%ibm_nxyz_interface_gpu, &
                 x_gpu => self%base_gpu%x_gpu, y_gpu => self%base_gpu%y_gpu, z_gpu => self%base_gpu%z_gpu, &
                 ibm_bc_relax_factor => self%equation_base%ibm_bc_relax_factor, &
                 ibm_parbc_gpu => self%ibm_parbc_gpu, &
                 fluid_mask_gpu => self%fluid_mask_gpu, &
                 ibm_num_bc => self%equation_base%ibm_num_bc, &
                 ibm_body_dist_gpu => self%ibm_body_dist_gpu, &
                 ibm_reflection_coeff_gpu => self%ibm_reflection_coeff_gpu, &
                 nsetcv => self%equation_base%nsetcv, trange_gpu => self%trange_gpu, &
                 rgas_gpu => self%rgas_gpu, &
                 ibm_eikonal_cfl => self%equation_base%ibm_eikonal_cfl, &
                 ibm_indx_eikonal => self%equation_base%ibm_indx_eikonal, &
                 ibm_dw_aux_eikonal_gpu => self%ibm_dw_aux_eikonal_gpu, &
                 randvar_a_gpu => self%randvar_a_gpu,randvar_p_gpu => self%randvar_p_gpu, time => self%equation_base%time)

        call ibm_eikonal_cons_cuf(nx,ny,nz,ng,w_gpu,x_gpu,y_gpu,z_gpu,ibm_dw_aux_eikonal_gpu,&
                                  ibm_body_dist_gpu,ibm_is_interface_node_gpu,ibm_reflection_coeff_gpu,&
                                  ibm_num_interface,ibm_w_refl_gpu,ibm_ijk_interface_gpu,ibm_bc_gpu,ibm_eikonal_cfl,ibm_indx_eikonal)

        call ibm_forcing_cuf(ibm_num_interface,nx,ny,nz,ng,nv,indx_cp_l,indx_cp_r,ibm_ijk_interface_gpu,w_gpu,w_aux_gpu,ibm_bc_gpu, &
                             cv_coeff_gpu,nsetcv,trange_gpu,ibm_w_refl_gpu, ibm_nxyz_interface_gpu,  & 
                             ibm_bc_relax_factor,rgas_gpu,ibm_parbc_gpu,ibm_num_bc,tol_iter_nr,randvar_a_gpu,randvar_p_gpu,time,x_gpu,y_gpu,z_gpu)

        endassociate
!
    end subroutine ibm_apply

    subroutine ibm_apply_old(self)
        class(equation_multideal_gpu_object), intent(inout) :: self
!
        associate(ibm_num_interface => self%equation_base%ibm_num_interface, &
                 ibm_ijk_refl_gpu => self%ibm_ijk_refl_gpu, ibm_refl_type_gpu => self%ibm_refl_type_gpu, &
                 ibm_coeff_d_gpu => self%ibm_coeff_d_gpu, ibm_coeff_n_gpu => self%ibm_coeff_n_gpu, &
                 ibm_is_interface_node_gpu => self%ibm_is_interface_node_gpu, ibm_bc_gpu => self%ibm_bc_gpu,  &
                 w_gpu => self%base_gpu%w_gpu, w_aux_gpu => self%w_aux_gpu, cv_coeff_gpu => self%cv_coeff_gpu, &
                 indx_cp_l => self%equation_base%indx_cp_l, ibm_w_refl_gpu => self%ibm_w_refl_gpu, ng => self%ng, &
                 nx => self%nx, ny => self%ny, nz => self%nz, &
                 nv => self%nv, nv_aux => self%nv_aux, &
                 indx_cp_r => self%equation_base%indx_cp_r, &
                 nsetcv => self%equation_base%nsetcv, &
                 trange_gpu => self%trange_gpu, &
                 ibm_ijk_interface_gpu => self%ibm_ijk_interface_gpu, ibm_nxyz_interface_gpu => self%ibm_nxyz_interface_gpu, &
                 ibm_aero_rad => self%equation_base%ibm_aero_rad, &
                 ibm_aero_pp => self%equation_base%ibm_aero_pp, &
                 ibm_aero_tt => self%equation_base%ibm_aero_tt, &
                 ibm_aero_modvel => self%equation_base%ibm_aero_modvel, &
                 x_gpu => self%base_gpu%x_gpu, y_gpu => self%base_gpu%y_gpu, z_gpu => self%base_gpu%z_gpu, &
                 ibm_bc_relax_factor => self%equation_base%ibm_bc_relax_factor, &
                 ibm_parbc_gpu => self%ibm_parbc_gpu, &
                 ibm_inside_moving_gpu => self%ibm_inside_moving_gpu, &
                 rmixt0 => self%equation_base%rmixt0, &
                 rgas_gpu => self%rgas_gpu, &
                 fluid_mask_gpu => self%fluid_mask_gpu, &
                 ibm_eikonal => self%equation_base%ibm_eikonal, &
                 ibm_vega_moving => self%equation_base%ibm_vega_moving, &
                 ibm_num_bc => self%equation_base%ibm_num_bc, &
                 ibm_vega_vel => self%equation_base%ibm_vega_vel, &
                 ibm_vega_species => self%equation_base%ibm_vega_species, &
                 ibm_vega_dist_gpu => self%ibm_vega_dist_gpu, &
                 ibm_body_dist_gpu => self%ibm_body_dist_gpu, &
                 ibm_dw_aux_vega_gpu => self%ibm_dw_aux_vega_gpu, &
                 time => self%equation_base%time, &
                 ibm_dw_aux_eikonal_gpu => self%ibm_dw_aux_eikonal_gpu, &
                 cp_coeff_gpu => self%cp_coeff_gpu, randvar_a_gpu => self%randvar_a_gpu,&
                 randvar_p_gpu => self%randvar_p_gpu)

        if (ibm_vega_moving>0) then
         call ibm_vega_old_cuf(nx,ny,nz,ng,nv,nv_aux,indx_cp_l,indx_cp_r,w_gpu,ibm_inside_moving_gpu, &
                           cv_coeff_gpu,nsetcv,trange_gpu,ibm_aero_rad,ibm_aero_pp,ibm_aero_tt, ibm_vega_vel, &
                           ibm_aero_modvel,x_gpu,y_gpu,z_gpu,ibm_bc_relax_factor,rgas_gpu, &
                           w_aux_gpu,ibm_dw_aux_vega_gpu,ibm_vega_dist_gpu,ibm_vega_species)
        endif
        if (ibm_eikonal>0) then
         call ibm_eikonal_old_cuf(nx,ny,nz,ng,indx_cp_l,indx_cp_r,w_gpu, &
                              cv_coeff_gpu,nsetcv,trange_gpu,x_gpu,y_gpu,z_gpu,ibm_bc_relax_factor,&
                              w_aux_gpu,ibm_dw_aux_eikonal_gpu,ibm_body_dist_gpu,ibm_is_interface_node_gpu,&
                              rgas_gpu,ibm_parbc_gpu)
        else
         call ibm_interpolation_old_cuf(ibm_num_interface,nx,ny,nz,ng,nv,nv_aux,indx_cp_l,indx_cp_r, &
              ibm_ijk_refl_gpu,ibm_refl_type_gpu,w_gpu,w_aux_gpu,ibm_is_interface_node_gpu, ibm_coeff_d_gpu, &
              ibm_coeff_n_gpu,ibm_bc_gpu,cv_coeff_gpu,nsetcv,trange_gpu,ibm_w_refl_gpu, &
              ibm_nxyz_interface_gpu,rgas_gpu,ibm_parbc_gpu)
        endif

        call ibm_forcing_old_cuf(ibm_num_interface,nx,ny,nz,ng,indx_cp_l,indx_cp_r,ibm_ijk_interface_gpu,w_gpu,w_aux_gpu,ibm_bc_gpu, &
             cp_coeff_gpu,cv_coeff_gpu,nsetcv,trange_gpu,ibm_w_refl_gpu,ibm_nxyz_interface_gpu,ibm_aero_rad,ibm_aero_pp,&
             ibm_aero_tt,ibm_aero_modvel,x_gpu,y_gpu,z_gpu,ibm_bc_relax_factor,rgas_gpu,ibm_parbc_gpu,ibm_eikonal,time,&
             randvar_a_gpu,randvar_p_gpu)
        endassociate
! 
    endsubroutine ibm_apply_old

    subroutine ibm_compute_force(self,istep)
        class(equation_multideal_gpu_object), intent(inout) :: self             
!
        integer, intent(in) :: istep
        real(rkind) :: ibm_force_x_s, ibm_force_y_s, ibm_force_z_s
!
        associate(ibm_num_interface => self%equation_base%ibm_num_interface, &
                 ibm_bc_gpu => self%ibm_bc_gpu,  &
                 w_gpu => self%base_gpu%w_gpu, w_aux_gpu => self%w_aux_gpu, &
                 ng => self%ng, &
                 ibm_ijk_interface_gpu => self%ibm_ijk_interface_gpu, &
                 ibm_force_x => self%equation_base%ibm_force_x, &
                 ibm_force_y => self%equation_base%ibm_force_y, &
                 ibm_force_z => self%equation_base%ibm_force_z, &
                 dcsidx_gpu => self%base_gpu%dcsidx_gpu, &
                 detady_gpu => self%base_gpu%detady_gpu, &
                 dzitdz_gpu => self%base_gpu%dzitdz_gpu, &
                 fluid_mask_gpu => self%fluid_mask_gpu,  &
                 fln_gpu => self%fln_gpu, nx => self%nx, ny => self%ny, nz => self%nz, iermpi => self%mpi_err, &
                 dt => self%equation_base%dt)

        call ibm_compute_force_cuf(ibm_num_interface,nx,ny,nz,ng,ibm_ijk_interface_gpu,fln_gpu,w_gpu,w_aux_gpu,ibm_bc_gpu,&
                                   dcsidx_gpu,detady_gpu,dzitdz_gpu,ibm_force_x_s,ibm_force_y_s,ibm_force_z_s,fluid_mask_gpu)
        ibm_force_x = ibm_force_x + ibm_force_x_s
        ibm_force_y = ibm_force_y + ibm_force_y_s
        ibm_force_z = ibm_force_z + ibm_force_z_s
!
        if (istep==self%equation_base%nrk) then
         call mpi_allreduce(mpi_in_place,ibm_force_x,1,mpi_prec,mpi_sum,mpi_comm_world,iermpi)
         call mpi_allreduce(mpi_in_place,ibm_force_y,1,mpi_prec,mpi_sum,mpi_comm_world,iermpi)
         call mpi_allreduce(mpi_in_place,ibm_force_z,1,mpi_prec,mpi_sum,mpi_comm_world,iermpi)
         ibm_force_x = -ibm_force_x/dt
         ibm_force_y = -ibm_force_y/dt
         ibm_force_z = -ibm_force_z/dt
        endif
!
        endassociate
!
    end subroutine ibm_compute_force

    subroutine insitu_coprocess(self)
        class(equation_multideal_gpu_object), intent(inout) :: self             
        integer :: l,i,j,k,ll
        call self%update_ghost()         
        !call self%base_gpu%bcswap()
        !call self%base_gpu%bcswap_corner()
        call self%base_gpu%bcswap_edges_corners()
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  w_aux_gpu  => self%w_aux_gpu, &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  indx_cp_l => self%equation_base%indx_cp_l, &
                  indx_cp_r => self%equation_base%indx_cp_r)
!
        if (self%equation_base%enable_les>0) then
         call self%compute_aux_les()
        else
         call self%compute_aux()
         call bcextr_var_cuf(nx, ny, nz, ng, self%w_aux_gpu(:,:,:,J_DIV:J_DIV))
         call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DIV:J_DIV)) ! div/3
         call self%base_gpu%bcswap_var(self%w_aux_gpu(:,:,:,J_DUC:J_DUC)) ! ducros
        endif
!
        endassociate
        ! filling psi (optionally we can fill ghosts and corners only of psi_pv_managed)
        if (self%equation_base%npsi > 0) then
         call self%insitu_compute_psi()
         do l=1,self%equation_base%npsi
          call self%base_gpu%bcswap_var(self%psi_gpu(:,:,:,l:l ))
         enddo
         do l=1,self%equation_base%npsi
          !call self%base_gpu%bcswap_corner_var(self%psi_gpu(:,:,:,l:l ))
          call self%base_gpu%bcswap_edges_corners_var(self%psi_gpu(:,:,:,l:l ))
         enddo
        endif
        associate( nxsl_ins => self%equation_base%nxsl_ins, nxel_ins => self%equation_base%nxel_ins, &
          nysl_ins => self%equation_base%nysl_ins, nyel_ins => self%equation_base%nyel_ins, &
          nzsl_ins => self%equation_base%nzsl_ins, nzel_ins => self%equation_base%nzel_ins, &
          npsi => self%equation_base%npsi, npsi_pv => self%equation_base%npsi_pv, &
          nv_aux => self%equation_base%nv_aux, n_aux_list => self%equation_base%n_aux_list, &
          psi_gpu => self%psi_gpu, n_add_list => self%equation_base%n_add_list, &
          w_aux_gpu => self%w_aux_gpu, aux_list_gpu => self%aux_list_gpu, add_list_gpu => self%add_list_gpu, &
          nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, &
          icyc => self%equation_base%icyc, time => self%equation_base%time, &
          i_insitu => self%equation_base%i_insitu, &
          flag => self%equation_base%insitu_flag, nrank => self%myrank, &
          ng => self%grid%ng, aux_list_name => self%equation_base%aux_list_name, add_list_name => &
          self%equation_base%add_list_name, time_insitu => self%equation_base%time_insitu)

        self%equation_base%w_aux = w_aux_gpu
        self%equation_base%psi = psi_gpu

        do k=nzsl_ins,nzel_ins
         do j=nysl_ins,nyel_ins
          do i=nxsl_ins,nxel_ins
           do l=1,n_aux_list
            ll = self%equation_base%aux_list(l)
            self%equation_base%psi_pv(i,j,k,l)  = self%equation_base%w_aux(i,j,k,ll)
           enddo
           do l=1,npsi
            self%equation_base%psi_pv(i,j,k,n_aux_list+l) = self%equation_base%psi(i,j,k,l)
           enddo
          enddo
         enddo
        enddo
     
        !NOMANAGEDcall copy_to_psi_pv_managed_cuf(nxsl_ins,nxel_ins,nysl_ins,nyel_ins,nzsl_ins,nzel_ins, &
        !NOMANAGED     ng,nx,ny,nz,npsi, npsi_pv, nv_aux, n_aux_list, &
        !NOMANAGED     psi_gpu,self%psi_pv_managed,w_aux_gpu,aux_list_gpu)
        !call requestdatadescription(icyc,time,flag)
        if(self%equation_base%insitu_platform == "catalyst-v1") then
            call requestdatadescription(i_insitu,time_insitu,flag)
            if (flag.ne.0) then
              call needtocreategrid(flag)
              if (flag.ne.0) then
                do l=1,n_aux_list
                 call addfieldtostructured("3d_struct"//c_null_char,self%equation_base%psi_pv(:,:,:,l), &
                         trim(adjustl(aux_list_name(l)))//c_null_char,nrank)
                 !NOMANAGEDcall addfieldtostructured("3d_struct"//c_null_char,self%psi_pv_managed(:,:,:,l), &
                 !NOMANAGED        trim(adjustl(aux_list_name(l)))//c_null_char,nrank)
                enddo
                do l=1,n_add_list
                 call addfieldtostructured("3d_struct"//c_null_char,self%equation_base%psi_pv(:,:,:,n_aux_list+l), &
                         trim(adjustl(add_list_name(l)))//c_null_char,nrank)
                 !NOMANAGEDcall addfieldtostructured("3d_struct"//c_null_char,self%psi_pv_managed(:,:,:,n_aux_list+l), &
                 !NOMANAGED        trim(adjustl(add_list_name(l)))//c_null_char,nrank)
                enddo
                call coprocess()
              end if
            end if
        elseif(self%equation_base%insitu_platform == "catalyst-v2") then
            call self%insitu_do_catalyst_execute()
        endif
        endassociate

    end subroutine insitu_coprocess

    subroutine insitu_do_catalyst_execute(self)
        class(equation_multideal_gpu_object), intent(inout) :: self             

        integer :: cycle
        real(real64) :: time
        type(C_PTR) :: catalyst_exec_params, mesh, info
        type(C_PTR) :: xt, yt, zt
        type(C_PTR), dimension(:), allocatable :: vx
        integer(kind(catalyst_status)) :: code
        integer :: exit_code
        integer :: l

        catalyst_exec_params = catalyst_conduit_node_create()
        call catalyst_conduit_node_set_path_int64(catalyst_exec_params, "catalyst/state/timestep", &
             int(self%equation_base%i_insitu,int64))
        ! one can also use "catalyst/cycle" for the same purpose.
        ! conduit_node_set_path_int64(catalyst_exec_params, "catalyst/state/cycle", cycle)
        call catalyst_conduit_node_set_path_float64(catalyst_exec_params, "catalyst/state/time", &
             self%equation_base%time_insitu)
    
        ! the data must be provided on a named channel. the name is determined by the
        ! simulation. for this one, we're calling it "grid".
    
        ! declare the type of the channel; we're using Conduit Mesh Blueprint
        ! to describe the mesh and fields.
        call catalyst_conduit_node_set_path_char8_str(catalyst_exec_params, "catalyst/channels/grid/type", "mesh")
    
        ! now, create the mesh.
        mesh = catalyst_conduit_node_create()
    
        associate(points_x => self%equation_base%points_x, &
                  points_y => self%equation_base%points_y, &
                  points_z => self%equation_base%points_z, &
                  n_points_x => self%equation_base%n_points_x, &
                  n_points_y => self%equation_base%n_points_y, &
                  n_points_z => self%equation_base%n_points_z, &
                  n_points => self%equation_base%n_points, &
                  n_aux_list => self%equation_base%n_aux_list, &
                  n_add_list => self%equation_base%n_add_list, &
                  aux_list_name => self%equation_base%aux_list_name, &
                  add_list_name => self%equation_base%add_list_name)

        allocate(vx(n_aux_list+n_add_list))
        
        !*************** STUCTURED START 
        ! add coordsets
        call catalyst_conduit_node_set_path_char8_str(mesh,"coordsets/coords/type","explicit")
        xt = catalyst_conduit_node_create()
        yt = catalyst_conduit_node_create()
        zt = catalyst_conduit_node_create()
        call catalyst_conduit_node_set_external_float64_ptr(xt, points_x, n_points)
        call catalyst_conduit_node_set_external_float64_ptr(yt, points_y, n_points)
        call catalyst_conduit_node_set_external_float64_ptr(zt, points_z, n_points)
        call catalyst_conduit_node_set_path_external_node(mesh, "coordsets/coords/values/x", xt)
        call catalyst_conduit_node_set_path_external_node(mesh, "coordsets/coords/values/y", yt)
        call catalyst_conduit_node_set_path_external_node(mesh, "coordsets/coords/values/z", zt)
        !call c_catalyst_conduit_node_set_path_external_float64_ptr(mesh, "coordsets/coords/values/x"//C_NULL_CHAR, &
        !    points_x, n_points)
        !call c_catalyst_conduit_node_set_path_external_float64_ptr(mesh, "coordsets/coords/values/y"//C_NULL_CHAR, &
        !    points_y, n_points)
        !call c_catalyst_conduit_node_set_path_external_float64_ptr(mesh, "coordsets/coords/values/z"//C_NULL_CHAR, &
        !    points_z, n_points)
        call catalyst_conduit_node_set_path_char8_str(mesh, "topologies/mesh/type", "structured")
        call catalyst_conduit_node_set_path_char8_str(mesh, "topologies/mesh/coordset", "coords")
        call catalyst_conduit_node_set_path_int32(mesh, "topologies/mesh/elements/dims/i", n_points_x-1)
        call catalyst_conduit_node_set_path_int32(mesh, "topologies/mesh/elements/dims/j", n_points_y-1)
        call catalyst_conduit_node_set_path_int32(mesh, "topologies/mesh/elements/dims/k", n_points_z-1)

        do l=1,n_aux_list
            vx(l) = catalyst_conduit_node_create()
            call catalyst_conduit_node_set_external_float64_ptr(vx(l), &
                self%equation_base%psi_pv(:,:,:,l), n_points)
            call catalyst_conduit_node_set_path_char8_str(mesh, &
                "fields/"//trim(adjustl(aux_list_name(l)))//"/association", "vertex")
            call catalyst_conduit_node_set_path_char8_str(mesh, &
                "fields/"//trim(adjustl(aux_list_name(l)))//"/topology", "mesh")
            call catalyst_conduit_node_set_path_char8_str(mesh, &
                "fields/"//trim(adjustl(aux_list_name(l)))//"/volume_dependent", "false")
            call catalyst_conduit_node_set_path_external_node(mesh, &
                "fields/"//trim(adjustl(aux_list_name(l)))//"/values", vx(l))
        enddo
        do l=1,n_add_list
            vx(n_aux_list+l) = catalyst_conduit_node_create()
            call catalyst_conduit_node_set_external_float64_ptr(vx(n_aux_list+l), &
                self%equation_base%psi_pv(:,:,:,n_aux_list+l), n_points)
            call catalyst_conduit_node_set_path_char8_str(mesh, &
                "fields/"//trim(adjustl(add_list_name(l)))//"/association", "vertex")
            call catalyst_conduit_node_set_path_char8_str(mesh, &
                "fields/"//trim(adjustl(add_list_name(l)))//"/topology", "mesh")
            call catalyst_conduit_node_set_path_char8_str(mesh, &
                "fields/"//trim(adjustl(add_list_name(l)))//"/volume_dependent", "false")
            call catalyst_conduit_node_set_path_external_node(mesh, &
                "fields/"//trim(adjustl(add_list_name(l)))//"/values", vx(n_aux_list+l))
        enddo

        !*************** STUCTURED END 

        call catalyst_conduit_node_set_path_external_node(catalyst_exec_params, "catalyst/channels/grid/data", mesh)
    
#if 1
        ! print for debugging purposes, if needed
        call catalyst_conduit_node_print(catalyst_exec_params)
         
        ! print information with details about memory allocation
        !info = catalyst_conduit_node_create()
        !call catalyst_conduit_node_info(catalyst_exec_params, info)
        !call catalyst_conduit_node_print(info)
        !call catalyst_conduit_node_destroy(info)
#endif
    
        code = c_catalyst_execute(catalyst_exec_params)
        if (code /= catalyst_status_ok) then
          write (error_unit, *) "failed to call `execute`:", code
          exit_code = 1
        end if
        call catalyst_conduit_node_destroy(catalyst_exec_params)
        call catalyst_conduit_node_destroy(mesh)
        call catalyst_conduit_node_destroy(xt)
        call catalyst_conduit_node_destroy(yt)
        call catalyst_conduit_node_destroy(zt)
        do l=1,n_aux_list+n_add_list
            call catalyst_conduit_node_destroy(vx(l))
        enddo

        deallocate(vx)

    endassociate

    endsubroutine insitu_do_catalyst_execute

    subroutine insitu_compute_psi(self)
        class(equation_multideal_gpu_object), intent(inout) :: self             
        integer :: l
        type(dim3) :: grid, tBlock
        associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                  nv => self%nv,  nv_aux => self%nv_aux, &
                  dt => self%equation_base%dt, &
                  visc_order => self%equation_base%visc_order, &
                  coeff_deriv1_gpu => self%coeff_deriv1_gpu, coeff_deriv2_gpu => self%coeff_deriv2_gpu, &
                  fhat_trans_gpu => self%fhat_trans_gpu, fl_trans_gpu => self%fl_trans_gpu, fl_gpu => self%fl_gpu, &
                  w_aux_gpu  => self%w_aux_gpu, w_aux_trans_gpu => self%w_aux_trans_gpu, &
                  dcsidx_gpu => self%base_gpu%dcsidx_gpu,   &  
                  detady_gpu => self%base_gpu%detady_gpu,   &
                  dzitdz_gpu => self%base_gpu%dzitdz_gpu,   &
                  dcsidxs_gpu => self%base_gpu%dcsidxs_gpu, &
                  detadys_gpu => self%base_gpu%detadys_gpu, &
                  dzitdzs_gpu => self%base_gpu%dzitdzs_gpu, &
                  dcsidx2_gpu => self%base_gpu%dcsidx2_gpu, &
                  detady2_gpu => self%base_gpu%detady2_gpu, &
                  dzitdz2_gpu => self%base_gpu%dzitdz2_gpu, &
                  eps_sensor  =>  self%equation_base%eps_sensor, &
                  eul_imin => self%equation_base%eul_imin, eul_imax => self%equation_base%eul_imax, &
                  eul_jmin => self%equation_base%eul_jmin, eul_jmax => self%equation_base%eul_jmax, &
                  eul_kmin => self%equation_base%eul_kmin, eul_kmax => self%equation_base%eul_kmax, &
                  cv_coeff_gpu => self%cv_coeff_gpu, &
                  indx_cp_l => self%equation_base%indx_cp_l, &
                  indx_cp_r => self%equation_base%indx_cp_r, &
                  npsi => self%equation_base%npsi, &
                  psi_gpu => self%psi_gpu, &
                  add_list => self%equation_base%add_list, &
                  x_gpu => self%base_gpu%x_gpu )
        do l=1,npsi
         if (add_list(l) == 1) then
             call insitu_div_cuf(nx, ny, nz, ng, visc_order, npsi,l, &
            w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, detady_gpu, dzitdz_gpu, psi_gpu )
         endif
         if (add_list(l) == 2) then
             call insitu_omega_cuf(nx, ny, nz, ng, visc_order, npsi,l, &
            w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, detady_gpu, dzitdz_gpu, psi_gpu )
         endif
         if (add_list(l) == 3) then 
             call insitu_ducros_cuf(nx, ny, nz, ng, visc_order, npsi,l, &
            w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, detady_gpu, dzitdz_gpu, psi_gpu, eps_sensor)
         endif
         if (add_list(l) == 4) then
          tBlock = dim3(EULERWENO_THREADS_X,EULERWENO_THREADS_Y,1)
          grid = dim3(ceiling(real(nx)/tBlock%x),ceiling(real(nz)/tBlock%y),1)
          call insitu_swirling_kernel<<<grid, tBlock>>>(nv, nx, ny, nz, visc_order, ng, npsi,l,&
          dcsidx_gpu, detady_gpu, dzitdz_gpu, w_aux_gpu, coeff_deriv1_gpu, psi_gpu, x_gpu )
         endif
         if (add_list(l) == 5) then
          tBlock = dim3(EULERWENO_THREADS_X,EULERWENO_THREADS_Y,1)
          grid = dim3(ceiling(real(nx)/tBlock%x),ceiling(real(nz)/tBlock%y),1)
          call insitu_schlieren_kernel<<<grid, tBlock>>>(nv, nx, ny, nz, visc_order, ng, npsi,l,&
          dcsidx_gpu, detady_gpu, dzitdz_gpu, w_aux_gpu, coeff_deriv1_gpu, psi_gpu, x_gpu )
         endif
        enddo
        endassociate

    end subroutine insitu_compute_psi

    subroutine manage_output(self)
        class(equation_multideal_gpu_object), intent(inout) :: self
!
        integer :: i,j,k,ii,jj,kk,l,n,lsp
        integer :: isize, jsize, ksize
        character(3) :: chx, chy, chz
        logical :: sliceyz_exist, slicexz_exist, slicexy_exist, probe_exist
        real(rkind) :: locgam
        logical :: output_t_exist
        integer :: idx1,idx2,idx3,done1=0,done2=0,done3=0
        integer :: m, mm 
!
        associate(time_from_last_rst    => self%equation_base%time_from_last_rst,            &
                  time_from_last_write  => self%equation_base%time_from_last_write,          &
                  time_from_last_stat   => self%equation_base%time_from_last_stat,           &
                  time_from_last_slice  => self%equation_base%time_from_last_slice,          &
                  time_from_last_slice_vtr  => self%equation_base%time_from_last_slice_vtr,  &
                  time_from_last_insitu => self%equation_base%time_from_last_insitu,         &
                  icyc0 => self%equation_base%icyc0, icyc => self%equation_base%icyc,        &
                  time => self%equation_base%time,                                           &
                  w_aux_gpu => self%w_aux_gpu,                                               &
                  ijk_probe_gpu => self%ijk_probe_gpu,                                       &
                  w_aux_probe_gpu => self%w_aux_probe_gpu,                                   &
                  probe_coeff_gpu => self%probe_coeff_gpu,                                   &
                  nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng,                &
                  nv_aux => self%nv_aux, w_aux_probe => self%equation_base%w_aux_probe,      &
                  time_insitu => self%equation_base%time_insitu)

!           Save flow samples
            time_from_last_write = time_from_last_write + self%equation_base%dt
            if (time_from_last_write >= self%equation_base%dtsave) then
                if (self%masterproc) write(*,*) 'time_from_last_write=',time_from_last_write
                if (self%masterproc) write(*,*) 'istore =', self%equation_base%istore
                call self%base_gpu%copy_gpu_cpu()
                self%equation_base%w_aux = self%w_aux_gpu

                if (self%equation_base%flow_init == 7 .or. self%equation_base%flow_init == 8) then
!               Save output for FERRER acoustic wave Test Case and perfectly stirred reactor Test Case 
                 if (self%equation_base%istore==1) then
                  open(12, file='output_t.dat', status='replace', action='write')
                 else
                  open(12, file='output_t.dat', status='old', position='append', action='write')
                 endif
                 write(12,100) time,(self%equation_base%w_aux(2,1,1,lsp),lsp=1,self%equation_base%nv_aux)
                 close(12)
                endif 

                open(12,file='output_1d.dat')

 100       format(50E30.15)
                if (self%nx>1) then
                 do i=1-ng,self%nx+ng
                  write(12,100) self%field%x(i),(self%equation_base%w_aux(i,1,1,lsp),lsp=1,self%equation_base%nv_aux)
                 enddo
                endif
                if (self%ny>1) then
                 do j=1,self%ny
                  write(12,100) self%field%y(j),(self%equation_base%w_aux(1,j,1,lsp),lsp=1,self%equation_base%nv_aux)
                 enddo
                endif
                if (self%nz>1) then
                 do k=1,self%nz
                  write(12,100) self%field%z(k),(self%equation_base%w_aux(1,1,k,lsp),lsp=1,self%equation_base%nv_aux)
                 enddo
                endif
                close(12)
                if (self%equation_base%enable_plot3d>0) then
                 call self%field%write_plot3d(mach=self%equation_base%Mach, reynolds=self%equation_base%Reynolds, &
                     time=time, istore=self%equation_base%istore, plot3dgrid=.false., plot3dfield=.true., &
                     w_aux_io=self%equation_base%w_aux(1:self%nx,1:self%ny,1:self%nz,1:self%nv_aux), &
                     l1=J_R,l2=J_U,l3=J_V,l4=J_W,l5=J_T)
                endif
                if (self%equation_base%enable_vtk>0) then
                 call self%field%write_vtk(N_S,self%equation_base%species_names,time=time, istore=self%equation_base%istore, &
                         w_aux_io=self%equation_base%w_aux(1:self%nx,1:self%ny,1:self%nz,1:self%nv_aux), &
                         l1=J_R,l2=J_U,l3=J_V,l4=J_W,l5=J_T,l6=J_P)

                endif
                time_from_last_write = 0._rkind !time_from_last_write - self%equation_base%dtsave
                self%equation_base%istore = self%equation_base%istore + 1
              endif
          if (self%equation_base%flow_init == 8) then
            call self%base_gpu%copy_gpu_cpu()
            self%equation_base%w_aux = self%w_aux_gpu
             if (nint(self%equation_base%time/self%equation_base%dt) .ge. 10 .and. &
             mod(nint(self%equation_base%time/self%equation_base%dt), 10) .lt. 1E-6_rkind) then
             open(22,file='output_batch.dat')
             write(22,'(3ES20.10)') self%equation_base%time, self%equation_base%w_aux(1,1,1,J_R), self%equation_base%w_aux(1,1,1,J_T)
             endif 
           elseif (self%equation_base%flow_init == 9) then
            idx1 = 0; idx2 = 0; idx3 = 0
            if (time >= 170D-6 .and. done1 == 0) then
              idx1 = 1; done1 = 1
            elseif (time >= 190D-6 .and. done2 == 0) then
              idx2 = 1; done2 = 1
            elseif (time >= 230D-6 .and. done3 == 0) then  
              idx3 = 1; done3 = 1
            endif 
            if (idx1 ==1 .or. idx2==1 .or. idx3==1) then 
             if (self%masterproc) write(*,*) 'time_from_last_write=',time_from_last_write
               if (self%masterproc) write(*,*) 'istore =', self%equation_base%istore
               call self%base_gpu%copy_gpu_cpu()
               self%equation_base%w_aux = self%w_aux_gpu
               if (idx1==1) open(22,file='output_1d_170micro_s.dat')
               if (idx2==1) open(22,file='output_1d_190micro_s.dat')
               if (idx3==1) open(22,file='output_1d_230micro_s.dat')
               if (self%nx>1) then
                do i=1,self%nx
                 write(22,'(4ES20.10)') self%field%x(i),self%equation_base%w_aux(i,1,1,J_T),self%equation_base%w_aux(i,1,1,J_U),self%equation_base%w_aux(i,1,1,3)
                enddo
               endif
               time_from_last_write = 0._rkind !time_from_last_write - self%equation_base%dtsave
               self%equation_base%istore = self%equation_base%istore + 1
             endif
            elseif (self%equation_base%flow_init == 10) then
             if (time >= 1D-3 .and. done1 == 0) then
              done1 = 1
              call self%base_gpu%copy_gpu_cpu()
              self%equation_base%w_aux = self%w_aux_gpu
              open(22,file='output_LPF.dat')
              if (self%nx>1) then
               do i=1,self%nx
                write(22,'(7ES20.10)') self%field%x(i),self%equation_base%w_aux(i,1,1,J_T),self%equation_base%w_aux(i,1,1,J_R),&
                self%equation_base%w_aux(i,1,1,1),self%equation_base%w_aux(i,1,1,4),self%equation_base%w_aux(i,1,1,5),self%equation_base%w_aux(i,1,1,2)        
               enddo
              endif
             endif
            endif

!           Compute stats
            time_from_last_stat = time_from_last_stat + self%equation_base%dt
            if (time_from_last_stat >= self%equation_base%dtstat) then
                if (self%masterproc) write(*,*) 'time_from_last_stat=',time_from_last_stat
                if (self%masterproc) write(*,*) 'itav =',self%equation_base%itav
                call self%base_gpu%copy_gpu_cpu()
                self%equation_base%w_aux = self%w_aux_gpu
                call self%equation_base%compute_stats()
                if (self%equation_base%enable_stat_3d>0) call self%equation_base%compute_stats_3d()
                if (self%equation_base%enable_ibm>0) then
                 if (self%equation_base%ibm_wm>0) then
                  if (self%equation_base%ibm_num_interface>0) then
                   self%equation_base%ibm_wm_wallprop = self%ibm_wm_wallprop_gpu
                  endif
                  call self%equation_base%ibm_compute_stat_wm()
                 endif
                endif
                time_from_last_stat = 0._rkind ! time_from_last_stat - self%equation_base%dtstat
                self%equation_base%itav = self%equation_base%itav + 1
            endif

!           Save insitu
            if (self%equation_base%enable_insitu > 0) then
             if (self%equation_base%time_is_freezed) then
              time_from_last_insitu = time_from_last_insitu + self%equation_base%dt_insitu
              time_insitu = time_insitu+self%equation_base%dt_insitu
             else
              time_from_last_insitu = time_from_last_insitu + self%equation_base%dt
              time_insitu = time_insitu+self%equation_base%dt
             endif
             if (time_from_last_insitu >= self%equation_base%dt_insitu) then
                 if (self%masterproc) write(*,*) 'time_from_last_insitu=',time_from_last_insitu
                 if (self%masterproc) write(*,*) 'i_insitu =', self%equation_base%i_insitu
                 call self%insitu_coprocess()
                 !time_from_last_insitu = 0._rkind !time_from_last_insitu - self%equation_base%dt_insitu
                 time_from_last_insitu = time_from_last_insitu - self%equation_base%dt_insitu
                 self%equation_base%i_insitu = self%equation_base%i_insitu + 1
             endif
            endif

!           Write slice
            time_from_last_slice = time_from_last_slice + self%equation_base%dt
            write(chx,'(I3.3)') self%field%ncoords(1)
            write(chy,'(I3.3)') self%field%ncoords(2)
            write(chz,'(I3.3)') self%field%ncoords(3) 
            if (icyc-icyc0 == 1) then
             if (allocated(self%equation_base%islice)) then
              inquire(file='sliceyz_'//chx//'_'//chy//'_'//chz//'.bin', exist=sliceyz_exist)
              if(.not.(sliceyz_exist)) then
               open(133,file='sliceyz_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted',                   asynchronous="no")
              else
               open(133,file='sliceyz_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted', position="append",asynchronous="no")
              endif
             endif
             if (allocated(self%equation_base%jslice)) then
              inquire(file='slicexz_'//chx//'_'//chy//'_'//chz//'.bin', exist=slicexz_exist)
              if(.not.(slicexz_exist)) then
               open(134,file='slicexz_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted',                   asynchronous="no")
              else
               open(134,file='slicexz_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted', position="append",asynchronous="no")
              endif
             endif
             if (allocated(self%equation_base%kslice)) then
              inquire(file='slicexy_'//chx//'_'//chy//'_'//chz//'.bin', exist=slicexy_exist)
              if(.not.(slicexy_exist)) then
               open(135,file='slicexy_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted',                   asynchronous="no")
              else
               open(135,file='slicexy_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted', position="append",asynchronous="no")
              endif
             endif
             if (self%equation_base%num_probe>0) then
              inquire(file='probe_'//chx//'_'//chy//'_'//chz//'.bin', exist=probe_exist)
              if(.not.(probe_exist)) then
               open(136,file='probe_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted')
               write(136) self%equation_base%num_probe
              else
               open(136,file='probe_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted', position="append")
              endif
             endif
            endif
!
            if (time_from_last_slice >= self%equation_base%dtslice) then
                call self%base_gpu%copy_gpu_cpu()
                self%equation_base%w_aux = self%w_aux_gpu
                if (self%masterproc) write(*,*) 'time_from_last_slice=',time_from_last_slice
                if (allocated(self%equation_base%islice)) then
                 isize = size(self%equation_base%islice)
                 do i=1,size(self%equation_base%islice)
                  ii = self%equation_base%islice(i)
                  do m=1,self%equation_base%num_aux_slice
                   mm = self%equation_base%list_aux_slice(m)
                   sliceyz_aux(i,:,:,m) = self%w_aux_gpu(ii,:,:,mm)
                  enddo
                 enddo 
                 !wait(133)
                 write(133,asynchronous="no") icyc,time
                 write(133,asynchronous="no") self%grid%nxmax,self%grid%nymax,self%grid%nzmax
                 write(133,asynchronous="no") isize,(self%equation_base%islice(i),i=1,isize)
                 write(133,asynchronous="no") isize, self%field%ny, self%field%nz, self%equation_base%num_aux_slice
                 write(133,asynchronous="no") sliceyz_aux !(1:isize,1:self%field%ny,1:self%field%nz,1:6)
                endif
                if (allocated(self%equation_base%jslice)) then
                 jsize = size(self%equation_base%jslice)
                 do j=1,size(self%equation_base%jslice)
                  jj = self%equation_base%jslice(j)
                  do m=1,self%equation_base%num_aux_slice
                   mm = self%equation_base%list_aux_slice(m)
                   slicexz_aux(:,j,:,m) = self%w_aux_gpu(:,jj,:,mm)
                  enddo
                 enddo 
                 !wait(134)
                 write(134,asynchronous="no") icyc,time
                 write(134,asynchronous="no") self%grid%nxmax,self%grid%nymax,self%grid%nzmax
                 write(134,asynchronous="no") jsize,(self%equation_base%jslice(j),j=1,jsize)
                 write(134,asynchronous="no") self%field%nx, jsize, self%field%nz, self%equation_base%num_aux_slice
                 write(134,asynchronous="no") slicexz_aux !(1:self%field%nx,1:jsize,1:self%field%nz,1:6)
                endif
                if (allocated(self%equation_base%kslice)) then
                 ksize = size(self%equation_base%kslice)
                 do k=1,size(self%equation_base%kslice)
                  kk = self%equation_base%kslice(k)
                  do m=1,self%equation_base%num_aux_slice
                   mm = self%equation_base%list_aux_slice(m)
                   slicexy_aux(:,:,k,m) = self%w_aux_gpu(:,:,kk,mm)
                  enddo
                 enddo 
                 !wait(135)
                 write(135,asynchronous="no") icyc,time
                 write(135,asynchronous="no") self%grid%nxmax,self%grid%nymax,self%grid%nzmax
                 write(135,asynchronous="no") ksize,(self%equation_base%kslice(k),k=1,ksize)
                 write(135,asynchronous="no") self%field%nx, self%field%ny, ksize, self%equation_base%num_aux_slice
                 write(135,asynchronous="no") slicexy_aux !(1:self%field%nx,1:self%field%ny,1:ksize,1:6)
                endif
!
                if (self%equation_base%num_probe>0) then
                 call probe_interpolation_cuf(self%equation_base%num_probe,nx,ny,nz,ng,nv_aux,ijk_probe_gpu,&
                         w_aux_probe_gpu,w_aux_gpu,probe_coeff_gpu)
!
                 w_aux_probe = w_aux_probe_gpu
!                write(136,100) time, ((w_aux_probe(l,n),l=1,6),n=1,self%equation_base%num_probe)
                 write(136) time, ((w_aux_probe(l,n),l=1,6),n=1,self%equation_base%num_probe)
                endif
!
                time_from_last_slice = 0._rkind ! time_from_last_slice - self%equation_base%dtslice
            endif
!
            time_from_last_slice_vtr = time_from_last_slice_vtr + self%equation_base%dt
            if (time_from_last_slice_vtr >= self%equation_base%dtslice_vtr) then
                self%equation_base%itslice_vtr = self%equation_base%itslice_vtr + 1
                if(self%equation_base%enable_slice_vtr > 0) then
                    if (self%masterproc) write(*,*) 'time_from_last_slice_vtr=',time_from_last_slice_vtr
                    self%equation_base%w_aux(:,:,:,:) = self%w_aux_gpu(:,:,:,:)
                    call self%equation_base%write_slice_vtr()
                endif
                if (self%equation_base%enable_ibm>0) then
                 if (self%equation_base%ibm_wm>0) then
                  self%equation_base%ibm_wm_wallprop = self%ibm_wm_wallprop_gpu
                  call self%equation_base%ibm_write_wm_wallprop()
                 endif
                endif
                time_from_last_slice_vtr = time_from_last_slice_vtr - self%equation_base%dtslice_vtr
            endif
!           Save restart & stats
            time_from_last_rst = time_from_last_rst + self%equation_base%dt
            if (time_from_last_rst >= self%equation_base%dtsave_restart) then
                if (self%masterproc) write(*,*) 'time_from_last_rst=',time_from_last_rst
                call self%base_gpu%copy_gpu_cpu()

                if (self%equation_base%io_type_w==1) then
                 call self%field%write_field_serial()
                 call self%equation_base%write_stats_serial()
                 if (self%equation_base%enable_stat_3d>0) call self%equation_base%write_stats_3d_serial()
                endif
                if (self%equation_base%io_type_w==2) then
                 call self%field%write_field()
                 call self%equation_base%write_stats()
                 if (self%equation_base%enable_stat_3d>0) call self%equation_base%write_stats_3d()
                endif
                if (self%equation_base%enable_ibm>0) then
                 if (self%equation_base%ibm_wm>0) then
                  call self%equation_base%ibm_write_wm_stat()
                 endif
                endif
                call self%equation_base%write_field_info()
                time_from_last_rst = 0._rkind ! time_from_last_rst - self%equation_base%dtsave_restart
            endif

        endassociate
!
    end subroutine manage_output
    
    subroutine tripping(self)
      class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
      real(rkind) :: asl, del0,zita,lamx,lamy,tau,pp,bt,rand
      integer :: i_rank_start, ierr
      real(rkind), dimension(2) :: rsend
     !
      associate(nx=>self%nx,ny=>self%ny,nz=>self%nz,ng=>self%ng,nv=>self%nx,w_gpu=>self%base_gpu%w_gpu,fl_gpu=>self%fl_gpu, &
                x_gpu => self%base_gpu%x_gpu, y_gpu => self%base_gpu%y_gpu, z_gpu => self%base_gpu%z_gpu, &
                rlz =>  self%grid%domain_size(3), &
                a_tr => self%equation_base%a_tr, u0 => self%equation_base%u0, time => self%equation_base%time, &
                xtr1 => self%equation_base%xtr1, xtr2 => self%equation_base%xtr2, &
                x0tr => self%equation_base%x0tr,&
                lamz => self%equation_base%lamz  , lamz1  => self%equation_base%lamz1 , &
                phiz => self%equation_base%phiz  , phiz1  => self%equation_base%phiz1 , &
                lamz_old => self%equation_base%lamz_old  , lamz1_old  => self%equation_base%lamz1_old , &
                phiz_old => self%equation_base%phiz_old  , phiz1_old  => self%equation_base%phiz1_old , &
                is => self%equation_base%is, is_old => self%equation_base%is_old)
     ! Tripping parameters based on delta
      asl  = a_tr*u0*u0 
      !del0 = 0.001_rkind ! approximation of delta* at x/c = 0.1
      del0 = self%equation_base%delta0star! approximation of delta* at x/c = 0.1
      zita = 1.7_rkind*del0 ! spanwise cutoff scale 
      lamx = 4._rkind*del0
      lamy = del0 ! lamx and lamy are the spatial Gaussian attenuation of the forcing region
      tau  = lamx/u0 ! temporal cutoff scale 
      is = int(time/tau)
      pp   = time/tau-is
      bt   = 3._rkind*pp*pp-2._rkind*pp*pp*pp
     !
     ! Time-evolution of the tripping is split in intervals of lenght tau
      if (is==is_old) then
       lamz  = lamz_old
       lamz1 = lamz1_old
       phiz  = phiz_old
       phiz1 = phiz1_old
      else
       lamz = lamz1_old
       phiz = phiz1_old
       if (self%masterproc) then
        call get_crandom_f(rand)
        lamz1 = 1._rkind/rlz*int(rand*rlz/zita) ! altering lambda, to match the periodicity boundary condition
        call get_crandom_f(rand)
        phiz1 = 2._rkind*pi*rand
        rsend(1) = lamz1
        rsend(2) = phiz1
       endif
       call mpi_bcast(rsend,2,mpi_prec,0,mpi_comm_world,ierr)
       if(.not.self%masterproc) then
         lamz1 = rsend(1)
         phiz1 = rsend(2)
       endif
       lamz_old  = lamz  ! needed for the continuity
       lamz1_old = lamz1 ! needed for the continuity
       phiz_old  = phiz  ! needed for the continuity
       phiz1_old = phiz1 ! needed for the continuity
     !
       is_old    = is    ! needed for the continuity
      endif

      call tripping_cuf(nx,ny,nz,ng,nv,pi,xtr1,xtr2,x0tr,lamx,lamy,lamz,lamz1,phiz,phiz1,asl,bt, &
                                       x_gpu,y_gpu,z_gpu,w_gpu,fl_gpu)

      endassociate
    endsubroutine tripping

    subroutine limiter(self)
     ! ==================================
     ! Lower limit for density and energy
     ! ==================================
     class(equation_multideal_gpu_object), intent(inout) :: self              !< The equation.
     integer :: iblock,kblock
     associate(nx=>self%nx,ny=>self%ny,nz=>self%nz,ng=>self%ng,nv=>self%nv,nv_aux=>self%nv_aux,&
               w_gpu=>self%base_gpu%w_gpu,w_aux_gpu=>self%w_aux_gpu,ncoords=>self%field%ncoords,&
               cv_coeff_gpu => self%cv_coeff_gpu, trange_gpu => self%trange_gpu, & 
               indx_cp_l=>self%equation_base%indx_cp_l, indx_cp_r=>self%equation_base%indx_cp_r, &
               nsetcv=>self%equation_base%nsetcv, rho_lim => self%equation_base%rho_lim, &
               tem_lim => self%equation_base%tem_lim, rho_lim_rescale => self%equation_base%rho_lim_rescale, &
               tem_lim_rescale => self%equation_base%tem_lim_rescale) 

     iblock  = ncoords(1)
     kblock  = ncoords(3)

     call limiter_cuf(nx,ny,nz,ng,nv,nv_aux,w_gpu,w_aux_gpu,iblock,kblock,&
                      indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu, &
                      rho_lim,tem_lim,rho_lim_rescale,tem_lim_rescale)

     endassociate

    endsubroutine limiter 

    subroutine energy_deposition(self)
       class(equation_multideal_gpu_object), intent(inout) :: self

       associate(nx => self%nx, ny => self%ny, nz => self%nz, ng => self%ng, &
                 nv => self%nv, nv_aux => self%nv_aux, nreactions => self%equation_base%nreactions, &
                 w_aux_gpu => self%w_aux_gpu, cp_coeff_gpu => self%cp_coeff_gpu, &
                 trange_gpu => self%trange_gpu, nsetcv => self%equation_base%nsetcv, &
                 indx_cp_l => self%equation_base%indx_cp_l, indx_cp_r => self%equation_base%indx_cp_r, &
                 x_gpu => self%base_gpu%x_gpu, y_gpu => self%base_gpu%y_gpu, z_gpu => self%base_gpu%z_gpu, &
                 time => self%equation_base%time, w_gpu => self%base_gpu%w_gpu, endepo_param_gpu => self%endepo_param_gpu)

         call energy_deposition_cuf(nx,ny,nz,ng,nv,nv_aux,nreactions,w_aux_gpu,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,&
               trange_gpu,x_gpu,y_gpu,z_gpu,time,w_gpu,endepo_param_gpu)
       endassociate
    endsubroutine energy_deposition

    subroutine ibm_inside_old(self)
        class(equation_multideal_gpu_object), intent(inout) :: self
!
        associate(ng => self%ng, nx => self%nx, ny => self%ny, nz => self%nz, &
                 x_gpu => self%base_gpu%x_gpu, y_gpu => self%base_gpu%y_gpu, z_gpu => self%base_gpu%z_gpu, &
                 ibm_inside_moving_gpu => self%ibm_inside_moving_gpu, &
!                ibm_vega_y_gpu => self%ibm_vega_y_gpu, &
!                ibm_vega_r_gpu => self%ibm_vega_r_gpu, &
!                ibm_vega_ny => self%equation_base%ibm_vega_ny, &
!                ibm_vega_dy => self%equation_base%ibm_vega_dy, &
                 ep_ord_change_gpu => self%ep_ord_change_gpu, &
                 ibm_order_reduce => self%equation_base%ibm_order_reduce, &
                 ibm_vega_displacement => self%equation_base%ibm_vega_displacement, &
                 ibm_vega_ymin => self%equation_base%ibm_vega_ymin,       &
                 ibm_vega_ymax => self%equation_base%ibm_vega_ymax,       &
                 ibm_vega_dist_gpu => self%ibm_vega_dist_gpu,             &
                 ibm_vega_distini_gpu => self%ibm_vega_distini_gpu,       &
                 fluid_mask_gpu => self%fluid_mask_gpu,                   &
                 fluid_mask_ini_gpu => self%fluid_mask_ini_gpu)

!        call ibm_inside_moving_old_cuf(nx,ny,nz,ng,ibm_inside_moving_gpu,  &
!                                   x_gpu,y_gpu,z_gpu,                  &
!                                   ep_ord_change_gpu,ibm_order_reduce, &
!                                   ibm_vega_dist_gpu,ibm_vega_distini_gpu,ibm_vega_displacement, &
!                                   ibm_vega_ymin, ibm_vega_ymax, &
!                                   fluid_mask_gpu,fluid_mask_ini_gpu)

        endassociate
!
    endsubroutine ibm_inside_old

    subroutine ibm_alloc_gpu_old(self)
        class(equation_multideal_gpu_object), intent(inout) :: self
        associate(ibm_num_interface => self%equation_base%ibm_num_interface, &
                  ibm_num_bc => self%equation_base%ibm_num_bc, &
                  nx => self%nx, ny => self%ny, nz => self%nz,  &
                  ng => self%ng, nv => self%nv)
!
        allocate(self%ibm_sbody_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
        allocate(self%ibm_is_interface_node_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
        if (self%equation_base%ibm_vega_moving>0) then
         allocate(self%ibm_inside_moving_gpu(nx,ny,nz))
         allocate(self%ibm_vega_dist_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
         allocate(self%ibm_vega_distini_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
         allocate(self%ibm_dw_aux_vega_gpu(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng,N_S+2))
         !notneeded self%ibm_inside_moving_gpu = 0
         self%ibm_vega_distini_gpu = self%equation_base%ibm_vega_dist
!        allocate(self%ibm_vega_y_gpu(1:self%equation_base%ibm_vega_ny+1))
!        self%ibm_vega_y_gpu = self%equation_base%ibm_vega_y
!        allocate(self%ibm_vega_r_gpu(1:self%equation_base%ibm_vega_ny+1))
!        self%ibm_vega_r_gpu = self%equation_base%ibm_vega_r
        endif
        if (self%equation_base%ibm_eikonal>0) then
         allocate(self%ibm_body_dist_gpu(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
         self%ibm_body_dist_gpu = self%equation_base%ibm_body_dist
         allocate(self%ibm_dw_aux_eikonal_gpu(1-ng:nx+ng, 1-ng:ny+ng,1-ng:nz+ng,N_S+5))
        endif
        allocate(self%ibm_parbc_gpu(ibm_num_bc,IBM_MAX_PARBC))
        if (ibm_num_interface>0) then
         allocate(self%ibm_ijk_interface_gpu (3,ibm_num_interface))      ! Local values of i,j,k for the interface node
         allocate(self%ibm_ijk_refl_gpu      (3,ibm_num_interface))      ! Reflected node bwtween i,i+1 and j,j+1 and k,k+1
         allocate(self%ibm_ijk_wall_gpu      (3,ibm_num_interface))      ! Wall node between i,i+1 and j,j+1 and k,k+1
         allocate(self%ibm_nxyz_interface_gpu(3,ibm_num_interface))      ! Wall-normal components
         allocate(self%ibm_bc_gpu            (2,ibm_num_interface))      ! Bc tag for interface nodes
         ! Distance between interface node and wall point (1) and reflected point and wall point (2)
         allocate(self%ibm_dist_gpu          (2,ibm_num_interface))
         allocate(self%ibm_coeff_d_gpu       (2,2,2,ibm_num_interface))  ! Coefficients for trilin interpolation (Dirichlet)
         allocate(self%ibm_coeff_n_gpu       (2,2,2,ibm_num_interface))  ! Coefficients for trilin interpolation (Neumann)
         allocate(self%ibm_refl_type_gpu(ibm_num_interface))
         allocate(self%ibm_w_refl_gpu(ibm_num_interface,nv))
        endif
        self%ibm_sbody_gpu             = self%equation_base%ibm_sbody
        self%ibm_is_interface_node_gpu = self%equation_base%ibm_is_interface_node
        self%ibm_parbc_gpu             = self%equation_base%ibm_parbc
        if (self%equation_base%turinf .eq. 1) then
         allocate(self%randvar_a_gpu(8))
         allocate(self%randvar_p_gpu(8))
         self%randvar_a_gpu = self%equation_base%randvar_a
         self%randvar_p_gpu = self%equation_base%randvar_p
!         self%avar_gpu = 0.5_rkind*0.05_rkind*(94.15_rkind-self%equation_base%u0)
        endif
        if (ibm_num_interface>0) then
            self%ibm_ijk_interface_gpu     = self%equation_base%ibm_ijk_interface
            self%ibm_ijk_refl_gpu          = self%equation_base%ibm_ijk_refl
            self%ibm_ijk_wall_gpu          = self%equation_base%ibm_ijk_wall
            self%ibm_nxyz_interface_gpu    = self%equation_base%ibm_nxyz_interface
            self%ibm_bc_gpu                = self%equation_base%ibm_bc
            self%ibm_dist_gpu              = self%equation_base%ibm_dist
            self%ibm_coeff_d_gpu           = self%equation_base%ibm_coeff_d
            self%ibm_coeff_n_gpu           = self%equation_base%ibm_coeff_n
            self%ibm_refl_type_gpu         = self%equation_base%ibm_refl_type
        endif
        endassociate
    endsubroutine ibm_alloc_gpu_old

endmodule streams_equation_multideal_gpu_object

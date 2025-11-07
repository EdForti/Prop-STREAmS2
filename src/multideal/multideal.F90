#include 'index_define.h'
module streams_equation_multideal_object
    !< STREAmS, Euler equations backend-independent variables and routines.

    use streams_field_object
    use streams_grid_object
    use streams_parameters
    use cfgio_mod, only: cfg_t, parse_cfg
    use, intrinsic :: iso_fortran_env, only : error_unit
    use crandom_f_mod
    use MPI
    use ISO_C_BINDING

!   IBM
    use cgal_wrappers
    
!   INSITU CATALYST2
    use catalyst_api
    use catalyst_conduit

!   Cantera
    use cantera
    use cantera_utils

    implicit none
    private
    public :: equation_multideal_object
    public :: RK_WRAY, RK_JAMESON, RK_SHU
    public :: slicexy_aux, slicexz_aux, sliceyz_aux
    public :: IBM_MAX_PARBC

    ! public constants

    integer(ikind), parameter :: RK_WRAY            = 1_ikind
    integer(ikind), parameter :: RK_JAMESON         = 2_ikind
    integer(ikind), parameter :: RK_SHU             = 3_ikind

    integer(ikind), parameter :: IBM_MAX_PARBC      = 5_ikind+N_S

    real(rkind), dimension(:,:,:,:), allocatable :: slicexy_aux, slicexz_aux, sliceyz_aux

    type :: equation_multideal_object
        type(grid_object)         :: grid
        type(field_object)        :: field
        type(cfg_t)               :: cfg
        type(cfg_t)               :: flow_params_cfg
        type(cfg_t)               :: field_info_cfg

        integer(ikind)            :: nx, ny, nz        ! replica from field
        integer(ikind)            :: ng                ! replica from grid
        integer(ikind)            :: mpi_err
        integer(ikind)            :: myrank
        integer(ikind)            :: nprocs
        logical                   :: masterproc

        integer(ikind)            :: nv       
        integer(ikind)            :: nv_aux   
        integer(ikind)            :: nv_stat
        integer(ikind)            :: nv_stat_3d
        integer(ikind)            :: enable_stat_3d

        real(rkind)               :: dt, dt_chem
        real(rkind)               :: cfl        
        real(rkind)               :: time0, time
        integer                   :: num_iter
        integer                   :: icyc0, icyc

        integer(ikind)            :: visc_order, conservative_viscous
        integer(ikind)            :: ep_order, nkeep
        integer(ikind)            :: weno_scheme, weno_version, flux_splitting
        real(rkind)               :: sensor_threshold
        integer(ikind)            :: sensor_type
        integer                   :: rand_type
        integer                   :: eul_imin, eul_imax, eul_jmin, eul_jmax, eul_kmin, eul_kmax
        integer, dimension(6)     :: force_zero_flux
        integer(ikind), dimension(:), allocatable :: supported_orders

        real(rkind), dimension(1:4,4) :: coeff_deriv1
        real(rkind), dimension(0:4,4) :: coeff_deriv2

        real(rkind)               :: ros_dttry, ros_dtmax, ros_eps
        integer                   :: ros_maxsteps, ros_maxtry

        integer(ikind)            :: rk_type
        integer(ikind)            :: nrk
        real(rkind), allocatable, dimension(:) :: rhork,gamrk,alprk
        real(rkind), allocatable, dimension(:) :: ark,brk,crk

        integer                   :: iter_dt_recompute, print_control

        real(rkind)               :: dtsave, dtsave_restart, dtstat, dtslice, dtslice_vtr
        real(rkind)               :: time_from_last_rst, time_from_last_write, time_from_last_stat
        real(rkind)               :: time_from_last_slice, time_from_last_slice_vtr
        integer                   :: restart_type
        integer                   :: io_type_r
        integer                   :: io_type_w
        integer                   :: istore
        integer                   :: itav, itslice_vtr
        integer                   :: enable_plot3d, enable_vtk, enable_slice_vtr
        real(rkind)               :: residual_rhou,vmax

        integer                   :: flow_init
        real(rkind)               :: Mach, Reynolds, Reynolds_friction, Prandtl
        real(rkind)               :: rmixt0, cp0, cv0
        real(rkind)               :: rho0
        real(rkind)               :: t0
        real(rkind)               :: u0, p0, mu0, k0, c0
        real(rkind)               :: t_min_tab, t_max_tab, dt_tab
        real(rkind)               :: gam, gm, gm1, rfac
        real(rkind)               :: rhomin, rhomax, tmin, tmax, pmin, pmax
        integer                   :: use_cantera
        integer                   :: num_t_tab
        real(rkind)               :: dpdx,rhobulk,ubulk,tbulk,volchan   ! for channel flow
        real(rkind)               :: T_wall,delta0,delta0star,theta_wall,T_recovery 
        logical                   :: channel_case, bl_case, bl_laminar 
        real(rkind)               :: x_recyc
        logical                   :: recyc
        integer                   :: i_recyc, ib_recyc, nv_recyc

!       double bl 
        logical                   :: double_bl_case
        real(rkind)               :: Reynolds2,Reynolds_friction2,delta02
!        
        integer     :: itr,its,ifilter,do_filter,is,is_old
        real(rkind) :: xtr1,xtr2,x0tr,xtrip,xtw1,xtw2,a_tr,a_tw,v_bs,kx_tw,om_tw,thic
        real(rkind) :: lamz,lamz1,lam,phiz,phiz1,del0_tr
        real(rkind) :: lamz_old,lamz1_old,phiz_old,phiz1_old

        integer :: enable_limiter
        real(rkind) :: rho_lim, tem_lim, rho_lim_rescale, tem_lim_rescale
 
        integer :: enable_chemistry, enable_soret, nreactions, rosenbrock_version, operator_splitting,enable_endepo

!       Bilger Mixture Fraction  
        integer :: enable_Zbil,N_EoI
        real(rkind), allocatable, dimension(:) :: aw_EoI,Beta0,coeff_EoI
        integer, allocatable, dimension(:,:) :: NainSp
!
        integer :: idx_N2,idx_N,idx_NO,idx_O2,idx_O
        real(rkind), dimension(:), allocatable :: mw, rgas, init_mf,endepo_param
        real(rkind), dimension(:), allocatable :: sigma, epsK, dipole, polariz
        integer, dimension(:), allocatable :: geometry
        real(rkind), dimension(:,:), allocatable :: visc_species, lambda_species
        real(rkind), dimension(:,:,:), allocatable :: diffbin_species

        integer, allocatable, dimension(:)       :: reac_ty,isRev
        real(rkind), allocatable, dimension(:,:) :: arr_a, arr_b, arr_ea
        real(rkind), allocatable, dimension(:)   :: h298
        real(rkind), allocatable, dimension(:,:) :: tb_eff
        real(rkind), allocatable, dimension(:,:) :: falloff_coeffs

        real(rkind), allocatable, dimension(:,:) :: r_coeffs, p_coeffs, kc_tab
        real(rkind), allocatable, dimension(:,:,:,:) :: abcstar_tab

        real(rkind), dimension(:), allocatable :: winf

        real(rkind), dimension(:,:,:,:), allocatable :: w_aux
        real(rkind), dimension(:,:,:), allocatable   :: wmean
        integer, dimension(:,:,:), allocatable       :: fluid_mask
        integer, dimension(:,:,:,:), allocatable     :: ep_ord_change
        integer :: correct_bound_ord

        real(rkind), dimension(:,:,:), allocatable   :: w_stat
        real(rkind), dimension(:,:,:,:), allocatable :: w_stat_3d

        integer, dimension(6) :: bctags, bctags_nr

        real(rkind)                            :: betarecyc, glund1
        real(rkind)                            :: betarecyc2, glund12
        real(rkind), dimension(:), allocatable :: deltavec, deltavvec, cfvec
        real(rkind), dimension(:), allocatable :: deltavec2, deltavvec2, cfvec2
        real(rkind), dimension(:), allocatable :: yplus_inflow,eta_inflow,yplus_recyc
        real(rkind), dimension(:), allocatable :: eta_recyc,eta_recyc_blend
        integer, dimension(:), allocatable     :: map_j_inn,map_j_out,map_j_out_blend
        real(rkind), dimension(:), allocatable :: weta_inflow
        real(rkind), dimension(:), allocatable :: yplus_inflow2,eta_inflow2,yplus_recyc2
        real(rkind), dimension(:), allocatable :: eta_recyc2,eta_recyc_blend2
        integer, dimension(:), allocatable     :: map_j_inn2,map_j_out2,map_j_out_blend2
        real(rkind), dimension(:), allocatable :: weta_inflow2

        real(rkind), dimension(:,:,:), allocatable :: inflow_random_plane
        integer :: jbl_inflow

        real(rkind) :: xbl 

        integer :: indx_cp_l, indx_cp_r, nsetcv
        real(rkind), allocatable, dimension(:,:,:) :: cp_coeff, cv_coeff
        real(rkind), allocatable, dimension(:,:) :: trange
!
        integer(ikind), dimension(:), allocatable :: igslice, jgslice, kgslice
        integer(ikind), dimension(:), allocatable :: islice, jslice, kslice
        integer(ikind) :: num_probe, num_probe_tot
        real(rkind), dimension(:,:), allocatable :: probe_coord, w_aux_probe
        real(rkind), dimension(:,:,:,:), allocatable :: probe_coeff
        integer, dimension(:,:), allocatable :: ijk_probe
        integer, dimension(:), allocatable :: moving_probe

        integer(ikind), dimension(:), allocatable :: list_aux_slice
        integer(ikind) :: num_aux_slice

        integer(ikind), dimension(:), allocatable :: igslice_vtr, jgslice_vtr, kgslice_vtr
        integer(ikind), dimension(:), allocatable :: islice_vtr, jslice_vtr, kslice_vtr

        integer :: debug_memory = 0
        integer :: mode_async = 0

        logical :: time_is_freezed=.false.

        real(rkind) :: eps_sensor
!
!       Cantera
        type(phase_t) :: mixture_yaml
        character(len=30), dimension(:), allocatable :: species_names
        character(len=30), dimension(:), allocatable :: aux_names
!
!       ibm_var_start
!
        integer :: enable_ibm = 0
        integer :: ibm_num_body, ibm_num_bc, ibm_internal_flow
        integer :: ibm_num_interface
        integer :: ibm_stencil_size
        integer :: ibm_methodology
        integer :: ibm_wm = 0
        integer :: ibm_fix_solid_refl = 0
        integer, allocatable, dimension(:,:,:)   :: ibm_sbody
        integer, allocatable, dimension(:,:,:)   :: ibm_is_interface_node
        type(c_ptr), allocatable, dimension(:)   :: ibm_ptree,ibm_ptree_patch
        integer, allocatable, dimension(:,:)     :: ibm_ijk_interface
        integer, allocatable, dimension(:,:)     :: ibm_bc
        integer, allocatable, dimension(:)       :: ibm_type_bc
        real(rkind), allocatable, dimension(:,:) :: ibm_parbc
        real(rkind), allocatable, dimension(:,:) :: ibm_nxyz_interface
        real(rkind), allocatable, dimension(:,:) :: ibm_bbox
        integer, allocatable, dimension(:,:)     :: ibm_ijk_hwm
        real(rkind), allocatable, dimension(:,:,:,:) :: ibm_coeff_hwm
        real(rkind), allocatable, dimension(:) ::       ibm_dist_hwm
        real(rkind), allocatable, dimension(:,:) ::     ibm_xyz_hwm,ibm_wm_wallprop,ibm_wm_stat
        real(rkind) :: ibm_bc_relax_factor
        integer :: ibm_order_reduce
        integer :: ibm_read,ibm_indx_eikonal
        integer :: turinf
        real(rkind), dimension(8) :: randvar_a, randvar_p
        real(rkind) :: ibm_hwm_dist
        real(rkind) :: ibm_eikonal_cfl
        real(rkind) :: ibm_force_x, ibm_force_y, ibm_force_z, ibm_mom_z
        real(rkind), allocatable, dimension(:,:,:) :: ibm_body_dist, ibm_reflection_coeff

        integer :: ibm_type
        integer :: ibm_eikonal
        real(rkind) :: ibm_tol_det_D, ibm_tol_det_N, ibm_tol_distance
        integer :: ibm_interpolation_id_d, ibm_interpolation_id_n
        integer, allocatable, dimension(:,:) :: ibm_ijk_refl,ibm_ijk_wall
        integer, allocatable, dimension(:)       :: ibm_refl_type
        real(rkind), allocatable, dimension(:,:) :: ibm_dist
        real(rkind), allocatable, dimension(:)   :: ibm_bbox_vega
        real(rkind), allocatable, dimension(:,:,:,:) :: ibm_coeff_d,ibm_coeff_n
        real(rkind), allocatable, dimension(:,:,:,:) :: ibm_coeff_tril_d,ibm_coeff_tril_n
        real(rkind), allocatable, dimension(:,:,:,:) :: ibm_coeff_idf,ibm_coeff_idfw
        real(rkind), allocatable, dimension(:,:) :: ibm_dets
        logical, allocatable, dimension(:) :: ibm_refl_insolid
        real(rkind) :: ibm_npr,ibm_ntr,ibm_timeshift,ibm_aero_rthroat,ibm_aero_rexit
        integer :: ibm_aero_nramp
!       integer :: ibm_vega_ny
        real(rkind) :: ibm_vega_ymin, ibm_vega_ymax
!       real(rkind) :: ibm_vega_dy
!       real(rkind), allocatable, dimension(:) :: ibm_vega_y, ibm_vega_r
        real(rkind), allocatable, dimension(:,:) :: ibm_aero_ramp
        real(rkind), allocatable, dimension(:,:) :: ibm_trajectory
        integer :: ibm_trajectory_points
        logical :: ibm_aeroacoustics
        integer :: ibm_vega_moving,ibm_vega_species
        real(rkind) :: ibm_vega_vel
        real(rkind) :: ibm_ramp_ptot, ibm_ramp_Mach
        real(rkind) :: ibm_aero_rad, ibm_aero_pp, ibm_aero_tt, ibm_aero_modvel
        type(c_ptr), allocatable, dimension(:) :: ibm_vega_ptree
        real(rkind), allocatable, dimension(:,:,:) :: ibm_vega_dist
        real(rkind) :: ibm_vega_displacement = 0._rkind

!       ibm_var_end
!
!       insitu_var_start
        integer :: enable_insitu = 0
        integer :: i_freeze = 0 ! to be clarified, maybe it could be done inside insitu first branch only
        integer :: i_insitu
        real(rkind) :: dt_insitu
        real(rkind) :: time_from_last_insitu
        real(rkind) :: time_insitu
        character(len=128) :: vtkpipeline
        integer, allocatable, dimension(:) :: aux_list
        integer, allocatable, dimension(:) :: add_list
        integer, allocatable, dimension(:) :: freeze_intervals
        character(len=64), allocatable, dimension(:) :: aux_list_name
        character(len=64), allocatable, dimension(:) :: add_list_name
        logical :: fcoproc
        logical :: enable_freeze_intervals
        real(rkind) :: perc_ny_cut
        integer :: ny_cut
        integer :: ngm, ngp
        integer :: nxsl_ins, nxel_ins, nysl_ins, nyel_ins, nzsl_ins, nzel_ins
        integer :: nxs_ins, nxe_ins, nys_ins, nye_ins, nzs_ins, nze_ins
        integer :: nxstartg, nxendg, nystartg, nyendg, nzstartg, nzendg
        integer :: npsi, npsi_pv, n_aux_list, n_add_list
        real(rkind), allocatable, dimension(:,:,:,:) :: psi, psi_pv
        real(rkind), allocatable, dimension(:) :: xyzc
        integer :: insitu_flag

        character(64) :: insitu_platform
        real(rkind), allocatable, dimension(:) :: points_x, points_y, points_z
        integer :: n_points_x, n_points_y, n_points_z
        integer(int64) :: n_points
!       insitu_var_end
!
!       les_var_start
!
        integer :: enable_les   = 0
        integer :: enable_pasr  = 0
        integer :: les_model
        real(rkind) :: les_c_wale, les_pr, les_sc, les_c_yoshi, les_c_eps, les_c_mix
        
!       jcf_var_start
        integer :: enable_jcf = 0
        integer :: jcf_jet_num
        real(rkind) :: jcf_jet_rad, jcf_relax_factor
        real(rkind), allocatable, dimension(:,:) :: jcf_parbc,jcf_coords
!       jcf_var_end
!        
    contains
        ! public methods
        procedure, pass(self) :: initialize              
        procedure, pass(self) :: read_input              
        procedure, pass(self) :: runge_kutta_initialize  
        procedure, pass(self) :: initial_conditions 
        procedure, pass(self) :: read_field_info    
        procedure, pass(self) :: write_field_info   
!       procedure, pass(self) :: set_oblique_shock
        procedure, pass(self) :: set_fluid_prop
        procedure, pass(self) :: set_fluid_prop_cantera
        procedure, pass(self) :: set_flow_params 
        procedure, pass(self) :: set_bl_prop 
        procedure, pass(self) :: set_double_bl_prop 
!       procedure, pass(self) :: set_chan_prop
!       procedure, pass(self) :: init_channel 
        procedure, pass(self) :: init_wind_tunnel
        procedure, pass(self) :: init_double_bl
        procedure, pass(self) :: compute_bl
        procedure, pass(self) :: init_sod
        procedure, pass(self) :: init_multi_diff
        procedure, pass(self) :: init_aw
        procedure, pass(self) :: init_reactor
        procedure, pass(self) :: init_reactive_tube
        procedure, pass(self) :: init_premix
        procedure, pass(self) :: init_scalability
!       procedure, pass(self) :: init_bl_old
        procedure, pass(self) :: init_bl
        procedure, pass(self) :: init_bl_lam
        procedure, pass(self) :: alloc
        procedure, pass(self) :: bc_preproc
        procedure, pass(self) :: compute_stats
        procedure, pass(self) :: write_stats
        procedure, pass(self) :: write_stats_serial
        procedure, pass(self) :: read_stats
        procedure, pass(self) :: read_stats_serial
        procedure, pass(self) :: compute_stats_3d
        procedure, pass(self) :: write_stats_3d
        procedure, pass(self) :: read_stats_3d
        procedure, pass(self) :: write_stats_3d_serial
        procedure, pass(self) :: read_stats_3d_serial
        procedure, pass(self) :: recyc_prepare 
        procedure, pass(self) :: add_synthetic_perturbations
        procedure, pass(self) :: add_synthetic_perturbations_double_bl
        procedure, pass(self) :: slice_prepare
        procedure, pass(self) :: probe_prepare
        procedure, pass(self) :: probe_compute_coeff
        procedure, pass(self) :: correct_bc_order
        procedure, pass(self) :: time_is_freezed_fun
        procedure, pass(self) :: write_slice_vtr 
        procedure, pass(self) :: compute_chemistry 
        procedure, pass(self) :: compute_collision_integrals 
!       Bilger Mixture Fraction
        procedure, pass(self) :: zbil_initialize
!       IBM
!       ibm_proc_start
        procedure, pass(self) :: ibm_initialize
        procedure, pass(self) :: ibm_readoff 
        procedure, pass(self) :: ibm_alloc
        procedure, pass(self) :: ibm_raytracing 
        procedure, pass(self) :: ibm_raytracing_write
        procedure, pass(self) :: ibm_raytracing_read
        procedure, pass(self) :: ibm_setup_geo
        procedure, pass(self) :: ibm_read_geo
        procedure, pass(self) :: ibm_correct_fields
        procedure, pass(self) :: ibm_setup_computation
        procedure, pass(self) :: ibm_compute_refl_coeff
        procedure, pass(self) :: ibm_prepare_wm
        procedure, pass(self) :: ibm_write_wm_wallprop
        procedure, pass(self) :: ibm_write_wm_stat
        procedure, pass(self) :: ibm_read_wm_stat
        procedure, pass(self) :: ibm_compute_stat_wm

!       ibm_proc_old
        procedure, pass(self) :: ibm_initialize_old        !Ok
        procedure, pass(self) :: ibm_readoff_old           !Ok
        procedure, pass(self) :: ibm_alloc_old             !Ok
        procedure, pass(self) :: ibm_raytracing_old        !Ok
        procedure, pass(self) :: ibm_raytracing_write_old  !Ok
        procedure, pass(self) :: ibm_raytracing_read_old   !Ok
        procedure, pass(self) :: ibm_compute_geo_old       !Ok
        procedure, pass(self) :: ibm_read_geo_old          !Ok
        procedure, pass(self) :: ibm_correct_fields_old    !Ok
        procedure, pass(self) :: ibm_setup_computation_old !Ok 
        procedure, pass(self) :: ibm_bc_prepare_old        !Ok
        procedure, pass(self) :: ibm_coeff_setup_old       !

!       ibm_proc_end
!
!       INSITU
!       insitu_proc_start
        procedure, pass(self) :: insitu_initialize
        procedure, pass(self) :: insitu_finalize
        procedure, pass(self) :: insitu_allocate
        procedure, pass(self) :: insitu_define_limits
        procedure, pass(self) :: insitu_do_catalyst_initialization
        procedure, pass(self) :: insitu_do_catalyst_finalization
!
!       JCF
        procedure, pass(self) :: jcf_initialize
!       insitu_proc_end
    endtype equation_multideal_object

contains

    subroutine initialize(self, filename)
        !< Initialize the equation.
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
        character(*)                    , intent(in)      :: filename          !< Input file name.
        logical, dimension(3) :: periodic
        integer, dimension(3) :: mpi_splits
        integer :: mpi_split_x ,mpi_split_y ,mpi_split_z
        integer :: nxmax, nymax, nzmax, ng 
        integer :: grid_type, metrics_order, ystag
        real(rkind) :: domain_size_x,  domain_size_y,  domain_size_z, rtemp
        real(rkind), dimension(:), allocatable :: grid_vars
        logical :: rebuild_ghost, ystaggering
        integer :: bctag_temp, order
        real(rkind) :: d1_temp(1:4), d2_temp(0:4)
        integer :: i,lsp

        ! Get MPI basic quantities
        call get_mpi_basic_info(self%nprocs, self%myrank, self%masterproc, self%mpi_err)

        ! Read input
        call self%read_input(filename)

        ! Enable memory debugger
        if (self%cfg%has_key("output","debug_memory")) call self%cfg%get("output","debug_memory",self%debug_memory)

        ! Enable async pattern
        if (self%cfg%has_key("numerics","mode_async")) call self%cfg%get("numerics","mode_async",self%mode_async) 

        ! Random numbers' configuration
        call self%cfg%get("numerics","rand_type",self%rand_type) 
        if(self%rand_type == 0) then
            call init_crandom_f(0,reproducible=.true.)
            if (self%masterproc) write(*,*) 'Random numbers disabled'
        elseif(self%rand_type < 0) then
            call init_crandom_f(self%myrank+1,reproducible=.false.)
            if (self%masterproc) write(*,*) 'Random numbers NOT reproducible'
        else
            call init_crandom_f(self%myrank+1,reproducible=.true.)
            if (self%masterproc) write(*,*) 'Random numbers reproducible'
        endif

        ! Reference state
        call self%cfg%get("ref_state","u0",self%u0)

        ! Flow init
        call self%cfg%get("flow","flow_init",self%flow_init)
        if (self%flow_init<-1 .or.self%flow_init>12) call fail_input_any("flow_init not implemented")
        self%channel_case = .false.
        if (self%flow_init == 0) self%channel_case = .true.
        self%bl_case = .false.
        if (self%flow_init == 1) then
         self%bl_case    = .true.
         self%bl_laminar = .false. ! Default is turbulent BL
        endif
        if (self%flow_init == 4) then
         self%double_bl_case = .true.
         if (self%bctags(1)==9) call fail_input_any("Laminar Double BL not implemented")
        endif

        ! Boundary conditions
        call self%cfg%get("bc","xmin",bctag_temp) ; self%bctags(1) = bctag_temp 
        if (self%bctags(1)==9) self%bl_laminar = .true.
        call self%cfg%get("bc","xmax",bctag_temp) ; self%bctags(2) = bctag_temp
        call self%cfg%get("bc","ymin",bctag_temp) ; self%bctags(3) = bctag_temp
        if (self%bctags(3)==36) self%bl_laminar = .true.
        call self%cfg%get("bc","ymax",bctag_temp) ; self%bctags(4) = bctag_temp
        call self%cfg%get("bc","zmin",bctag_temp) ; self%bctags(5) = bctag_temp
        call self%cfg%get("bc","zmax",bctag_temp) ; self%bctags(6) = bctag_temp
        periodic(1:3) = .false.
        if(any(self%bctags(1:2) == 0)) then
            self%bctags(1:2) = 0
            periodic(1) = .true.
        endif
        if(any(self%bctags(3:4) == 0)) then
            self%bctags(3:4) = 0
            periodic(2) = .true.
        endif
        if(any(self%bctags(5:6) == 0)) then
            self%bctags(5:6) = 0
            periodic(3) = .true.
        endif
        call self%cfg%get("bc","xmin_nr",bctag_temp) ; self%bctags_nr(1) = bctag_temp 
        call self%cfg%get("bc","xmax_nr",bctag_temp) ; self%bctags_nr(2) = bctag_temp
        call self%cfg%get("bc","ymin_nr",bctag_temp) ; self%bctags_nr(3) = bctag_temp
        call self%cfg%get("bc","ymax_nr",bctag_temp) ; self%bctags_nr(4) = bctag_temp
        call self%cfg%get("bc","zmin_nr",bctag_temp) ; self%bctags_nr(5) = bctag_temp
        call self%cfg%get("bc","zmax_nr",bctag_temp) ; self%bctags_nr(6) = bctag_temp
        ! Restart
        call self%cfg%get("controls","restart_type",self%restart_type)
        call self%cfg%get("grid","grid_type",grid_type)
        rebuild_ghost = .false.
        if (grid_type == GRID_FROMFILE) rebuild_ghost = .true.
        if (self%restart_type > 0) grid_type = GRID_FROMFILE

        ! Grid
        call self%cfg%get("grid","nxmax",nxmax)
        call self%cfg%get("grid","nymax",nymax)
        call self%cfg%get("grid","nzmax",nzmax)
        call self%cfg%get("grid","ng",ng)
        call self%cfg%get("grid","domain_size_x",domain_size_x)
        call self%cfg%get("grid","domain_size_y",domain_size_y)
        call self%cfg%get("grid","domain_size_z",domain_size_z)
        call self%cfg%get("grid","metrics_order",metrics_order)

        ystag = 0
        if (self%cfg%has_key("grid","ystag")) then
         call self%cfg%get("grid","ystag",ystag)
        endif
        ystaggering = .false.
        if (ystag>0) ystaggering = .true.
        if (self%channel_case) ystaggering = .true.

        call self%cfg%get("numerics","ep_order",self%ep_order)
        if (self%ep_order /= metrics_order) then
         if (self%masterproc) write(*,*) 'Changing metrics_order = ep_order ', self%ep_order
         self%grid%metrics_order = self%ep_order
        endif

        select case (grid_type)
            case(GRID_FROMFILE)
            case(GRID_UNIFORM)
            case(GRID_CHA)
                allocate(grid_vars(4))
                call self%cfg%get("grid","jbgrid",rtemp)     ; grid_vars(1) = rtemp 
                call self%cfg%get("grid","dyptarget",rtemp)  ; grid_vars(2) = rtemp 
                call self%cfg%get("grid","ysmoosteps",rtemp) ; grid_vars(3) = rtemp 
                call self%cfg%get("flow","Reynolds",rtemp)   ; grid_vars(4) = rtemp 
                call fail_input_any("channel grid not implemented")
            case(GRID_BL)
                allocate(grid_vars(7))
                call self%cfg%get("grid","jbgrid",rtemp)     ; grid_vars(1) = rtemp
                call self%cfg%get("grid","dyptarget",rtemp)  ; grid_vars(2) = rtemp
                call self%cfg%get("grid","nywr",rtemp)       ; grid_vars(3) = rtemp
                call self%cfg%get("grid","lywr",rtemp)       ; grid_vars(4) = rtemp
                call self%cfg%get("grid","ysmoosteps",rtemp) ; grid_vars(5) = rtemp 
                call self%cfg%get("flow","delta0",rtemp)     ; grid_vars(6) = rtemp 
                call self%cfg%get("flow","Reynolds",rtemp)   ; grid_vars(7) = rtemp
            case(GRID_2BL)
                allocate(grid_vars(9))
                call self%cfg%get("grid","jbgrid",rtemp)     ; grid_vars(1) = rtemp
                call self%cfg%get("grid","dyptarget",rtemp)  ; grid_vars(2) = rtemp
                call self%cfg%get("grid","nywr",rtemp)       ; grid_vars(3) = rtemp
                call self%cfg%get("grid","lywr",rtemp)       ; grid_vars(4) = rtemp
                call self%cfg%get("grid","ysmoosteps",rtemp) ; grid_vars(5) = rtemp 
                call self%cfg%get("flow","delta0",rtemp)     ; grid_vars(6) = rtemp 
                call self%cfg%get("flow","Reynolds",rtemp)   ; grid_vars(7) = rtemp
                call self%cfg%get("flow","delta02",rtemp)    ; grid_vars(8) = rtemp 
                call self%cfg%get("flow","Reynolds2",rtemp)  ; grid_vars(9) = rtemp
            case(GRID_BL_LAM)
                allocate(grid_vars(7))
                call self%cfg%get("grid","jbgrid",rtemp)     ; grid_vars(1) = rtemp
                call self%cfg%get("grid","dyptarget",rtemp)  ; grid_vars(2) = rtemp
                call self%cfg%get("grid","nywr",rtemp)       ; grid_vars(3) = rtemp
                call self%cfg%get("grid","lywr",rtemp)       ; grid_vars(4) = rtemp
                call self%cfg%get("grid","ysmoosteps",rtemp) ; grid_vars(5) = rtemp 
                call self%cfg%get("grid","delta0",rtemp)     ; grid_vars(6) = rtemp 
                call self%cfg%get("grid","retau",rtemp)      ; grid_vars(7) = rtemp

        end select

        call self%grid%initialize(periodic, nxmax, nymax, nzmax, ng, grid_type, &
                domain_size_x, domain_size_y, domain_size_z, &
                grid_vars, metrics_order, rebuild_ghost, ystaggering)

        ! Number of variables and auxiliary variables
        self%nv         = N_S+4
        self%nv_aux     = 2*N_S+13
        self%nv_stat    = 86+2*N_S
        self%nv_stat_3d = 16+2*N_S
  
        if (self%cfg%has_key("fluid","enable_chemistry")) then
         call self%cfg%get("fluid","enable_chemistry",self%enable_chemistry)
        endif

        ! Energy deposition
        if (self%cfg%has_key("fluid","enable_endepo")) then
         call self%cfg%get("fluid","enable_endepo",self%enable_endepo)
         if (self%enable_endepo .eq. 1) then
          allocate(self%endepo_param(11))
          call self%cfg%get("fluid","endepo_param",self%endepo_param)
         endif
        endif

        ! LES model
        if (self%cfg%has_key("lespar","enable_les")) then
         call self%cfg%get("lespar","enable_les",self%enable_les)
        endif
        self%enable_pasr = 0
        if (self%enable_les>0) then
         call self%cfg%get("lespar","les_model",self%les_model)
         call self%cfg%get("lespar","les_pr",self%les_pr)
         call self%cfg%get("lespar","les_sc",self%les_sc)
         select case (self%les_model)
         case(1)
          if (self%masterproc) write(*,*) 'LES model: WALE'
          call self%cfg%get("lespar","les_c_wale",self%les_c_wale)
         case(2)
          if (self%enable_chemistry .ne. 2) call fail_input_any("PaSR cannot be used without chemistry enabled or for explicit integration")
          if (self%masterproc) write(*,*) 'LES model: WALE + PaSR'
          self%enable_pasr = 1
          call self%cfg%get("lespar","les_c_wale",self%les_c_wale)
          call self%cfg%get("lespar","les_c_yoshi",self%les_c_yoshi)
          call self%cfg%get("lespar","les_c_eps",self%les_c_eps)
          call self%cfg%get("lespar","les_c_mix",self%les_c_mix)
         case default
          call fail_input_any("LES model not implemented")
         end select
         if (self%mode_async/=0) then
          if (self%masterproc) write(*,*) 'Changing mode async to 0 for LES'
          self%mode_async = 0
         endif
        else
         self%les_pr = 0._rkind
         self%les_sc = 0._rkind
        endif

        self%enable_soret = 0
        if (self%cfg%has_key("fluid","enable_soret")) then
         call self%cfg%get("fluid","enable_soret",self%enable_soret)
        endif 
        if (self%enable_les>0) self%nv_aux = self%nv_aux+1
        if (self%enable_chemistry>0) self%nv_aux = self%nv_aux+N_S+1
        if (self%enable_pasr>0) self%nv_aux = self%nv_aux+1
        if (self%enable_chemistry>0) self%nv_stat = self%nv_stat+2*N_S+2
        if (self%enable_les>0) self%nv_stat = self%nv_stat + 11

        ! Set fluid properties
        call self%cfg%get("fluid","use_cantera",self%use_cantera)

        if (self%use_cantera==1) then
         call self%set_fluid_prop_cantera()
        else
         call self%set_fluid_prop()
        endif
        ! Set flow parameters
        call self%set_flow_params()

        ! Numerics
        if (self%cfg%has_key("numerics","nkeep")) then
         call self%cfg%get("numerics","nkeep",self%nkeep)
        else
         self%nkeep = 0
        endif
        call self%cfg%get("numerics","weno_scheme",self%weno_scheme)
        call self%cfg%get("numerics","weno_version",self%weno_version)
        if (self%cfg%has_key("numerics","flux_splitting")) then
         call self%cfg%get("numerics","flux_splitting",self%flux_splitting)
        else
         self%flux_splitting = 0
        endif
        call self%cfg%get("numerics","visc_order",self%visc_order)
        if (self%visc_order /= self%ep_order) then
         if (self%masterproc) write(*,*) 'Changing visc_order = ep_order ', self%ep_order
         self%visc_order = self%ep_order
        endif
        call self%cfg%get("numerics","conservative_viscous",self%conservative_viscous)
        allocate(self%supported_orders(1:4))
        self%supported_orders = [2,4,6,8]
        do i=1,size(self%supported_orders)
            order = self%supported_orders(i)
            call self%grid%get_deriv_coeffs(d1_temp, d2_temp, order, order)
            self%coeff_deriv1(:,i) = d1_temp
            self%coeff_deriv2(:,i) = d2_temp
        enddo
        call self%cfg%get("numerics","sensor_threshold",self%sensor_threshold)
        self%sensor_type = 1 ! default is modified ducros sensor
        if (self%cfg%has_key("numerics","sensor_type")) call self%cfg%get("numerics","sensor_type",self%sensor_type)

        ! Stats 3D
         call self%cfg%get("output","enable_plot3d",self%enable_plot3d)
        if (self%cfg%has_key("output","enable_stat_3d")) then
         call self%cfg%get("output","enable_stat_3d",self%enable_stat_3d)
        else
         self%enable_stat_3d = 0
        endif

        ! MPI split
        call self%cfg%get("mpi","x_split",mpi_split_x) ; mpi_splits(1) = mpi_split_x
        call self%cfg%get("mpi","y_split",mpi_split_y) ; mpi_splits(2) = mpi_split_y
        call self%cfg%get("mpi","z_split",mpi_split_z) ; mpi_splits(3) = mpi_split_z

        ! Field initialize
        call self%field%initialize(self%grid, self%nv, mpi_splits)

        ! Allocate variables
        call self%alloc()

        ! Fix boundary conditions considering MPI neighbours
        !print*,'before rank= ',self%myrank, ' - bc: ',self%bctags
        call self%field%correct_bc(self%bctags)
        call self%field%correct_bc(self%bctags_nr)

        ! Pre-process boundary conditions
        call self%bc_preproc()
        !print*,'rank= ',self%myrank, ' - bc: ',self%bctags,' - bc_nr: ',self%bctags_nr

        ! Initialize temporal integrator
        if (self%enable_chemistry == 2) then
         call self%cfg%get("numerics","ros_dttry",self%ros_dttry)
         call self%cfg%get("numerics","ros_dtmax",self%ros_dtmax)
         call self%cfg%get("numerics","ros_eps",self%ros_eps)
         call self%cfg%get("numerics","ros_maxsteps",self%ros_maxsteps)
         call self%cfg%get("numerics","ros_maxtry",self%ros_maxtry)
        endif

        self%enable_limiter = 0
        if (self%cfg%has_key("limiter","enable_limiter")) then
         call self%cfg%get("limiter","enable_limiter",self%enable_limiter)
         if (self%enable_limiter > 0) then
          if (self%masterproc) write(*,*) 'Limiter enabled'
         endif
        endif
        self%rho_lim = 0.0_rkind
        if (self%cfg%has_key("limiter","rho_lim")) then
         call self%cfg%get("limiter","rho_lim",self%rho_lim)
        endif
        if (self%cfg%has_key("limiter","tem_lim")) then
         call self%cfg%get("limiter","tem_lim",self%tem_lim)
        endif
        self%rho_lim_rescale = 1.0_rkind
        if (self%cfg%has_key("limiter","rho_lim_rescale")) then
         call self%cfg%get("limiter","rho_lim_rescale",self%rho_lim_rescale)
        endif
        self%tem_lim_rescale = 1.0_rkind
        if (self%cfg%has_key("limiter","tem_lim_rescale")) then
         call self%cfg%get("limiter","tem_lim_rescale",self%tem_lim_rescale)
        endif
        if(self%enable_limiter > 0 .and. self%masterproc) then
            print*,'Limited minimum rho :',self%rho_lim
            print*,'Limited minimum tem :',self%tem_lim
            print*,'Limited minimum rho rescale :',self%rho_lim_rescale
            print*,'Limited minimum tem rescale :',self%tem_lim_rescale
        endif

        call self%cfg%get("numerics","rk_type",self%rk_type)
        call self%runge_kutta_initialize()
        call self%cfg%get("controls","cfl",self%cfl)
        call self%cfg%get("controls","num_iter",self%num_iter)

        ! IO type
        self%io_type_r = 2 ! (0 => no IO, 1 => serial, 2 => parallel)
        if (self%cfg%has_key("output","io_type_r")) then
         call self%cfg%get("output","io_type_r",self%io_type_r)
        endif
        self%io_type_w = 2 ! (0 => no IO, 1 => serial, 2 => parallel)
        if (self%cfg%has_key("output","io_type_w")) then
         call self%cfg%get("output","io_type_w",self%io_type_w)
        endif

        ! enable_insitu must be known here to garantee correct reading of read_field_info
        if (self%cfg%has_key("insitu","enable_insitu")) then
         call self%cfg%get("insitu","enable_insitu",self%enable_insitu)
         self%insitu_platform = "catalyst-v2"
         if (self%cfg%has_key("insitu","insitu_platform")) then
          call self%cfg%get("insitu","insitu_platform",self%insitu_platform)
         endif
         if (self%enable_insitu>0) then
          if (self%masterproc) print*,'Insitu-platform :',self%insitu_platform
         endif
        endif
        call self%initial_conditions()
        select case(self%restart_type)
            case(0)
                 self%time0       = 0._rkind
                 self%icyc0       = 0
                 self%itav        = 0
                 self%itslice_vtr = 0
                 self%istore      = 1
                 self%time_from_last_rst       = 0._rkind
                 self%time_from_last_write     = 0._rkind
                 self%time_from_last_stat      = 0._rkind
                 self%time_from_last_slice     = 0._rkind
                 self%time_from_last_slice_vtr = 0._rkind
                 if(self%enable_insitu > 0) then
                     self%i_insitu = 1
                     self%time_from_last_insitu = 0._rkind
                     self%time_insitu = 0._rkind
                 endif
                 self%w_stat = 0._rkind
                 if (self%enable_stat_3d>0)  self%w_stat_3d = 0._rkind 
            case(1)
                 if (self%io_type_r==1) call self%field%read_field_serial()
                 if (self%io_type_r==2) call self%field%read_field()
                 call self%read_field_info()
                 self%itav        = 0
                 !self%itslice_vtr = 0
                 self%w_stat      = 0._rkind
                 if (self%enable_stat_3d>0)  self%w_stat_3d = 0._rkind 
            case(2)
                 if (self%io_type_r==1) then
                  call self%field%read_field_serial()
                  call self%read_stats_serial()
                  if (self%enable_stat_3d>0) call self%read_stats_3d_serial()
                 endif
                 if (self%io_type_r==2) then
                  call self%field%read_field()
                  call self%read_stats()
                  if (self%enable_stat_3d>0) call self%read_stats_3d()
                 endif
                 call self%read_field_info()
        endselect

        self%recyc = .false.
        ! only ncoords(1) procs have bctags 10, otherwise MPI tag replaced it
        if (self%field%ncoords(1) == 0) then
            if(self%bctags(1) == 10) then
                self%recyc = .true.
            endif
        endif
        call mpi_bcast(self%recyc,1,mpi_logical,0,self%field%mp_cartx,self%mpi_err)
        if (self%recyc) call self%recyc_prepare()

        self%time     = self%time0
        self%icyc     = self%icyc0

        ! Save field and restart intervals
        call self%cfg%get("output","dtsave",self%dtsave)
        call self%cfg%get("output","dtsave_restart",self%dtsave_restart)
        call self%cfg%get("output","dtstat",self%dtstat)
        call self%cfg%get("output","enable_plot3d",self%enable_plot3d)
        call self%cfg%get("output","enable_vtk",self%enable_vtk)
        call self%cfg%get("output","dtslice",self%dtslice)
        call self%probe_prepare()
        call self%probe_compute_coeff()

        if (self%cfg%has_key("output","list_aux_slice")) then
         call self%cfg%get("output","list_aux_slice",self%list_aux_slice)
         self%num_aux_slice = size(self%list_aux_slice)
        else
         ! default: mass fraction, velocities, P, T, rho, speed of sound, viscosity
         self%num_aux_slice = N_S+8
         allocate(self%list_aux_slice(self%num_aux_slice))
         do lsp=1,N_S
          self%list_aux_slice(lsp) = lsp
         enddo
         self%list_aux_slice(N_S+1)  = J_U
         self%list_aux_slice(N_S+2)  = J_V
         self%list_aux_slice(N_S+3)  = J_W
         self%list_aux_slice(N_S+4)  = J_C
         self%list_aux_slice(N_S+5)  = J_P
         self%list_aux_slice(N_S+6)  = J_T
         self%list_aux_slice(N_S+7)  = J_R
         self%list_aux_slice(N_S+8) = J_MU
        endif

        allocate(self%aux_names(self%nv_aux))
        do lsp=1,N_S
         write(self%aux_names(lsp)             ,'(A,A)') 'Y_',trim(self%species_names(lsp))
         write(self%aux_names(J_D_START+lsp)   ,'(A,A)') 'D_',trim(self%species_names(lsp))
         if (self%enable_chemistry > 0) write(self%aux_names(J_WDOT_START+lsp),'(A,A)') 'WDOT_',trim(self%species_names(lsp))
        enddo
        self%aux_names(J_U)      = "velocity_x" 
        self%aux_names(J_V)      = "velocity_y" 
        self%aux_names(J_W)      = "velocity_z" 
        self%aux_names(J_H)      = "enthalpy" 
        self%aux_names(J_T)      = "temperature" 
        self%aux_names(J_P)      = "pressure" 
        self%aux_names(J_C)      = "speed of sound" 
        if (self%enable_les == 0) then 
         self%aux_names(J_MU)    = "viscosity"
        else
         self%aux_names(J_MU)    = "viscosity_total" 
         self%aux_names(J_LES1)  = "viscosity_sgs" 
        endif
        self%aux_names(J_DUC)    = "ducros"
        self%aux_names(J_DIV)    = "div3"
        self%aux_names(J_R)      = "density"
        self%aux_names(J_Z)      = 'Z'
        self%aux_names(J_K_COND) = "thermal_conductivity"
        if (self%enable_chemistry > 0) self%aux_names(J_HRR) = 'heat release rate'
        if (self%enable_pasr > 0) self%aux_names(J_GAM_PASR) = 'gam_pasr'
        call self%cfg%get("output","igslice",self%igslice)
        call self%cfg%get("output","jgslice",self%jgslice)
        call self%cfg%get("output","kgslice",self%kgslice)
        call self%slice_prepare(self%igslice, self%jgslice, self%kgslice, &
                 self%islice, self%jslice, self%kslice, alloc_aux=.true.)

        self%enable_slice_vtr = 0
        if (self%cfg%has_key("output","enable_slice_vtr")) then
         call self%cfg%get("output","enable_slice_vtr",self%enable_slice_vtr)
        endif
        if(self%enable_slice_vtr > 0) then
         call self%cfg%get("output","igslice_vtr",self%igslice_vtr)
         call self%cfg%get("output","jgslice_vtr",self%jgslice_vtr)
         call self%cfg%get("output","kgslice_vtr",self%kgslice_vtr)
         call self%slice_prepare(self%igslice_vtr, self%jgslice_vtr, self%kgslice_vtr, &
                  self%islice_vtr, self%jslice_vtr, self%kslice_vtr, alloc_aux = .false.)
         if (self%cfg%has_key("output","dtslice_vtr")) then
          call self%cfg%get("output","dtslice_vtr",self%dtslice_vtr)
         else
          self%dtslice_vtr = self%dtslice
         endif
        endif

        call self%cfg%get("output","print_control",self%print_control)
        call self%cfg%get("controls","iter_dt_recompute",self%iter_dt_recompute)

!       Correct order of accuracy at the borders 
        self%correct_bound_ord = 0
        if (self%cfg%has_key("bc","correct_bound_ord")) then
         call self%cfg%get("bc","correct_bound_ord",self%correct_bound_ord)
        endif
        if (self%correct_bound_ord>0) call self%correct_bc_order()

        if (self%cfg%has_key("ibmpar","enable_ibm")) then
         call self%cfg%get("ibmpar","enable_ibm",self%enable_ibm)
        endif
        if (self%enable_ibm>0) then
         call self%cfg%get("ibmpar","ibm_num_body",self%ibm_num_body)
         !call self%ibm_initialize()
         call self%ibm_initialize_old()
        endif

        !put as type default self%i_freeze = 0
        if (self%enable_insitu>0) call self%insitu_initialize()

!       JCF
        if(self%cfg%has_key("jcfpar","enable_jcf")) then
         call self%cfg%get("jcfpar","enable_jcf",self%enable_jcf)
         if (self%enable_jcf > 0) call self%jcf_initialize
        endif

        ! Mixture Fraction
        if (self%cfg%has_key("Zpar","enable_Zbil")) then
         call self%cfg%get("Zpar","enable_Zbil",self%enable_Zbil)
         if (self%enable_Zbil>0 .and. self%use_cantera==1) then
          call self%zbil_initialize()
         elseif (self%enable_Zbil>0 .and. self%use_cantera==0) then
          call fail_input_any("Zbil implemented only using Cantera!")
         elseif (self%enable_Zbil == 0) then
          self%N_EoI = 1
         endif
        else
         self%N_EoI = 1
        endif

        ! LES model
!       if (self%cfg%has_key("lespar","enable_les")) then
!        call self%cfg%get("lespar","enable_les",self%enable_les)
!       endif
       
    endsubroutine initialize

    subroutine zbil_initialize(self)
        class(equation_multideal_object), intent(inout) :: self
     
        integer :: N_E,le,lsp,lezbil,l,idx
        real(rkind), dimension(:), allocatable :: aw
        character(len=30), dimension(:), allocatable :: element_names
        character*3, allocatable, dimension(:) :: Names_EoI
        real(rkind) :: yatm_inflow,yatm_ambient
        logical :: found

        call self%cfg%get("Zpar","N_EoI",self%N_EoI)
        call self%cfg%get("Zpar","Names_EoI",Names_EoI)
        call self%cfg%get("Zpar","coeff_EoI",self%coeff_EoI)

        N_E = nElements(self%mixture_yaml)
        allocate(aw(N_E), element_names(N_E), self%aw_EoI(self%N_EoI))
        allocate(self%NainSp(self%N_EoI,N_S),self%Beta0(2))
        call getAtomicWeights(self%mixture_yaml,aw)
        do le = 1,N_E
         call getElementName(self%mixture_yaml, le, element_names(le))
         do lezbil = 1,self%N_EoI
          if (element_names(le) .eq. Names_EoI(lezbil)) then
           self%aw_EoI(lezbil) = aw(le)
           do lsp = 1,N_S
            self%NainSp(lezbil,lsp) = nAtoms(self%mixture_yaml,lsp,le)
           enddo
          endif
         enddo
        enddo
        deallocate(aw,element_names,Names_EoI)
        !Find inflow on IBM        
        found = .false.
        do l = 1, self%ibm_num_bc
         select case ( self%ibm_type_bc(l) )
         case (1:4)          ! match 1,2,3,4 - New IBM - Not Workind
          idx   = l
          found = .true.
         exit
         endselect
        end do
        if (.not. found) call fail_input_any("No inflow on IBM")

        self%Beta0 = 0._rkind
        do lezbil = 1,self%N_EoI
         yatm_inflow  = 0._rkind
         yatm_ambient = 0._rkind
         do lsp = 1,N_S
          yatm_inflow  = yatm_inflow  + self%NainSp(lezbil,lsp)*self%aw_EoI(lezbil)*self%ibm_parbc(idx,5+lsp)/self%mw(lsp)
          yatm_ambient = yatm_ambient + self%NainSp(lezbil,lsp)*self%aw_EoI(lezbil)*self%init_mf(lsp)/self%mw(lsp)
         enddo
         self%Beta0(1) = self%Beta0(1) + self%coeff_EoI(lezbil)*yatm_inflow/self%aw_EoI(lezbil)
         self%Beta0(2) = self%Beta0(2) + self%coeff_EoI(lezbil)*yatm_ambient/self%aw_EoI(lezbil)
        enddo

    endsubroutine zbil_initialize

    subroutine correct_bc_order(self)
        class(equation_multideal_object), intent(inout) :: self
!
        integer :: i,j,k
        integer :: stencil_size
!
        associate(nx => self%field%nx, ny => self%field%ny,nz => self%field%nz, &
                  ng => self%grid%ng, ep_order => self%ep_order, weno_scheme => self%weno_scheme, &
                  ncoords => self%field%ncoords, bctags => self%bctags, nblocks => self%field%nblocks)

        stencil_size = max(ep_order/2, weno_scheme)
!
!       IMIN
        if ((ncoords(1)==0).and.(any(bctags(1:2)/=0))) then
         self%ep_ord_change(0,:,:,1)  = -stencil_size+1
         do i=1,stencil_size-1
          self%ep_ord_change(i,:,:,1) = -stencil_size+i
         enddo
        endif
!
!       IMAX
        if ((ncoords(1)==(nblocks(1)-1)).and.(any(bctags(1:2)/=0))) then
         self%ep_ord_change(nx,:,:,1) = -stencil_size+1
         do i=1,stencil_size-1
          self%ep_ord_change(nx-i,:,:,1) = -stencil_size+i
         enddo
        endif
!
!       JMIN
        if ((ncoords(2)==0).and.(any(bctags(3:4)/=0))) then
         self%ep_ord_change(:,0,:,2)  = -stencil_size+1
         do j=1,stencil_size-1
          self%ep_ord_change(:,j,:,2) = -stencil_size+j
         enddo
        endif
!
!       JMAX
        if ((ncoords(2)==(nblocks(2)-1)).and.(any(bctags(3:4)/=0))) then
         self%ep_ord_change(:,ny,:,2) = -stencil_size+1
         do j=1,stencil_size-1
          self%ep_ord_change(:,ny-j,:,2) = -stencil_size+j
         enddo
        endif
!
!       KMIN
        if ((ncoords(3)==0).and.(any(bctags(5:6)/=0))) then
         self%ep_ord_change(:,:,0,3)  = -stencil_size+1
         do k=1,stencil_size-1
          self%ep_ord_change(:,:,k,3) = -stencil_size+k
         enddo
        endif
!
!       KMAX
        if ((ncoords(3)==(nblocks(3)-1)).and.(any(bctags(5:6)/=0))) then
         self%ep_ord_change(:,:,nz,3) = -stencil_size+1
         do k=1,stencil_size-1
          self%ep_ord_change(:,:,nz-k,3) = -stencil_size+k
         enddo
        endif

        endassociate
    end subroutine correct_bc_order

    subroutine insitu_do_catalyst_initialization(self)
        class(equation_multideal_object), intent(inout) :: self 
        type(C_PTR) :: params, about, exec, final, results
        integer, parameter :: f64 = selected_real_kind(8)
        integer(kind(catalyst_status)) :: code
        integer :: exit_code
        character(128) :: PARAVIEW_IMPL_DIR=""
      
        params = catalyst_conduit_node_create()
        call catalyst_conduit_node_set_path_char8_str(params, "catalyst/scripts/script0", trim(adjustl(self%vtkpipeline)))
        call catalyst_conduit_node_set_path_char8_str(params, "catalyst_load/implementation", "paraview")
        call catalyst_conduit_node_set_path_char8_str(params, "catalyst_load/search_paths/paraview", PARAVIEW_IMPL_DIR)
        code = c_catalyst_initialize(params)
        if (code /= catalyst_status_ok) then
            write (error_unit, *) "failed to initialize: ", code
            exit_code = 1
        end if
        call catalyst_conduit_node_destroy(params)
    endsubroutine insitu_do_catalyst_initialization

    subroutine insitu_initialize(self)
        use tcp
        class(equation_multideal_object), intent(inout) :: self 
        integer :: counter, i, j, k
        call self%cfg%get("insitu","vtkpipeline",self%vtkpipeline)

        if(self%insitu_platform == "catalyst-v1") then
            call insitu_start(self%fcoproc,trim(adjustl(self%vtkpipeline)),self%masterproc)
        elseif(self%insitu_platform == "catalyst-v2") then
            call self%insitu_do_catalyst_initialization()
        else
            print*,'insitu_platform not available. Aborting!'
            call MPI_ABORT(mpi_comm_world,-40,self%mpi_err)
        endif

        call self%cfg%get("insitu","dt_insitu",self%dt_insitu)
        call self%cfg%get("insitu","perc_ny_cut",self%perc_ny_cut)
        call self%cfg%get("insitu","freeze_intervals",self%freeze_intervals)
        if (allocated(self%freeze_intervals)) then
         if (size(self%freeze_intervals) > 0) then
          self%enable_freeze_intervals = .true.
         endif
        endif
        call self%insitu_define_limits()
        call self%insitu_allocate()
          associate( x => self%field%x, y => self%field%y, z => self%field%z, &
            nrank => self%myrank, nproc => self%nprocs, &
            nxsl_ins => self%nxsl_ins, nxel_ins => self%nxel_ins, &
            nysl_ins => self%nysl_ins, nyel_ins => self%nyel_ins, &
            nzsl_ins => self%nzsl_ins, nzel_ins => self%nzel_ins, &
            nxs_ins => self%nxs_ins, nxe_ins => self%nxe_ins, &
            nys_ins => self%nys_ins, nye_ins => self%nye_ins, &
            nzs_ins => self%nzs_ins, nze_ins => self%nze_ins, &
            nxstartg => self%nxstartg, nxendg => self%nxendg, &
            nystartg => self%nystartg, nyendg => self%nyendg, &
            nzstartg => self%nzstartg, nzendg => self%nzendg, &
            xyzc => self%xyzc, &
            points_x   => self%points_x,   points_y   => self%points_y,   points_z   => self%points_z, &
            n_points_x => self%n_points_x, n_points_y => self%n_points_y, n_points_z => self%n_points_z, &
            n_points => self%n_points)
        if(self%insitu_platform == "catalyst-v1") then
          call createcpstructureddata(nxel_ins-nxsl_ins+1&
                                     ,nyel_ins-nysl_ins+1&
                                     ,nzel_ins-nzsl_ins+1&
                                     ,nxs_ins,nys_ins,nzs_ins&
                                     ,nxe_ins,nye_ins,nze_ins&
                                     ,nxendg-nxstartg+1&
                                     ,nyendg-nystartg+1&
                                     ,nzendg-nzstartg+1&
                                     ,x(nxsl_ins:nxel_ins)&
                                     ,y(nysl_ins:nyel_ins)&
                                     ,z(nzsl_ins:nzel_ins) &
                                     ,nrank,nproc &
                                     ,"3d_struct"//c_null_char,xyzc)
        elseif(self%insitu_platform == "catalyst-v2") then
            counter = 1
            n_points_x = nxel_ins-nxsl_ins+1
            n_points_y = nyel_ins-nysl_ins+1
            n_points_z = nzel_ins-nzsl_ins+1
            n_points = n_points_x * n_points_y * n_points_z
            do k=nzsl_ins,nzel_ins
            do j=nysl_ins,nyel_ins
            do i=nxsl_ins,nxel_ins
                points_x(counter) = x(i)
                points_y(counter) = y(j)
                points_z(counter) = z(k)
                counter = counter + 1
            enddo
            enddo
            enddo
        endif
        endassociate
    endsubroutine insitu_initialize

    subroutine insitu_define_limits(self)
        class(equation_multideal_object), intent(inout) :: self 
        integer :: ngg, j
        integer :: i, ii
        real(rkind) :: ycut
        call self%cfg%get("insitu","aux_list",self%aux_list)
        self%n_aux_list = size(self%aux_list)
        allocate(self%aux_list_name(self%n_aux_list))
        call self%cfg%get("insitu","add_list",self%add_list)
        self%n_add_list = size(self%add_list)
        allocate(self%add_list_name(self%n_add_list))
        associate( aux_list => self%aux_list, aux_list_name => self%aux_list_name, &
          mpi_err => self%mpi_err, n_aux_list => self%n_aux_list, &
          add_list => self%add_list, add_list_name => self%add_list_name, &
          n_add_list => self%n_add_list, &
          ngm => self%ngm, ngp => self%ngp, &
          yg => self%grid%yg, nxmax => self%grid%nxmax, nymax => self%grid%nymax, nzmax => self%grid%nzmax, &
          perc_ny_cut => self%perc_ny_cut, ny_cut => self%ny_cut, &
          nxsl_ins => self%nxsl_ins, nxel_ins => self%nxel_ins, &
          nysl_ins => self%nysl_ins, nyel_ins => self%nyel_ins, &
          nzsl_ins => self%nzsl_ins, nzel_ins => self%nzel_ins, &
          nxs_ins => self%nxs_ins, nxe_ins => self%nxe_ins, &
          nys_ins => self%nys_ins, nye_ins => self%nye_ins, &
          nzs_ins => self%nzs_ins, nze_ins => self%nze_ins, &
          nxstartg => self%nxstartg, nxendg => self%nxendg, &
          nystartg => self%nystartg, nyendg => self%nyendg, &
          nzstartg => self%nzstartg, nzendg => self%nzendg, &
          npsi => self%npsi, npsi_pv => self%npsi_pv, &
          nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, &
          ncoords => self%field%ncoords )
        do i=1,n_aux_list
         ii = aux_list(i)
         aux_list_name(i) = self%aux_names(ii)
         !if (ii == 1) aux_list_name(i) = "density"
         !if (ii == 2) aux_list_name(i) = "u"
         !if (ii == 3) aux_list_name(i) = "v"
         !if (ii == 4) aux_list_name(i) = "w"
         !if (ii == 5) aux_list_name(i) = "h"
         !if (ii == 6) aux_list_name(i) = "T"
         !if (ii == 7) aux_list_name(i) = "viscosity"
         !if (ii >  7) call MPI_ABORT(mpi_comm_world,-3,mpi_err)
        enddo
        do i=1,n_add_list
         ii = add_list(i)
         if (ii == 1) add_list_name(i) = "div"
         if (ii == 2) add_list_name(i) = "abs_omega"
         if (ii == 3) add_list_name(i) = "ducros"
         if (ii == 4) add_list_name(i) = "swirling_strength"
         if (ii == 5) add_list_name(i) = "schlieren"
         if (ii >  5) call MPI_ABORT(mpi_comm_world,-3,mpi_err)
        enddo
!       
!        only ngg=1 fully supported (from catalyst examples it appears that conversion to structured grids is required)
        !ngg = 0 ! rectilinear
        ngg = 1 ! structured
        if (ngg == 0) then ! i want 1 overlapping node
         ngm = 0
         ngp = 1
        else
         ngm = ngg
         ngp = ngg
        endif
!       
        ny_cut=nymax
        if(perc_ny_cut > 0._rkind) then
          ycut = yg(nymax)*(1.-perc_ny_cut/100.)
          do j=1,nymax
           if (yg(j)>ycut) then
            ny_cut = j
            exit
           endif
          enddo
        endif
!       
!        Extremal y excluded, since no partition in y allowed
        nxsl_ins = 1-ngm
        nxel_ins = nx+ngp
        nysl_ins = 1
        nyel_ins = ny_cut
        nzsl_ins = 1-ngm
        nzel_ins = nz+ngp
!       
        nxs_ins = nxsl_ins + ncoords(1)*nx
        nxe_ins = nxel_ins + ncoords(1)*nx
        nys_ins = nysl_ins + ncoords(2)*ny
        nye_ins = nyel_ins + ncoords(2)*ny
        nzs_ins = nzsl_ins + ncoords(3)*nz
        nze_ins = nzel_ins + ncoords(3)*nz
!       
        !nxsize  = nxmax+(ngp+ngm)
        !nysize  = nymax
        !nzsize  = nzmax+(ngp+ngm)
        nxstartg = 1-ngm
        nxendg   = nxmax+ngp
        nystartg = 1
        nyendg   = ny_cut
        nzstartg = 1-ngm
        nzendg   = nzmax+ngp
!
!       Variables to be stored
        npsi    = size(self%add_list) ! additional variables (with respect to w) to export
        npsi_pv = npsi+size(self%aux_list) ! total variables to export
!       
        if (self%masterproc) print*, "Insitu variable: ", aux_list_name, add_list_name
!       
        endassociate
    endsubroutine insitu_define_limits

    subroutine insitu_allocate(self)
        class(equation_multideal_object), intent(inout) :: self 
        integer :: n_points_insitu
        associate( nxsl_ins => self%nxsl_ins, nxel_ins => self%nxel_ins, &
          nysl_ins => self%nysl_ins, nyel_ins => self%nyel_ins, &
          nzsl_ins => self%nzsl_ins, nzel_ins => self%nzel_ins, &
          npsi => self%npsi, npsi_pv => self%npsi_pv, &
          nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, &
          ng => self%grid%ng, insitu_platform => self%insitu_platform )
        if (npsi_pv > 0) allocate(self%psi_pv(nxsl_ins:nxel_ins,nysl_ins:nyel_ins,nzsl_ins:nzel_ins,npsi_pv))
        if (npsi > 0) allocate(self%psi(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,npsi))
        if(insitu_platform == "catalyst-v1") then
            allocate(self%xyzc(3*(nx+2)*(ny)*(nz+2))) ! josh style
            !allocate(self%xyzc(3*(nx+1)*(ny)*(nz+1))) ! old rectilinear style
        elseif(insitu_platform == "catalyst-v2") then
            n_points_insitu = (nxel_ins-nxsl_ins+1)*(nyel_ins-nysl_ins+1)*(nzel_ins-nzsl_ins+1)
            allocate(self%points_x(n_points_insitu))
            allocate(self%points_y(n_points_insitu))
            allocate(self%points_z(n_points_insitu))
        endif
        endassociate
    endsubroutine insitu_allocate

    subroutine insitu_finalize(self)
        class(equation_multideal_object), intent(inout) :: self 
        if(self%insitu_platform == "catalyst-v1") then
            call insitu_end(self%fcoproc)
        elseif(self%insitu_platform == "catalyst-v2") then
            call self%insitu_do_catalyst_finalization()
        endif
    endsubroutine insitu_finalize

    subroutine insitu_do_catalyst_finalization(self)
        class(equation_multideal_object), intent(inout) :: self 
        type(C_PTR) :: params
        integer(kind(catalyst_status)) :: code
        integer :: exit_code
        print*,'start do_catalyst_finalization'
        params = catalyst_conduit_node_create()
        code = c_catalyst_finalize(params)
        if (code /= catalyst_status_ok) then
          write (error_unit, *) "failed to call finalize:", code
          exit_code = 1
        end if
        call catalyst_conduit_node_destroy(params)
        print*,'end do_catalyst_finalization'
    endsubroutine insitu_do_catalyst_finalization

    function time_is_freezed_fun(self)
        class(equation_multideal_object), intent(inout) :: self 
        integer :: i
        logical :: time_is_freezed_fun

        time_is_freezed_fun = .false.
        if (self%enable_insitu > 0) then
         do i=1,size(self%freeze_intervals),2
          if (self%icyc == self%freeze_intervals(i)) then
           self%i_freeze = self%i_freeze + 1
           time_is_freezed_fun = .true.
           if (self%i_freeze == self%freeze_intervals(i+1) + 1) then
            self%i_freeze = 0
            time_is_freezed_fun = .false.
            return
           endif
           return  ! already found the freezed icyc, no need to loop
          endif
         enddo
        endif
    endfunction time_is_freezed_fun

    subroutine slice_prepare(self, igslice, jgslice, kgslice, islice, jslice, kslice, alloc_aux)
        class(equation_multideal_object), intent(inout) :: self

        integer, dimension(:), intent(in) :: igslice, jgslice, kgslice
        integer, allocatable, dimension(:), intent(out):: islice, jslice, kslice
        logical, intent(in) :: alloc_aux
        integer :: i, j, k, l, ip, jp, kp, n
        integer :: ii, jj, kk, ll
        integer :: icord, jcord, kcord
        integer :: inum, jnum, knum

        associate(ng => self%grid%ng, nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, &
                  x => self%field%x, y => self%field%y, z => self%field%z, num_aux_slice => self%num_aux_slice)

        inum = 0
        do i = 1,size(igslice)
         if (igslice(i)>0) then
          icord = (igslice(i)-1)/nx
          if (self%field%ncoords(1)==icord) inum = inum + 1
         endif
        enddo
        jnum = 0
        do j = 1,size(jgslice)
         if (jgslice(j)>0) then
          jcord = (jgslice(j)-1)/ny
          if (self%field%ncoords(2)==jcord) jnum = jnum + 1
         endif
        enddo
        knum = 0
        do k = 1,size(kgslice)
         if (kgslice(k)>0) then
          kcord = (kgslice(k)-1)/nz
          if (self%field%ncoords(3)==kcord) knum = knum + 1
         endif
        enddo
!
        if (inum>0) then
         allocate(islice(inum))
         if (alloc_aux) allocate(sliceyz_aux(inum,1-ng:ny+ng,1-ng:nz+ng,num_aux_slice))
        endif
        if (jnum>0) then
         allocate(jslice(jnum))
         if (alloc_aux) allocate(slicexz_aux(1-ng:nx+ng,jnum,1-ng:nz+ng,num_aux_slice))
        endif
        if (knum>0) then
         allocate(kslice(knum))
         if (alloc_aux) allocate(slicexy_aux(1-ng:nx+ng,1-ng:ny+ng,knum,num_aux_slice))
        endif
!
        inum = 0
        do i = 1,size(igslice)
         if (igslice(i)>0) then
          icord = (igslice(i)-1)/nx
          if (self%field%ncoords(1)==icord) then
           inum = inum+1
           islice(inum) = igslice(i)-self%field%ncoords(1)*nx
          endif
         endif
        enddo
        jnum = 0
        do j = 1,size(jgslice)
         if (jgslice(j)>0) then
          jcord = (jgslice(j)-1)/ny
          if (self%field%ncoords(2)==jcord) then
           jnum = jnum + 1
           jslice(jnum) = jgslice(j)-self%field%ncoords(2)*ny
          endif
         endif
        enddo
        knum = 0
        do k = 1,size(kgslice)
         if (kgslice(k)>0) then
          kcord = (kgslice(k)-1)/nz
          if (self%field%ncoords(3)==kcord) then
           knum = knum + 1
           kslice(knum) = kgslice(k)-self%field%ncoords(3)*nz
          endif
         endif
        enddo

        endassociate

    endsubroutine slice_prepare

    subroutine probe_prepare(self)
        class(equation_multideal_object), intent(inout) :: self 

        integer :: i, j, k, l, ip, jp, kp, n
        integer :: ii, jj, kk, ll
        integer :: inum, jnum, knum
        real(rkind) :: xp, yp, zp
        logical :: probe_exists
        logical :: in_i, in_j, in_k
        logical, dimension(:), allocatable :: in_ijk
        integer :: mp

        associate(ng => self%grid%ng, nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, &
                  x => self%field%x, y => self%field%y, z => self%field%z)

        inquire(file='probe_list.dat',exist=probe_exists)
        self%num_probe = 0
        if (probe_exists) then
         open(18,file='probe_list.dat')
         read(18,*) self%num_probe_tot
         allocate(in_ijk(self%num_probe_tot))
         in_ijk = .false.
         do l=1,self%num_probe_tot
          read(18,*) xp, yp, zp
          in_i   = .false.
          in_j   = .false.
          in_k   = .false.
          if (xp>=self%field%x(1).and.xp<self%field%x(nx+1)) in_i = .true.
          if (yp>=self%field%y(1).and.yp<self%field%y(ny+1)) in_j = .true.
          if (zp>=self%field%z(1).and.zp<self%field%z(nz+1)) in_k = .true.
          if (in_i.and.in_j.and.in_k) then
           in_ijk(l) = .true.
           self%num_probe = self%num_probe+1
          endif
         enddo
         close(18)

         if (self%num_probe>0) then
          allocate(self%moving_probe(self%num_probe))
          allocate(self%probe_coord(3,self%num_probe))
          allocate(self%w_aux_probe(6,self%num_probe))
          allocate(self%ijk_probe(3,self%num_probe))
          allocate(self%probe_coeff(2,2,2,self%num_probe))
         endif

         open(18,file='probe_list.dat')
         read(18,*) self%num_probe_tot
         n = 0
         do l=1,self%num_probe_tot
          read(18,*) xp, yp, zp, mp
          if (in_ijk(l)) then
           n = n+1
           self%moving_probe(n)  = mp 
           self%probe_coord(1,n) = xp
           self%probe_coord(2,n) = yp
           self%probe_coord(3,n) = zp
          endif
         enddo
         close(18)
        endif
        endassociate

    endsubroutine probe_prepare

    subroutine probe_compute_coeff(self)
        class(equation_multideal_object), intent(inout) :: self 

        integer :: i, j, k, l, ip, jp, kp, n
        integer :: ii, jj, kk, ll
        integer :: inum, jnum, knum
        real(rkind) :: xp, yp, zp
        real(rkind) :: x0, y0, z0, dxloc, dyloc, dzloc
        real(rkind) :: xyz1, xyz2, xyz3
        logical :: probe_exists
        real(rkind), dimension(8,8) :: amat3d
        real(rkind), dimension(1,8) :: xtrasp3d,alftrasp3d
        integer :: moving_probe

        associate(ng => self%grid%ng, nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, &
                  x => self%field%x, y => self%field%y, z => self%field%z)

         do n=1,self%num_probe
          xp = self%probe_coord(1,n)
          yp = self%probe_coord(2,n)
          zp = self%probe_coord(3,n)
          call locateval(x(1:nx),nx,xp,ip)
          call locateval(y(1:ny),ny,yp,jp)
          call locateval(z(1:nz),nz,zp,kp)
          self%ijk_probe(1,n) = ip
          self%ijk_probe(2,n) = jp
          self%ijk_probe(3,n) = kp
!
!         Find coefficients for trilinear interpolation
!
          x0 = x(ip)
          y0 = y(jp)
          z0 = z(kp)
          dxloc = x(ip+1)-x(ip)
          dyloc = y(jp+1)-y(jp)
          dzloc = z(kp+1)-z(kp)
          xp = (xp-x0)/dxloc
          yp = (yp-y0)/dyloc
          zp = (zp-z0)/dzloc
!
          xtrasp3d(1,1) = xp*yp*zp
          xtrasp3d(1,2) = xp*yp
          xtrasp3d(1,3) = xp*zp
          xtrasp3d(1,4) = yp*zp
          xtrasp3d(1,5) = xp
          xtrasp3d(1,6) = yp
          xtrasp3d(1,7) = zp
          xtrasp3d(1,8) = 1._rkind
!
!         Dirichlet
          ll = 0
          do kk=0,1
           do jj=0,1
            do ii=0,1
             ll = ll+1
             xyz1 = x(ip+ii)
             xyz2 = y(jp+jj)
             xyz3 = z(kp+kk)
             xyz1 = (xyz1-x0)/dxloc
             xyz2 = (xyz2-y0)/dyloc
             xyz3 = (xyz3-z0)/dzloc
             amat3d(ll,:) = [xyz1*xyz2*xyz3,xyz1*xyz2,xyz1*xyz3,xyz2*xyz3,xyz1,xyz2,xyz3,1._rkind]
            enddo
           enddo
          enddo
!
          call invmat(amat3d,8)
          alftrasp3d = matmul(xtrasp3d,amat3d)
          self%probe_coeff(1,1,1,n) = alftrasp3d(1,1)
          self%probe_coeff(2,1,1,n) = alftrasp3d(1,2)
          self%probe_coeff(1,2,1,n) = alftrasp3d(1,3)
          self%probe_coeff(2,2,1,n) = alftrasp3d(1,4)
          self%probe_coeff(1,1,2,n) = alftrasp3d(1,5)
          self%probe_coeff(2,1,2,n) = alftrasp3d(1,6)
          self%probe_coeff(1,2,2,n) = alftrasp3d(1,7)
          self%probe_coeff(2,2,2,n) = alftrasp3d(1,8)
         enddo
        
        endassociate

    endsubroutine probe_compute_coeff

    subroutine alloc(self)
        class(equation_multideal_object), intent(inout) :: self 
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz,  &
                  ng => self%grid%ng, nv => self%nv, nv_aux => self%nv_aux, nv_stat => self%nv_stat, &
                  nv_stat_3d => self%nv_stat_3d, enable_stat_3d => self%enable_stat_3d)
        
        allocate(self%w_aux(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng, nv_aux))
        allocate(self%w_stat(nv_stat, 1:nx, 1:ny))
        allocate(self%fluid_mask(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng))
        allocate(self%ep_ord_change(0:nx, 0:ny, 0:nz, 1:3))
        if (enable_stat_3d>0) allocate(self%w_stat_3d(nv_stat_3d, 1:nx, 1:ny, 1:nz))
        self%fluid_mask    = 0
        self%ep_ord_change = 0
        endassociate
    endsubroutine alloc

    subroutine read_stats(self)
        class(equation_multideal_object), intent(inout) :: self 
        integer, dimension(3) :: sizes    ! Dimensions of the total grid
        integer, dimension(3) :: subsizes ! Dimensions of grid local to a procs
        integer, dimension(3) :: starts   ! Starting coordinates
        integer :: ntotxy
        integer, dimension(mpi_status_size) :: istatus
        integer :: mpi_io_file
        integer :: filetype, l, m
        integer :: size_real
        integer (kind=mpi_offset_kind) :: offset
        character(len=256) :: oldname, newname
        associate(nx => self%field%nx, ny => self%field%ny, nv_stat => self%nv_stat, &
                  w_stat => self%w_stat, mp_cartx => self%field%mp_cartx, mp_cartz => self%field%mp_cartz, &
                  ncoords => self%field%ncoords, nblocks => self%field%nblocks, iermpi => self%mpi_err)
        if (ncoords(3)==0) then
            sizes(1) = nblocks(1)*nx
            sizes(2) = nblocks(2)*ny
            sizes(3) = 1
            subsizes(1) = nx
            subsizes(2) = ny
            subsizes(3) = 1 
            starts(1) = 0 + ncoords(1)*subsizes(1)
            starts(2) = 0 + ncoords(2)*subsizes(2)
            starts(3) = 0
            ntotxy = nx*ny

            call mpi_type_create_subarray(3,sizes,subsizes,starts,mpi_order_fortran,mpi_prec,filetype,iermpi)
            call mpi_type_commit(filetype,iermpi)
            call mpi_file_open(mp_cartx,'stat.bin',mpi_mode_rdonly,mpi_info_null,mpi_io_file,iermpi)
            offset = 0
            do l=1,nv_stat
             call mpi_file_set_view(mpi_io_file,offset,mpi_prec,filetype,"native",mpi_info_null,iermpi)
             call mpi_file_read_all(mpi_io_file,w_stat(l,1:nx,1:ny),ntotxy,mpi_prec,istatus,iermpi)
             call mpi_type_size(mpi_prec,size_real,iermpi)
             do m=1,nblocks(1)*nblocks(2)
              offset = offset+size_real*ntotxy
             enddo
            enddo

            call mpi_file_close(mpi_io_file,iermpi)
            call mpi_type_free(filetype,iermpi)
        endif
!
        call mpi_bcast(w_stat,nv_stat*nx*ny,mpi_prec,0,mp_cartz,iermpi)
!       
        call mpi_barrier(mpi_comm_world, iermpi)
        if (self%masterproc) then
         oldname = c_char_"stat.bin"//c_null_char
         newname = c_char_"stat.bak"//c_null_char
         iermpi = rename_wrapper(oldname, newname)
         if (iermpi /= 0) write(error_unit,*) "Warning! Cannot rename file stat.bin to stat.bak"
        endif
        call mpi_barrier(mpi_comm_world, iermpi)
        endassociate

    endsubroutine read_stats

    subroutine read_stats_serial(self)
        class(equation_multideal_object), intent(inout) :: self 
        character(4) :: chx,chy
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat => self%nv_stat, &
                  w_stat => self%w_stat, ncoords => self%field%ncoords, mp_cartz => self%field%mp_cartz, &
                  iermpi => self%mpi_err)

        if (ncoords(3)==0) then

         if (self%masterproc) write(*,*) 'Reading stat0_XXX_XXX.bin'
 1004 format(I4.4)
         write(chx,1004) ncoords(1)
         write(chy,1004) ncoords(2)

         open (11,file='stat0_'//chx//'_'//chy//'.bin',form='unformatted')
         read(11) w_stat(1:nv_stat,1:nx,1:ny)
         close(11)

        endif
!
        call mpi_bcast(w_stat,nv_stat*nx*ny,mpi_prec,0,mp_cartz,iermpi)
!       
        endassociate

    endsubroutine read_stats_serial

    subroutine read_stats_3d(self)
        class(equation_multideal_object), intent(inout) :: self 
        integer, dimension(3) :: sizes    ! Dimensions of the total grid
        integer, dimension(3) :: subsizes ! Dimensions of grid local to a procs
        integer, dimension(3) :: starts   ! Starting coordinates
        integer :: ntot3d
        integer, dimension(mpi_status_size) :: istatus
        integer :: mpi_io_file
        integer :: filetype, l, m
        integer :: size_real
        integer (kind=mpi_offset_kind) :: offset
        character(len=256) :: oldname, newname

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat_3d => self%nv_stat_3d, &
                  w_stat_3d => self%w_stat_3d, mp_cart => self%field%mp_cart, &
                  ncoords => self%field%ncoords, nblocks => self%field%nblocks, iermpi => self%mpi_err)
        
        sizes(1) = nblocks(1)*nx
        sizes(2) = nblocks(2)*ny
        sizes(3) = nblocks(3)*nz
        subsizes(1) = nx
        subsizes(2) = ny
        subsizes(3) = nz
        starts(1) = 0 + ncoords(1)*subsizes(1)
        starts(2) = 0 + ncoords(2)*subsizes(2)
        starts(3) = 0 + ncoords(3)*subsizes(3)
        ntot3d = nx*ny*nz

        call mpi_type_create_subarray(3,sizes,subsizes,starts,mpi_order_fortran,mpi_prec,filetype,iermpi)
        call mpi_type_commit(filetype,iermpi)
        call mpi_file_open(mp_cart,'stat3d.bin',mpi_mode_rdonly,mpi_info_null,mpi_io_file,iermpi)
        offset = 0
        do l=1,nv_stat_3d
         call mpi_file_set_view(mpi_io_file,offset,mpi_prec,filetype,"native",mpi_info_null,iermpi)
         call mpi_file_read_all(mpi_io_file,w_stat_3d(l,1:nx,1:ny,1:nz),ntot3d,mpi_prec,istatus,iermpi)
         call mpi_type_size(mpi_prec,size_real,iermpi)
         do m=1,nblocks(1)*nblocks(2)*nblocks(3)
          offset = offset+size_real*ntot3d
         enddo
        enddo

        call mpi_file_close(mpi_io_file,iermpi)
        call mpi_type_free(filetype,iermpi)
!        
        if (self%masterproc) then
         oldname = c_char_"stat3d.bin"//c_null_char
         newname = c_char_"stat3d.bak"//c_null_char
         iermpi = rename_wrapper(oldname, newname)
         if (iermpi /= 0) write(error_unit,*) "Warning! Cannot rename file stat3d.bin to stat3d.bak"
        endif
        call mpi_barrier(mpi_comm_world, iermpi)
!
        endassociate

    endsubroutine read_stats_3d

    subroutine write_stats(self)
        class(equation_multideal_object), intent(inout) :: self 
        integer, dimension(3) :: sizes    ! Dimensions of the total grid
        integer, dimension(3) :: subsizes ! Dimensions of grid local to a procs
        integer, dimension(3) :: starts   ! Starting coordinates
        integer :: ntotxy
        integer, dimension(mpi_status_size) :: istatus
        integer :: mpi_io_file
        integer :: filetype, l, m
        integer :: size_real
        integer (kind=mpi_offset_kind) :: offset
        associate(nx => self%field%nx, ny => self%field%ny, nv_stat => self%nv_stat, &
                  w_stat => self%w_stat, mp_cartx => self%field%mp_cartx, &
                  ncoords => self%field%ncoords, nblocks => self%field%nblocks, iermpi => self%mpi_err)
        if (ncoords(3)==0) then
            sizes(1) = nblocks(1)*nx
            sizes(2) = nblocks(2)*ny
            sizes(3) = 1
            subsizes(1) = nx
            subsizes(2) = ny
            subsizes(3) = 1 
            starts(1) = 0 + ncoords(1)*subsizes(1)
            starts(2) = 0 + ncoords(2)*subsizes(2)
            starts(3) = 0
            ntotxy = nx*ny

            call mpi_type_create_subarray(3,sizes,subsizes,starts,mpi_order_fortran,mpi_prec,filetype,iermpi)
            call mpi_type_commit(filetype,iermpi)
            call mpi_file_open(mp_cartx,'stat.bin',mpi_mode_create+mpi_mode_wronly,mpi_info_null,mpi_io_file,iermpi)
            offset = 0
            do l=1,nv_stat
             call mpi_file_set_view(mpi_io_file,offset,mpi_prec,filetype,"native",mpi_info_null,iermpi)
             call mpi_file_write_all(mpi_io_file,w_stat(l,1:nx,1:ny),ntotxy,mpi_prec,istatus,iermpi)
             call mpi_type_size(mpi_prec,size_real,iermpi)
             do m=1,nblocks(1)*nblocks(2)
              offset = offset+size_real*ntotxy
             enddo
            enddo

            call mpi_file_close(mpi_io_file,iermpi)
            call mpi_type_free(filetype,iermpi)
        endif
        endassociate
    endsubroutine write_stats

    subroutine write_stats_serial(self)
        class(equation_multideal_object), intent(inout) :: self 
        character(4) :: chx,chy
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat => self%nv_stat, &
                  w_stat => self%w_stat, ncoords => self%field%ncoords, mp_cartz => self%field%mp_cartz)

        if (ncoords(3)==0) then

         if (self%masterproc) write(*,*) 'Writing stat1_XXX_XXX.bin'
 1004 format(I4.4)
         write(chx,1004) ncoords(1)
         write(chy,1004) ncoords(2)

         open (11,file='stat1_'//chx//'_'//chy//'.bin',form='unformatted')
         write(11) w_stat(1:nv_stat,1:nx,1:ny)
         close(11)

        endif
!
        endassociate

    endsubroutine write_stats_serial

    subroutine write_stats_3d(self)
        class(equation_multideal_object), intent(inout) :: self 
        integer, dimension(3) :: sizes    ! Dimensions of the total grid
        integer, dimension(3) :: subsizes ! Dimensions of grid local to a procs
        integer, dimension(3) :: starts   ! Starting coordinates
        integer :: ntot3d
        integer, dimension(mpi_status_size) :: istatus
        integer :: mpi_io_file
        integer :: filetype, l, m
        integer :: size_real
        integer (kind=mpi_offset_kind) :: offset
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat_3d => self%nv_stat_3d, &
                  w_stat_3d => self%w_stat_3d, mp_cart => self%field%mp_cart, &
                  ncoords => self%field%ncoords, nblocks => self%field%nblocks, iermpi => self%mpi_err)
        
        sizes(1) = nblocks(1)*nx
        sizes(2) = nblocks(2)*ny
        sizes(3) = nblocks(3)*nz
        subsizes(1) = nx
        subsizes(2) = ny
        subsizes(3) = nz
        starts(1) = 0 + ncoords(1)*subsizes(1)
        starts(2) = 0 + ncoords(2)*subsizes(2)
        starts(3) = 0 + ncoords(3)*subsizes(3)
        ntot3d = nx*ny*nz

        call mpi_type_create_subarray(3,sizes,subsizes,starts,mpi_order_fortran,mpi_prec,filetype,iermpi)
        call mpi_type_commit(filetype,iermpi)
        call mpi_file_open(mp_cart,'stat3d.bin',mpi_mode_create+mpi_mode_wronly,mpi_info_null,mpi_io_file,iermpi)
        offset = 0
        do l=1,nv_stat_3d
         call mpi_file_set_view(mpi_io_file,offset,mpi_prec,filetype,"native",mpi_info_null,iermpi)
         call mpi_file_write_all(mpi_io_file,w_stat_3d(l,1:nx,1:ny,1:nz),ntot3d,mpi_prec,istatus,iermpi)
         call mpi_type_size(mpi_prec,size_real,iermpi)
         do m=1,nblocks(1)*nblocks(2)*nblocks(3)
          offset = offset+size_real*ntot3d
         enddo
        enddo

        call mpi_file_close(mpi_io_file,iermpi)
        call mpi_type_free(filetype,iermpi)

        endassociate
    endsubroutine write_stats_3d
!
    subroutine write_stats_3d_serial(self)
     class(equation_multideal_object), intent(inout) :: self 
     character(4) :: chx,chy,chz
     associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat_3d => self%nv_stat_3d, &
               w_stat_3d => self%w_stat_3d, ncoords => self%field%ncoords)

     if (self%masterproc) write(*,*) 'Writing stat3d1_XXX_XXX_XXX.bin'
 1004 format(I4.4)
     write(chx,1004) ncoords(1)
     write(chy,1004) ncoords(2)
     write(chz,1004) ncoords(3)

     open (11,file='stat3d1_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted')
     write(11) w_stat_3d(1:nv_stat_3d,1:nx,1:ny,1:nz)
     close(11)

     endassociate
    endsubroutine write_stats_3d_serial
!
    subroutine read_stats_3d_serial(self)
     class(equation_multideal_object), intent(inout) :: self 
     character(4) :: chx,chy,chz
     associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat_3d => self%nv_stat_3d, &
                  w_stat_3d => self%w_stat_3d, ncoords => self%field%ncoords)

     if (self%masterproc) write(*,*) 'Reading stat3d0_XXX_XXX_XXX.bin'
 1004 format(I4.4)
     write(chx,1004) ncoords(1)
     write(chy,1004) ncoords(2)
     write(chz,1004) ncoords(3)

     open (11,file='stat3d0_'//chx//'_'//chy//'_'//chz//'.bin',form='unformatted')
     read(11) w_stat_3d(1:nv_stat_3d,1:nx,1:ny,1:nz)
     close(11)

     endassociate
    endsubroutine read_stats_3d_serial
!
    subroutine compute_stats(self)
        class(equation_multideal_object), intent(inout) :: self 
        real(rkind), dimension(self%nv_stat, self%field%nx, self%field%ny) :: w_stat_z
        real(rkind) :: rho, rhou, rhov, rhow, rhoe, ri, uu, vv, ww, qq, pp, tt, rho2, uu2, vv2, ww2, pp2, tt2, mu, nu, uv
        real(rkind) :: ux,uy,uz,vx,vy,vz,wx,wy,wz,omx,omy,omz,divl,div3l,machlocal,gamloc,c,t_tot,ccl,cploc,rmixt
        real(rkind) :: tx, ty, tz, kloc, qx, qy, qz, hh, mu_sgs, k_sgs, a1, a2, a3, qx_sgs, qy_sgs, qz_sgs, qx2, qy2, qz2, hh2, hrr
        real(rkind) :: d1,d2,dratio,dprodd,dsum
        real(rkind), dimension(3,3) :: sig, sig_sgs
        real(rkind), dimension(N_S) :: omdot
        integer :: i,j,l,k,npt,lmax,lm,lsp,idx_les
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat => self%nv_stat, &
                  ng => self%grid%ng, nv_aux => self%nv_aux, w_stat => self%w_stat, nzmax => self%grid%nzmax, &
                  w => self%field%w, mu0 => self%mu0, itav => self%itav, mp_cartz => self%field%mp_cartz, &
                  mpi_err => self%mpi_err, cv_coeff => self%cv_coeff, w_aux => self%w_aux, &
                  dcsidx => self%field%dcsidx, detady => self%field%detady, dzitdz => self%field%dzitdz, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cp_coeff => self%cp_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, init_mf => self%init_mf, &
                  rmixt0 => self%rmixt0, t0 => self%t0)

           lmax = self%visc_order/2
           ! Centered with variable spacing (2nd order)
           !dratio = (y(2)-y(1))/y(1)                ! dy2/dy1
           !a1  = - dratio/y(2)                      ! -dy2/dy1/(dy1+dy2)
           !a2  = (dratio - 1._rkind/dratio)/y(2)    ! (dy2/dy1 - dy1/dy2)/(dy1+dy2)
           !a3  = 1._rkind/y(2)/dratio               ! dy1/dy2/(dy1+dy2)
           ! Downwind with variable spacing (2nd order)
           d1     = self%field%y(2)-self%field%y(1)
           d2     = self%field%y(3)-self%field%y(2)
           dratio = d1/d2
           dprodd = d1*d2
           dsum   = d1+d2
           a1     = dratio/dsum - dsum/dprodd
           a2     = dsum/dprodd
           a3     = - dratio/dsum

           w_stat_z = 0._rkind
           do k=1,nz
            do j=1,ny
             do i=1,nx
              ux = 0._rkind
              vx = 0._rkind
              wx = 0._rkind
              uy = 0._rkind
              vy = 0._rkind
              wy = 0._rkind
              uz = 0._rkind
              vz = 0._rkind
              wz = 0._rkind
              do l=1,lmax
               ccl = self%coeff_deriv1(l,lmax)
               ux = ux+ccl*(w(I_U,i+l,j,k)/w_aux(i+l,j,k,J_R)-w(I_U,i-l,j,k)/w_aux(i-l,j,k,J_R))
               vx = vx+ccl*(w(I_V,i+l,j,k)/w_aux(i+l,j,k,J_R)-w(I_V,i-l,j,k)/w_aux(i-l,j,k,J_R))
               wx = wx+ccl*(w(I_W,i+l,j,k)/w_aux(i+l,j,k,J_R)-w(I_W,i-l,j,k)/w_aux(i-l,j,k,J_R))
               uy = uy+ccl*(w(I_U,i,j+l,k)/w_aux(i,j+l,k,J_R)-w(I_U,i,j-l,k)/w_aux(i,j-l,k,J_R))
               vy = vy+ccl*(w(I_V,i,j+l,k)/w_aux(i,j+l,k,J_R)-w(I_V,i,j-l,k)/w_aux(i,j-l,k,J_R))
               wy = wy+ccl*(w(I_W,i,j+l,k)/w_aux(i,j+l,k,J_R)-w(I_W,i,j-l,k)/w_aux(i,j-l,k,J_R))
               uz = uz+ccl*(w(I_U,i,j,k+l)/w_aux(i,j,k+l,J_R)-w(I_U,i,j,k-l)/w_aux(i,j,k-l,J_R))
               vz = vz+ccl*(w(I_V,i,j,k+l)/w_aux(i,j,k+l,J_R)-w(I_V,i,j,k-l)/w_aux(i,j,k-l,J_R))
               wz = wz+ccl*(w(I_W,i,j,k+l)/w_aux(i,j,k+l,J_R)-w(I_W,i,j,k-l)/w_aux(i,j,k-l,J_R))
              enddo
              ux = ux*dcsidx(i)
              vx = vx*dcsidx(i)
              wx = wx*dcsidx(i)
              uy = uy*detady(j)
              vy = vy*detady(j)
              wy = wy*detady(j)
              uz = uz*dzitdz(k)
              vz = vz*dzitdz(k)
              wz = wz*dzitdz(k)
              omx = wy-vz
              omy = uz-wx
              omz = vx-uy
              rho  = w_aux(i,j,k,J_R)
              rhou = w(I_U,i,j,k)
              rhov = w(I_V,i,j,k)
              rhow = w(I_W,i,j,k)
              rhoe = w(I_E,i,j,k)
              ri   = 1._rkind/rho
              uu   = rhou*ri
              vv   = rhov*ri
              ww   = rhow*ri
              qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)

              tt = w_aux(i,j,k,J_T)
              pp = w_aux(i,j,k,J_P)
              mu = w_aux(i,j,k,J_MU)
              nu = mu*ri
!
              cploc     = get_cp(tt,indx_cp_l,indx_cp_r,cp_coeff,nsetcv,trange,w_aux(i,j,k,1:N_S))
              kloc      = w_aux(i,j,k,J_K_COND)
              rmixt     = get_rmixture(N_S,self%rgas,w_aux(i,j,k,1:N_S))
              gamloc    = cploc/(cploc-rmixt)
              c         = w_aux(i,j,k,J_C)
              machlocal = sqrt(2._rkind*qq)/c
              t_tot     = tt+qq*(gamloc-1._rkind)/gamloc
!
              if (self%enable_les>0) then
                mu_sgs = w_aux(i,j,k,J_LES1)
                k_sgs  = mu_sgs*cploc/self%les_pr ! 
                mu     = mu - mu_sgs              ! Only molecular
                kloc   = kloc - k_sgs             ! 
              endif
              nu = mu*ri
!
              tx = 0._rkind
              ty = 0._rkind
              tz = 0._rkind
              do l=1,lmax
               ccl = self%coeff_deriv1(l,lmax)
               tx = tx+ccl*(w_aux(i+l,j,k,J_T)-w_aux(i-l,j,k,J_T))
               ty = ty+ccl*(w_aux(i,j+l,k,J_T)-w_aux(i,j-l,k,J_T))
               tz = tz+ccl*(w_aux(i,j,k+l,J_T)-w_aux(i,j,k-l,J_T))
              enddo
              tx = tx*dcsidx(i)
              ty = ty*detady(j)
              tz = tz*dzitdz(k)

              !! ty with reduced order close to boundary (no wall cross + centred diff.)
              !lm = lmax
!              if ((j > 1) .and. (j < lm+1) .and. (self%enable_cht==1)) lm = j-1
              !do l=1,lm
              ! ccl = self%coeff_deriv1(l,lm)
              ! ty = ty+ccl*(w_aux(i,j+l,k,J_T)-w_aux(i,j-l,k,J_T))
              !enddo
              !ty = ty*detady(j)

              !if (j == 1) then
               !if (self%enable_cht==1) then
               !   tw = t_fti(i,k)
               !else
               !   tw = self%T_wall
               !endif
               !ty  = a1*tw+a2*tt+a3*w_aux(i,2,k,6) ! 2nd order centered
              ! ty   = a1*tt + a2*w_aux(i,2,k,J_T) + a3*w_aux(i,3,k,J_T) ! 2nd order upwind
              !endif
!
              qx   = kloc * tx
              qy   = kloc * ty
              qz   = kloc * tz
              if (self%enable_les > 0) then
                qx_sgs = k_sgs * tx
                qy_sgs = k_sgs * ty
                qz_sgs = k_sgs * tz
              endif
              hh   = (rhoe+pp)*ri-qq
              qx2  = qx*qx
              qy2  = qy*qy
              qz2  = qz*qz
              hh2  = hh*hh
!
              rho2 = rho*rho
              uu2  = uu*uu
              vv2  = vv*vv
              ww2  = ww*ww
              uv   = uu*vv
              pp2  = pp*pp
              tt2  = tt*tt
!
              divl  = (ux+vy+wz)
              div3l = divl/3._rkind
              sig(1,1) = 2._rkind*(ux-div3l)
              sig(1,2) = uy+vx
              sig(1,3) = uz+wx
              sig(2,1) = sig(1,2)
              sig(2,2) = 2._rkind*(vy-div3l)
              sig(2,3) = vz+wy
              sig(3,1) = sig(1,3)
              sig(3,2) = sig(2,3)
              sig(3,3) = 2._rkind*(wz-div3l)
              if (self%enable_les > 0) sig_sgs = sig*mu_sgs
              sig      = sig*mu
!
              w_stat_z(1,i,j)  = w_stat_z(1,i,j)+rho
              w_stat_z(2,i,j)  = w_stat_z(2,i,j)+uu
              w_stat_z(3,i,j)  = w_stat_z(3,i,j)+vv
              w_stat_z(4,i,j)  = w_stat_z(4,i,j)+ww
              w_stat_z(5,i,j)  = w_stat_z(5,i,j)+pp
              w_stat_z(6,i,j)  = w_stat_z(6,i,j)+tt
              w_stat_z(7,i,j)  = w_stat_z(7,i,j)+rho2
              w_stat_z(8,i,j)  = w_stat_z(8,i,j)+uu2
              w_stat_z(9,i,j)  = w_stat_z(9,i,j)+vv2
              w_stat_z(10,i,j) = w_stat_z(10,i,j)+ww2
              w_stat_z(11,i,j) = w_stat_z(11,i,j)+pp2
              w_stat_z(12,i,j) = w_stat_z(12,i,j)+tt2
              w_stat_z(13,i,j) = w_stat_z(13,i,j)+rhou
              w_stat_z(14,i,j) = w_stat_z(14,i,j)+rhov
              w_stat_z(15,i,j) = w_stat_z(15,i,j)+rhow
              w_stat_z(16,i,j) = w_stat_z(16,i,j)+rhou*uu
              w_stat_z(17,i,j) = w_stat_z(17,i,j)+rhov*vv
              w_stat_z(18,i,j) = w_stat_z(18,i,j)+rhow*ww
              w_stat_z(19,i,j) = w_stat_z(19,i,j)+rhou*vv
              w_stat_z(20,i,j) = w_stat_z(20,i,j)+mu
              w_stat_z(21,i,j) = w_stat_z(21,i,j)+nu
!
              !! Vorticity fluctuations terms
              w_stat_z(22,i,j) = w_stat_z(22,i,j)+omx**2
              w_stat_z(23,i,j) = w_stat_z(23,i,j)+omy**2
              w_stat_z(24,i,j) = w_stat_z(24,i,j)+omz**2
              !! Temperature fluctuations terms
              w_stat_z(25,i,j) = w_stat_z(25,i,j)+rho*tt
              w_stat_z(26,i,j) = w_stat_z(26,i,j)+rho*tt2
              w_stat_z(27,i,j) = w_stat_z(27,i,j)+t_tot
              w_stat_z(28,i,j) = w_stat_z(28,i,j)+rho*t_tot
              w_stat_z(29,i,j) = w_stat_z(29,i,j)+t_tot**2
              w_stat_z(30,i,j) = w_stat_z(30,i,j)+rhou*tt
              w_stat_z(31,i,j) = w_stat_z(31,i,j)+rhov*tt
              w_stat_z(32,i,j) = w_stat_z(32,i,j)+rhow*tt
              ! Fluctuating Mach number
              w_stat_z(33,i,j) = w_stat_z(33,i,j)+machlocal
              w_stat_z(34,i,j) = w_stat_z(34,i,j)+machlocal**2
!             Turbulent transport 
              w_stat_z(35,i,j) = w_stat_z(35,i,j)+rhou*uu2
              w_stat_z(36,i,j) = w_stat_z(36,i,j)+rhov*uu2
              w_stat_z(37,i,j) = w_stat_z(37,i,j)+rhou*vv2
              w_stat_z(38,i,j) = w_stat_z(38,i,j)+rhov*vv2
              w_stat_z(39,i,j) = w_stat_z(39,i,j)+rhou*ww2
              w_stat_z(40,i,j) = w_stat_z(40,i,j)+rhov*ww2
!             Pressure transport 
              w_stat_z(41,i,j) = w_stat_z(41,i,j)+pp*uu
              w_stat_z(42,i,j) = w_stat_z(42,i,j)+pp*vv
!             Useful for compressibility terms
              w_stat_z(43,i,j) = w_stat_z(43,i,j)+sig(1,1)
              w_stat_z(44,i,j) = w_stat_z(44,i,j)+sig(1,2)
              w_stat_z(45,i,j) = w_stat_z(45,i,j)+sig(1,3)
              w_stat_z(46,i,j) = w_stat_z(46,i,j)+sig(2,2)
              w_stat_z(47,i,j) = w_stat_z(47,i,j)+sig(2,3)
              w_stat_z(48,i,j) = w_stat_z(48,i,j)+sig(3,3)
!             ! Viscous transport (diffusion)
              ! 11
              w_stat_z(49,i,j) = w_stat_z(49,i,j)+sig(1,1)*uu
              w_stat_z(50,i,j) = w_stat_z(50,i,j)+sig(1,2)*uu
              ! 22
              w_stat_z(51,i,j) = w_stat_z(51,i,j)+sig(2,1)*vv
              w_stat_z(52,i,j) = w_stat_z(52,i,j)+sig(2,2)*vv
              ! 33
              w_stat_z(53,i,j) = w_stat_z(53,i,j)+sig(3,1)*ww
              w_stat_z(54,i,j) = w_stat_z(54,i,j)+sig(3,2)*ww
              ! 12
              w_stat_z(55,i,j) = w_stat_z(55,i,j)+sig(1,1)*vv+sig(2,1)*uu
              w_stat_z(56,i,j) = w_stat_z(56,i,j)+sig(1,2)*vv+sig(2,2)*uu
!             Dissipation
              !! 11
              w_stat_z(57,i,j) = w_stat_z(57,i,j)+sig(1,1)*ux+sig(1,2)*uy+sig(1,3)*uz  
              !! 22
              w_stat_z(58,i,j) = w_stat_z(58,i,j)+sig(2,1)*vx+sig(2,2)*vy+sig(2,3)*vz 
              !! 33
              w_stat_z(59,i,j) = w_stat_z(59,i,j)+sig(3,1)*wx+sig(3,2)*wy+sig(3,3)*wz
              !! 12
              w_stat_z(60,i,j) = w_stat_z(60,i,j)+sig(1,1)*vx+sig(1,2)*(ux+vy)+sig(2,2)*uy+sig(1,3)*vz+sig(2,3)*uz
!             Pressure-strain redistribution
              w_stat_z(61,i,j) = w_stat_z(61,i,j)+pp*ux
              w_stat_z(62,i,j) = w_stat_z(62,i,j)+pp*vy
              w_stat_z(63,i,j) = w_stat_z(63,i,j)+pp*wz
              w_stat_z(64,i,j) = w_stat_z(64,i,j)+pp*(uy+vx)
!             Compressible dissipation
              w_stat_z(65,i,j)  = w_stat_z(65,i,j)+divl*divl
!          
              w_stat_z(66,i,j) = w_stat_z(66,i,j)+rho*tt2*tt
              w_stat_z(67,i,j) = w_stat_z(67,i,j)+rho*tt2*tt2
              w_stat_z(68,i,j) = w_stat_z(68,i,j)+rhou*uu2*uu
              w_stat_z(69,i,j) = w_stat_z(69,i,j)+cploc
              w_stat_z(70,i,j) = w_stat_z(70,i,j)+gamloc
              w_stat_z(71,i,j) = w_stat_z(71,i,j)+kloc
!            
!             Calorically imperfect gas stats
!
              w_stat_z(72,i,j) = w_stat_z(72,i,j)+gamloc*gamloc
              w_stat_z(73,i,j) = w_stat_z(73,i,j)+cploc*cploc
              w_stat_z(74,i,j) = w_stat_z(74,i,j)+kloc*kloc
              w_stat_z(75,i,j) = w_stat_z(75,i,j)+qx
              w_stat_z(76,i,j) = w_stat_z(76,i,j)+qy
              w_stat_z(77,i,j) = w_stat_z(77,i,j)+qz
              w_stat_z(78,i,j) = w_stat_z(78,i,j)+qx2
              w_stat_z(79,i,j) = w_stat_z(79,i,j)+qy2
              w_stat_z(80,i,j) = w_stat_z(80,i,j)+qz2
              w_stat_z(81,i,j) = w_stat_z(81,i,j)+rho*hh
              w_stat_z(82,i,j) = w_stat_z(82,i,j)+rho*hh2
              w_stat_z(83,i,j) = w_stat_z(83,i,j)+rho*uu*hh
              w_stat_z(84,i,j) = w_stat_z(84,i,j)+rho*vv*hh
              w_stat_z(85,i,j) = w_stat_z(85,i,j)+rho*ww*hh
              w_stat_z(86,i,j) = w_stat_z(86,i,j)+mu*mu

              do lsp=1,N_S
               w_stat_z(86+lsp,i,j) = w_stat_z(86+lsp,i,j)+w(lsp,i,j,k)
               w_stat_z(86+N_S+lsp,i,j) = w_stat_z(86+N_S+lsp,i,j)+w(lsp,i,j,k)**2/rho
               if (self%enable_chemistry > 0) then 
                 w_stat_z(86+2*N_S+lsp,i,j) = w_stat_z(86+2*N_S+lsp,i,j)+w_aux(i,j,k,J_WDOT_START+lsp)
                 w_stat_z(86+3*N_S+lsp,i,j) = w_stat_z(86+3*N_S+lsp,i,j)+w_aux(i,j,k,J_WDOT_START+lsp)**2
               endif
              enddo
              if (self%enable_chemistry > 0) then
               w_stat_z(86+4*N_S+1,i,j) = w_stat_z(86+4*N_S+1,i,j)+w_aux(i,j,k,J_HRR) 
               w_stat_z(86+4*N_S+2,i,j) = w_stat_z(86+4*N_S+2,i,j)+w_aux(i,j,k,J_HRR)**2
              endif
!
!             LES basic stats
!
              if (self%enable_les>0) then
                idx_les = 86+2*N_S
                if (self%enable_chemistry > 0) idx_les = idx_les+2*N_S+2
                w_stat_z(idx_les+1 ,i,j) = w_stat_z(idx_les+1 ,i,j) + mu_sgs
                w_stat_z(idx_les+2 ,i,j) = w_stat_z(idx_les+2 ,i,j) + k_sgs
                w_stat_z(idx_les+3 ,i,j) = w_stat_z(idx_les+3 ,i,j) + sig_sgs(1,1)
                w_stat_z(idx_les+4 ,i,j) = w_stat_z(idx_les+4 ,i,j) + sig_sgs(1,2)
                w_stat_z(idx_les+5 ,i,j) = w_stat_z(idx_les+5 ,i,j) + sig_sgs(1,3)
                w_stat_z(idx_les+6 ,i,j) = w_stat_z(idx_les+6 ,i,j) + sig_sgs(2,2)
                w_stat_z(idx_les+7 ,i,j) = w_stat_z(idx_les+7 ,i,j) + sig_sgs(2,3)
                w_stat_z(idx_les+8 ,i,j) = w_stat_z(idx_les+8 ,i,j) + sig_sgs(3,3)
                w_stat_z(idx_les+9 ,i,j) = w_stat_z(idx_les+9 ,i,j) + qx_sgs
                w_stat_z(idx_les+10,i,j) = w_stat_z(idx_les+10,i,j) + qy_sgs
                w_stat_z(idx_les+11,i,j) = w_stat_z(idx_les+11,i,j) + qz_sgs
              endif
!
             enddo
            enddo
           enddo

           npt      = nv_stat*nx*ny
           call mpi_allreduce(MPI_IN_PLACE,w_stat_z,npt,mpi_prec,mpi_sum,mp_cartz,mpi_err)
           w_stat_z = w_stat_z/nzmax

           w_stat = w_stat*itav + w_stat_z
           w_stat = w_stat/(itav+1)
        endassociate
    endsubroutine compute_stats

    subroutine compute_stats_3d(self)
        class(equation_multideal_object), intent(inout) :: self 
        integer :: i,j,k,lsp
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, itav => self%itav, &
                  w_stat_3d => self%w_stat_3d, w => self%field%w, w_aux => self%w_aux) 

           do k=1,nz
            do j=1,ny
             do i=1,nx
              w_stat_3d(1,i,j,k)  = w_stat_3d(1,i,j,k) *itav + w_aux(i,j,k,J_R) 
              w_stat_3d(2,i,j,k)  = w_stat_3d(2,i,j,k) *itav + w(I_U,i,j,k) 
              w_stat_3d(3,i,j,k)  = w_stat_3d(3,i,j,k) *itav + w(I_V,i,j,k) 
              w_stat_3d(4,i,j,k)  = w_stat_3d(4,i,j,k) *itav + w(I_W,i,j,k) 
              w_stat_3d(5,i,j,k)  = w_stat_3d(5,i,j,k) *itav + w_aux(i,j,k,J_R)*w_aux(i,j,k,J_T) 
              w_stat_3d(6,i,j,k)  = w_stat_3d(6,i,j,k) *itav + w_aux(i,j,k,J_R)**2 
              w_stat_3d(7,i,j,k)  = w_stat_3d(7,i,j,k) *itav + w(I_U,i,j,k)**2/w_aux(i,j,k,J_R) 
              w_stat_3d(8,i,j,k)  = w_stat_3d(8,i,j,k) *itav + w(I_V,i,j,k)**2/w_aux(i,j,k,J_R) 
              w_stat_3d(9,i,j,k)  = w_stat_3d(9,i,j,k) *itav + w(I_W,i,j,k)**2/w_aux(i,j,k,J_R) 
              w_stat_3d(10,i,j,k) = w_stat_3d(10,i,j,k)*itav + w(I_U,i,j,k)*w(I_V,i,j,k)/w_aux(i,j,k,J_R) 
              w_stat_3d(11,i,j,k) = w_stat_3d(11,i,j,k)*itav + w(I_U,i,j,k)*w(I_W,i,j,k)/w_aux(i,j,k,J_R) 
              w_stat_3d(12,i,j,k) = w_stat_3d(12,i,j,k)*itav + w(I_V,i,j,k)*w(I_W,i,j,k)/w_aux(i,j,k,J_R) 
              w_stat_3d(13,i,j,k) = w_stat_3d(13,i,j,k)*itav + w_aux(i,j,k,J_R)*w_aux(i,j,k,J_T)**2 
              w_stat_3d(14,i,j,k) = w_stat_3d(14,i,j,k)*itav + (w_aux(i,j,k,J_R)*w_aux(i,j,k,J_T))**2 
              w_stat_3d(15,i,j,k) = w_stat_3d(15,i,j,k)*itav + w_aux(i,j,k,J_P)
              w_stat_3d(16,i,j,k) = w_stat_3d(16,i,j,k)*itav + w_aux(i,j,k,J_P)**2
              do lsp=1,N_S
               w_stat_3d(16+lsp,i,j,k)     = w_stat_3d(16+    lsp,i,j,k)*itav + w(lsp,i,j,k)
               w_stat_3d(16+N_S+lsp,i,j,k) = w_stat_3d(16+N_S+lsp,i,j,k)*itav + w(lsp,i,j,k)**2/w_aux(i,j,k,J_R)
              enddo
             enddo
            enddo
           enddo
           w_stat_3d = w_stat_3d/(itav+1)

        end associate
    end subroutine compute_stats_3d

    subroutine ibm_compute_stat_wm(self)
        class(equation_multideal_object), intent(inout) :: self
        integer :: l
        associate(ibm_wm_wallprop => self%ibm_wm_wallprop, ibm_wm_stat => self%ibm_wm_stat, &
                  ibm_num_interface => self%ibm_num_interface, itav => self%itav)

           if (ibm_num_interface>0) then 
            do l=1,ibm_num_interface
             ibm_wm_stat(1,l) = ibm_wm_stat(1,l)*itav+ibm_wm_wallprop(1,l)     ! tauw
             ibm_wm_stat(2,l) = ibm_wm_stat(2,l)*itav+ibm_wm_wallprop(2,l)     ! qw
             ibm_wm_stat(3,l) = ibm_wm_stat(3,l)*itav+ibm_wm_wallprop(1,l)**2  ! tauw2
             ibm_wm_stat(4,l) = ibm_wm_stat(4,l)*itav+ibm_wm_wallprop(2,l)**2  ! qw2
            enddo
            ibm_wm_stat = ibm_wm_stat/(itav+1)
           endif

        end associate
    end subroutine ibm_compute_stat_wm

    subroutine compute_chemistry(self,i,j,k,omdot,hrr)
        class(equation_multideal_object), intent(inout) :: self
        integer, intent(in) :: i,j,k
        integer :: lr,lsp,itt
        real(rkind) :: tt,rho,ttleft,dtt,dkc,kc,kf,kb,conc,qlr,tb,q1,q2,wdtkj
        real(rkind), intent(out) :: hrr
        real(rkind), dimension(N_S), intent(out) :: omdot

        associate(t_min_tab => self%t_min_tab, dt_tab => self%dt_tab, &
                  num_t_tab => self%num_t_tab, kc_tab => self%kc_tab, &
                  arr_a => self%arr_a, arr_b => self%arr_b, arr_ea => self%arr_ea,&
                  p_coeffs => self%p_coeffs, r_coeffs => self%r_coeffs, &
                  tb_eff => self%tb_eff, reac_ty => self%reac_ty, mw => self%mw, &
                  nreactions => self%nreactions, w_aux => self%w_aux)

        tt  = self%t0 
        rho = self%rho0

        itt = int((tt-t_min_tab)/dt_tab)+1
        itt = max(itt,1)
        itt = min(itt,num_t_tab)
        ttleft = (itt-1)*dt_tab + t_min_tab
        dtt = (tt-ttleft)/dt_tab
        ! interpola kc

        do lr=1,nreactions
         dkc = kc_tab(itt+1,lr)-kc_tab(itt,lr)
         kc  = kc_tab(itt,lr)+dkc*dtt

         kf  = arr_a(lr,1)*(tt**arr_b(lr,1))*exp(-arr_ea(lr,1)/R_univ/tt)
         kb  = kf/kc

         !if (i == 1) print *, kc

         q1  = 1._rkind
         q2  = 1._rkind

         do lsp=1,N_S
          conc = rho*self%init_mf(lsp)/mw(lsp)
          q1 = q1*conc**r_coeffs(lr,lsp)
          q2 = q2*conc**p_coeffs(lr,lsp)
          !if (i == 1) print *,w_aux(i,j,k,lsp)
         enddo
         if (reac_ty(lr) == 1) then
          tb = 0._rkind
          do lsp=1,N_S
           tb = tb + tb_eff(lr,lsp)*rho*self%init_mf(lsp)/mw(lsp)
          enddo
         else
          tb = 1._rkind
         endif
         qlr = tb*(kf*q1-kb*q2)
         do lsp=1,N_S
          wdtkj = (p_coeffs(lr,lsp)-r_coeffs(lr,lsp))*qlr
          omdot(lsp) = omdot(lsp) - wdtkj*mw(lsp)
         enddo
        enddo

        hrr = 0._rkind
        do lsp=1,N_S
         hrr = hrr - omdot(lsp)*self%h298(lsp)
        enddo
        endassociate

    endsubroutine compute_chemistry

    subroutine read_field_info(self)
        class(equation_multideal_object), intent(inout) :: self 
        character(len=256) :: oldname, newname

        if (self%io_type_r==1) self%field_info_cfg = parse_cfg("field_info0.dat")
        if (self%io_type_r==2) self%field_info_cfg = parse_cfg("field_info0.dat")
        call self%field_info_cfg%get("field_info","icyc0",                self%icyc0)
        call self%field_info_cfg%get("field_info","time0",                self%time0)
        call self%field_info_cfg%get("field_info","itav",                 self%itav)
        call self%field_info_cfg%get("field_info","itslice_vtr",          self%itslice_vtr)
        call self%field_info_cfg%get("field_info","time_from_last_rst",   self%time_from_last_rst)
        call self%field_info_cfg%get("field_info","time_from_last_write", self%time_from_last_write)
        call self%field_info_cfg%get("field_info","time_from_last_stat",  self%time_from_last_stat)
        call self%field_info_cfg%get("field_info","time_from_last_slice", self%time_from_last_slice)
        call self%field_info_cfg%get("field_info","istore",               self%istore)
        if (self%field_info_cfg%has_key("field_info","time_from_last_slice_vtr")) then
            call self%field_info_cfg%get("field_info","time_from_last_slice_vtr", self%time_from_last_slice_vtr)
        else
            self%time_from_last_slice_vtr = self%time_from_last_slice
        endif
        if (self%enable_insitu > 0) then
            if (self%field_info_cfg%has_key("field_info","time_from_last_insitu")) then
             call self%field_info_cfg%get("field_info","time_from_last_insitu",self%time_from_last_insitu)
            else
             self%time_from_last_insitu = 0._rkind
            endif
            if (self%field_info_cfg%has_key("field_info","i_insitu")) then
             call self%field_info_cfg%get("field_info","i_insitu",self%i_insitu)
            else
             self%i_insitu = 1
            endif
            if (self%field_info_cfg%has_key("field_info","time_insitu")) then
             call self%field_info_cfg%get("field_info","time_insitu",self%time_insitu)
            else
             self%time_insitu = 0._rkind
            endif
        endif
!
        !open(unit=15, file="finaltime.dat")
        !read(15,*) self%icyc0
        !read(15,*) self%time0
        !read(15,*) self%itav
        !read(15,*) self%time_from_last_rst
        !read(15,*) self%time_from_last_write
        !read(15,*) self%time_from_last_stat
        !read(15,*) self%time_from_last_slice
        !read(15,*) self%istore
        !if (self%enable_insitu > 0) then
        ! read(15,*,iostat=ier) self%time_from_last_insitu
        ! if (ier /= 0) self%time_from_last_insitu = 0.
        ! read(15,*,iostat=ier) self%i_insitu
        ! if (ier /= 0) self%i_insitu = 1
        ! read(15,*,iostat=ier) self%time_insitu
        ! if (ier /= 0) self%time_insitu = 0. ! insitu is just starting
        !endif
        !close(15)
!
        call mpi_barrier(mpi_comm_world, self%mpi_err)
        if (self%io_type_r==2) then
         if (self%masterproc) then
             oldname = c_char_"field_info0.dat"//c_null_char
             newname = c_char_"field_info0.bak"//c_null_char
             self%mpi_err = rename_wrapper(oldname, newname)
             if (self%mpi_err /= 0) write(error_unit,*) "Warning! Cannot rename file field_info.dat to field_info.bak"
         endif
         call mpi_barrier(mpi_comm_world, self%mpi_err)
        endif  
!
    endsubroutine read_field_info

    subroutine write_field_info(self)
        class(equation_multideal_object), intent(inout) :: self 
        if(self%masterproc) then
         call self%field_info_cfg%set("field_info","icyc0",                    self%icyc)
         call self%field_info_cfg%set("field_info","time0",                    self%time)
         call self%field_info_cfg%set("field_info","itav",                     self%itav)
         call self%field_info_cfg%set("field_info","itslice_vtr",              self%itslice_vtr)
         call self%field_info_cfg%set("field_info","time_from_last_rst",       self%time_from_last_rst)
         call self%field_info_cfg%set("field_info","time_from_last_write",     self%time_from_last_write)
         call self%field_info_cfg%set("field_info","time_from_last_stat",      self%time_from_last_stat)
         call self%field_info_cfg%set("field_info","time_from_last_slice",     self%time_from_last_slice)
         call self%field_info_cfg%set("field_info","time_from_last_slice_vtr", self%time_from_last_slice_vtr)
         call self%field_info_cfg%set("field_info","istore",                   self%istore)
         if (self%enable_insitu > 0) then
          call self%field_info_cfg%set("field_info","time_from_last_insitu",self%time_from_last_insitu)
          call self%field_info_cfg%set("field_info","i_insitu",self%i_insitu)
          call self%field_info_cfg%set("field_info","time_insitu",self%time_insitu)
         endif
         if (self%io_type_w==1) call self%field_info_cfg%write("field_info1.dat")
         if (self%io_type_w==2) call self%field_info_cfg%write("field_info0.dat")
         !open(unit=15, file="finaltime.dat")
         !write(15,*) self%icyc
         !write(15,*) self%time
         !write(15,*) self%itav
         !write(15,*) self%time_from_last_rst
         !write(15,*) self%time_from_last_write
         !write(15,*) self%time_from_last_stat
         !write(15,*) self%time_from_last_slice
         !write(15,*) self%istore
         !if (self%enable_insitu > 0) then
         ! write(15,*) self%time_from_last_insitu
         ! write(15,*) self%i_insitu
         ! write(15,*) self%time_insitu
         !endif
         !close(15)
        endif
    endsubroutine write_field_info

    function get_rmixture(n, rgas, massfrac)
        ! Compute mixture gas constant
        integer, intent(in) :: n
        real(rkind), dimension(n), intent(in) :: massfrac, rgas
        integer :: l
        real(rkind) :: get_rmixture
        real(rkind) :: rmix
!
        rmix = massfrac(1)*rgas(1)
        do l=2,n
         rmix = rmix+massfrac(l)*rgas(l)
        enddo
        get_rmixture = rmix
!
    endfunction get_rmixture
        
    function get_mixture_viscosity(tt,t_min_tab,dt_tab,num_t_tab,&
                                   init_mf,mw,visc_species)
        integer :: num_t_tab
        real(rkind), intent(in) :: tt, t_min_tab,dt_tab
        real(rkind), dimension(N_S) :: mw,init_mf
        real(rkind), dimension(num_t_tab+1,N_S) :: visc_species
        integer :: itt, lsp, msp
        real(rkind) :: tloc, dtt, mu, dmu, mulsp, mumsp, phi_lm, mu_den
        real(rkind) :: get_mixture_viscosity

        itt = int((tt-t_min_tab)/dt_tab)+1
        itt = max(itt,1)
        itt = min(itt,num_t_tab)
        tloc = t_min_tab+(itt-1)*dt_tab
        dtt = (tt-tloc)/dt_tab
        ! Wilke (1950) mixture dynamic viscosity            
        mu   = 0._rkind
        do lsp=1,N_S
         dmu   = visc_species(itt+1,lsp)-visc_species(itt,lsp)
         mulsp = visc_species(itt,lsp)+dmu*dtt
         mu_den = 0._rkind
         do msp=1,N_S
          dmu    = visc_species(itt+1,msp)-visc_species(itt,msp)
          mumsp  = visc_species(itt,msp)+dmu*dtt
          phi_lm = 1._rkind/sqrt(1._rkind+mw(lsp)/mw(msp))*&
                  (1._rkind + sqrt(mumsp/mulsp)*(mw(msp)/mw(lsp))**0.25_rkind)**2
          mu_den = mu_den + init_mf(msp)*phi_lm/mw(msp)
         enddo
         mu = mu + mulsp*init_mf(lsp)/mw(lsp)/mu_den
        enddo
        mu = sqrt(8._rkind)*mu
        get_mixture_viscosity = mu

    endfunction get_mixture_viscosity 

    function get_mixture_lambda(tt,t_min_tab,dt_tab,num_t_tab,&
                                init_mf,mw,rmixtloc,lambda_species)
        integer :: num_t_tab
        real(rkind), intent(in) :: tt, t_min_tab,dt_tab,rmixtloc
        real(rkind), dimension(N_S) :: mw,init_mf
        real(rkind), dimension(num_t_tab+1,N_S) :: lambda_species
        integer :: itt, lsp, msp
        real(rkind) :: tloc, dtt, k_cond, k_cond2, dlam, klsp, mwmixt, xlsp
        real(rkind) :: get_mixture_lambda

        itt = int((tt-t_min_tab)/dt_tab)+1
        itt = max(itt,1)
        itt = min(itt,num_t_tab)
        tloc = t_min_tab+(itt-1)*dt_tab
        dtt = (tt-tloc)/dt_tab

        mwmixt   = R_univ/rmixtloc
        k_cond   = 0._rkind
        k_cond2  = 0._rkind
        do lsp = 1,N_S
         dlam    = lambda_species(itt+1,lsp)-lambda_species(itt,lsp)
         klsp    = lambda_species(itt,lsp)+dlam*dtt
         xlsp    = init_mf(lsp)*mwmixt/mw(lsp)
         k_cond  = k_cond  + xlsp*klsp
         k_cond2 = k_cond2 + xlsp/klsp
        enddo
        k_cond = 0.5_rkind*(k_cond + 1._rkind/k_cond2)
        get_mixture_lambda = k_cond 
              
    endfunction get_mixture_lambda 

    function get_species_diff(lsp,pp,p0,rho,tt,t_min_tab,dt_tab,num_t_tab,&
                              init_mf,mw,rmixtloc,diffbin_species)
        integer :: num_t_tab,lsp
        real(rkind), intent(in) :: pp,p0,rho,tt, t_min_tab,dt_tab,rmixtloc
        real(rkind), dimension(N_S) :: mw,init_mf
        real(rkind), dimension(num_t_tab+1,N_S,N_S) :: diffbin_species
        integer :: itt, msp
        real(rkind) :: tloc, dtt, diff_den, diff, diff_ij, ddiff, xmsp, mwmixt
        real(rkind) :: get_species_diff

        itt = int((tt-t_min_tab)/dt_tab)+1
        itt = max(itt,1)
        itt = min(itt,num_t_tab)
        tloc = t_min_tab+(itt-1)*dt_tab
        dtt = (tt-tloc)/dt_tab

        mwmixt   = R_univ/rmixtloc
!       Bird (1960) species' diffusion into mixture    
        diff_den = 0._rkind
        do msp = 1,N_S
         if (msp /= lsp) then
          ddiff = diffbin_species(itt+1,lsp,msp)-diffbin_species(itt,lsp,msp)
          diff_ij = diffbin_species(itt,lsp,msp)+ddiff*dtt
          xmsp = init_mf(msp)*mwmixt/mw(msp)
          diff_den = diff_den + xmsp/diff_ij
         endif
        enddo
        if (diff_den > 1.0D-015) then
         diff = (1._rkind-init_mf(lsp))/diff_den*p0/pp*rho ! rho * diffusion
        else
         diff = 0._rkind
        endif
        get_species_diff = diff
              
    endfunction get_species_diff

    function findrho(n, rhopartial)
        ! Compute mixture density from partial densities    
        integer, intent(in) :: n
        real(rkind), dimension(n), intent(in) :: rhopartial
        integer :: l
        real(rkind) :: findrho
        real(rkind) :: rho

        rho = rhopartial(1)
        do l=2,n
         rho = rho+rhopartial(l)
        enddo
        findrho = rho
    endfunction findrho

    function get_e_from_temperature(tt,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,massfrac)
        ! Compute mixture energy at temperature tt
        integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
        real(rkind), intent(in) :: tt
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in) :: cv_coeff
        real(rkind), dimension(N_S), intent(in) :: massfrac
        real(rkind), dimension(N_S,nsetcv+1), intent(in) :: trange
        real(rkind) :: get_e_from_temperature
        real(rkind) :: ee,tprod,cv_l
        integer, dimension(N_S) :: nrange
        integer :: lsp,l,jl,ju,jm
!
        nrange = 1
        if (nsetcv>1) then ! Replicate locate function of numerical recipes
         do lsp=1,N_S
          jl = 0
          ju = nsetcv+1+1
          do 
           if (ju-jl <= 1) exit
           jm = (ju+jl)/2
           if (tt>= trange(lsp,jm)) then
            jl=jm
           else
            ju=jm
           endif
          enddo
          nrange(lsp) = jl
         enddo
        endif
!
        ee = 0._rkind
        do lsp=1,N_S
         ee = ee+cv_coeff(indx_cp_r+1,lsp,nrange(lsp))*massfrac(lsp)
        enddo
        do l=indx_cp_l,indx_cp_r
         if (l==-1) then
          tprod = log(tt)
         else
          tprod = (tt**(l+1))/(l+1._rkind)
         endif
         cv_l = 0._rkind
         do lsp=1,N_S
          cv_l = cv_l+cv_coeff(l,lsp,nrange(lsp))*massfrac(lsp)
         enddo
         ee = ee+cv_l*tprod
        enddo
!        
        get_e_from_temperature = ee
!        
    endfunction get_e_from_temperature

    function h_species_dimensional_NASA(tt,indx_cp_l,indx_cp_r,cp_coeff,nsetcv,trange)
        ! Compute enthalpy at tt for a single species
        real(rkind), intent(in) :: tt
        integer, intent(in) :: indx_cp_l, indx_cp_r, nsetcv
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,nsetcv), intent(in) :: cp_coeff
        real(rkind), dimension(nsetcv+1), intent(in) :: trange
        real(rkind) :: h_species_dimensional_NASA
        real(rkind) :: hh, tprod
        integer :: nrange
        integer :: l,ju,jl,jm
!
        nrange = 1
        if (nsetcv>1) then ! Replicate locate function of numerical recipes
         jl = 0
         ju = nsetcv+1+1
         do 
          if (ju-jl <= 1) exit
          jm = (ju+jl)/2
          if (tt>= trange(jm)) then
           jl=jm
          else
           ju=jm
          endif
         enddo
         nrange = jl
        endif
!
        hh = cp_coeff(indx_cp_r+1,nrange)
        do l=indx_cp_l,indx_cp_r
         if (l==-1) then
          tprod = log(tt)
         else
          tprod = tt**(l+1)/(l+1._rkind)
         endif
         hh = hh+cp_coeff(l,nrange)*tprod
        enddo
!        
        h_species_dimensional_NASA = hh
!        
    endfunction h_species_dimensional_NASA
!
     subroutine set_fluid_prop(self)
        class(equation_multideal_object), intent(inout) :: self
!
        integer :: lsp,msp,m,l,lr
        integer :: number_species, number_cp_coeff
        real(rkind) :: cploc, cploc_spec
        real(rkind), allocatable, dimension(:) :: tem_ranges, tmp, tmp2
        real(rkind), allocatable, dimension(:,:) :: cp_coeff_matrix
        character(1) :: chnset
        character(2) :: chspec
        character(40) :: ReacType, filename
        integer, dimension(N_S) :: nrange
        real(rkind) :: tloc, p_tab
        real(rkind), dimension(N_S) :: ytmp
        real(rkind), dimension(N_S*N_S) :: diff_ij
        logical :: present_t0,present_p0,present_rho0
        real(rkind), allocatable, dimension(:)   :: q
        real(rkind), dimension(N_S) :: wdot, wdot_from_cantera, omdot
        real(rkind)                 :: q1,q2,tb,kb,kf,hrr
        integer :: nset, itt
        real(rkind) :: dtt, dkc, kc,ttleft, arr_a_mult, dlsp
!
        associate(gam => self%gam, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, nsetcv => self%nsetcv, &
                  gm => self%gm, gm1 => self%gm1, rfac => self%rfac, rmixt0 => self%rmixt0, mu0 => self%mu0, &
                  rho0 => self%rho0, p0 => self%p0, k0 => self%k0, &
                  cfg => self%cfg,t0 => self%t0, mw => self%mw, rgas => self%rgas, trange => self%trange, &
                  t_min_tab => self%t_min_tab, t_max_tab => self%t_max_tab, dt_tab => self%dt_tab, num_t_tab => self%num_t_tab, &
                  init_mf => self%init_mf, cp0 => self%cp0, cv0 => self%cv0, enable_chemistry => self%enable_chemistry)
!
        call cfg%get("fluid","enable_chemistry",enable_chemistry)

        if (enable_chemistry == 1) then
         if (self%masterproc) print *, 'Chemistry enabled: explicit integration'
        elseif (enable_chemistry == 2) then
         if (self%masterproc) print *, 'Chemistry enabled: implicit Rosenbrock integration'
        endif

        if (enable_chemistry == 2) then
         call cfg%get("numerics","rosenbrock_version",self%rosenbrock_version)
         if (self%rosenbrock_version .ne. 1 .and. self%rosenbrock_version .ne. 2) call fail_input_any("rosenbrock_version must be 1 or 2")

         call cfg%get("numerics","operator_splitting",self%operator_splitting)
         if (self%operator_splitting .ne. 1 .and. self%operator_splitting .ne. 2) call fail_input_any("operator_splitting must be 1 or 2")

         if (self%rosenbrock_version == 1 .and. self%operator_splitting == 1) call fail_input_any("operator_splitting 1 not implement for rosenbrock_version 1")
        endif


        open(10, file='fluid_prop.bin', form='unformatted', status='old')
        read(10) number_species, self%nreactions
        if (number_species/=N_S) call fail_input_any("Error! Check number of species in fluid_prop.bin or index_define.h")
        allocate(self%mw(N_S),self%rgas(N_S),self%init_mf(N_S),self%h298(N_S),self%species_names(N_S))
        allocate(self%sigma(N_S),self%epsK(N_S),self%dipole(N_S),self%polariz(N_S),self%geometry(N_S))
        if (self%enable_chemistry > 0 .and. self%nreactions == 0) then
         call fail_input_any("Error! Chemistry enabled but no reaction provided in yaml")
        endif

        read(10) self%species_names
        read(10) self%mw
        read(10) self%rgas
        read(10) self%h298
        read(10) self%sigma
        read(10) self%epsK
        read(10) self%dipole
        read(10) self%polariz
        read(10) self%geometry
        if(self%masterproc) print *, 'species :', self%species_names
        if(self%masterproc) print *, 'Number of reactions :', self%nreactions

        read(10) number_cp_coeff, nsetcv
        select case (number_cp_coeff)
        case(7) ! NASA 7
         indx_cp_l = 0
         indx_cp_r = 4
        case(9) ! NASA 9
         indx_cp_l = -2
         indx_cp_r = 4
        case default
         call fail_input_any("Error! Cp model not supported")
        end select
        allocate(self%trange(N_S,nsetcv+1))
        allocate(self%cp_coeff(indx_cp_l:indx_cp_r+2,N_S,nsetcv))
        allocate(self%cv_coeff(indx_cp_l:indx_cp_r+2,N_S,nsetcv))
        read(10) self%trange
        read(10) self%cp_coeff

        do lsp=1,N_S ! Loop on species
         write(chspec,'(I2.2)') lsp
         call cfg%get("ref_state","init_Y_"//chspec,self%init_mf(lsp)) ! Initial mass fraction
        enddo

        do lsp=1,N_S
         if (self%masterproc) write(*,*) 'h(298.15) (J/Kmol):', self%h298(lsp)
         do m=1,nsetcv
          do l=indx_cp_l,indx_cp_r+2
           self%cp_coeff(l,lsp,m) = self%cp_coeff(l,lsp,m)*self%rgas(lsp)
           self%cv_coeff(l,lsp,m) = self%cp_coeff(l,lsp,m)
          enddo
          self%cv_coeff(0,lsp,m)  = self%cp_coeff(0,lsp,m) - self%rgas(lsp)
         enddo
        enddo

!
!       Define mixture properties with initial mass fractions
!
        rmixt0 = get_rmixture(N_S,self%rgas,self%init_mf)
!
        present_t0   = .false.
        present_p0   = .false.
        present_rho0 = .false.
        if (self%cfg%has_key("ref_state","t0"))   present_t0 = .true.
        if (self%cfg%has_key("ref_state","p0"))   present_p0 = .true.
        if (self%cfg%has_key("ref_state","rho0")) present_rho0 = .true.
        if (.not.present_t0.and.present_p0.and.present_rho0) then
         call self%cfg%get("ref_state","p0",self%p0)
         call self%cfg%get("ref_state","rho0",self%rho0)
         self%t0 = self%p0/self%rho0/rmixt0
        elseif (.not.present_p0.and.present_t0.and.present_rho0) then
         call self%cfg%get("ref_state","t0",self%t0)
         call self%cfg%get("ref_state","rho0",self%rho0)
         self%p0 = self%rho0*rmixt0*self%t0
        elseif (.not.present_rho0.and.present_t0.and.present_p0) then
         call self%cfg%get("ref_state","t0",self%t0)
         call self%cfg%get("ref_state","p0",self%p0)
         self%rho0 = self%p0/rmixt0/self%t0
        else
         call fail_input_any("Problem with the reference state")
        endif

        cp0    = get_cp(t0,indx_cp_l,indx_cp_r,self%cp_coeff,nsetcv,self%trange,self%init_mf)
        cv0    = cp0-rmixt0
        gam    = cp0/cv0
        gm1    = gam-1._rkind
        gm     = 1._rkind/gm1
        if (self%masterproc) write(*,*) 'Mixture gas constant: ', rmixt0
        if (self%masterproc) write(*,*) 'Mixture cp0: ', cp0
        if (self%masterproc) write(*,*) 'Mixture cv0: ', cv0
        if (self%masterproc) write(*,*) 'Mixture Gamma: ', gam

        read(10) t_min_tab,t_max_tab,dt_tab
        num_t_tab = nint(t_max_tab-t_min_tab)/dt_tab
        dt_tab    = (t_max_tab-t_min_tab)/num_t_tab
        allocate(self%visc_species(num_t_tab+1,N_S))
        allocate(self%lambda_species(num_t_tab+1,N_S))
        allocate(self%diffbin_species(num_t_tab+1,N_S,N_S))
        read(10) self%visc_species
        read(10) self%lambda_species
        read(10) self%diffbin_species

        self%diffbin_species = self%diffbin_species*101325._rkind/p0

        mu0 = get_mixture_viscosity(t0,t_min_tab,dt_tab,num_t_tab,&
                                    self%init_mf,self%mw,self%visc_species)

        k0 = get_mixture_lambda(t0,t_min_tab,dt_tab,num_t_tab,&
                                    self%init_mf,self%mw,rmixt0,self%lambda_species)
!
!        call setState_TPY(self%mixture_yaml,t0,p0,self%init_mf)
!        mu0 = viscosity(self%mixture_yaml)
!        k0  = thermalConductivity(self%mixture_yaml)
!
        if (self%masterproc) write(*,*) 'Mixture pressure: ', p0
        if (self%masterproc) write(*,*) 'Mixture density: ', rho0
        if (self%masterproc) write(*,*) 'Mixture temperature: ', t0
        if (self%masterproc) write(*,*) 'Mixture viscosity: ', mu0
        if (self%masterproc) write(*,*) 'Mixture th conductivityi: ', k0

        !do lsp=1,N_S
        ! dlsp = get_species_diff(lsp,p0,p0,rho0,t0,t_min_tab,dt_tab,num_t_tab,&
        !                        self%init_mf,self%mw,rmixt0,self%diffbin_species)
        !enddo


        if (self%enable_soret) call self%compute_collision_integrals()

        if (self%enable_chemistry > 0) then
         allocate(self%arr_a(self%nreactions,2))
         allocate(self%arr_b(self%nreactions,2))
         allocate(self%arr_ea(self%nreactions,2))
         allocate(self%falloff_coeffs(self%nreactions,5))
         allocate(self%tb_eff(self%nreactions,N_S))
         allocate(self%reac_ty(self%nreactions))
         allocate(self%isRev(self%nreactions))
         allocate(self%r_coeffs(self%nreactions,N_S))
         allocate(self%p_coeffs(self%nreactions,N_S))
         allocate(self%kc_tab(num_t_tab+1,self%nreactions))

         read(10) self%reac_ty
         read(10) self%arr_a
         read(10) self%arr_b
         read(10) self%arr_ea
         read(10) self%falloff_coeffs
         read(10) self%tb_eff
         read(10) self%r_coeffs
         read(10) self%p_coeffs
         read(10) self%kc_tab
   
         arr_a_mult = 1._rkind
         if (self%cfg%has_key("fluid","arr_preexp_multiplier")) then
          call cfg%get("fluid","arr_preexp_multiplier",arr_a_mult)
         endif
         self%arr_a = arr_a_mult*self%arr_a

         do l=1,num_t_tab+1
          do lr = 1,self%nreactions
           if (self%kc_tab(l,lr) > huge(1._rkind)) self%kc_tab(l,lr) = huge(1._rkind)
           if (self%kc_tab(l,lr) < tiny(1._rkind)) self%kc_tab(l,lr) = tiny(1._rkind)
          enddo
         enddo
        endif

       close(10)
        endassociate
    endsubroutine set_fluid_prop
!
    subroutine set_fluid_prop_cantera(self)
        class(equation_multideal_object), intent(inout) :: self
!
        integer :: lsp,msp,m,l,lr
        integer :: number_species, number_cp_coeff
        real(rkind) :: cploc, cploc_spec
        real(rkind), allocatable, dimension(:) :: tem_ranges
        real(rkind), allocatable, dimension(:,:) :: cp_coeff_matrix ! 1st=coeffs, 2nd=ranges
        character(2) :: chspec
        character(40) :: ReacType, filename
        integer, dimension(N_S) :: nrange
        real(rkind) :: tloc, p_tab
        real(rkind), dimension(N_S) :: ytmp
        real(rkind), dimension(N_S*N_S) :: diff_ij
        logical :: present_t0,present_p0,present_rho0
        real(rkind), allocatable, dimension(:)   :: q,q_cant
        real(rkind), dimension(N_S) :: wdot, wdot_from_cantera,omdot
        real(rkind)                 :: q1,q2,tb,kb,kf,hrr
        integer :: itt
        real(rkind) :: dtt, dkc, kc,ttleft, arr_a_mult
        real(rkind) :: conc,fcent,rpres,coff,doff,foff,noff,xoff,k0,kinf
!

        associate(gam => self%gam, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, nsetcv => self%nsetcv, &
                  gm => self%gm, gm1 => self%gm1, rfac => self%rfac, rmixt0 => self%rmixt0, mu0 => self%mu0, &
                  rho0 => self%rho0, p0 => self%p0, k0 => self%k0, &
                  cfg => self%cfg,t0 => self%t0, mw => self%mw, rgas => self%rgas, trange => self%trange, &
                  t_min_tab => self%t_min_tab, t_max_tab => self%t_max_tab, dt_tab => self%dt_tab, num_t_tab => self%num_t_tab, &
                  init_mf => self%init_mf, cp0 => self%cp0, cv0 => self%cv0, enable_chemistry => self%enable_chemistry)
!
        call cfg%get("fluid","enable_chemistry",enable_chemistry)

        if (enable_chemistry == 1) then
         if (self%masterproc) print *, 'Chemistry enabled: explicit integration'
        elseif (enable_chemistry == 2) then
         if (self%masterproc) print *, 'Chemistry enabled: implicit Rosenbrock integration'
        endif

        if (enable_chemistry == 2) then 
         call cfg%get("numerics","rosenbrock_version",self%rosenbrock_version)
         if (self%rosenbrock_version .ne. 1 .and. self%rosenbrock_version .ne. 2) call fail_input_any("rosenbrock_version must be 1 or 2")

         call cfg%get("numerics","operator_splitting",self%operator_splitting)
         if (self%operator_splitting .ne. 1 .and. self%operator_splitting .ne. 2) call fail_input_any("operator_splitting must be 1 or 2")

         if (self%rosenbrock_version == 1 .and. self%operator_splitting == 1) call fail_input_any("operator_splitting 1 not implement for rosenbrock_version 1")
        endif
!
        self%mixture_yaml = importphase('input_cantera.yaml') 
        allocate(self%species_names(N_S))
        do lsp=1,N_S
         call getSpeciesName(self%mixture_yaml, lsp, self%species_names(lsp))
        enddo
        if(self%masterproc) print *, 'species :', self%species_names
        number_species = nSpecies(self%mixture_yaml)
        self%nreactions = nReactions(self%mixture_yaml)
        if(self%masterproc) print *, 'Number of reactions :', self%nreactions

        if (self%enable_chemistry > 0 .and. self%nreactions == 0) then
         call fail_input_any("Error! Chemistry enabled but no reaction provided in yaml")
        endif

        if (number_species/=N_S) call fail_input_any("Error! Check number of species in yaml or index_define.h")

        if (any(self%bctags(1:6) == 56)) then
         do lsp=1,N_S
          if (self%species_names(lsp)=='N2') self%idx_N2 = lsp
          if (self%species_names(lsp)=='N')  self%idx_N  = lsp
          if (self%species_names(lsp)=='NO') self%idx_NO = lsp
          if (self%species_names(lsp)=='O2') self%idx_O2 = lsp
          if (self%species_names(lsp)=='O')  self%idx_O  = lsp
         enddo
         if (self%masterproc) print *, 'idx N2,N,NO,O2,O',self%idx_N2,self%idx_N,self%idx_NO,self%idx_O2,self%idx_O
        endif

        allocate(self%mw(N_S),self%rgas(N_S),self%init_mf(N_S),self%h298(N_S)) ! Molecular weights, specific gas constants and initial mass fractions
!
        call get_cp_from_cantera(1,tem_ranges,cp_coeff_matrix)
        nsetcv = size(tem_ranges)-1
        number_cp_coeff = size(cp_coeff_matrix,1)
        do lsp=2,N_S
         call get_cp_from_cantera(lsp,tem_ranges,cp_coeff_matrix)
         if (nsetcv/=(size(tem_ranges)-1)) call fail_input_any("Error! Different number of temperature ranges not supported")
         if (number_cp_coeff/=(size(cp_coeff_matrix,1))) call fail_input_any("Error! Different models for cp not supported")
        enddo
!
        select case (number_cp_coeff)
        case(7) ! NASA 7
         indx_cp_l = 0
         indx_cp_r = 4
        case(9) ! NASA 9
         indx_cp_l = -2
         indx_cp_r = 4
        case default
         call fail_input_any("Error! Cp model not supported")
        end select
        allocate(self%cp_coeff(indx_cp_l:indx_cp_r+2,N_S,nsetcv))
        allocate(self%cv_coeff(indx_cp_l:indx_cp_r+2,N_S,nsetcv))
        allocate(self%trange(N_S,nsetcv+1))
!
        do lsp=1,N_S
         call get_cp_from_cantera(lsp,tem_ranges,cp_coeff_matrix)
         self%trange(lsp,:) = tem_ranges
         self%cp_coeff(:,lsp,:) = cp_coeff_matrix
        enddo
!        
        call getMolecularWeights(self%mixture_yaml,self%mw)
!
        do lsp=1,N_S ! Loop on species
         write(chspec,'(I2.2)') lsp
         call cfg%get("ref_state","init_Y_"//chspec,self%init_mf(lsp)) ! Initial mass fraction
        enddo
        do lsp=1,N_S
         self%rgas(lsp) = R_univ/self%mw(lsp)
         call setState_TPY(self%mixture_yaml,298.15_rkind,p0,self%init_mf)
         call getPartialMolarEnthalpies(self%mixture_yaml,self%h298)
         !self%h298(lsp) = h_species_dimensional_NASA(298.15_rkind,indx_cp_l,indx_cp_r, &
         !       self%cp_coeff(indx_cp_l:indx_cp_r+2,lsp,1:nsetcv),nsetcv,self%trange(lsp,1:nsetcv+1))
         !self%h298(lsp) = self%h298(lsp)*R_univ
         if (self%masterproc) write(*,*) 'h(298.15) (J/Kmol):', self%h298(lsp)
!
         do m=1,nsetcv
          do l=indx_cp_l,indx_cp_r+2
           self%cp_coeff(l,lsp,m) = self%cp_coeff(l,lsp,m)*self%rgas(lsp)
           self%cv_coeff(l,lsp,m) = self%cp_coeff(l,lsp,m)
          enddo
          self%cv_coeff(0,lsp,m)  = self%cp_coeff(0,lsp,m) - self%rgas(lsp)
         enddo
!
        enddo
!
!       Define mixture properties with initial mass fractions
!
        rmixt0 = get_rmixture(N_S,self%rgas,self%init_mf)
!
        present_t0   = .false.
        present_p0   = .false.
        present_rho0 = .false.
        if (self%cfg%has_key("ref_state","t0"))   present_t0 = .true.
        if (self%cfg%has_key("ref_state","p0"))   present_p0 = .true.
        if (self%cfg%has_key("ref_state","rho0")) present_rho0 = .true.
        if (.not.present_t0.and.present_p0.and.present_rho0) then
         call self%cfg%get("ref_state","p0",self%p0)
         call self%cfg%get("ref_state","rho0",self%rho0)
         self%t0 = self%p0/self%rho0/rmixt0
        elseif (.not.present_p0.and.present_t0.and.present_rho0) then
         call self%cfg%get("ref_state","t0",self%t0)
         call self%cfg%get("ref_state","rho0",self%rho0)
         self%p0 = self%rho0*rmixt0*self%t0
        elseif (.not.present_rho0.and.present_t0.and.present_p0) then
         call self%cfg%get("ref_state","t0",self%t0)
         call self%cfg%get("ref_state","p0",self%p0)
         self%rho0 = self%p0/rmixt0/self%t0
        else
         call fail_input_any("Problem with the reference state")
        endif
!
        cp0    = get_cp(t0,indx_cp_l,indx_cp_r,self%cp_coeff,nsetcv,self%trange,self%init_mf)
        cv0    = cp0-rmixt0
        gam    = cp0/cv0
        gm1    = gam-1._rkind
        gm     = 1._rkind/gm1
        if (self%masterproc) write(*,*) 'Mixture gas constant: ', rmixt0
        if (self%masterproc) write(*,*) 'Mixture cp0: ', cp0
        if (self%masterproc) write(*,*) 'Mixture cv0: ', cv0
        if (self%masterproc) write(*,*) 'Mixture Gamma: ', gam
!
        call setState_TPY(self%mixture_yaml,t0,p0,self%init_mf)
        mu0 = viscosity(self%mixture_yaml)
        k0  = thermalConductivity(self%mixture_yaml)
!
        if (self%masterproc) write(*,*) 'Mixture pressure: ', p0
        if (self%masterproc) write(*,*) 'Mixture density: ', rho0
        if (self%masterproc) write(*,*) 'Mixture temperature: ', t0
        if (self%masterproc) write(*,*) 'Mixture viscosity: ', mu0
!
        call cfg%get("fluid","t_min_tab",t_min_tab)
        call cfg%get("fluid","t_max_tab",t_max_tab)
        call cfg%get("fluid","dt_tab",dt_tab)
        num_t_tab = nint(t_max_tab-t_min_tab)/dt_tab
        dt_tab    = (t_max_tab-t_min_tab)/num_t_tab
        allocate(self%visc_species(num_t_tab+1,N_S))
        allocate(self%lambda_species(num_t_tab+1,N_S))
        do lsp=1,N_S
         ytmp = 0._rkind
         ytmp(lsp) = 1._rkind
         do l=1,num_t_tab+1
          tloc = t_min_tab+(l-1)*dt_tab
          call setState_TPY(self%mixture_yaml,tloc,p0,ytmp)
          self%visc_species(l,lsp) = viscosity(self%mixture_yaml)
          self%lambda_species(l,lsp) = thermalConductivity(self%mixture_yaml)
         enddo
        enddo
!
        allocate(self%diffbin_species(num_t_tab+1,N_S,N_S))
        do l=1,num_t_tab+1
         tloc = t_min_tab+(l-1)*dt_tab
         call setState_TPY(self%mixture_yaml,tloc,p0,self%init_mf)
         call getBinDiffCoeffs(self%mixture_yaml,N_S,diff_ij)
         do lsp=1,N_S
          do msp=1,N_S
           self%diffbin_species(l,lsp,msp) = diff_ij(msp+(lsp-1)*N_S)
          enddo
         enddo
        enddo

        if (self%enable_soret) then
         call get_transport_data_from_cantera(self%sigma,self%epsK,self%dipole,self%polariz,self%geometry)
         call self%compute_collision_integrals()
        endif

        if (self%enable_chemistry > 0) then
         allocate(self%arr_a(self%nreactions,2))
         allocate(self%arr_b(self%nreactions,2))
         allocate(self%arr_ea(self%nreactions,2))
         allocate(self%falloff_coeffs(self%nreactions,5))
         allocate(self%tb_eff(self%nreactions,N_S))
         allocate(self%reac_ty(self%nreactions))
         allocate(self%isRev(self%nreactions))

         do lr=1,self%nreactions
          call getReactionType(self%mixture_yaml,lr,ReacType)
          if (ReacType == "Arrhenius") then
           self%reac_ty(lr) = 0
          else if (ReacType == "three-body-Arrhenius") then
           self%reac_ty(lr) = 1
          else if (ReacType == "falloff-Lindemann") then
           self%reac_ty(lr) = 2
          else if (ReacType == "falloff-Troe") then
           self%reac_ty(lr) = 3
          else if (ReacType == "falloff-SRI") then
           self%reac_ty(lr) = 4
          end if
          self%isRev(lr) = isReversible(self%mixture_yaml,lr-1)
         enddo

         call get_arrhenius_from_cantera(self%arr_a, self%arr_b, self%arr_ea, self%tb_eff, self%falloff_coeffs)
         

         arr_a_mult = 1._rkind
         if (self%cfg%has_key("fluid","arr_preexp_multiplier")) then
          call cfg%get("fluid","arr_preexp_multiplier",arr_a_mult)
         endif
         self%arr_a = arr_a_mult*self%arr_a

         allocate(self%r_coeffs(self%nreactions,N_S))
         allocate(self%p_coeffs(self%nreactions,N_S))

         do lr = 1,self%nreactions
          do lsp = 1,N_S
           self%r_coeffs(lr,lsp) = reactantStoichCoeff(self%mixture_yaml,lsp,lr)
           self%p_coeffs(lr,lsp) = productStoichCoeff (self%mixture_yaml,lsp,lr)
          enddo
         enddo

         allocate(self%kc_tab(num_t_tab+1,self%nreactions))
         do l=1,num_t_tab+1
          tloc = t_min_tab+(l-1)*dt_tab
          call setState_TPY(self%mixture_yaml,tloc,p0,self%init_mf)
          call getEquilibriumConstants(self%mixture_yaml,self%kc_tab(l,:))
          do lr = 1,self%nreactions
           if (self%kc_tab(l,lr) > huge(1._rkind)) self%kc_tab(l,lr) = huge(1._rkind)
           if (self%kc_tab(l,lr) < tiny(1._rkind)) self%kc_tab(l,lr) = tiny(1._rkind)
          enddo
         enddo
        !endif

!         !Test OmegaDot
!         allocate(q(self%nreactions))
!         allocate(q_cant(self%nreactions))
!         !allocate(kc2(self%nreactions))
!         call setState_TPY(self%mixture_yaml,t0,p0,ytmp)
!         !call getEquilibriumConstants(self%mixture_yaml,kc2)
! 
!         itt = int((t0-t_min_tab)/dt_tab)+1
!         ttleft = (itt-1)*dt_tab + t_min_tab
!         dtt = (t0-ttleft)/dt_tab
!
!         do lr=1,self%nreactions
!          select case (self%reac_ty(lr))
!           case (0) !Arrhenius
!            kf = self%arr_a(lr,1)*(t0**self%arr_b(lr,1))*exp(-self%arr_ea(lr,1)/R_univ/t0)
!           case (1) !Three-body
!            kf = self%arr_a(lr,1)*(t0**self%arr_b(lr,1))*exp(-self%arr_ea(lr,1)/R_univ/t0)
!           case (2) !falloff-Lindemann
!            k0 = self%arr_a(lr,1)*(t0**self%arr_b(lr,1))*exp(-self%arr_ea(lr,1)/R_univ/t0)
!            kinf = self%arr_a(lr,2)*(t0**self%arr_b(lr,2))*exp(-self%arr_ea(lr,2)/R_univ/t0)
!            tb = 0._rkind
!            do lsp=1,N_S
!             tb = tb + self%tb_eff(lr,lsp)*(rho0*self%init_mf(lsp)/self%mw(lsp))
!            enddo
!            rpres = k0*tb/kinf
!            foff = 1
!            kf = kinf*(rpres/(1 + rpres))*foff
!            case (3) !falloff-Troe
!             k0 = self%arr_a(lr,1)*(t0**self%arr_b(lr,1))*exp(-self%arr_ea(lr,1)/R_univ/t0)
!             kinf = self%arr_a(lr,2)*(t0**self%arr_b(lr,2))*exp(-self%arr_ea(lr,2)/R_univ/t0)
!             tb = 0._rkind
!             do lsp=1,N_S
!              tb = tb + self%tb_eff(lr,lsp)*(rho0*self%init_mf(lsp)/self%mw(lsp))
!             enddo
!             rpres = k0*tb/kinf
!             if (rpres .lt. 1E-30_rkind) then
!              kf = 0._rkind
!             else
!             fcent = (1 - self%falloff_coeffs(lr,1))*exp(-t0/self%falloff_coeffs(lr,2)) + &
!             self%falloff_coeffs(lr,1)*exp(-t0/self%falloff_coeffs(lr,3))
!             if (self%falloff_coeffs(lr,4) .ne. -3.14_rkind) then
!              fcent = fcent + exp(-self%falloff_coeffs(lr,4)/t0)
!             end if
!             coff = -0.4 - 0.67*log10(fcent)
!             noff = 0.75 - 1.27*log10(fcent)
!             doff = 0.14
!             foff = 10**((((1 + ((log10(rpres) + coff)/(noff - doff*(log10(rpres) + &
!             coff)))**2)**(-1))*log10(fcent)))
!             kf = kinf*(rpres/(1 + rpres))*foff
!             endif
!            case(4) !falloff-SRI
!            k0 = self%arr_a(lr,1)*(t0**self%arr_b(lr,1))*exp(-self%arr_ea(lr,1)/R_univ/t0)
!            kinf = self%arr_a(lr,2)*(t0**self%arr_b(lr,2))*exp(-self%arr_ea(lr,2)/R_univ/t0)
!            tb = 0._rkind
!            do lsp=1,N_S
!             tb = tb + self%tb_eff(lr,lsp)*(rho0*self%init_mf(lsp)/self%mw(lsp))
!            enddo
!            rpres = k0*tb/kinf
!            if (rpres .lt. 1E-30_rkind) then
!             kf = 0._rkind
!            else
!             xoff = 1/(1 + log10(rpres)**2)
!             foff = self%falloff_coeffs(lr,4)*((self%falloff_coeffs(lr,1) * exp(-self%falloff_coeffs(lr,2)/t0) + &
!             exp(-t0/self%falloff_coeffs(lr,3)))**xoff)*(t0**self%falloff_coeffs(lr,5))
!             kf = kinf*(rpres/(1 + rpres))*foff
!            endif
!            end select
!
!            if (self%isRev(lr) .eq. 1) then                        
!             dkc = self%kc_tab(itt+1,lr)-self%kc_tab(itt,lr)
!             kc  = self%kc_tab(itt,lr)+dkc*dtt
!             kb = kf/kc
!            else
!             kb = 0._rkind
!            endif
!
!            q1 = 1._rkind
!            q2 = 1._rkind
!            do lsp=1,N_S
!             conc = rho0*self%init_mf(lsp)/self%mw(lsp)
!             q1 = q1*conc**self%r_coeffs(lr,lsp)
!             q2 = q2*conc**self%p_coeffs(lr,lsp)
!            enddo
!            if (self%reac_ty(lr) == 1) then
!             tb = 0._rkind
!             do lsp=1,N_S
!              tb = tb + self%tb_eff(lr,lsp)*(rho0*self%init_mf(lsp)/self%mw(lsp))
!             enddo
!            else
!             tb = 1._rkind
!            endif
!            q(lr) = tb*(kf*q1-kb*q2)
!           enddo
!           do lsp=1,N_S
!            wdot(lsp) = 0._rkind
!            do lr=1,self%nreactions
!             wdot(lsp) = wdot(lsp) + (self%p_coeffs(lr,lsp)-self%r_coeffs(lr,lsp))*q(lr)
!            enddo
!           enddo
!
!           call setState_TPY(self%mixture_yaml,t0,p0,self%init_mf)
!           call getNetProductionRates(self%mixture_yaml,wdot_from_cantera)
!           call getNetRatesOfProgress(self%mixture_yaml,q_cant)
!
!           if (self%masterproc) then
!            print *, 'STREAmS-2 vs. Cantera'
!            print '(1x,A)', '------------- qjNet ------------'
!            do lr=1,self%nreactions
!             write(*,'(1x,A,I3,A,g20.10,A,g20.10,A,g20.10)') 'nRx ', lr, ': ', q(lr),' - ', q_cant(lr), ' - ', q(lr)-q_cant(lr)
!            end do
!         !Meglio stampare le differenze e stampare solo se diff > FixedEps?
!            print '(1x,A)', '---------- OmegaDotNet ----------'
!             do lsp=1,N_S
!              write(*,'(1x,A,I3,A,A,A,1x,g20.10,A,g20.10)') 'nSp ',lsp,' -',trim(self%species_names(lsp)),": ",wdot(lsp)," - ", wdot_from_cantera(lsp)
!             enddo
!            endif
!        !end OmegaDot Test
        endif
        endassociate
        
    endsubroutine set_fluid_prop_cantera
!
   subroutine compute_collision_integrals(self)
        class(equation_multideal_object), intent(inout) :: self
        real(rkind), dimension(8)    :: delta
        real(rkind), dimension(39)   :: tstar
        real(rkind), dimension(39,8) :: astar_table,bstar_table,cstar_table
        real(rkind), dimension(39,7) :: astar_coeff,bstar_coeff,cstar_coeff
        real(rkind), parameter :: kB   = 1.380649D-23
        real(rkind), parameter :: eps0 = 8.8541878188D-12
        integer :: i,j,it,kp,knp,l,i1,i2,k
        logical :: i_is_polar, j_is_polar
        real(rkind) :: ts,reduced_mass,sigmaij,epsij,deltastar,d3np,d3p,alpha_star,mu_p_star,xi,f_sigma,f_eps
        real(rkind) :: tloc,astar,bstar,cstar
        real(rkind), dimension(3) :: logtstarvec,values_astar,values_bstar,values_cstar

        associate(mw => self%mw, sigma => self%sigma, epsK => self%epsK, dipole => self%dipole, polariz => self%polariz)

        call get_abcstar_tables(delta,tstar,astar_table,bstar_table,cstar_table) ! in parameters!

        ! for each temperature in the table, fit a 6 degree polynomial in delta
        do it=1,39
         call polyfit(delta,astar_table(it,:),8,6,astar_coeff(it,:)) ! in parameters!
         call polyfit(delta,bstar_table(it,:),8,6,bstar_coeff(it,:)) ! in parameters!
         call polyfit(delta,cstar_table(it,:),8,6,cstar_coeff(it,:)) ! in parameters!
        enddo

        allocate(self%abcstar_tab(self%num_t_tab+1,N_S,N_S,3))
        do j=1,N_S
         do i=1,N_S
          reduced_mass = mw(i)*mw(j)/(mw(i)+mw(j))
          sigmaij      = 0.5_rkind*(sigma(i) + sigma(j))
          epsij        = sqrt(epsK(i)*epsK(j)) * kB ! my epsK is already divided by kB
          deltastar    = 0.5_rkind*dipole(i)*dipole(j)
          deltastar    = deltastar/(4._rkind*pi*eps0*epsij*sigmaij**3)
          ! compute polar correction
          i_is_polar = .false.; j_is_polar = .false.
          if (dipole(i) .ne. 0._rkind) i_is_polar = .true.
          if (dipole(j) .ne. 0._rkind) j_is_polar = .true.
          if (i_is_polar == j_is_polar) then
           f_eps = 1._rkind
           f_sigma = 1._rkind
          else
           if (i_is_polar) then
            kp = i; knp = j
           else
            kp = j; knp = i
           endif
           d3np = sigma(knp)**3
           d3p = sigma(kp)**3
           alpha_star = polariz(knp)/d3np
           mu_p_star = dipole(kp) / sqrt(4._rkind*pi*eps0*d3p*epsK(kp)*kB)
           xi = 1._rkind + 0.25_rkind * alpha_star * mu_p_star**2 * sqrt(epsK(kp)/epsK(knp))
           f_sigma = xi**(-1._rkind/6._rkind)
           f_eps = xi*xi
          endif
          epsij   = epsij*f_eps
          sigmaij = sigmaij*f_sigma
          ! tabulation
          do l=1,self%num_t_tab+1
           tloc = self%t_min_tab+(l-1)*self%dt_tab
           ts = tloc*kB/epsij
           do it = 1,39
            if (ts < tstar(it)) exit
           enddo
           !if (i1 < 1 .and. self%masterproc) write(*,'(A,F10.4)') 'WARNING: tstar out of bounds, limited to ', tstar(1)
           i1 = max(it-1,1); i2 = i1+2
           if (i2 > 39) then
            !if (self%masterproc) write(*,'(A,F10.4)') 'WARNING: tstar out of bounds, limited to ', tstar(39) 
            i2 = 39; i1 = 39-2
           endif
           do k=i1,i2
            if (deltastar == 0._rkind) then
             values_astar(k-i1+1) = astar_table(k,1)
             values_bstar(k-i1+1) = bstar_table(k,1)
             values_cstar(k-i1+1) = cstar_table(k,1)
            else
             values_astar(k-i1+1) = poly6(deltastar, astar_coeff(k,:))
             values_bstar(k-i1+1) = poly6(deltastar, bstar_coeff(k,:))
             values_cstar(k-i1+1) = poly6(deltastar, cstar_coeff(k,:))
            endif
            logtstarvec(k-i1+1) = log(tstar(k))
           enddo
           astar = quadInterp(log(ts), logtstarvec, values_astar)
           bstar = quadInterp(log(ts), logtstarvec, values_bstar)
           cstar = quadInterp(log(ts), logtstarvec, values_cstar)
           self%abcstar_tab(l,i,j,1) = astar
           self%abcstar_tab(l,i,j,2) = bstar
           self%abcstar_tab(l,i,j,3) = cstar
          enddo

         enddo
        enddo
        endassociate

    endsubroutine compute_collision_integrals
!
    subroutine set_flow_params(self)
        class(equation_multideal_object), intent(inout) :: self 
!        
        integer :: lsp
        real(rkind) :: gm1h
        real(rkind) :: e0, etot0
!
        allocate(self%winf(self%nv))
!
        associate(Mach => self%Mach, Reynolds => self%Reynolds, &
                  t0 => self%t0, rho0 => self%rho0, p0 => self%p0, &
                  u0 => self%u0, mu0 => self%mu0, k0 => self%k0, c0 => self%c0, &
                  rmixt0 => self%rmixt0, rfac => self%rfac, gm1 => self%gm1, gam => self%gam, &
                  winf => self%winf, flow_params_cfg => self%flow_params_cfg, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, Prandtl => self%Prandtl, &
                  T_recovery => self%T_recovery, theta_wall => self%theta_wall, &
                  cv_coeff => self%cv_coeff, cfg => self%cfg, cp_coeff => self%cp_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, cp0 => self%cp0, cv0 => self%cv0, &
                  init_mf => self%init_mf, delta0 => self%delta0, delta0star => self%delta0star, &
                  T_wall => self%T_wall, eps_sensor => self%eps_sensor)
!
        p0    = rho0*rmixt0*t0
        c0    = sqrt(gam*rmixt0*t0)
        if (self%masterproc) write(*,*) 'Speed of sound', c0
        Prandtl = mu0*cp0/k0
        rfac    = Prandtl**(1._rkind/3._rkind)

        if (self%bl_case .or. self%double_bl_case .or. self%channel_case) then
         call cfg%get("flow","Reynolds",Reynolds)
         call cfg%get("flow","Mach",Mach)
         u0 = Mach*c0
         if (self%masterproc) write(*,*) 'u0 based on Mach and c0: ',u0
         T_recovery = t0*(1._rkind+0.5_rkind*(gam-1._rkind)*rfac*Mach**2)

!        Wall state
         if (self%cfg%has_key("flow","T_wall")) then
          call self%cfg%get("flow","T_wall",T_wall)
          if (any(self%bctags(1:6) == 6) .or. any(self%bctags(1:6) == 26) &
             .or. any(self%bctags(1:6) == 36) .or. any(self%bctags(1:6) == 46) .or. any(self%bctags(1:6) == 56)) then
           theta_wall = (T_wall-t0)/(T_recovery-t0)
          else
           call fail_input_any("Error! Found T_wall key but no isothermal wall has been selected")
          endif
         else
          theta_wall = 1._rkind
          if(any(self%bctags(1:6) == 6) .or. any(self%bctags(1:6) == 26)) then
           call fail_input_any("Error! T_wall key not found but at least one isothermal wall has been selected")
          endif
          T_wall = T_recovery
          if (self%masterproc) write(*,*) "Setting T_wall == T_recovery"
         endif
!
        else
         Mach  = u0/c0
        endif

        select case(self%flow_init)
        case(0)
!        call self%set_chan_prop()
        case(1)
         call self%set_bl_prop()
        case(4)
         call self%set_double_bl_prop()
        end select
!
        e0    = get_e_from_temperature(t0,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,init_mf)
        etot0 = e0+0.5_rkind*u0**2

        winf = 0._rkind
        do lsp=1,N_S
         winf(lsp) = rho0*init_mf(lsp)
        enddo
        winf(I_U) = rho0*u0
        winf(I_V) = rho0*0._rkind
        winf(I_W) = rho0*0._rkind
        winf(I_E) = rho0*etot0
!
        if (abs(u0)<tol_iter) then
         u0 = c0
         if (self%masterproc) write(*,*) 'u0 too small, set to c0'
        endif

!       sensor constant
        if (self%bl_case .or.  self%double_bl_case .or. self%channel_case) then
         eps_sensor = (u0/delta0)**2
        else
         eps_sensor = u0**2
        endif
!
        if(self%masterproc) then
           call flow_params_cfg%set("flow_params","u0",             u0)
           call flow_params_cfg%set("flow_params","p0",             p0)
           call flow_params_cfg%set("flow_params","rho0",           rho0)
           call flow_params_cfg%set("flow_params","t0",             t0)
           call flow_params_cfg%set("flow_params","c0",             c0)
           call flow_params_cfg%set("flow_params","cp0",            cp0)
           call flow_params_cfg%set("flow_params","cv0",            cv0)
           call flow_params_cfg%set("flow_params","mu0",            mu0)
           call flow_params_cfg%set("flow_params","k0",             k0)
           call flow_params_cfg%set("flow_params","gam",            gam)
           call flow_params_cfg%set("flow_params","rfac",           rfac)
           call flow_params_cfg%set("flow_params","Reynolds",       Reynolds)
           call flow_params_cfg%set("flow_params","Mach",           Mach)
           call flow_params_cfg%set("flow_params","Prandtl",        Prandtl)
           if (self%bl_case) then
            call flow_params_cfg%set("flow_params","theta_wall",    theta_wall)
            call flow_params_cfg%set("flow_params","T_wall",        T_wall)
            call flow_params_cfg%set("flow_params","delta0",        delta0)
            if (self%bl_laminar) &
            call flow_params_cfg%set("flow_params","delta0star",    delta0star)
           endif
           if (self%double_bl_case) then
            call flow_params_cfg%set("flow_params","delta0",        delta0)
            call flow_params_cfg%set("flow_params","Reynolds2",     self%Reynolds2)
            call flow_params_cfg%set("flow_params","delta02",       self%delta02)
            call flow_params_cfg%set("flow_params","theta_wall",    theta_wall)
            call flow_params_cfg%set("flow_params","T_wall",        T_wall)
           endif
           call flow_params_cfg%set("flow_params","nv_stat",        self%nv_stat)
           call flow_params_cfg%write("flow_params.dat")
        endif

        endassociate

    endsubroutine set_flow_params

    subroutine set_bl_prop(self)
        class(equation_multideal_object), intent(inout) :: self
        real(rkind) :: Re_out, Trat
        real(rkind), allocatable, dimension(:) :: uvec, rhovec, tvec, viscvec, yl0, yvec
        real(rkind) :: cf,thrat,retheta,redelta,th,ch,mtau,deltav,spr
        integer :: l,i,j,m,imode,icompute,ne
        real(rkind), dimension(:), allocatable :: rbl,eta,ubl
        real(rkind) :: etaedge,deta
        logical :: file_exists

        associate(nymax => self%grid%nymax, Reynolds_friction => self%Reynolds_friction, &
                  Reynolds => self%Reynolds, &
                  Mach => self%Mach, u0 => self%u0, rho0 => self%rho0, mu0 => self%mu0, &
                  gam => self%gam, rfac => self%rfac, yg => self%grid%yg, &
                  T_wall => self%T_wall, theta_wall => self%theta_wall, Prandtl => self%Prandtl, &
                  delta0 => self%delta0)
!
        allocate(uvec(nymax),tvec(nymax),rhovec(nymax),viscvec(nymax),yl0(nymax),yvec(nymax))
!
        if (self%bl_case) then
         if (self%bl_laminar) then
          self%delta0star = Reynolds*self%mu0/(self%u0*self%rho0)
          ! replicate init_bl_lam first part only to compute delta 0 and store it in flow params'
          inquire(file='similar_profiles.prof',exist=file_exists)
          if (.not. file_exists) call fail_input_any('similar_profiles.prof needed for init_bl_lam')
          open(10,file='similar_profiles.prof',action='read',form='formatted', status='old')
          read(10,*) ne,deta
          allocate(rbl(0:ne))
          allocate(ubl(0:ne))
          allocate(eta(0:ne))
 
          do i=0,ne
           read(10,*) eta(i),rbl(i),ubl(i)
          enddo
          close(10)
 
          etaedge = 0._rkind ! delta0star in eta coords
          do i=0,ne-1
           etaedge = etaedge + 0.5_rkind*((1._rkind - ubl(i+1)*rbl(i+1)/(ubl(ne)*rbl(ne)))*rbl(ne)/rbl(i+1) + &
                                          (1._rkind - ubl(i  )*rbl(i  )/(ubl(ne)*rbl(ne)))*rbl(ne)/rbl(i  ) )*(eta(i+1) - eta(i))
          enddo
          self%xbl = Reynolds*self%delta0star/(2._rkind*(etaedge**2))
          
          ! compute delta0
          self%delta0 = 0._rkind
          do i=1,ne 
           !                                             Reynolds_x0
           self%delta0 =self%delta0 + self%xbl*sqrt(2._rkind/(Reynolds*self%xbl/self%delta0star))*&
                                      rbl(ne)*(eta(i)-eta(i-1))*0.5_rkind*(1._rkind/rbl(i-1)+1._rkind/rbl(i))
           if (ubl(i)>0.99_rkind) exit
          enddo
 
          if (self%masterproc) write(*,*) 'XBL/delta0star', self%xbl/self%delta0star
          if (self%masterproc) write(*,*) 'delta0, delta0star', self%delta0, self%delta0star

          !self%delta0 = 5._rkind/1.721_rkind*self%delta0star
         else
          if (self%masterproc) write(*,*) 'Input friction Reynolds number: ', Reynolds
          if (self%cfg%has_key("flow","delta0")) then
           call self%cfg%get("flow","delta0",delta0)
          else
           call fail_input_any("Error! Inflow bl thickness need to be specified") 
          endif          
          Reynolds_friction  = Reynolds
          !Trat = T_wall/self%T_recovery
          !yl0  = yg(1:nymax)
          !call meanvelocity_bl(Reynolds_friction,Mach,Trat,s2tinf,powerlaw_vtexp,visc_exp,gam,rfac,nymax,yl0(1:nymax),uvec,&
          !                         rhovec,tvec,viscvec,Re_out,cf,thrat)
          imode = 0
          icompute = 0
          yvec = yg(1:nymax)
          spr = 0.8_rkind
          call hasan_meanprofile(self,nymax,delta0,Mach,theta_wall,Reynolds_friction,retheta,redelta,Prandtl,spr,rfac,gam,&
                                 yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
          Reynolds = redelta
         endif
         if (self%masterproc) write(*,*) 'Reynolds based on free-stream properties: ', Reynolds
        endif
!
        endassociate
!
    endsubroutine set_bl_prop

    subroutine set_double_bl_prop(self)
        class(equation_multideal_object), intent(inout) :: self
        real(rkind) :: Re_out, Trat
        real(rkind), allocatable, dimension(:) :: uvec, rhovec, tvec, viscvec, yl0, yvec
        real(rkind) :: cf,thrat,retheta,redelta,th,ch,mtau,deltav,spr
        integer :: l,i,j,m,imode,icompute

        associate(nymax => self%grid%nymax, Reynolds_friction => self%Reynolds_friction, &
                  Reynolds => self%Reynolds, Reynolds2 => self%Reynolds2, &
                  Reynolds_friction2 => self%Reynolds_friction2, &
                  Mach => self%Mach, u0 => self%u0, rho0 => self%rho0, mu0 => self%mu0, &
                  gam => self%gam, rfac => self%rfac, yg => self%grid%yg, &
                  theta_wall => self%theta_wall, Prandtl => self%Prandtl, &
                  delta0 => self%delta0, delta02 => self%delta02)
!
        allocate(uvec(nymax),tvec(nymax),rhovec(nymax),viscvec(nymax),yl0(nymax),yvec(nymax))
!
        if (self%double_bl_case) then

!
!        lower wall
!
         if (self%masterproc) write(*,*) 'Input friction Reynolds number (lower wall): ', Reynolds
         if (self%cfg%has_key("flow","delta0")) then
          call self%cfg%get("flow","delta0",delta0)
         else
          call fail_input_any("Error! Inflow bl thickness need to be specified (lower wall), add delta0 key") 
         endif          
!
         Reynolds_friction  = Reynolds
         !Trat = self%T_wall/self%T_recovery
         !yl0  = yg(1:nymax)
         !call meanvelocity_bl(Reynolds_friction,Mach,Trat,s2tinf,powerlaw_vtexp,visc_exp,gam,rfac,nymax,yl0(1:nymax),uvec,&
         !                         rhovec,tvec,viscvec,Re_out,cf,thrat)
         imode = 0
         icompute = 0
         yvec = yg(1:nymax)
         spr = 0.8_rkind
         call hasan_meanprofile(self,nymax,delta0,Mach,theta_wall,Reynolds_friction,retheta,redelta,Prandtl,spr,rfac,gam,&
                                 yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
         Reynolds = redelta
         if (self%masterproc) write(*,*) 'Reynolds based on free-stream properties: ', Reynolds
!
!        upper wall
!
         if (self%cfg%has_key("flow","Reynolds2")) then
          call self%cfg%get("flow","Reynolds2",self%Reynolds2)
          if (self%cfg%has_key("flow","delta02")) then
           call self%cfg%get("flow","delta02",delta02)
          else
           call fail_input_any("Error! Inflow bl thickness need to be specified (upper wall), add delta02 key") 
          endif
          if (self%masterproc) write(*,*) 'Input friction Reynolds number (upper wall): ', Reynolds2
          Reynolds_friction2 = Reynolds2
          imode = 0
          icompute = 0
          yvec =  yg(1:nymax)
          spr = 0.8_rkind
          call hasan_meanprofile(self,nymax,delta02,Mach,theta_wall,Reynolds_friction2,retheta,redelta,Prandtl,spr,rfac,gam,&
                                  yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
          Reynolds2 = redelta

         else
          if (self%masterproc) print *, "No info given for upper bl, considered the same as the lower one"
          Reynolds2 = Reynolds
          Reynolds_friction2 = Reynolds_friction
          delta02 = delta0
         endif
!
        endif
        endassociate
!
    endsubroutine set_double_bl_prop
!
    subroutine initial_conditions(self)
        class(equation_multideal_object), intent(inout) :: self 
!
        integer :: l
!
        do l=1,self%nv
         self%field%w(l,:,:,:) = self%winf(l)
        enddo
!       ! Temperature initialized so that compute_aux_gpu is able to have the iteration first value (for thermally perfect)
        self%w_aux(:,:,:,J_T)   = self%t0
!
        select case (self%flow_init)
            case(-1) ! wind tunnel
                call self%init_wind_tunnel()
            case(1)
                if (self%bl_laminar) then
                 call self%init_bl_lam()
                 if (self%masterproc) print *, 'case: bl laminar'
                else
                 if (self%masterproc) print *, 'case: bl turb'
                 if (self%masterproc) print *, 'Computing init_bl'
                 call self%init_bl()
                 if (self%masterproc) print *, 'Done with init_bl'
                endif
            case(4) ! HYSHOT
                if (self%masterproc) print *, 'case: double_bl turb'
                if (self%masterproc) print *, 'Computing init_double_bl'
                call  self%init_double_bl()
                if (self%masterproc) print *, 'Done with init_double_bl'
            case(5) ! SOD 
                call self%init_sod()
            case(6) ! FERRER 4.4 multicomponent diffusion
                call self%init_multi_diff()
            case(7) ! FERRER 4.5 acoustic wave
                call self%init_aw()
            case(8) ! FERRER 4.6 perfectly stirred reactor
                call self%init_reactor()
            case(9) ! FERRER 4.7 reactive shock tube
                call self%init_reactive_tube
            case(10) ! FERRER 4.8 
                call self%init_premix()
            case(11) !Scalability
                call self%init_scalability()
        endselect
    endsubroutine initial_conditions
!
    subroutine init_wind_tunnel(self)
        class(equation_multideal_object), intent(inout) :: self 

        integer :: i,j,k,l
        ! only ghost in x to be similar to bl needs
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, nv => self%nv, &
                  wmean => self%wmean, w => self%field%w, winf => self%winf)

!
        wmean = 0._rkind
        do j=1,ny
         do i=1-ng,nx+ng+1
          do l=1,nv
           wmean(i,j,l) = winf(l)
          enddo
         enddo
        enddo
!
        do k=1,nz
         do j=1,ny
          do i=1,nx
           do l=1,nv
            w(l,i,j,k) = winf(l)
           enddo
          enddo
         enddo
        enddo
!
        endassociate
    endsubroutine init_wind_tunnel

    subroutine init_bl_lam(self)
        class(equation_multideal_object), intent(inout) :: self

        integer :: ie,i,j,k,n,nst,ii,l
        integer :: ne,lsp,m
        logical :: file_exists 
        real(rkind) :: etad, deta, etaedge 
        real(rkind) :: xx, yy, etast, wl, wr, ust, vst, tst, rst
        real(rkind) :: rho,uu,vv,ww,rhouu,rhovv,rhoww,ee,tt
        real(rkind), allocatable, dimension(:) :: rbl,ubl,tbl,vbl,eta,ybl,yblloc
        real(rkind), allocatable, dimension(:,:) :: speciesbl 
        real(rkind) :: errmin, error, xtrip, rtemp, rmixtloc
        real(rkind), dimension(N_S) :: yylsp 

        call self%cfg%get("bl_trip","xtr1",rtemp)     ; self%xtr1  = rtemp
        call self%cfg%get("bl_trip","xtr2",rtemp)     ; self%xtr2  = rtemp
        call self%cfg%get("bl_trip","xtw1",rtemp)     ; self%xtw1  = rtemp
        call self%cfg%get("bl_trip","xtw2",rtemp)     ; self%xtw2  = rtemp
        call self%cfg%get("bl_trip","a_tr",rtemp)     ; self%a_tr  = rtemp
        call self%cfg%get("bl_trip","a_tw",rtemp)     ; self%a_tw  = rtemp
        !if (self%cfg%has_key("bl_trip","del0_tr")) then
        ! call self%cfg%get("bl_trip","del0_tr",rtemp) ; self%del0_tr  = rtemp
        !else
        ! self%del0_tr = 0.001_rkind
        !endif
        if (self%cfg%has_key("bl_trip","v_bs")) then
         call self%cfg%get("bl_trip","v_bs",rtemp)    ; self%v_bs  = rtemp
        else
         self%v_bs = 0._rkind
        endif
        call self%cfg%get("bl_trip","kx_tw",rtemp)    ; self%kx_tw  = rtemp
        call self%cfg%get("bl_trip","om_tw",rtemp)    ; self%om_tw  = rtemp
        call self%cfg%get("bl_trip","thic" ,rtemp)    ; self%thic   = rtemp
       
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))

        associate(gam => self%gam, Mach => self%Mach, Prandtl => self%Prandtl, &
                  t0 => self%t0, T_wall => self%T_wall, Reynolds => self%Reynolds, &
                  p0 => self%p0, u0 => self%u0, mu0 => self%mu0, rho0 => self%rho0, &
                  rmixt0 => self%rmixt0, wmean => self%wmean, w => self%field%w, &
                  delta0 => self%delta0, delta0star => self%delta0star, nx => self%field%nx, ny => self%field%ny, &
                  nz => self%field%nz, ng => self%grid%ng, x => self%field%x, y => self%field%y, &
                  z => self%field%z, xg => self%grid%xg, nxmax => self%grid%nxmax, &
                  xtw1 => self%xtw1, xtw2 => self%xtw2, a_tr => self%a_tr, a_tw => self%a_tw, v_bs => self%v_bs,&
                  lamz_old   => self%lamz_old  , lamz1_old  => self%lamz1_old , &
                  phiz_old   => self%phiz_old  , phiz1_old  => self%phiz1_old , &
                  xtr1 => self%xtr1, xtr2 => self%xtr2, xtrip => self%xtrip, masterproc => self%masterproc, &
                  x0tr => self%x0tr, is_old => self%is_old, itr => self%itr, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, xbl => self%xbl)


        
!        etad = 20._rkind
!        deta = 0.04_rkind
!        ne = nint(etad/deta)
         inquire(file='similar_profiles.prof',exist=file_exists)
         if (.not. file_exists) call fail_input_any('similar_profiles.prof needed for init_bl_lam')

         open(10,file='similar_profiles.prof',action='read',form='formatted', status='old')
         read(10,*) ne,deta

         allocate(rbl(0:ne))
         allocate(ubl(0:ne))
         allocate(vbl(0:ne))
         allocate(tbl(0:ne))
         allocate(speciesbl(N_S,0:ne))
         allocate(eta(0:ne))
         allocate(ybl(0:ne))
         allocate(yblloc(0:ne))

         do i=0,ne
          read(10,*) eta(i),rbl(i),ubl(i),vbl(i),tbl(i),speciesbl(:,i) 
         enddo
         close(10)

         ! compute y coord 
         ybl = 0._rkind
         do i=1,ne 
          !                                             Reynolds_x0
          ybl(i) = ybl(i-1) + xbl*sqrt(2._rkind/(Reynolds*xbl/delta0star))*&
                              rbl(ne)*(eta(i)-eta(i-1))*0.5_rkind*(1._rkind/rbl(i-1)+1._rkind/rbl(i))
         enddo
!
!         call compressible_blasius(ne,etad,deta,Mach,t0,T_wall,eta,ubl,vbl,tbl,etaedge)
!
         open(10,file='blasius_inlet.dat',action='write')
         open(11,file='blasius_outlet.dat',action='write')
         wmean = 0._rkind
         do i=1-ng,nx+ng+1
          ii = self%field%ncoords(1)*nx+i
          xx = xg(ii)+xbl
          do ie=0,ne
           yblloc(ie) = ybl(ie)*sqrt(xx/xbl)
          enddo
          do j=1,ny
           yy = y(j)
           call locateval(yblloc,ne+1,yy,nst)
           nst = min(nst,ne)
           ust = ubl(nst) * u0
           vst = vbl(nst) * sqrt(xbl/xx) * u0
           tst = tbl(nst) * self%t0 /tbl(ne)
           !rst = (wr*rbl(nst)+wl*rbl(nst+1))/deta * self%rho0
           rmixtloc = 0._rkind
           do lsp=1,N_S
            yylsp(lsp) = speciesbl(lsp,nst)
            rmixtloc = rmixtloc + self%rgas(lsp)*yylsp(lsp)
           enddo
           rst = p0/rmixtloc/tst
           do lsp=1,N_S
            wmean(i,j,lsp) = rst*yylsp(lsp)
           enddo
           ee = get_e_from_temperature(tst,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,yylsp)
           wmean(i,j,I_U) = rst*ust
           wmean(i,j,I_V) = rst*vst
           wmean(i,j,I_W) = 0._rkind
           wmean(i,j,I_E) = rst*(ee + 0.5_rkind*(ust**2+vst**2))
           if (ii==1) write(10,200) yy,ust,vst,tst,rst,p0,rmixtloc
           if (ii==self%grid%nxmax) write(11,200) yy,ust,vst,tst,rst,p0,rmixtloc
          enddo
         enddo
         close(10)
         close(11)
200 format(20E20.10)
!
         do k=1,nz
          do j=1,ny
           do i=1,nx
            do m=1,self%nv
             w(m,i,j,k) = wmean(i,j,m)
            enddo
           enddo
          enddo
         enddo
!
!        Find the tripping center
         xtrip = xtr1+0.5_rkind*abs(xtr2-xtr1)
!
         errmin = 100.
         do i=1,nx ! pressure side of the airfoil
          error = abs(x(i)-xtrip)
          if (error < errmin) then
           errmin = error
           itr = i
          endif
         enddo
!
!        Center of the tripping
         x0tr = x(itr)        
!
         if (masterproc) then
           if (a_tr > 0.) then
             print*, 'Tripping center', x0tr
             print*, 'Global tripping center nodes    ', itr
           else
             print*, 'No tripping'
           endif
           if (a_tw > 0. .and. abs(v_bs) > 0.) then
            print*, 'Error: traveling waves and blowing/suction cannot be both active'
            call mpi_abort(mpi_comm_world,99,self%mpi_err)
           endif
           if (a_tw > 0.) then
             print*, 'Traveling waves active with amplitude =', a_tw
           elseif (v_bs > 0.) then
             print*, 'Blowing active with velocity =', v_bs
           elseif (v_bs < 0.) then
             print*, 'Suction active with velocity =', v_bs
           endif
           if (a_tw > 0. .or. abs(v_bs) > 0.) then
             print*, 'Start - end of actuated region  ', xtw1,xtw2
           else
             print*, 'No actuation'
           endif
         endif

         lamz_old  = 0._rkind
         lamz1_old = 0._rkind
         phiz_old  = 0._rkind
         phiz1_old = 0._rkind
         is_old    = 0
        endassociate

    endsubroutine init_bl_lam

    subroutine compressible_blasius(n,etad,deta,Mach,Te,Twall,eta,u,v,tbl,delta1)
     integer, intent(in) :: n
     real(rkind), intent(in) :: etad, deta, Mach, Te, Twall
     real(rkind), intent(out) :: delta1
     real(rkind), dimension(0:n), intent(inout) :: u,v,tbl,eta
     real(rkind), dimension(0:n) :: f,t,g,h,a1,a2,a3,a4,a5,a6,a7,a8,s,r,visc,tcr
    !
     integer :: iflag,nm,i,j,kk
     real(rkind) :: Rg,s_suth,eps,z1,z2,z3,gm,adiab,T0e,Ae,Cp,He,H0e,visce,s0,r0,tt,vis
     real(rkind) :: u0,f0,t0,g0,h0
     real(rkind) :: u1,f1,t1,g1,h1
     real(rkind) :: u2,f2,t2,g2,h2
     real(rkind) :: u3,f3,t3,g3,h3
     real(rkind) :: a10,a20,a30,a40,a50,a60,a70,a80
     real(rkind) :: a11,a21,a31,a41,a51,a61,a71,a81
     real(rkind) :: a12,a22,a32,a42,a52,a62,a72,a82
     real(rkind) :: a13,a23,a33,a43,a53,a63,a73,a83
     real(rkind) :: eq1,eq2,b1,b2,b3,b4,det,c1,c2,c3,c4,da,db,dd,rho,delta2
     real(rkind) :: T_ref_dim, gam, Prandtl

     Rg  = 287.15_rkind
     gam = 1.4_rkind
     Prandtl = 0.72_rkind
     s_suth = 110.4_rkind
     eps    = 0.000001_rkind
     z1     = 0.334_rkind
     z2     = 0.82_rkind
     z3     = 0.22_rkind
     iflag  =  1
    !
     nm = n-1
    !
    !  *************************************  fluid conditions
    !
     gm    = gam / ( gam -1._rkind)
     adiab = (1._rkind+ (gam - 1._rkind) / 2._rkind *Mach*Mach)
     T0e   = Te * adiab
     Ae    = sqrt (gam * Rg * Te)
!     Ue    = Mach * ae
     Cp    = gm * Rg
     He    = Cp * Te
     H0e   = Cp * T0e
    !
     if (Te.lt.110.4_rkind) then
      visce = .693873D-6*te
     else
      visce = 1.458D-5 * te**1.5_rkind/( te + s_suth )
     end if
    !
    !  ************************************* initial conditions
    !
     j = 0
     eta(0)  = 0._rkind
     u(0)    = 0._rkind
     h(0)    = 0._rkind
     a1(0)   = 0._rkind
     a2(0)   = 0._rkind
     a3(0)   = 0._rkind
     a5(0)   = 1._rkind
     a6(0)   = 0._rkind
     a7(0)   = 0._rkind
     s0      = z1
     if(iflag.eq.0) then
      g(0)  = 0._rkind
      a4(0) = 1._rkind
      a8(0) = 0._rkind
      r0    = z2
     else
      t(0)  = (Twall - Te )/(T0e-te)
      a4(0) = 0._rkind
      a8(0) = 1._rkind
      r0    = z3
     end if
    !
     do 

      f(0) = s0
      if ( iflag.eq.0)  t(0) = r0
      if ( iflag.eq.1)  g(0) = r0
    !
      Tbl(0)  = Te + (T0e - Te) * t(0)
      tt      = tbl(0)
      if (tt<110.4_rkind) then
       visc(0) = .693873D-6*tt
      else
       visc(0) = 1.458D-5 * tt**1.5_rkind  / ( tt + s_suth )
      end if
      vis     = visc(0) / visce
    !
    !  ******************************** Runge-Kutta integration
    !
      do i = 0,nm
    !  
       u0 =   f(i) / vis * deta
       f0 = - h(i) * f(i) / vis* deta
       t0 =   g(i) * Prandtl / vis * deta
       g0 = -(h(i) * g(i) * Prandtl + 2._rkind * f(i) **2._rkind) / vis * deta
       h0 =  .5_rkind*u(i) / (1 + (T0e/Te - 1) * t(i)) * deta
       u1 =   (f(i) + .5_rkind*f0) / vis * deta
       f1 = - (h(i) + .5_rkind*h0) * (f(i) + .5_rkind*f0) /vis * deta
       t1 =   (g(i) + .5_rkind*g0) * Prandtl / vis * deta
       g1 = -((h(i) + .5_rkind*h0) * (g(i) + .5_rkind*g0) * Prandtl + 2._rkind * (f(i) + .5_rkind*f0)**2._rkind) /vis*deta
       h1 =  .5_rkind*(u(i)+.5_rkind*u0) / (1 + (T0e/Te - 1) * (t(i)+.5_rkind*t0)) *deta
       u2 =   (f(i) + .5_rkind*f1) / vis * deta
       f2 = - (h(i) + .5_rkind*h1) * (f(i) + .5_rkind*f1) /vis * deta
       t2 =   (g(i) + .5_rkind*g1) * Prandtl / vis * deta
       g2 = -((h(i) + .5_rkind*h1) * (g(i) + .5_rkind*g1) * Prandtl + 2._rkind * (f(i) + .5_rkind*f1)**2._rkind) /vis*deta
       h2 =  .5_rkind*(u(i)+.5_rkind*u1) / (1 + (T0e/Te - 1) * (t(i)+.5*t1)) *deta
       u3 =   (f(i) + f2) / vis * deta
       f3 = - (h(i) + h2) * (f(i) + f2) /vis * deta
       t3 =   (g(i) + g2) * Prandtl / vis * deta
       g3 = -((h(i) + h2) * (g(i) + g2) * Prandtl + 2._rkind * (f(i) + f2)**2._rkind) /vis*deta
       h3 = .5_rkind*(u(i)+u2) / (1 + (T0e/Te - 1) *(t(i)+t2))*deta
    !
       a10 =   a5(i)/vis *deta
       a20 =   a6(i) / vis *deta
       a30 =   a7(i) * Prandtl / vis *deta
       a40 =   a8(i) *Prandtl / vis *deta
       a50 = - a5(i) * h(i) / vis *deta
       a60 = - a6(i) * h(i) / vis *deta
       a70 = -(4*f(i)*a5(i)+ Prandtl * h(i) *a7(i))/vis *deta
       a80 = -(4*f(i)*a6(i)+ Prandtl * h(i)*a8(i))/vis *deta
    !
       a11 =   (a5(i) + .5_rkind*a50) / vis *deta
       a21 =   (a6(i) + .5_rkind*a60) / vis *deta
       a31 =   (a7(i) + .5_rkind*a70) * Prandtl / vis *deta
       a41 =   (a8(i) + .5_rkind*a80) * Prandtl / vis *deta
       a51 = - (a5(i) + .5_rkind*a50) * (h(i) + .5_rkind*h0) / vis *deta
       a61 = - (a6(i) + .5_rkind*a60) * (h(i) + .5_rkind*h0) / vis *deta
       a71 = -(4* (f(i) + .5_rkind*f0) * (a5(i) + .5_rkind*a50)+Prandtl * (h(i) + .5_rkind*h0) *(a7(i) + .5_rkind*a70))/vis *deta
       a81 = -(4* (f(i) + .5_rkind*f0) * (a6(i) + .5_rkind*a60)+Prandtl * (h(i) + .5_rkind*h0) *(a8(i) + .5_rkind*a80))/vis *deta
    !
       a12 =   (a5(i) + .5_rkind*a51) / vis *deta
       a22 =   (a6(i) + .5_rkind*a61) / vis *deta
       a32 =   (a7(i) + .5_rkind*a71) * Prandtl / vis *deta
       a42 =   (a8(i) + .5_rkind*a81) * Prandtl / vis *deta
       a52 = - (a5(i) + .5_rkind*a51) * (h(i) + .5_rkind*h1) / vis *deta
       a62 = - (a6(i) + .5_rkind*a61) * (h(i) + .5_rkind*h1) / vis *deta
       a72 = -(4* (f(i) + .5_rkind*f1) * (a5(i) + .5_rkind*a51)+Prandtl * (h(i) + .5_rkind*h1) *(a7(i) + .5_rkind*a71))/vis *deta
       a82 = -(4* (f(i) + .5_rkind*f1) * (a6(i) + .5_rkind*a61)+Prandtl * (h(i) + .5_rkind*h1) *(a8(i) + .5_rkind*a81))/vis *deta
    !
       a13 =   (a5(i) + a52) / vis *deta
       a23 =   (a6(i) + a62) / vis *deta
       a33 =   (a7(i) + a72) * Prandtl / vis *deta
       a43 =   (a8(i) + a82) * Prandtl / vis *deta
       a53 = - (a5(i) + a52) * (h(i) + .5_rkind*h2) / vis *deta
       a63 = - (a6(i) + a62) * (h(i) + .5_rkind*h2) / vis *deta
       a73 = -(4* (f(i) + f2) * (a5(i) + a52) + Prandtl * (h(i) + h2) *(a7(i) + a72))/vis *deta
       a83 = -(4* (f(i) + f2) * (a6(i) + a62) + Prandtl * (h(i) + h2) *(a8(i) + a82))/vis *deta
    !
       f(i+1) = f(i) + (f0 + 2._rkind*f1 + 2*f2 + f3) / 6._rkind
       u(i+1) = u(i) + (u0 + 2._rkind*u1 + 2*u2 + u3) / 6._rkind
       t(i+1) = t(i) + (t0 + 2._rkind*t1 + 2*t2 + t3) / 6._rkind
       g(i+1) = g(i) + (g0 + 2._rkind*g1 + 2*g2 + g3) / 6._rkind
       h(i+1) = h(i) + (h0 + 2._rkind*h1 + 2*h2 + h3) / 6._rkind
    !
       a1(i+1) = a1(i) + (a10 + 2._rkind*a11 + 2._rkind*a12 + a13) / 6._rkind
       a2(i+1) = a2(i) + (a20 + 2._rkind*a21 + 2._rkind*a22 + a23) / 6._rkind
       a3(i+1) = a3(i) + (a30 + 2._rkind*a31 + 2._rkind*a32 + a33) / 6._rkind
       a4(i+1) = a4(i) + (a40 + 2._rkind*a41 + 2._rkind*a42 + a43) / 6._rkind
       a5(i+1) = a5(i) + (a50 + 2._rkind*a51 + 2._rkind*a52 + a53) / 6._rkind
       a6(i+1) = a6(i) + (a60 + 2._rkind*a61 + 2._rkind*a62 + a63) / 6._rkind
       a7(i+1) = a7(i) + (a70 + 2._rkind*a71 + 2._rkind*a72 + a73) / 6._rkind
       a8(i+1) = a8(i) + (a80 + 2._rkind*a81 + 2._rkind*a82 + a83) / 6._rkind
       eta(i+1)  = eta(i) + deta
    !
    !  **************************************   new value of visc  
    !
       Tbl(i+1)  = Te + (T0e - Te) * t(i+1)
       tt      = Tbl(i+1)
       if (tt<110.4_rkind) then
        visc(i+1) = .693873D-6*tt
       else
        visc(i+1) = 1.458D-5 * tt**1.5_rkind  / ( tt + s_suth )
       end if
       vis     = visc(i+1) / visce
    !
      end do
    !
    !  ******************************************* shooting method
    !  ******************************************* with Newton Raphson
    !
      eq1 = 1._rkind - u(n)
      eq2 = - t(n)
      b1  =   a1(n)
      b2  =   a1(n)
      b3  =   a3(n)
      b4  =   a4(n)
      det =   b1 * b4 - b2 * b3
      c1  =   b4 / det
      c2  = - b2 / det
      c3  = - b3 / det
      c4  =   b1 / det
      da  =   c1 * eq1 + c2 * eq2
      db  =   c3 * eq1 + c4 * eq2
      s0  =   s0 + da
      r0  =   r0 + db
      j = j + 1
      if (abs(u(n)-1._rkind)<eps.and.abs(t(n))<eps) exit
     enddo
    !
    !  ********************************* b.l. thickness
    !
     dd = .99_rkind
     do i = 0, nm
      if (u(i).ge.dd) exit
     end do
     delta1 = eta(i-1)
     kk = i - 1
    !
    !******************************* displacement thickness
    !
     delta2 = delta1 -2*h(kk)
    !
    !********************************* Crocco's temperature profile
    !
     do i = 0,n
      tcr(i) = tbl(0)/Te  + (1._rkind - tbl(0)/Te) * u(i) + (gam - 1._rkind) / 2._rkind *Mach**2. * u(i) * (1._rkind - u(i))
     end do
    !
    !********************************************* printing
    !
     do i=0,n
      g1  = .5_rkind*u(i) / tbl(i) * Te
      rho = Te/tbl(i)
      v(i) =  eta(i)*g1 - h(i)
     end do
!
     return     

     end subroutine compressible_blasius

 subroutine hasan_meanprofile(self,n,d99,mach,theta,retau,retheta,redelta,pr,spr,rfac,gam,&
                 ye,ue,te,rhoe,mue,th,cf,ch,mtau,deltav,imode,icompute)
 implicit none
 class(equation_multideal_object), intent(inout) :: self
 type(phase_t) :: mixture_hasan
 !integer, parameter :: rkind = REAL64
 integer, intent(in) :: n,imode,icompute
 real(rkind), intent(in) :: d99,mach,theta,pr,spr,rfac,gam
 real(rkind), intent(inout) :: retau,retheta
 real(rkind), intent(inout) :: cf,ch,th,mtau,deltav,redelta
 real(rkind), dimension(n+1), intent(in) :: ye
 real(rkind), dimension(n+1), intent(out) :: ue,te,rhoe,mue

 integer, parameter :: ny    = 10000
 integer, parameter :: nyext = 2*ny
 integer :: j,niter,nitermax,l
 real(rkind) :: pi
 real(rkind) :: gm1h
 real(rkind) :: rethetaold,retauold
 real(rkind) :: tr,tw,picole,apl,vkc,z1
 real(rkind) :: absdiff,tol
 real(rkind) :: uinf,uinf_star,uinf_plus,ufac
 real(rkind) :: dy,dudy
 real(rkind) :: thint,thint_j,thint_jm
 real(rkind) :: uint,uint_j,uint_jm,uint_inn_j,uint_inn_jm,uint_out_j,uint_out_jm
 real(rkind), dimension(ny+1)    :: y,u,t,rho,mu
 real(rkind), dimension(ny+1)    :: ypl,upl
 real(rkind), dimension(ny+1)    :: yst,d,mut,wake
 real(rkind), dimension(nyext+1) :: yext,uext,text,rhoext,muext
!
 integer :: jj,jjj,m
 real(rkind) :: yy,u_u0
!
 associate(u0 => self%u0, t0 => self%t0, rho0 => self%rho0, p0 => self%p0, mu0 => self%mu0, &
           rmixt0 => self%rmixt0, init_mf => self%init_mf) 
!
 mixture_hasan = importphase('input_cantera.yaml')
!
 pi       = 4._rkind*atan(1._rkind)        ! Greek Pi
 gm1h     = 0.5_rkind*(gam-1._rkind)       ! (gamma-1)/2
 tr       = self%T_recovery                ! T_rec / T_infty
 tw       = self%T_wall                    ! T_wall / T_infty
 apl      = 17._rkind                      ! Van Driest damping
 vkc      = 0.41_rkind                     ! Von Karman constant
 nitermax = 10000                          ! Maximum number of iterations for convergence
 tol      = 0.00001_rkind                  ! Tolerance on retau
!
! Wall-normal grid (uniform)
!
 dy = d99/ny
 do j=1,ny+1
  y(j) = (j-1)*dy
 enddo
 do j=1,nyext+1
  yext(j) = (j-1)*dy
 enddo
!
 ufac = 0.99_rkind
 do j=1,ny+1
  u(j) = y(j)/d99
  u(j) = min(1._rkind,u(j))
 enddo
 uinf = u(ny+1)/ufac
 u    = u/uinf*u0
!
 niter   = 0
 if (imode==0) then ! Find retheta for a fixed retau
  retheta = 1000._rkind ! Initial retheta
 else ! Find retau for a fixed retheta
  retau = 200._rkind    ! Initial retau
 endif
 mtau = 0.1_rkind        ! Initial mtau
!
 absdiff = huge(1._rkind)
 do while (absdiff > tol)
  niter = niter+1
  if (niter>nitermax) exit
  if (imode==0) then
   rethetaold = retheta
  else
   retauold = retau
  endif
  do j=1,ny+1
   u_u0    = u(j)/u0
   t(j)    = tw+(tr-tw)*(spr*u_u0+(1._rkind-spr)*u_u0**2)+(t0-tr)*u_u0**2
   rho(j)  = p0/(rmixt0*t(j))
   call setState_TPY(mixture_hasan,t(j),p0,init_mf)
   mu(j)   = viscosity(mixture_hasan)
!   mu(j)   = t(j)**1.5_rkind*(1._rkind+s2tinf)/(t(j)+s2tinf)
   ypl(j)  = (y(j)/d99)*retau
   yst(j)  = ypl(j)*sqrt(rho(j)/rho(1))*mu(1)/mu(j)
   d(j)    = (1._rkind-exp(-yst(j)/(apl+19.3_rkind*mtau)))**2
   mut(j)  = mu(j)/mu(1)*vkc*yst(j)*d(j)
   z1      = retheta/425._rkind-1._rkind
   picole  = 0.69_rkind*(1._rkind-exp(-0.243_rkind*sqrt(z1)-0.15_rkind*z1))
   wake(j) = picole/vkc*pi*sin(pi*(y(j)/d99))
  enddo
!
! Compute u plus
!
  upl = 0._rkind
  do j=2,ny+1
   uint_inn_j   = 1._rkind/(mu(j)/mu(1)+mut(j))
   uint_inn_jm  = 1._rkind/(mu(j-1)/mu(1)+mut(j-1))
   uint_out_j   = sqrt(rho(1)/rho(j))/retau*wake(j)
   uint_out_jm  = sqrt(rho(1)/rho(j-1))/retau*wake(j-1)
   uint_j       = uint_inn_j +uint_out_j
   uint_jm      = uint_inn_jm+uint_out_jm
   uint         = 0.5_rkind*(uint_j+uint_jm)
   upl(j)       = upl(j-1)+uint*(ypl(j)-ypl(j-1))
  enddo
  uinf_plus = upl(ny+1)/ufac
  u         = upl/uinf_plus*u0
!
  th = 0._rkind
  do j=2,ny+1
   thint_j  = rho(j)/rho0*u(j)/u0*(1._rkind-u(j)/u0)
   thint_jm = rho(j-1)/rho0*u(j-1)/u0*(1._rkind-u(j-1)/u0)
   thint    = 0.5_rkind*(thint_j+thint_jm)
   th       = th+thint*(y(j)-y(j-1))
  enddo
!
  cf = 2._rkind/uinf_plus**2*rho(1)/rho0
  ch = 0.5_rkind*cf*spr/pr
  mtau  = mach*sqrt(0.5_rkind*cf)
  if (imode==0) then
   retheta = retau*th/d99*(mu(1)/mu0)*(rho0/rho(1))*uinf_plus
   absdiff = abs(retheta-rethetaold)
  else
   retau = retheta*d99/th*(mu0/mu(1))*(rho(1)/rho0)/uinf_plus
   absdiff = abs(retau-retauold)
  endif
  redelta = retheta*d99/th
 enddo
!
 deltav = d99/retau
 if (niter>nitermax) write(*,*) 'Maximum number of iterations in Hasan!',absdiff
!
 do j=1,ny+1
  uext(j) = u(j)
  text(j) = t(j)
 enddo
 dudy = (uext(ny+1)-uext(ny))/(yext(ny+1)-yext(ny))
 do j=ny+2,nyext+1
  uext(j) = uext(ny+1)+dudy*(yext(j)-yext(ny+1))
  uext(j) = min(uext(j),u0)
  u_u0    = uext(j)/u0
  text(j) = tw+(tr-tw)*(spr*u_u0+(1._rkind-spr)*u_u0**2)+(t0-tr)*u_u0**2
 enddo
 do j=1,nyext+1
  rhoext(j)  = p0/(rmixt0*text(j))
!  muext(j)   = text(j)**1.5_rkind*(1._rkind+s2tinf)/(text(j)+s2tinf)
  call setState_TPY(mixture_hasan,text(j),p0,init_mf)
  muext(j)   = viscosity(mixture_hasan)
 enddo
 ue   = u0 
 rhoe = rho0
 te   = t0
 mue  = mu0
 if (icompute/=0) then
  ue(1)   = uext(1)
  rhoe(1) = rhoext(1)
  te(1)   = text(1)
  mue(1)  = muext(1)
  do j=1,n
   yy = ye(j)
   if (yy<yext(nyext+1)) then
    call locateval(yext,nyext+1,yy,jj)
    m = 2
    jjj = min(max(jj-(m-1)/2,1),nyext+1+1-m)
    call pol_int(yext(jjj),uext(jjj),m,yy,ue(j))
    call pol_int(yext(jjj),rhoext(jjj),m,yy,rhoe(j))
    call pol_int(yext(jjj),text(jjj),m,yy,te(j))
    call pol_int(yext(jjj),muext(jjj),m,yy,mue(j))
   endif
  enddo
 endif
!
 return
 endassociate
 endsubroutine hasan_meanprofile

     subroutine init_bl(self)
        class(equation_multideal_object), intent(inout) :: self

        real(rkind), dimension(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1) :: thvec, retauvec
        real(rkind), dimension(self%field%ny) :: yvec, uvec, rhovec, tvec, viscvec
        real(rkind), dimension(3) :: rr
        real(rkind) :: spr
        real(rkind) :: cf,ch,mtau,deltav
        real(rkind) :: retheta,retau,redelta,retauold,delta,th,retheta_inflow,deltaold
        real(rkind) :: vi,vi_j,vi_jm
        real(rkind) :: rho,uu,vv,ww,rhouu,rhovv,rhoww,ee,tt
        real(rkind) :: u0_02
        integer :: i,j,k,ii,imode,icompute,lsp,counter
        logical :: file_exists

        ! ghost only on x dir
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))
        allocate(self%deltavec(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1))
        allocate(self%deltavvec(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1))
        allocate(self%cfvec(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, &
                  y => self%field%y, z => self%field%z, Reynolds_friction => self%Reynolds_friction, &
                  xg => self%grid%xg, nxmax => self%grid%nxmax, &
                  rho0 => self%rho0, u0 => self%u0, p0 => self%p0, gm => self%gm, &
                  wmean => self%wmean, w => self%field%w, Mach => self%Mach, gam => self%gam, &
                  rfac => self%rfac, Prandtl => self%Prandtl, w_aux => self%w_aux,  &
                  deltavec => self%deltavec ,deltavvec => self%deltavvec, cfvec => self%cfvec, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  rmixt0 => self%rmixt0, t0 => self%t0, delta0 => self%delta0, &
                  jbl_inflow => self%jbl_inflow, theta_wall => self%theta_wall)

        call locateval(y(1:ny),ny,delta0,jbl_inflow) ! l0 is between yvec(jbl_inflow) and yvec(jbl_inflow+1)
        spr = 0.8_rkind
        yvec = y(1:ny)

        inquire(file='blvec.bin',exist=file_exists)
        if (file_exists) then
         open(183,file='blvec.bin',form='unformatted')
         read(183) cfvec,thvec,deltavec,deltavvec
         close(183)
        else
         imode = 0
         icompute = 0
         call hasan_meanprofile(self,ny,delta0,Mach,theta_wall,Reynolds_friction,retheta,redelta,Prandtl,spr,rfac,gam,&
                                yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
         retheta_inflow = retheta
         deltavec(1)  = delta0
         deltavvec(1) = deltav
         cfvec(1)     = cf
         thvec(1)     = th
         retauvec(1)  = Reynolds_friction

         retheta = retheta_inflow
         do i=1,ng
           thvec(1-i) = thvec(2-i)-0.5_rkind*abs((xg(1-i)-xg(2-i)))*cfvec(2-i)
           retheta    = retheta/thvec(2-i)*thvec(1-i)
           delta      = deltavec(2-i)/thvec(2-i)*thvec(1-i)
           do
            deltaold = delta
            imode = 1 
            icompute = 0
            call hasan_meanprofile(self,ny,delta,Mach,theta_wall,retau,retheta,redelta,Prandtl,spr,rfac,gam,&
                                   yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
            delta = deltavec(2-i)*retau/retauvec(2-i)*sqrt(cfvec(2-i)/cf)
            if (abs(delta-deltaold)<0.000000001_rkind) exit
           enddo
           deltavec (1-i) = delta
           cfvec    (1-i) = cf
           deltavvec(1-i) = deltav
           retauvec (1-i) = retau
         enddo

         if (self%masterproc) open(182,file='cfstart.dat')
         if (self%masterproc) write(182,100) xg(1),deltavec(1),deltavvec(1),cfvec(1),thvec(1)
         retheta = retheta_inflow
         do i=2,nxmax+ng+1
          thvec(i) = thvec(i-1)+0.5_rkind*abs((xg(i)-xg(i-1)))*cfvec(i-1)
          retheta  = retheta/thvec(i-1)*thvec(i)
          delta    = deltavec(i-1)/thvec(i-1)*thvec(i)
          do 
           deltaold = delta
           imode = 1
           icompute = 0
           call hasan_meanprofile(self,ny,delta,Mach,theta_wall,retau,retheta,redelta,Prandtl,spr,rfac,gam,&
                                  yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
           delta = deltavec(i-1)*retau/retauvec(i-1)*sqrt(cfvec(i-1)/cf)
           if (abs(delta-deltaold)<0.000000001_rkind) exit
          enddo
          deltavec(i)  = delta
          cfvec    (i) = cf
          deltavvec(i) = deltav
          retauvec (i) = retau
          if (self%masterproc) write(182,100) xg(i),delta,deltav,cf,th
100  format(20ES20.10)
         enddo

         if (self%masterproc) close(182)
         if (self%masterproc) then
          open(183,file='blvec.bin',form='unformatted')
          write(183) cfvec,thvec,deltavec,deltavvec
          close(183)
         endif
        endif
!
!        Compute locally wmean from 1-ng to nx+ng+1
!
         wmean = 0._rkind
         do i=1-ng,nx+ng+1
          ii = self%field%ncoords(1)*nx+i
          delta  = deltavec(ii)
          deltav = deltavvec(ii)
          retau  = delta/deltav
          imode = 0
          icompute = 1
          call hasan_meanprofile(self,ny,delta,Mach,theta_wall,retau,retheta,redelta,Prandtl,spr,rfac,gam,&
                                 yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
          do j=1,ny
           do lsp=1,N_S
            wmean(i,j,lsp) = rhovec(j)*self%init_mf(lsp)
           enddo
           wmean(i,j,I_U)  = rhovec(j)*uvec(j)
          enddo
         enddo

         do i=1-ng,nx+ng
          ii = self%field%ncoords(1)*nx+i
          do j=2,ny
           vi_j  = -(wmean(i+1,j,I_U)-wmean(i,j,I_U))/(xg(ii+1)-xg(ii))
           vi_jm = -(wmean(i+1,j-1,I_U)-wmean(i,j-1,I_U))/(xg(ii+1)-xg(ii))
           vi    = 0.5_rkind*(vi_j+vi_jm)
           wmean(i,j,I_V) = wmean(i,j-1,I_V)+vi*(y(j)-y(j-1))
          enddo
         enddo

         u0_02 = 0.02_rkind*u0
         if (self%rand_type==0) u0_02 = 0._rkind

         do k=1,nz
          do j=1,ny
           do i=1,nx
            rho = wmean(i,j,1)
            do lsp=2,N_S
             rho = rho + wmean(i,j,lsp)
            enddo
            call get_crandom_f(rr)
!           rr = 0.5_rkind
            rr = rr-0.5_rkind
            uu  = wmean(i,j,I_U)/rho+u0_02*rr(1)
            vv  = wmean(i,j,I_V)/rho+u0_02*rr(2)
            ww  = wmean(i,j,I_W)/rho+u0_02*rr(3)

            rhouu = rho*uu
            rhovv = rho*vv
            rhoww = rho*ww
            do lsp=1,N_S
             w(lsp,i,j,k) = rho*self%init_mf(lsp)
            enddo
            w(I_U,i,j,k) = rhouu
            w(I_V,i,j,k) = rhovv
            w(I_W,i,j,k) = rhoww
            tt           = p0/rmixt0/rho
            w_aux(i,j,k,J_T) = tt
            ee = get_e_from_temperature(tt,indx_cp_l,indx_cp_r,cv_coeff,self%nsetcv,self%trange,self%init_mf)
            w(I_E,i,j,k) = rho*ee + 0.5_rkind*(rhouu**2+rhovv**2+rhoww**2)/rho
           enddo
          enddo
         enddo
         if (self%masterproc) then
          open(183,file='bl1.dat',form='formatted')
          do j=1,ny
           rho = wmean(1,j,1)
           do lsp=2,N_S
            rho = rho + wmean(1,j,lsp)
           enddo
           write(183,*) y(j),rho,wmean(1,j,I_U)/rho
          enddo
          close(183)
         endif

         call self%add_synthetic_perturbations()

        endassociate
     endsubroutine init_bl 

     subroutine add_synthetic_perturbations(self)
        class(equation_multideal_object), intent(inout) :: self
        real(rkind), dimension(:,:), allocatable :: synth_params
        real(rkind) :: rho_wall, tau_wall, u_tau, lz_plus, rr, rhofac, up, vp, wp
        real(rkind) :: arg_sin, arg_cos, ys , ufy , vfy , dup_dx , dvfy_dy, dvp_dy
        real(rkind) :: rho,ee,tt
        real(rkind) :: delta, u0_03, yy
        real(rkind), dimension(3) :: rr3
        integer :: i,j,k,l,n_streaks,ii,lsp

        allocate(synth_params(5,7))

        synth_params(:,1) = [  12._rkind,   0.3_rkind,  0.45_rkind,   0.6_rkind,   0.5_rkind]
        synth_params(:,2) = [  1.2_rkind,   0.3_rkind,   0.2_rkind,  0.08_rkind,  0.04_rkind]
        synth_params(:,3) = [-0.25_rkind, -0.06_rkind, -0.05_rkind, -0.04_rkind, -0.03_rkind]
        synth_params(:,4) = [ 0.12_rkind,   1.2_rkind,   0.6_rkind,   0.4_rkind,   0.2_rkind]
        synth_params(:,5) = [ 10.0_rkind,   0.9_rkind,   0.9_rkind,   0.9_rkind,   0.9_rkind]
        synth_params(:,6) = [120.0_rkind, 0.333_rkind,  0.25_rkind,   0.2_rkind, 0.166_rkind]
        synth_params(:,7) = [  0.0_rkind,    1._rkind,    1._rkind,    1._rkind,    1._rkind]

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, &
                  deltavec => self%deltavec, deltavvec => self%deltavvec, wmean => self%wmean, cfvec => self%cfvec, &
                  rho0 => self%rho0, u0 => self%u0, lz => self%grid%domain_size(3), w => self%field%w, &
                  x => self%field%x, y => self%field%y, z => self%field%z, p0 => self%p0, gm => self%gm, &
                  w_aux => self%w_aux, indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  trange => self%trange, nsetcv => self%nsetcv, rmixt0 => self%rmixt0, t0 => self%t0, &
                  init_mf => self%init_mf, delta0 => self%delta0)

        rho_wall = wmean(1,1,1)
        do lsp=2,N_S
         rho_wall = rho_wall + wmean(1,1,lsp)
        enddo
        tau_wall = cfvec(1)*rho0*u0**2*0.5_rkind
        u_tau    = sqrt(tau_wall/rho_wall)

        synth_params(1,1) = synth_params(1,1) * deltavvec(1)
        synth_params(1,4) = synth_params(1,4) * u_tau / deltavvec(1)
        synth_params(1,5) = synth_params(1,5) * u_tau

        lz_plus   = lz/deltavvec(1)
        n_streaks = nint(lz_plus / synth_params(1,6))
        synth_params(1,6) = lz / n_streaks

        do l=2,5
            synth_params(l,1) = synth_params(l,1) * deltavec(1)
            synth_params(l,4) = synth_params(l,4) * u0 / deltavec(1)
            synth_params(l,5) = synth_params(l,5) * u0
            synth_params(l,6) = synth_params(l,6) * lz
            call get_crandom_f(rr)
!           rr = 0._rkind
            synth_params(l,7) = synth_params(l,7) * 2._rkind*pi *rr
        enddo
        ! random must be synced across processes
        call mpi_bcast(synth_params(2:5,7),4,mpi_prec,0,self%field%mp_cart,self%mpi_err)

        u0_03 = 0.03_rkind*u0
        if (self%rand_type==0) u0_03 = 0._rkind
        do k=1,nz
         do j=1,ny
          do i=1,nx
!
           yy = y(j)
!
           rho      = wmean(i,j,1)
           rho_wall = wmean(i,1,1)
           do lsp=2,N_S
            rho      = rho      + wmean(i,j,lsp)
            rho_wall = rho_wall + wmean(i,1,lsp)
           enddo
           rhofac  = rho_wall/rho
           rhofac  = sqrt(rhofac)*u0
           up      = 0._rkind
           vp      = 0._rkind
           wp      = 0._rkind
           do l=2,5
            arg_sin = synth_params(l,4)*x(i)/synth_params(l,5)
            arg_cos = 2._rkind*pi*z(k)/synth_params(l,6) + synth_params(l,7)
            ys      = yy/synth_params(l,1)
            ufy     = ys    * exp(-ys)
            vfy     = ys**2 * exp(-(ys**2))
            up      = up + synth_params(l,2) * ufy * sin(arg_sin) * cos(arg_cos)
            vp      = vp + synth_params(l,3) * vfy * sin(arg_sin) * cos(arg_cos)

            dup_dx  = synth_params(l,2) * ufy * synth_params(l,4)/synth_params(l,5) * cos(arg_sin)
            dvfy_dy = 2._rkind*ys/synth_params(l,1)*exp(-(ys**2))*(1._rkind-ys**2)
            dvp_dy  = synth_params(l,3) * dvfy_dy * sin(arg_sin)
            wp      = wp-(dup_dx+dvp_dy)*sin(arg_cos)*synth_params(l,6)/(2._rkind*pi)
           enddo
           up = up * rhofac
           vp = vp * rhofac
           wp = wp * rhofac
!
           ii     = self%field%ncoords(1)*nx+i
           delta  = deltavec(ii)
           if (yy<delta) then
            call get_crandom_f(rr3)
            rr3 = rr3-0.5_rkind
            up = up+u0_03*rr3(1)*(yy/delta0)
            vp = vp+u0_03*rr3(2)*(yy/delta0)
            wp = wp+u0_03*rr3(3)*(yy/delta0)
           endif
!
           rho = w(1,i,j,k)
           do lsp=2,N_S
            rho = rho + w(lsp,i,j,k)
           enddo          
           w(I_U,i,j,k) = w(I_U,i,j,k) + rho*up
           w(I_V,i,j,k) = w(I_V,i,j,k) + rho*vp
           w(I_W,i,j,k) = w(I_W,i,j,k) + rho*wp
           tt           = p0/rmixt0/rho
           w_aux(i,j,k,J_T) = tt
           ee = get_e_from_temperature(tt,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,init_mf)
           w(I_E,i,j,k) = rho*ee + 0.5_rkind*(w(I_U,i,j,k)**2 + w(I_V,i,j,k)**2 + w(I_W,i,j,k)**2)/rho
!
          enddo
         enddo
        enddo
        endassociate

     endsubroutine add_synthetic_perturbations

     subroutine init_double_bl(self)
        class(equation_multideal_object), intent(inout) :: self

        real(rkind), dimension(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1) :: thvec, retauvec
        real(rkind), dimension(self%field%ny) :: yvec, uvec, rhovec, tvec, viscvec
        real(rkind), dimension(3) :: rr
        real(rkind) :: spr
        real(rkind) :: cf,ch,mtau,deltav
        real(rkind) :: retheta,retau,redelta,retauold,delta,th,retheta_inflow,deltaold
        real(rkind) :: vi,vi_j,vi_jm
        real(rkind) :: rho,uu,vv,ww,rhouu,rhovv,rhoww,ee,tt
        real(rkind) :: u0_02
        integer :: i,j,k,ii,imode,icompute,lsp,counter,jj
        logical :: file_exists
        real(rkind), dimension(self%field%ny) :: dist
        real(rkind), dimension(:,:,:), allocatable :: wmean1,wmean2 
        real(rkind) :: delta0_1,retau_1,theta_wall_1
        real(rkind) :: delta0_2,retau_2,theta_wall_2
        real(rkind) :: f, tanhly, a, b, L

        ! ghost only on x dir
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))
        allocate(self%deltavec(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1))
        allocate(self%deltavvec(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1))
        allocate(self%cfvec(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1))
        allocate(self%deltavec2(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1))
        allocate(self%deltavvec2(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1))
        allocate(self%cfvec2(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, &
                  y => self%field%y, z => self%field%z, Reynolds_friction => self%Reynolds_friction, &
                  xg => self%grid%xg, nxmax => self%grid%nxmax, Ly => self%grid%domain_size(2), &
                  rho0 => self%rho0, u0 => self%u0, p0 => self%p0, gm => self%gm, &
                  wmean => self%wmean, w => self%field%w, Mach => self%Mach, gam => self%gam, &
                  rfac => self%rfac, Prandtl => self%Prandtl, w_aux => self%w_aux,  &
                  deltavec => self%deltavec ,deltavvec => self%deltavvec, cfvec => self%cfvec, &
                  deltavec2 => self%deltavec2 ,deltavvec2 => self%deltavvec2, cfvec2 => self%cfvec2, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  rmixt0 => self%rmixt0, t0 => self%t0, delta0 => self%delta0, &
                  Reynolds_friction2 => self%Reynolds_friction2, delta02 => self%delta02, &
                  jbl_inflow => self%jbl_inflow, theta_wall => self%theta_wall)

        ! already read: delta0, retau, theta_wall
        ! need to be read: delta0_sup, retau_sup, theta_wall_sup


        allocate(wmean1(1-ng:nx+ng+1, 1:ny, 3)) ; wmean1 = 0._rkind
        allocate(wmean2(1-ng:nx+ng+1, 1:ny, 3)) ; wmean2 = 0._rkind

        delta0_1     = delta0            ;  delta0_2     = delta02
        retau_1      = Reynolds_friction ;  retau_2      = Reynolds_friction2
        theta_wall_1 = theta_wall        ;  theta_wall_2 = theta_wall ! only walls at the same temperature supported

        if (self%masterproc) print *, 'Initializing bl #1...'
        call self%compute_bl(1,delta0_1,retau_1,theta_wall_1,wmean1,deltavec ,deltavvec ,cfvec)
        if (self%masterproc) print *, 'Done with bl #1'
        if (self%masterproc) print *, 'Initializing bl #2...'
        call self%compute_bl(2,delta0_2,retau_2,theta_wall_2,wmean2,deltavec2,deltavvec2,cfvec2)
        if (self%masterproc) print *, 'Done with bl #2'

        wmean = 0._rkind
        ! blending
        do i=1-ng,nx+ng+1
         ii = self%field%ncoords(1)*nx+i
         a = min(deltavec (ii),0.8_rkind*Ly/2._rkind) ! ensuring enough space for blending
         b = min(deltavec2(ii),0.8_rkind*Ly/2._rkind) ! ensuring enough space for blending
         b = Ly-b
         L = b - a 
         do j=1,ny
          jj = ny+1-j

          if (y(j) <= a) then
            f = 1._rkind
          elseif ( a < y(j) .and. y(j) < b) then
            f = 0.5_rkind * ( 1._rkind - tanh (2.5_rkind/L*(y(j)-a-L/2._rkind))/tanh(2.5_rkind/L*L/2._rkind) )
          else
            f = 0._rkind
          endif

          rho   = f*wmean1(i,j,1) + (1-f)*wmean2(i,jj,1)
          rhouu = f*wmean1(i,j,2) + (1-f)*wmean2(i,jj,2)
          rhovv = f*wmean1(i,j,3) - (1-f)*wmean2(i,jj,3)

          do lsp=1,N_S
           wmean(i,j,lsp) = rho*self%init_mf(lsp)
          enddo
          wmean(i,j,I_U) = rhouu
          wmean(i,j,I_V) = rhovv
         enddo
        enddo

        u0_02 = 0.02_rkind*u0
        if (self%rand_type==0) u0_02 = 0._rkind

        do k=1,nz
         do j=1,ny
          do i=1,nx
           rho = wmean(i,j,1)
           do lsp=2,N_S
            rho = rho + wmean(i,j,lsp)
           enddo
           call get_crandom_f(rr)
!          rr = 0.5_rkind
           rr = rr-0.5_rkind
           uu  = wmean(i,j,I_U)/rho+u0_02*rr(1)
           vv  = wmean(i,j,I_V)/rho+u0_02*rr(2)
           ww  = wmean(i,j,I_W)/rho+u0_02*rr(3)

           rhouu = rho*uu
           rhovv = rho*vv
           rhoww = rho*ww
           do lsp=1,N_S
            w(lsp,i,j,k) = rho*self%init_mf(lsp)
           enddo
           w(I_U,i,j,k) = rhouu
           w(I_V,i,j,k) = rhovv
           w(I_W,i,j,k) = rhoww
           tt           = p0/rmixt0/rho
           w_aux(i,j,k,J_T) = tt
           ee = get_e_from_temperature(tt,indx_cp_l,indx_cp_r,cv_coeff,self%nsetcv,self%trange,self%init_mf)
           w(I_E,i,j,k) = rho*ee + 0.5_rkind*(rhouu**2+rhovv**2+rhoww**2)/rho
          enddo
         enddo
        enddo

        open(183,file='bl1.dat',form='formatted')
        open(184,file='blnx.dat',form='formatted')
        do i=1,nx
         ii = self%field%ncoords(1)*nx+i
         if     (ii==1) then
          do j=1,ny
          rho = wmean(1,j,1)
          do lsp=2,N_S
           rho = rho + wmean(1,j,lsp)
          enddo
          write(183,100) y(j),rho,wmean(1,j,I_U)/rho,wmean(1,j,I_V)/rho
         enddo
         elseif (ii==nxmax) then
          do j=1,ny
           rho = wmean(nx,j,1)
           do lsp=2,N_S
            rho = rho + wmean(nx,j,lsp)
           enddo
           write(184,100) y(j),rho,wmean(nx,j,I_U)/rho,wmean(nx,j,I_V)/rho
          enddo
         endif
        enddo
        close(183)
        close(184)

100  format(20ES20.10)

        deallocate(wmean1,wmean2)

        call self%add_synthetic_perturbations_double_bl()
!        call self%add_synthetic_perturbations_double_bl(2)

        endassociate
     endsubroutine init_double_bl

     subroutine compute_bl(self,idx,delta0,retau0,theta_wall,wmean_loc,deltavec_loc,deltavvec_loc,cfvec_loc)
         class(equation_multideal_object), intent(inout) :: self

        integer, intent(in) :: idx
        real(rkind), intent(in) :: delta0,retau0,theta_wall
        real(rkind), dimension(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1) :: thvec, retauvec
        real(rkind), intent(inout), dimension(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, 3) :: wmean_loc
        real(rkind), intent(inout), dimension(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1) :: deltavec_loc,deltavvec_loc,cfvec_loc
        real(rkind), dimension(self%field%ny) :: yvec, uvec, rhovec, tvec, viscvec
        real(rkind), dimension(3) :: rr
        real(rkind) :: spr
        real(rkind) :: cf,ch,mtau,deltav
        real(rkind) :: retheta,retau,redelta,retauold,delta,th,retheta_inflow,deltaold
        real(rkind) :: vi,vi_j,vi_jm
        real(rkind) :: rho,uu,vv,ww,rhouu,rhovv,rhoww,ee,tt
        real(rkind) :: u0_02
        integer :: i,j,k,ii,imode,icompute,lsp,counter
        real(rkind) :: Reynolds_friction
        logical :: file_exists
        character(2) :: chblnum

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, &
                  y => self%field%y, z => self%field%z, &
                  xg => self%grid%xg, nxmax => self%grid%nxmax, &
                  rho0 => self%rho0, u0 => self%u0, p0 => self%p0, gm => self%gm, &
                  w => self%field%w, Mach => self%Mach, gam => self%gam, &
                  rfac => self%rfac, Prandtl => self%Prandtl, w_aux => self%w_aux,  &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  rmixt0 => self%rmixt0, t0 => self%t0, &
                  jbl_inflow => self%jbl_inflow)

        Reynolds_friction = retau0
        write(chblnum,'(I2.2)') idx

        call locateval(y(1:ny),ny,delta0,jbl_inflow) ! l0 is between yvec(jbl_inflow) and yvec(jbl_inflow+1)
        spr = 0.8_rkind
        yvec = y(1:ny)

        inquire(file='blvec_'//chblnum//'.bin',exist=file_exists)
        if (file_exists) then
         open(183,file='blvec_'//chblnum//'.bin',form='unformatted')
         read(183) cfvec_loc,thvec,deltavec_loc,deltavvec_loc
         close(183)
        else
         imode = 0
         icompute = 0
         call hasan_meanprofile(self,ny,delta0,Mach,theta_wall,Reynolds_friction,retheta,redelta,Prandtl,spr,rfac,gam,&
                                yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
         retheta_inflow   = retheta
         deltavec_loc(1)  = delta0
         deltavvec_loc(1) = deltav
         cfvec_loc(1)     = cf
         thvec(1)         = th
         retauvec(1)      = Reynolds_friction

         retheta = retheta_inflow
         do i=1,ng
           thvec(1-i) = thvec(2-i)-0.5_rkind*abs((xg(1-i)-xg(2-i)))*cfvec_loc(2-i)
           retheta    = retheta/thvec(2-i)*thvec(1-i)
           delta      = deltavec_loc(2-i)/thvec(2-i)*thvec(1-i)
           do
            deltaold = delta
            imode = 1 
            icompute = 0
            call hasan_meanprofile(self,ny,delta,Mach,theta_wall,retau,retheta,redelta,Prandtl,spr,rfac,gam,&
                                   yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
            delta = deltavec_loc(2-i)*retau/retauvec(2-i)*sqrt(cfvec_loc(2-i)/cf)
            if (abs(delta-deltaold)<0.000000001_rkind) exit
           enddo
           deltavec_loc (1-i) = delta
           cfvec_loc    (1-i) = cf
           deltavvec_loc(1-i) = deltav
           retauvec     (1-i) = retau
         enddo

         if (self%masterproc) open(182,file='cfstart_'//chblnum//'.dat')
         if (self%masterproc) write(182,100) xg(1),deltavec_loc(1),deltavvec_loc(1),cfvec_loc(1),thvec(1)
         retheta = retheta_inflow
         do i=2,nxmax+ng+1
          thvec(i) = thvec(i-1)+0.5_rkind*abs((xg(i)-xg(i-1)))*cfvec_loc(i-1)
          retheta  = retheta/thvec(i-1)*thvec(i)
          delta    = deltavec_loc(i-1)/thvec(i-1)*thvec(i)
          do 
           deltaold = delta
           imode = 1
           icompute = 0
           call hasan_meanprofile(self,ny,delta,Mach,theta_wall,retau,retheta,redelta,Prandtl,spr,rfac,gam,&
                                  yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
           delta = deltavec_loc(i-1)*retau/retauvec(i-1)*sqrt(cfvec_loc(i-1)/cf)
           if (abs(delta-deltaold)<0.000000001_rkind) exit
          enddo
          deltavec_loc (i) = delta
          cfvec_loc    (i) = cf
          deltavvec_loc(i) = deltav
          retauvec     (i) = retau
          if (self%masterproc) write(182,100) xg(i),delta,deltav,cf,th
100  format(20ES20.10)
         enddo

         if (self%masterproc) close(182)
         if (self%masterproc) then
          open(183,file='blvec_'//chblnum//'.bin',form='unformatted')
          write(183) cfvec_loc,thvec,deltavec_loc,deltavvec_loc
          close(183)
         endif
        endif
!
!        Compute locally wmean from 1-ng to nx+ng+1
!
         wmean_loc = 0._rkind
         do i=1-ng,nx+ng+1
          ii = self%field%ncoords(1)*nx+i
          delta  = deltavec_loc(ii)
          deltav = deltavvec_loc(ii)
          retau  = delta/deltav
          imode = 0
          icompute = 1
          call hasan_meanprofile(self,ny,delta,Mach,theta_wall,retau,retheta,redelta,Prandtl,spr,rfac,gam,&
                                 yvec,uvec,tvec,rhovec,viscvec,th,cf,ch,mtau,deltav,imode,icompute)
          do j=1,ny
           wmean_loc(i,j,1) = rhovec(j)
           wmean_loc(i,j,2) = rhovec(j)*uvec(j)
          enddo
         enddo

         do i=1-ng,nx+ng
          ii = self%field%ncoords(1)*nx+i
          do j=2,ny
           vi_j  = -(wmean_loc(i+1,j,2)-wmean_loc(i,j,2))/(xg(ii+1)-xg(ii))
           vi_jm = -(wmean_loc(i+1,j-1,2)-wmean_loc(i,j-1,2))/(xg(ii+1)-xg(ii))
           vi    = 0.5_rkind*(vi_j+vi_jm)
           wmean_loc(i,j,3) = wmean_loc(i,j-1,3)+vi*(y(j)-y(j-1))
          enddo
         enddo

    endassociate

    endsubroutine compute_bl

     subroutine add_synthetic_perturbations_double_bl(self)
        class(equation_multideal_object), intent(inout) :: self
        real(rkind), dimension(:,:), allocatable :: synth_params,synth_params1,synth_params2
        real(rkind), dimension(5,7) :: synth_params_init
        real(rkind), dimension(1-self%grid%ng:self%grid%nxmax+self%grid%ng+1) :: deltavec_loc
        real(rkind) :: delta0_loc
        real(rkind) :: rho_wall1, tau_wall1, u_tau1, lz_plus1  
        real(rkind) :: rho_wall2, tau_wall2, u_tau2, lz_plus2  
        real(rkind) :: rr, rhofac, up, vp, wp, rho_wall
        real(rkind) :: arg_sin, arg_cos, ys , ufy , vfy , dup_dx , dvfy_dy, dvp_dy
        real(rkind) :: rho,ee,tt
        real(rkind) :: delta, u0_03, yy
        real(rkind), dimension(3) :: rr3
        integer :: i,j,k,l,n_streaks1,n_streaks2,ii,lsp,jwall

        allocate(synth_params1(5,7),synth_params2(5,7),synth_params(5,7))

        synth_params_init(:,1) = [  12._rkind,   0.3_rkind,  0.45_rkind,   0.6_rkind,   0.5_rkind]
        synth_params_init(:,2) = [  1.2_rkind,   0.3_rkind,   0.2_rkind,  0.08_rkind,  0.04_rkind]
        synth_params_init(:,3) = [-0.25_rkind, -0.06_rkind, -0.05_rkind, -0.04_rkind, -0.03_rkind]
        synth_params_init(:,4) = [ 0.12_rkind,   1.2_rkind,   0.6_rkind,   0.4_rkind,   0.2_rkind]
        synth_params_init(:,5) = [ 10.0_rkind,   0.9_rkind,   0.9_rkind,   0.9_rkind,   0.9_rkind]
        synth_params_init(:,6) = [120.0_rkind, 0.333_rkind,  0.25_rkind,   0.2_rkind, 0.166_rkind]
        synth_params_init(:,7) = [  0.0_rkind,    1._rkind,    1._rkind,    1._rkind,    1._rkind]

        synth_params  = 0._rkind ; synth_params1 = 0._rkind ; synth_params2 = 0._rkind
        synth_params1 = synth_params_init ; synth_params2 = synth_params_init

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, &
                  wmean => self%wmean, init_mf => self%init_mf, &
                  rho0 => self%rho0, u0 => self%u0, lz => self%grid%domain_size(3), w => self%field%w, &
                  x => self%field%x, y => self%field%y, z => self%field%z, p0 => self%p0, gm => self%gm, &
                  w_aux => self%w_aux, indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  trange => self%trange, nsetcv => self%nsetcv, rmixt0 => self%rmixt0, t0 => self%t0)

        rho_wall1 = wmean(1,1 ,1)
        rho_wall2 = wmean(1,ny,1)
        do lsp=2,N_S
         rho_wall1 = rho_wall1 + wmean(1,1 ,lsp)
         rho_wall2 = rho_wall2 + wmean(1,ny,lsp)
        enddo
        tau_wall1 = self%cfvec (1)*rho0*u0**2*0.5_rkind
        tau_wall2 = self%cfvec2(1)*rho0*u0**2*0.5_rkind
        u_tau1    = sqrt(tau_wall1/rho_wall1)
        u_tau2    = sqrt(tau_wall2/rho_wall2)

        synth_params1(1,1) = synth_params1(1,1) * self%deltavvec (1)
        synth_params2(1,1) = synth_params2(1,1) * self%deltavvec2(1)
        synth_params1(1,4) = synth_params1(1,4) * u_tau1 / self%deltavvec (1)
        synth_params2(1,4) = synth_params2(1,4) * u_tau2 / self%deltavvec2(1)
        synth_params1(1,5) = synth_params1(1,5) * u_tau1
        synth_params2(1,5) = synth_params2(1,5) * u_tau2

        lz_plus1   = lz/self%deltavvec (1)
        lz_plus2   = lz/self%deltavvec2(1)
        n_streaks1 = nint(lz_plus1 / synth_params1(1,6))
        n_streaks2 = nint(lz_plus2 / synth_params2(1,6))
        synth_params1(1,6) = lz  / n_streaks1
        synth_params2(1,6) = lz  / n_streaks2

        do l=2,5
            synth_params1(l,1) = synth_params1(l,1) * self%deltavec (1)
            synth_params2(l,1) = synth_params2(l,1) * self%deltavec2(1)
            synth_params1(l,4) = synth_params1(l,4) * u0 / self%deltavec (1)
            synth_params2(l,4) = synth_params2(l,4) * u0 / self%deltavec2(1)
            synth_params1(l,5) = synth_params1(l,5) * u0
            synth_params2(l,5) = synth_params2(l,5) * u0
            synth_params1(l,6) = synth_params1(l,6) * lz
            synth_params2(l,6) = synth_params2(l,6) * lz

            call get_crandom_f(rr)
!           rr = 0._rkind
            synth_params1(l,7) = synth_params1(l,7) * 2._rkind*pi*rr
            synth_params2(l,7) = synth_params2(l,7) * 2._rkind*pi*rr
        enddo

        ! random must be synced across processes
        call mpi_bcast(synth_params1(2:5,7),4,mpi_prec,0,self%field%mp_cart,self%mpi_err)
        call mpi_bcast(synth_params2(2:5,7),4,mpi_prec,0,self%field%mp_cart,self%mpi_err)

        u0_03 = 0.03_rkind*u0
        if (self%rand_type==0) u0_03 = 0._rkind
        do k=1,nz
         do j=1,ny
!
          yy = min(y(j), self%grid%domain_size(2)-y(j))
          if (j < ny/2) then
           synth_params = synth_params1
           delta0_loc   = self%delta0
           deltavec_loc = self%deltavec
           jwall = 1
          else
           synth_params = synth_params2
           delta0_loc   = self%delta02
           deltavec_loc = self%deltavec2
           jwall = ny
          endif
!
          do i=1,nx
           rho      = wmean(i,j,1)
           rho_wall = wmean(i,jwall,1)
           do lsp=2,N_S
            rho      = rho      + wmean(i,j,lsp)
            rho_wall = rho_wall + wmean(i,jwall,lsp)
           enddo
           rhofac  = rho_wall/rho
           rhofac  = sqrt(rhofac)*u0
           up      = 0._rkind
           vp      = 0._rkind
           wp      = 0._rkind
           do l=2,5
            arg_sin = synth_params(l,4)*x(i)/synth_params(l,5)
            arg_cos = 2._rkind*pi*z(k)/synth_params(l,6) + synth_params(l,7)
            ys      = yy/synth_params(l,1)
            ufy     = ys    * exp(-ys)
            vfy     = ys**2 * exp(-(ys**2))
            up      = up + synth_params(l,2) * ufy * sin(arg_sin) * cos(arg_cos)
            vp      = vp + synth_params(l,3) * vfy * sin(arg_sin) * cos(arg_cos)

            dup_dx  = synth_params(l,2) * ufy * synth_params(l,4)/synth_params(l,5) * cos(arg_sin)
            dvfy_dy = 2._rkind*ys/synth_params(l,1)*exp(-(ys**2))*(1._rkind-ys**2)
            dvp_dy  = synth_params(l,3) * dvfy_dy * sin(arg_sin)
            wp      = wp-(dup_dx+dvp_dy)*sin(arg_cos)*synth_params(l,6)/(2._rkind*pi)
           enddo
           up = up * rhofac
           vp = vp * rhofac
           wp = wp * rhofac
!
           ii     = self%field%ncoords(1)*nx+i
           delta  = deltavec_loc(ii)
           if (yy<delta) then
            call get_crandom_f(rr3)
            rr3 = rr3-0.5_rkind
            up = up+u0_03*rr3(1)*(yy/delta0_loc)
            vp = vp+u0_03*rr3(2)*(yy/delta0_loc)
            wp = wp+u0_03*rr3(3)*(yy/delta0_loc)
           endif
!
           if (j < ny/2) vp = -vp
!
           rho = w(1,i,j,k)
           do lsp=2,N_S
            rho = rho + w(lsp,i,j,k)
           enddo          
           w(I_U,i,j,k) = w(I_U,i,j,k) + rho*up
           w(I_V,i,j,k) = w(I_V,i,j,k) + rho*vp
           w(I_W,i,j,k) = w(I_W,i,j,k) + rho*wp
           tt           = p0/rmixt0/rho
           w_aux(i,j,k,J_T) = tt
           ee = get_e_from_temperature(tt,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,init_mf)
           w(I_E,i,j,k) = rho*ee + 0.5_rkind*(w(I_U,i,j,k)**2 + w(I_V,i,j,k)**2 + w(I_W,i,j,k)**2)/rho
!
          enddo
         enddo
        enddo

        deallocate(synth_params1,synth_params2,synth_params)

 400    format(100ES20.10)
        endassociate

     endsubroutine add_synthetic_perturbations_double_bl

    subroutine init_sod(self)
        class(equation_multideal_object), intent(inout) :: self 

        integer :: i,j,k,l,lsp
        real(rkind) :: rho_sod, u_sod, p_sod, t_sod, rmixt_sod
        real(rkind) :: e_sod, etot_sod
        real(rkind), allocatable, dimension(:) :: winf_sod, y_sod 
        ! only ghost in x to be similar to bl needs
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, nv => self%nv, &
                  wmean => self%wmean, w => self%field%w, winf => self%winf, rgas => self%rgas, init_mf => self%init_mf, &
                  x => self%field%x, &
                  p0 => self%p0, u0 => self%u0, rho0 => self%rho0, t0 => self%t0, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, &
                  masterproc => self%masterproc)

        allocate(winf_sod(nv),y_sod(N_S))
        rho_sod   = 0.125_rkind*rho0
        u_sod     = 0._rkind*u0
        p_sod     = 0.1_rkind*p0
        rmixt_sod = 0._rkind
        do lsp=1,N_S
         y_sod(lsp) = init_mf(lsp)
         rmixt_sod  = rmixt_sod+rgas(lsp)*y_sod(lsp)
        enddo
        t_sod = p_sod/rmixt_sod/rho_sod
        e_sod = get_e_from_temperature(t_sod,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,y_sod)
        etot_sod = e_sod + 0.5_rkind*u_sod**2  
!
        winf_sod = 0._rkind
        do lsp=1,N_S
         winf_sod(lsp) = rho_sod*init_mf(lsp)
        enddo
        winf_sod(I_U) = rho_sod*u_sod
        winf_sod(I_V) = rho_sod*0._rkind
        winf_sod(I_W) = rho_sod*0._rkind
        winf_sod(I_E) = rho_sod*etot_sod
!
        wmean = 0._rkind
        do j=1,ny
         do i=1-ng,nx+ng+1
          do l=1,nv
           wmean(i,j,l) = winf(l)
          enddo
         enddo
        enddo

        do k=1,nz
         do j=1,ny
          do i=1,nx
           if ((x(i))<0.5_rkind) then
            do l=1,nv
             w(l,i,j,k) = winf(l)
            enddo
           else
            do l=1,nv
             w(l,i,j,k) = winf_sod(l)
            enddo
           endif
          enddo
         enddo
        enddo
!
        if (masterproc) then
         open(12,file='init_sod.dat')
         do i=1,nx
          write(12,100) x(i),(w(l,i,1,1),l=1,nv)
         enddo
         close(12)
        endif
!
 100    format(100ES20.10)
!
        endassociate
    endsubroutine init_sod
!   
    subroutine init_scalability(self)
        class(equation_multideal_object), intent(inout) :: self

        integer :: i,j,k,l,lsp
        real(rkind) :: rho_tube, u_tube, p_tube, t_tube, rmixt_tube
        real(rkind) :: e_tube, etot_tube
        real(rkind), allocatable, dimension(:) :: winf_tube, y_tube
        ! only ghost in x to be similar to bl needs
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, nv => self%nv, &
                  wmean => self%wmean, w => self%field%w, winf => self%winf, rgas => self%rgas, init_mf => self%init_mf, &
                  x => self%field%x, &
                  p0 => self%p0, u0 => self%u0, rho0 => self%rho0, t0 => self%t0, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, &
                  masterproc => self%masterproc)

        allocate(winf_tube(nv),y_tube(N_S))

        rho_tube   = 1.707_rkind
        u_tube= -400._rkind
        p_tube= 202650._rkind
        rmixt_tube= 0._rkind

        y_tube(:) = 0._rkind
        y_tube(7) = 1._rkind

        do lsp=1,N_S
         rmixt_tube= rmixt_tube+rgas(lsp)*y_tube(lsp)
        enddo
        t_tube = p_tube/rmixt_tube/rho_tube
        e_tube = get_e_from_temperature(t_tube,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,y_tube)
        etot_tube = e_tube + 0.5_rkind*u_tube**2
!
        winf_tube = 0._rkind
        do lsp=1,N_S
         winf_tube(lsp) = 0._rkind
        enddo
        winf_tube(7) = rho_tube
        winf_tube(I_U) = rho_tube*u_tube
        winf_tube(I_V) = rho_tube*0._rkind
        winf_tube(I_W) = rho_tube*0._rkind
        winf_tube(I_E) = rho_tube*etot_tube
!
        wmean = 0._rkind
        do j=1,ny
         do i=1-ng,nx+ng+1
          do l=1,nv
           wmean(i,j,l) = winf(l)
          enddo
         enddo
        enddo

        do k=1,nz
         do j=1,ny
          do i=1,nx
           if ((x(i)) .le. 0.5_rkind*x(nx)) then
            do l=1,nv
             w(l,i,j,k) = winf(l)
            enddo
           else
            do l=1,nv
             w(l,i,j,k) = winf_tube(l)
            enddo
           endif
          enddo
         enddo
        enddo
!
        if (masterproc) then
         open(12,file='init_scalability.dat')
         do i=1,nx
          write(12,100) x(i),(w(l,i,1,1),l=1,nv)
         enddo
         close(12)
        endif
!
 100    format(100ES20.10)
!
        endassociate
    endsubroutine init_scalability

!
    subroutine init_reactive_tube(self)
        class(equation_multideal_object), intent(inout) :: self 

        integer :: i,j,k,l,lsp
        real(rkind) :: rho_tube, u_tube, p_tube, t_tube, rmixt_tube
        real(rkind) :: e_tube, etot_tube
        real(rkind), allocatable, dimension(:) :: winf_tube, y_tube
        ! only ghost in x to be similar to bl needs
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, nv => self%nv, &
                  wmean => self%wmean, w => self%field%w, winf => self%winf, rgas => self%rgas, init_mf => self%init_mf, &
                  x => self%field%x, &
                  p0 => self%p0, u0 => self%u0, rho0 => self%rho0, t0 => self%t0, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, &
                  masterproc => self%masterproc)

        allocate(winf_tube(nv),y_tube(N_S))
        rho_tube   = 0.18075_rkind
        u_tube= -487.34_rkind
        p_tube= 35594._rkind
        rmixt_tube= 0._rkind
        do lsp=1,N_S
         y_tube(lsp) = init_mf(lsp)
         rmixt_tube= rmixt_tube+rgas(lsp)*y_tube(lsp)
        enddo
        t_tube = p_tube/rmixt_tube/rho_tube
        e_tube = get_e_from_temperature(t_tube,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,y_tube)
        etot_tube = e_tube + 0.5_rkind*u_tube**2  
!
        winf_tube = 0._rkind
        do lsp=1,N_S
         winf_tube(lsp) = rho_tube*init_mf(lsp)
        enddo
        winf_tube(I_U) = rho_tube*u_tube
        winf_tube(I_V) = rho_tube*0._rkind
        winf_tube(I_W) = rho_tube*0._rkind
        winf_tube(I_E) = rho_tube*etot_tube
!
        wmean = 0._rkind
        do j=1,ny
         do i=1-ng,nx+ng+1
          do l=1,nv
           wmean(i,j,l) = winf(l)
          enddo
         enddo
        enddo

        do k=1,nz
         do j=1,ny
          do i=1,nx
           if ((x(i))<0.5_rkind*x(nx)) then
            do l=1,nv
             w(l,i,j,k) = winf(l)
            enddo
           else
            do l=1,nv
             w(l,i,j,k) = winf_tube(l)
            enddo
           endif
          enddo
         enddo
        enddo
!
        if (masterproc) then
         open(12,file='init_reactive_tube.dat')
         do i=1,nx
          write(12,100) x(i),(w(l,i,1,1),l=1,nv)
         enddo
         close(12)
        endif
!
 100    format(100ES20.10)
!
        endassociate
    endsubroutine init_reactive_tube

!
    subroutine init_aw(self)
        class(equation_multideal_object), intent(inout) :: self

        integer :: i,j,k,l,lsp
        real(rkind) :: rmixt_aw, u_aw, pp_aw, rho_aw, tt_aw, ee_aw, etot_aw
        real(rkind) :: x0, d
        
        ! only ghost in x to be similar to bl needs
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, nv => self%nv, &
                  wmean => self%wmean, w => self%field%w, winf => self%winf, rgas => self%rgas, x => self%field%x, &
                  p0 => self%p0, u0 => self%u0, rho0 => self%rho0, t0 => self%t0, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, init_mf => self%init_mf, &
                  masterproc => self%masterproc)
!
        x0   = 0.0025_rkind
        d    = 0.0005_rkind
        u0 = 734.6_rkind
        rho0 = 0.24_rkind
        do k=1,nz
         do j=1,ny
          do i=1,nx
!
           rmixt_aw = 0._rkind
           do lsp = 1,N_S
            rmixt_aw  = rmixt_aw + rgas(lsp)*init_mf(lsp)
           enddo

           u_aw    = 0.01_rkind*u0*exp(-(x(i)-x0)**2/d**2)
           pp_aw   = p0+rho0*u0*u_aw
           rho_aw  = rho0*(1._rkind+u_aw/u0)
           tt_aw   = pp_aw/rmixt_aw/rho_aw
           ee_aw   = get_e_from_temperature(tt_aw,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,init_mf)
           etot_aw = ee_aw + 0.5_rkind*u_aw**2

           do lsp = 1,N_S
            w(lsp,i,j,k) = rho_aw*init_mf(lsp)
           enddo
           w(I_U,i,j,k) = rho_aw*u_aw
           w(I_V,i,j,k) = rho_aw*0._rkind
           w(I_W,i,j,k) = rho_aw*0._rkind
           w(I_E,i,j,k) = rho_aw*etot_aw
          enddo
         enddo
        enddo
!
        if (masterproc) then
         open(12,file='init_aw.dat')
         do i=1,nx
          write(12,100) x(i),(w(l,i,1,1),l=1,nv)
         enddo
         close(12)
        endif
!
 100    format(100ES20.10)
!
        endassociate
    endsubroutine init_aw

    subroutine init_multi_diff(self)
        class(equation_multideal_object), intent(inout) :: self

        integer :: i,j,k,l,lsp
        real(rkind) :: rho_multi_diff, u_multi_diff, p_multi_diff, t_multi_diff, rmixt_multi_diff
        real(rkind) :: e_multi_diff, etot_multi_diff
        real(rkind), dimension(4) :: yk_ox, yk_fuel
        real(rkind), allocatable, dimension(:) :: y_multi_diff
        real(rkind) :: T_ox, T_fuel, x0, d, dm1s, fx
        
        ! only ghost in x to be similar to bl needs
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, nv => self%nv, &
                  wmean => self%wmean, w => self%field%w, winf => self%winf, rgas => self%rgas, x => self%field%x, &
                  y => self%field%y, z => self%field%z, p0 => self%p0, u0 => self%u0, rho0 => self%rho0, &
                  t0 => self%t0, indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, &
                  masterproc => self%masterproc)
!
        u_multi_diff = 0._rkind
        p_multi_diff = 101325._rkind
!        
        allocate(y_multi_diff(N_S))
!
        yk_ox   = (/ 0.142_rkind, 0.758_rkind, 0.1_rkind, 0._rkind /)
        yk_fuel = (/0.195_rkind, 0.591_rkind, 0._rkind , 0.214_rkind/)
        T_ox    = 1350.0_rkind
        T_fuel  = 320.0_rkind
!        
        x0   = 0.025_rkind
        d    = 0.0025_rkind
        dm1s = 1._rkind/d**2
        do k=1,nz
         do j=1,ny
          do i=1,nx
           rmixt_multi_diff = 0._rkind
           if (nx>1) fx = 1._rkind - 0.5_rkind*exp(-(x(i)-x0)**2*dm1s)
           if (ny>1) fx = 1._rkind - 0.5_rkind*exp(-(y(j)-x0)**2*dm1s)
           if (nz>1) fx = 1._rkind - 0.5_rkind*exp(-(z(k)-x0)**2*dm1s)
           do lsp = 1,N_S
            y_multi_diff(lsp) = yk_ox(lsp) + (yk_fuel(lsp) - yk_ox(lsp))*fx
            rmixt_multi_diff  = rmixt_multi_diff + rgas(lsp)*y_multi_diff(lsp)
           enddo
           T_multi_diff    = T_ox + (T_fuel - T_ox)*fx
           rho_multi_diff  = p_multi_diff/rmixt_multi_diff/T_multi_diff
           e_multi_diff    = get_e_from_temperature(T_multi_diff,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,y_multi_diff)
           etot_multi_diff = e_multi_diff + 0.5_rkind*u_multi_diff**2

           do lsp = 1,N_S
            w(lsp,i,j,k) = rho_multi_diff*y_multi_diff(lsp)
           enddo
           w(I_U,i,j,k) = rho_multi_diff*u_multi_diff
           w(I_V,i,j,k) = rho_multi_diff*0._rkind
           w(I_W,i,j,k) = rho_multi_diff*0._rkind
           w(I_E,i,j,k) = rho_multi_diff*etot_multi_diff
          enddo
         enddo
        enddo
!
        if (masterproc) then
         if (nx>1) then
          open(12,file='init_multi_diff.dat')
          do i=1,nx
           write(12,100) x(i),(w(l,i,1,1),l=1,nv)
          enddo
          close(12)
         endif
         if (ny>1) then
          open(12,file='init_multi_diff.dat')
          do j=1,ny
           write(12,100) y(j),(w(l,1,j,1),l=1,nv)
          enddo
          close(12)
         endif
         if (nz>1) then
          open(12,file='init_multi_diff.dat')
          do k=1,nz
           write(12,100) z(k),(w(l,1,1,k),l=1,nv)
          enddo
          close(12)
         endif
        endif
!
 100    format(100ES20.10)
!
        endassociate
    endsubroutine init_multi_diff

    subroutine init_premix(self)
        class(equation_multideal_object), intent(inout) :: self

        integer :: i,j,k,l,lsp
        real(rkind) :: rho_reactor, u_reactor, p_reactor, t_reactor, rmixt_reactor
        real(rkind) :: e_reactor, etot_reactor, ylsp, xx
        real(rkind), allocatable, dimension(:) :: y_reactor
        real(rkind), allocatable, dimension(:,:) :: wpremix

        ! only ghost in x to be similar to bl needs
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, nv => self%nv, &
                  wmean => self%wmean, w => self%field%w, winf => self%winf, rgas => self%rgas, &
                  x => self%field%x, y => self%field%y, z => self%field%z, &
                  p0 => self%p0, u0 => self%u0, rho0 => self%rho0, t0 => self%t0, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, &
                  masterproc => self%masterproc)
!
        u_reactor = u0
        p_reactor = p0
        T_reactor = t0
        allocate(y_reactor(N_S))
        allocate(wpremix(2+N_S,nx))
        y_reactor = self%init_mf
        open(12,file='premix.dat')
        do i=1,nx
         read(12,*) xx,(wpremix(l,i),l=1,2+N_S)
        enddo
        close(12)
!
        do k=1,nz
         do j=1,ny
          do i=1,nx
           rmixt_reactor = 0._rkind
           u_reactor = wpremix(1,i)
           T_reactor = wpremix(2,i)
           do lsp = 1,N_S
            ylsp = wpremix(2+lsp,i)
            rmixt_reactor = rmixt_reactor + rgas(lsp)*ylsp
            y_reactor(lsp) = ylsp
            if (abs(y_reactor(lsp) ) .lt. 1E-25) y_reactor(lsp) = 0._rkind
           enddo
           rho_reactor  = p_reactor/rmixt_reactor/T_reactor
           e_reactor    = get_e_from_temperature(T_reactor,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,y_reactor)
           etot_reactor = e_reactor + 0.5_rkind*u_reactor**2


           do lsp = 1,N_S
            ylsp = wpremix(2+lsp,i)
            w(lsp,i,j,k) = rho_reactor*ylsp
           enddo
           w(I_U,i,j,k) = rho_reactor*u_reactor
           w(I_V,i,j,k) = rho_reactor*0._rkind
           w(I_W,i,j,k) = rho_reactor*0._rkind
           w(I_E,i,j,k) = rho_reactor*etot_reactor
          enddo
         enddo
        enddo
!
        if (masterproc) then
         open(12,file='init_premix.dat')
         do i=1,nx
          write(12,100) x(i),(w(l,i,1,1),l=1,nv)
         enddo
         close(12)
        endif
!
 100    format(100ES20.10)
!
        endassociate
    endsubroutine init_premix

    subroutine init_reactor(self)
        class(equation_multideal_object), intent(inout) :: self

        integer :: i,j,k,l,lsp
        real(rkind) :: rho_reactor, u_reactor, p_reactor, t_reactor, rmixt_reactor
        real(rkind) :: e_reactor, etot_reactor
        real(rkind), allocatable, dimension(:) :: y_reactor

        ! only ghost in x to be similar to bl needs
        allocate(self%wmean(1-self%grid%ng:self%field%nx+self%grid%ng+1, 1:self%field%ny, self%nv))

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, nv => self%nv, &
                  wmean => self%wmean, w => self%field%w, winf => self%winf, rgas => self%rgas, &
                  x => self%field%x, y => self%field%y, z => self%field%z, &
                  p0 => self%p0, u0 => self%u0, rho0 => self%rho0, t0 => self%t0, &
                  indx_cp_l => self%indx_cp_l, indx_cp_r => self%indx_cp_r, cv_coeff => self%cv_coeff, &
                  nsetcv => self%nsetcv, trange => self%trange, &
                  masterproc => self%masterproc)
!
        u_reactor = 0._rkind
        p_reactor = p0
!
        allocate(y_reactor(N_S))
!
        y_reactor = self%init_mf
        T_reactor = t0
!
        do k=1,nz
         do j=1,ny
          do i=1,nx
           rmixt_reactor = 0._rkind
           do lsp = 1,N_S
            rmixt_reactor = rmixt_reactor + rgas(lsp)*y_reactor(lsp)
           enddo
           rho_reactor  = p_reactor/rmixt_reactor/T_reactor
           e_reactor    = get_e_from_temperature(T_reactor,indx_cp_l,indx_cp_r,cv_coeff,nsetcv,trange,y_reactor)
           etot_reactor = e_reactor + 0.5_rkind*u_reactor**2


           do lsp = 1,N_S
            w(lsp,i,j,k) = rho_reactor*y_reactor(lsp)
           enddo
           w(I_U,i,j,k) = rho_reactor*u_reactor
           w(I_V,i,j,k) = rho_reactor*0._rkind
           w(I_W,i,j,k) = rho_reactor*0._rkind
           w(I_E,i,j,k) = rho_reactor*etot_reactor
          enddo
         enddo
        enddo
!
        if (masterproc) then
         open(12,file='init_reactor.dat')
         do i=1,nx
          write(12,100) x(i),(w(l,i,1,1),l=1,nv)
         enddo
         close(12)
        endif
!
 100    format(100ES20.10)
!
        endassociate
    endsubroutine init_reactor

    subroutine bc_preproc(self)
        class(equation_multideal_object), intent(inout) :: self      
        integer :: ilat, offset

        self%force_zero_flux = [0,0,0,0,0,0]
        self%eul_imin = 1
        self%eul_imax = self%field%nx
        self%eul_jmin = 1
        self%eul_jmax = self%field%ny
        self%eul_kmin = 1
        self%eul_kmax = self%field%nz
        do ilat=1,6 ! loop on all sides of the boundary (3D -> 6)
          select case(self%bctags(ilat))
            case(0)
            case(1)
            case(2)
            case(4)
            case(5)
            case(6) !viscous isothermal wall
             if (self%grid%is_y_staggered) then
              self%force_zero_flux(ilat) = 1
              self%bctags_nr(ilat) = 0
             endif
            case(7)
            case(8) 
            case(9)
          endselect
          if(self%bctags_nr(ilat) /= 0) then
            offset = 1
          else
            offset = 0
          endif
          select case(ilat)
            case(1)
                self%eul_imin = self%eul_imin + offset
            case(2)
                self%eul_imax = self%eul_imax - offset
            case(3)
                self%eul_jmin = self%eul_jmin + offset
            case(4)
                self%eul_jmax = self%eul_jmax - offset
            case(5)
                self%eul_kmin = self%eul_kmin + offset
            case(6)
                self%eul_kmax = self%eul_kmax - offset
          endselect
        enddo
    endsubroutine bc_preproc

    function get_cp(tt,indx_cp_l,indx_cp_r,cp_coeff,nsetcv,trange,massfrac)
    ! Compute mixture cp
    real(rkind) :: get_cp
    integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
    real(rkind), intent(in) :: tt
    real(rkind), dimension(N_S), intent(in) :: massfrac
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in) :: cp_coeff
    real(rkind), dimension(N_S,nsetcv+1), intent(in) :: trange
    integer, dimension(N_S) :: nrange
    real(rkind) :: cp_l,cploc,tprod
    integer :: l,lsp,jl,ju,jm
!
    nrange = 1
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (tt>= trange(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
     enddo
    endif
!
    cploc = 0._rkind
    do l=indx_cp_l,indx_cp_r
     tprod = tt**l
     cp_l = 0._rkind
     do lsp=1,N_S
      cp_l = cp_l+cp_coeff(l,lsp,nrange(lsp))*massfrac(lsp)
     enddo
     cploc = cploc+cp_l*tprod
    enddo
    get_cp = cploc
!
    endfunction get_cp

    subroutine runge_kutta_initialize(self)
        !< Initialize Runge-Kutta data.
        class(equation_multideal_object), intent(inout) :: self !< The equation.

        select case(self%rk_type)
            case(RK_WRAY)
                self%nrk = 3
                allocate(self%rhork(self%nrk),self%gamrk(self%nrk),self%alprk(self%nrk))
                self%rhork(:) = [0._rkind, -17._rkind/60._rkind , -5._rkind /12._rkind]
                self%gamrk(:) = [8._rkind  /15._rkind, 5._rkind  /12._rkind, 3._rkind  /4._rkind]
                self%alprk(:) = self%rhork(:) + self%gamrk(:) 
            case(RK_JAMESON)
                self%nrk = 4
                allocate(self%alprk(self%nrk))
                self%alprk(:) = [1._rkind/4._rkind, 1._rkind/3._rkind, 1._rkind/2._rkind, 1._rkind]
            case(RK_SHU)
                self%nrk = 3
                allocate(self%ark(self%nrk),self%brk(self%nrk),self%crk(self%nrk))
                self%ark(:) = [1._rkind, 0.75_rkind, 1._rkind /3._rkind]
                self%brk(:) = [0._rkind, 0.25_rkind, 2._rkind /3._rkind]
                self%crk(:) = [1._rkind, 0.25_rkind, 2._rkind /3._rkind]
        endselect

    endsubroutine runge_kutta_initialize

    subroutine read_input(self, filename)
        !< Initialize the equation.
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
        character(*)                    , intent(in)      :: filename          !< Input file name.

        ! Test and possibly use .ini file as input format
        ! https://github.com/pkgpl/cfgio
        self%cfg=parse_cfg(filename)

    endsubroutine read_input
!
!   ibm routines
!
    subroutine ibm_initialize(self)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        integer :: n
        integer, allocatable, dimension(:) :: ibm_type_bc_tmp
!
!       ibm_num_body =  1 ==> Single-body (multi patch)
!       ibm_num_body >  1 ==> Multi-body (single patch)
!
        call self%cfg%get("ibmpar","ibm_internal_flow",self%ibm_internal_flow)
        if (self%ibm_num_body>1) self%ibm_internal_flow  = 0 ! External flow
!
        call self%cfg%get("ibmpar","ibm_num_bc",self%ibm_num_bc)
!
        call self%cfg%get("ibmpar","ibm_stencil_size",self%ibm_stencil_size)
!
        if (self%cfg%has_key("ibmpar","ibm_bc_relax_factor")) then
         call self%cfg%get("ibmpar","ibm_bc_relax_factor",self%ibm_bc_relax_factor)
        else
         self%ibm_bc_relax_factor = 1._rkind
        endif
!
        if (self%cfg%has_key("ibmpar","ibm_order_reduce")) then
         call self%cfg%get("ibmpar","ibm_order_reduce",self%ibm_order_reduce)
        else
         self%ibm_order_reduce = 0
        endif
!
        if (self%cfg%has_key("ibmpar","ibm_eikonal_cfl")) then
         call self%cfg%get("ibmpar","ibm_eikonal_cfl",self%ibm_eikonal_cfl)
        else
         self%ibm_eikonal_cfl = 0.9_rkind
        endif
!
        if (self%cfg%has_key("ibmpar","ibm_indx_eikonal")) then
         call self%cfg%get("ibmpar","ibm_indx_eikonal",self%ibm_indx_eikonal)
        else
         self%ibm_indx_eikonal = 3
        endif
!
        self%ibm_methodology = 0
!
        call self%ibm_alloc(step=1)
!
        call self%cfg%get("ibmpar","ibm_type_bc",ibm_type_bc_tmp)
        if (size(ibm_type_bc_tmp)==self%ibm_num_bc) then
         self%ibm_type_bc = ibm_type_bc_tmp
        else
         call fail_input_any("Error! Check number of bc for IBM")
        endif
        if (any(self%ibm_type_bc == 16)) self%ibm_wm = 1
        if (any(self%ibm_type_bc == 18)) self%ibm_wm = 1
!
        call self%ibm_readoff() ! Read geometry in off format
!
        if (self%cfg%has_key("ibmpar","ibm_read")) then
         call self%cfg%get("ibmpar","ibm_read",self%ibm_read)
        else
         self%ibm_read = 0
        endif
        
        !if (self%restart_type==0 .or. (self%ibm_read == 0)) then
        if (self%ibm_read == 0) then
         call self%ibm_raytracing()
         call self%ibm_raytracing_write()
        else
         call self%ibm_raytracing_read()
        endif
!
        !if (self%restart_type==0 .or. (self%ibm_read == 0)) then
        if (self%ibm_read == 0) then
         call self%ibm_setup_geo()
        else
         call self%ibm_read_geo()
        endif
!
        if (self%ibm_wm>0) then
         self%ibm_hwm_dist = 2._rkind
         if (self%cfg%has_key("ibmpar","ibm_hwm_dist")) then
          call self%cfg%get("ibmpar","ibm_hwm_dist",self%ibm_hwm_dist)
         endif
         call self%ibm_prepare_wm()
        endif
!
        call self%ibm_correct_fields()
        call self%ibm_setup_computation()

        if (self%ibm_wm>0) then
         select case(self%restart_type)
         case(0,1)
          if (self%ibm_num_interface > 0) then
           self%ibm_wm_stat = 0._rkind
          endif
         case(2)
          call self%ibm_read_wm_stat()
         endselect
        endif
!
    end subroutine ibm_initialize
!
    subroutine ibm_alloc(self,step)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
        integer, intent(in) :: step
!
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, &
                  ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc, &
                  ibm_methodology => self%ibm_methodology)

        select case(step)
        case(1)
         allocate(self%ibm_sbody(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
         allocate(self%ibm_is_interface_node(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
         if (ibm_num_body>0) then
          allocate(self%ibm_ptree(ibm_num_body))
          allocate(self%ibm_bbox(6,ibm_num_body))
         endif
         allocate(self%ibm_body_dist(0-ng:nx+ng+1,0-ng:ny+ng+1,0-ng:nz+ng+1))
         allocate(self%ibm_reflection_coeff(nx,ny,nz))
         if (ibm_num_bc>1) allocate(self%ibm_ptree_patch(ibm_num_bc)) ! Patch needed only when ibm_num_bc > 1
         if (ibm_num_bc>0) then
          allocate(self%ibm_parbc(ibm_num_bc,IBM_MAX_PARBC))
          allocate(self%ibm_type_bc(ibm_num_bc))
         endif
        case(2)
         if (self%ibm_num_interface>0) then
          allocate(self%ibm_ijk_interface (3,self%ibm_num_interface))     ! Local values of i,j,k for the interface node
          allocate(self%ibm_nxyz_interface(3,self%ibm_num_interface))     ! Wall-normal components
          allocate(self%ibm_bc            (2,self%ibm_num_interface))     ! Bc tag (1,:) and patch index (2,:) for interface nodes
          allocate(self%ibm_ijk_hwm       (3,self%ibm_num_interface))     ! Local values of i,j,k for external points for wm
          allocate(self%ibm_dist_hwm      (self%ibm_num_interface))
          allocate(self%ibm_xyz_hwm       (3,self%ibm_num_interface))
          allocate(self%ibm_wm_wallprop   (2,self%ibm_num_interface))
          allocate(self%ibm_wm_stat       (4,self%ibm_num_interface))
          allocate(self%ibm_coeff_hwm     (2,2,2,self%ibm_num_interface))  ! Coefficients for trilin interpolation (Dirichlet)
         endif
        end select
!
        end associate
    endsubroutine ibm_alloc
!
    subroutine ibm_compute_refl_coeff(self)
        class(equation_multideal_object), intent(inout) :: self
!
        integer :: i,j,k,ii,jj,kk,iii,jjj,kkk,numfluids
        real(rkind) :: normx,normy,normz,dist,xwall,ywall,zwall,xx,yy,zz,densum,minrefl
!
        associate(nx => self%field%nx,ny => self%field%ny,nz => self%field%nz, &
                  ng => self%grid%ng, x => self%field%x, y => self%field%y, z => self%field%z, &
                  ibm_reflection_coeff => self%ibm_reflection_coeff, &
                  ibm_body_dist => self%ibm_body_dist)
!
        minrefl = 0._rkind
        do k=1,nz
         do j=1,ny
          do i=1,nx
           ibm_reflection_coeff(i,j,k) = 0._rkind
           if (self%ibm_sbody(i,j,k)>0) then ! solid
            if (self%ibm_is_interface_node(i,j,k) == 1) then ! interface node
             xx = x(i)
             yy = y(j)
             zz = z(k)
             dist  = ibm_body_dist(i,j,k)
             normx = ibm_body_dist(i+1,j,k)-ibm_body_dist(i-1,j,k)
             normy = ibm_body_dist(i,j+1,k)-ibm_body_dist(i,j-1,k)
             normz = ibm_body_dist(i,j,k+1)-ibm_body_dist(i,j,k-1)
             normx = normx/(x(i+1)-x(i-1))
             normy = normy/(y(j+1)-y(j-1))
             normz = normz/(z(k+1)-z(k-1))
             xwall = xx-normx*dist
             ywall = yy-normy*dist
             zwall = zz-normz*dist
             call locateval(x(1-ng:nx+ng),nx+2*ng,xwall,ii)
             call locateval(y(1-ng:ny+ng),ny+2*ng,ywall,jj)
             call locateval(z(1-ng:nz+ng),nz+2*ng,zwall,kk)
             ii = ii-ng ; jj = jj-ng ; kk = kk-ng

             numfluids = 0
             densum = 0._rkind
             do kkk=0,1
              do jjj=0,1
               do iii=0,1
                if (ibm_body_dist(ii+iii,jj+jjj,kk+kkk)<0._rkind) then
                 numfluids = numfluids+1
                 densum = densum+ibm_body_dist(ii+iii,jj+jjj,kk+kkk)
                endif
               enddo
              enddo
             enddo
             if (numfluids>0) then
              densum = densum/numfluids
              ibm_reflection_coeff(i,j,k) = dist/densum
              minrefl = min(minrefl,ibm_reflection_coeff(i,j,k))
              ibm_reflection_coeff(i,j,k) = max(-1._rkind,ibm_reflection_coeff(i,j,k))
             else
              ibm_reflection_coeff(i,j,k) = -1._rkind
             endif
            endif
           endif
          enddo
         enddo
        enddo
!
        endassociate
!
    end subroutine ibm_compute_refl_coeff
!
    subroutine ibm_correct_fields(self)
        class(equation_multideal_object), intent(inout) :: self
!
        integer :: i,j,k,l
        integer :: stencil_size
        real(rkind) :: rho,rhou,rhov,rhow,rhoe,uu,vv,ww,qq
!
        associate(nx => self%field%nx,ny => self%field%ny,nz => self%field%nz, &
                  ng => self%grid%ng)
!
        do k=1-ng,nz+ng
         do j=1-ng,ny+ng
          do i=1-ng,nx+ng
           if (self%ibm_sbody(i,j,k)>0) then ! solid
            self%fluid_mask(i,j,k) = 1
            if (self%ibm_is_interface_node(i,j,k) == 0) then ! no interface
             rho  = findrho(N_S,self%field%w(1:N_S,i,j,k))
             rhou = self%field%w(I_U,i,j,k)
             rhov = self%field%w(I_V,i,j,k)
             rhow = self%field%w(I_W,i,j,k)
             rhoe = self%field%w(I_E,i,j,k)
             uu   = rhou/rho
             vv   = rhov/rho
             ww   = rhow/rho
             qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
             self%field%w(I_U,i,j,k) = 0._rkind
             self%field%w(I_V,i,j,k) = 0._rkind
             self%field%w(I_W,i,j,k) = 0._rkind
             self%field%w(I_E,i,j,k) = rhoe-rho*qq
            else
            endif
           endif
          enddo
         enddo
        enddo
!
        self%ibm_order_reduce = min(self%ibm_stencil_size-self%ep_order/2,self%ibm_order_reduce)
!
!       i direction
        stencil_size = self%ibm_stencil_size
        do k=1,nz
         do j=1,ny
          idir: do i=0,nx
           do l=1,stencil_size
            if (self%ibm_sbody(i+l,j,k)>0) then
             self%ep_ord_change(i,j,k,1) = self%ibm_order_reduce
             cycle idir
            endif
            if (self%ibm_sbody(i-l+1,j,k)>0) then
             self%ep_ord_change(i,j,k,1) = self%ibm_order_reduce
             cycle idir
            endif
           enddo
          enddo idir
         enddo
        enddo
!
!       j direction
        do k=1,nz
         do i=1,nx
          jdir: do j=0,ny
           do l=1,stencil_size
            if (self%ibm_sbody(i,j+l,k)>0) then
             self%ep_ord_change(i,j,k,2) = self%ibm_order_reduce
             cycle jdir
            endif
            if (self%ibm_sbody(i,j-l+1,k)>0) then
             self%ep_ord_change(i,j,k,2) = self%ibm_order_reduce
             cycle jdir
            endif
           enddo
          enddo jdir
         enddo
        enddo
!       k direction
        do j=1,ny
         do i=1,nx
          kdir: do k=0,nz
           do l=1,stencil_size
            if (self%ibm_sbody(i,j,k+l)>0) then
             self%ep_ord_change(i,j,k,3) = self%ibm_order_reduce
             cycle kdir
            endif
            if (self%ibm_sbody(i,j,k-l+1)>0) then
             self%ep_ord_change(i,j,k,3) = self%ibm_order_reduce
             cycle kdir
            endif
           enddo
          enddo kdir
         enddo
        enddo

        endassociate
!
    end subroutine ibm_correct_fields
!
    subroutine ibm_setup_computation(self)
        class(equation_multideal_object), intent(inout) :: self
!
        integer :: i,j,k,l,num_par,ierr
        real(rkind), allocatable, dimension(:) :: tmp_arr
        character(2) :: chbcnum
!
        associate(ibm_num_bc => self%ibm_num_bc, ibm_type_bc => self%ibm_type_bc)
!
!       ibm_type_bc dictionary
!
!       1 supersonic inflow, required: p, T, U, Y
!       2 subsonic inflow, required: p0, T0, Y
!       3 supersonic inflow powerlaw+turbulence, required: yp,zp,p,T,U,Y
!       4 subsonic inflow powerlaw+turbulence, required: yp,zp,p0,T0,Y
!       5 inviscid, adiabatic wall
!       6 viscous, isothermal wall, required: T
!       16 viscous, isothermal wall with wall function, required: T
!       8 viscous, adiabatic wall
!       18 viscous, adiabatic wall with wall function
!       9 Vega bc, required: NPR, NTR, timeshift
!
        if (any(ibm_type_bc == 16)) self%ibm_wm = 1
        if (any(ibm_type_bc == 18)) self%ibm_wm = 1
!
        do l=1,ibm_num_bc
         write(chbcnum,'(I2.2)') l
         select case (ibm_type_bc(l))
         case (5,8,18)
          num_par = 0
         case (6,16)
          num_par = 1
         case(4) 
          num_par = 4+N_S
          self%turinf = 1
          if (self%masterproc) then
           call get_crandom_f(self%randvar_a)
           call get_crandom_f(self%randvar_p)
           self%randvar_a = self%randvar_a*8._rkind*datan(1._rkind)
           self%randvar_p = self%randvar_p*1.E5_rkind
          endif
          call mpi_bcast(self%randvar_a, 8, mpi_prec, 0, MPI_COMM_WORLD, ierr)
          call mpi_bcast(self%randvar_p, 8, mpi_prec, 0, MPI_COMM_WORLD, ierr)
         case(3)
          num_par = 5+N_S
          self%turinf = 1
          if (self%masterproc) then
           call get_crandom_f(self%randvar_a)
           call get_crandom_f(self%randvar_p)
           self%randvar_a = self%randvar_a*8._rkind*datan(1._rkind)
           self%randvar_p = self%randvar_p*1.E5_rkind
          endif       
          call mpi_bcast(self%randvar_a, 8, mpi_prec, 0, MPI_COMM_WORLD, ierr)
          call mpi_bcast(self%randvar_p, 8, mpi_prec, 0, MPI_COMM_WORLD, ierr)
         case (2) 
          num_par = 2+N_S
         case(1)
          num_par = 3+N_S
         endselect
         if (num_par > 0) then
          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,tmp_arr)
          self%ibm_parbc(l,1:num_par) = tmp_arr
          deallocate(tmp_arr)
         endif
        enddo
                 
!        do l=1,ibm_num_bc
!         write(chbcnum,'(I2.2)') l
!         select case (ibm_type_bc(l))
!         case (1)
!          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,tmp_arr3)
!          self%ibm_parbc(l,1:3+N_S) = tmp_arr3
!          deallocate(tmp_arr3)
!         case (2)
!          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,tmp_arr2)
!          self%ibm_parbc(l,1:2+N_S) = tmp_arr2
!          deallocate(tmp_arr2)
!         case(6,16)
!          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,tmp_arr1)
!          self%ibm_parbc(l,1:1) = tmp_arr1
!          deallocate(tmp_arr1)
!         endselect
!        enddo
!
        endassociate
!
    end subroutine ibm_setup_computation
!
    subroutine ibm_readoff(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.

        integer     :: ib,ip
        real(rkind) :: db_x, db_y, db_z
        real(rkind) :: c_x, c_y, c_z
        real(rkind) :: bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z
        character(128) :: filename
!
        associate(masterproc => self%masterproc, ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc, &
                  ibm_ptree => self%ibm_ptree, ibm_ptree_patch => self%ibm_ptree_patch, ibm_bbox => self%ibm_bbox)
!
        ! Read geometry (ibm_num_body is the number of solid objects)
        do ib=1,ibm_num_body
         filename = "X_YYYYY.off"
         write(filename(3:7), "(I5.5)") ib
         if (masterproc) write(*,*) 'Reading geometry: ', filename
         call cgal_polyhedron_read(ibm_ptree(ib),filename)
        enddo
!
        if (ibm_num_bc>1) then
        ! Read patches (ibm_num_bc, active only for single body)
         do ip=1,ibm_num_bc
          if (masterproc) write(*,*) 'Reading patch #', ip
          filename = "patchxx.off"
          write(filename(6:7),"(I2.2)") ip
          call cgal_polyhedron_read(ibm_ptree_patch(ip),filename)
         enddo
        endif
!
!       Compute bounding box (only for ib = 1)
        do ib=1,ibm_num_body
         call polyhedron_bbox(ibm_ptree(ib),bmin_x,bmin_y,bmin_z,bmax_x,bmax_y,bmax_z)
         ibm_bbox(1:6,ib) = [bmin_x,bmin_y,bmin_z,bmax_x,bmax_y,bmax_z]
         db_x = bmax_x - bmin_x ; db_y = bmax_y - bmin_y ; db_z = bmax_z - bmin_z
         c_x = 0.5*(bmax_x + bmin_x) ; c_y = 0.5*(bmax_y + bmin_y) ; c_z = 0.5*(bmax_z + bmin_z)
         if (ib==1) then
          ! if (masterproc) write(*,*) 'Solid center: ', c_x, c_y, c_z
          ! if (masterproc) write(*,*) 'Solid bounding box: ', db_x, db_y, db_z
          if (masterproc) write(*,*) 'Solid limits x: ', c_x-0.5_rkind*db_x,c_x+0.5_rkind*db_x
          if (masterproc) write(*,*) 'Solid limits y: ', c_y-0.5_rkind*db_y,c_y+0.5_rkind*db_y
          if (masterproc) write(*,*) 'Solid limits z: ', c_z-0.5_rkind*db_z,c_z+0.5_rkind*db_z
         endif
        enddo
!
        if (masterproc) write(*,*) 'Done with ibm_readoff'
!
        endassociate
    endsubroutine ibm_readoff
!

    subroutine ibm_raytracing(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.

        integer     :: i,j,k
        integer     :: ii,jj,kk
        integer     :: iii,jjj,kkk
        integer     :: ib
        integer     :: imin,imax,jmin,jmax,kmin,kmax
        integer     :: stencil_size
        real(rkind) :: query_x, query_y, query_z
        real(rkind) :: near_x,near_y,near_z,ib_dist
        real(rkind) :: ib_dist_min
        logical     :: is_inside
        logical, dimension(:,:,:), allocatable :: is_body_node
        integer, dimension(:,:,:), allocatable :: ibm_sbody_extended
        integer :: num_in_solid, num_in_solid_tot
        integer :: num_interface_tot
!
        associate(masterproc => self%masterproc, ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc, &
                  xg => self%grid%xg, nxmax => self%grid%nxmax, nx => self%field%nx,              &
                  yg => self%grid%yg, nymax => self%grid%nymax, ny => self%field%ny,              &
                  zg => self%grid%zg, nzmax => self%grid%nzmax, nz => self%field%nz,              &
                  x => self%field%x, y => self%field%y, z => self%field%z,                        &
                  ng => self%grid%ng, ncoords => self%field%ncoords,                              &
                  ibm_sbody => self%ibm_sbody, is_xyz_periodic => self%grid%is_xyz_periodic,      &
                  ibm_is_interface_node => self%ibm_is_interface_node, ibm_internal_flow => self%ibm_internal_flow, &
                  nblocks => self%field%nblocks, ibm_ptree => self%ibm_ptree, &
                  ibm_bbox => self%ibm_bbox, &
                  ibm_ptree_patch => self%ibm_ptree_patch, iermpi => self%mpi_err, &
                  ibm_num_interface => self%ibm_num_interface, &
                  ibm_body_dist => self%ibm_body_dist, &
                  ibm_methodology => self%ibm_methodology)
!
        stencil_size = self%ibm_stencil_size
        imin =  1-ng-stencil_size
        imax = nx+ng+stencil_size
        jmin =  1-ng-stencil_size
        jmax = ny+ng+stencil_size
        kmin =  1-ng-stencil_size
        kmax = nz+ng+stencil_size
!
        if (ncoords(1)==0)  imin = 1-ng
        if (ncoords(2)==0)  jmin = 1-ng
        if (ncoords(3)==0)  kmin = 1-ng
        if (ncoords(1)==(nblocks(1)-1)) imax = nx+ng
        if (ncoords(2)==(nblocks(2)-1)) jmax = ny+ng
        if (ncoords(3)==(nblocks(3)-1)) kmax = nz+ng
!
        allocate(is_body_node(imin:imax,jmin:jmax,kmin:kmax))
        allocate(ibm_sbody_extended(imin:imax,jmin:jmax,kmin:kmax))
!
!       Default fluid
        is_body_node = .false.
        ibm_sbody_extended = 0
        ibm_sbody          = 0
!
        if (masterproc) write(*,*) 'Start raytracing with CGAL'
!
        do k=kmin,kmax
         kk = ncoords(3)*nz+k
         query_z = zg(kk)
         do j=jmin,jmax
          jj = ncoords(2)*ny+j
          query_y = yg(jj)
          do i=imin,imax
           ii = ncoords(1)*nx+i
           query_x = xg(ii)
           loopib: do ib=1,ibm_num_body
            if (query_x>ibm_bbox(1,ib).and.query_x<ibm_bbox(4,ib).and.&
                query_y>ibm_bbox(2,ib).and.query_y<ibm_bbox(5,ib).and.&
                query_z>ibm_bbox(3,ib).and.query_z<ibm_bbox(6,ib)) then
             is_inside = cgal_polyhedron_inside(ibm_ptree(ib),query_x,query_y,query_z)
             if (is_inside) then
              is_body_node(i,j,k) = .true.
              ibm_sbody_extended(i,j,k) = ib
              exit loopib
             endif
            endif
           enddo loopib
          enddo
         enddo
        enddo
!
        if (ibm_internal_flow==1) then
         if (ibm_num_body==1) then
          is_body_node = .not.is_body_node ! Fluid is inside stl
          ibm_sbody_extended = 1-ibm_sbody_extended
         else
          if (masterproc) write(*,*) 'Error, ibm_internal_flow=1 not supported for multi body'
         endif
        endif
!
        ibm_sbody = ibm_sbody_extended(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng)
!
!       Loop to identify interface nodes
        ibm_is_interface_node = 0
        do k=1-ng,nz+ng
         do j=1-ng,ny+ng
          found: do i=1-ng,nx+ng
!
           if (is_body_node(i,j,k)) then ! Solid node
!
            do kk=-stencil_size,stencil_size
             kkk = k+kk
             if (kkk<kmin.or.kkk>kmax) cycle
             if (.not.is_body_node(i,j,kkk)) then
              ibm_is_interface_node(i,j,k) = 1
              cycle found
             endif
            enddo
            do jj=-stencil_size,stencil_size
             jjj = j+jj
             if (jjj<jmin.or.jjj>jmax) cycle
             if (.not.is_body_node(i,jjj,k)) then
              ibm_is_interface_node(i,j,k) = 1
              cycle found
             endif
            enddo
            do ii=-stencil_size,stencil_size
             iii = i+ii
             if (iii<imin.or.iii>imax) cycle
             if (.not.is_body_node(iii,j,k)) then
              ibm_is_interface_node(i,j,k) = 1
              cycle found
             endif
            enddo
!
           endif
          enddo found
         enddo
        enddo
!
!       Count the number of local interface nodes (ghost are excluded)
!
        ibm_num_interface = 0
        num_in_solid = 0
        do k=1,nz
         do j=1,ny
          do i=1,nx
           if (ibm_is_interface_node(i,j,k)==1) ibm_num_interface = ibm_num_interface+1
           if (ibm_sbody(i,j,k) > 0) num_in_solid = num_in_solid+1
          enddo
         enddo
        enddo
!
        call mpi_allreduce(num_in_solid,num_in_solid_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        if (masterproc) write(*,*) 'Total number of nodes in the solid body = ', num_in_solid_tot
        call mpi_allreduce(ibm_num_interface,num_interface_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        if (masterproc) write(*,*) 'Total number of interface nodes = ', num_interface_tot
!
        if (masterproc) write(*,*) 'Done with ibm_raytracing'
!
        deallocate(is_body_node)
!
        imin =  0-ng
        imax = nx+ng+1
        jmin =  0-ng
        jmax = ny+ng+1
        kmin =  0-ng
        kmax = nz+ng+1
!
        if (ncoords(1)==0)  imin = 1-ng
        if (ncoords(2)==0)  jmin = 1-ng
        if (ncoords(3)==0)  kmin = 1-ng
        if (ncoords(1)==(nblocks(1)-1)) imax = nx+ng
        if (ncoords(2)==(nblocks(2)-1)) jmax = ny+ng
        if (ncoords(3)==(nblocks(3)-1)) kmax = nz+ng
!
        if (masterproc) write(*,*) 'Computing wall distance'
!
        do k=kmin,kmax
         kk = ncoords(3)*nz+k
         do j=jmin,jmax
          jj = ncoords(2)*ny+j
          do i=imin,imax
           ii = ncoords(1)*nx+i
           query_x = xg(ii)
           query_y = yg(jj)
           query_z = zg(kk)
           if (ibm_sbody_extended(i,j,k) > 0) then ! Positive distance inside the body
            ib = ibm_sbody_extended(i,j,k)
            call polyhedron_closest(ibm_ptree(ib),query_x,query_y,query_z,near_x,near_y,near_z)
            ib_dist = sqrt((near_x-query_x)**2+(near_y-query_y)**2+(near_z-query_z)**2)
            ibm_body_dist(i,j,k) = ib_dist
           else ! Negative distance outside the body
            ib_dist_min = huge(1._rkind)
            do ib=1,ibm_num_body ! Inefficient, to be improved in the future
!                                (compute distance from bbox, sort and then compute distance)
             call polyhedron_closest(ibm_ptree(ib),query_x,query_y,query_z,near_x,near_y,near_z)
             ib_dist = sqrt((near_x-query_x)**2+(near_y-query_y)**2+(near_z-query_z)**2)
             ib_dist_min = min(ib_dist,ib_dist_min)
            enddo
            ibm_body_dist(i,j,k) = -ib_dist_min
           endif
          enddo
         enddo
        enddo
!
        if (masterproc) write(*,*) 'Done with wall distance'
!
        call self%ibm_compute_refl_coeff()
!
        deallocate(ibm_sbody_extended)
!
        endassociate
    endsubroutine ibm_raytracing
!
    subroutine ibm_raytracing_write(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        character(3) :: chx,chz
!
        associate(ncoords => self%field%ncoords,   &
                  ibm_sbody => self%ibm_sbody,     &
                  ibm_is_interface_node => self%ibm_is_interface_node, &
                  ibm_reflection_coeff => self%ibm_reflection_coeff, &
                  ibm_methodology => self%ibm_methodology, &
                  ibm_body_dist => self%ibm_body_dist, &
                  ibm_num_interface => self%ibm_num_interface)
!
         write(chx,"(I3.3)") ncoords(1)
         write(chz,"(I3.3)") ncoords(3)
!
         open(444,file='ibm_raytracing_'//chx//'_'//chz//'.bin',form='unformatted')
         write(444) ibm_num_interface
         write(444) ibm_sbody
         write(444) ibm_is_interface_node
         write(444) ibm_body_dist
         write(444) ibm_reflection_coeff
         close(444)
!
        endassociate
!
    endsubroutine ibm_raytracing_write
!
    subroutine ibm_raytracing_read(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        character(3) :: chx,chz
!
        associate(ncoords => self%field%ncoords,   &
                  ibm_sbody => self%ibm_sbody,     &
                  ibm_is_interface_node => self%ibm_is_interface_node, &
                  ibm_reflection_coeff => self%ibm_reflection_coeff, &
                  ibm_methodology => self%ibm_methodology, &
                  ibm_body_dist => self%ibm_body_dist, &
                  ibm_num_interface => self%ibm_num_interface)
!
         write(chx,"(I3.3)") ncoords(1)
         write(chz,"(I3.3)") ncoords(3)
!
         open(444,file='ibm_raytracing_'//chx//'_'//chz//'.bin',form='unformatted')
         read(444) ibm_num_interface
         read(444) ibm_sbody
         read(444) ibm_is_interface_node
         read(444) ibm_body_dist
         read(444) ibm_reflection_coeff
         close(444)
!
        endassociate
!
    endsubroutine ibm_raytracing_read
!
    subroutine ibm_setup_geo(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
        integer :: num_interface_tot
        integer :: i,j,k,n,l,typebcloc,indx_patch,ib
        real(rkind) :: query_x,query_y,query_z,near_x,near_y,near_z,near2_x,near2_y,near2_z
        real(rkind) :: rnx,rny,rnz,dist,dbc,dminbc
        real(rkind) :: dxl,dyl,dzl,dlcell,refl_x,refl_y,refl_z
        real(rkind) :: dxloc,dyloc,dzloc,x0,y0,z0,xref,yref,zref,xyz1,xyz2,xyz3
        real(rkind), dimension(8,8) :: amat3d,amat3dtmp
        real(rkind), dimension(1,8) :: xtrasp3d,alftrasp3d
        integer :: sumnei,ii,jj,kk,iii,jjj,kkk,lll
        character(3) :: chx,chz
        logical :: solidcell
!
        call self%ibm_alloc(step=2)
!
        associate(masterproc => self%masterproc, iermpi => self%mpi_err, ibm_num_interface => self%ibm_num_interface, &
                  nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng,      &
                  ncoords => self%field%ncoords, x => self%field%x, y => self%field%y, z => self%field%z, &
                  ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc,                       &
                  ibm_is_interface_node => self%ibm_is_interface_node, ibm_ijk_interface => self%ibm_ijk_interface, &
                  ibm_bc => self%ibm_bc, ibm_nxyz_interface => self%ibm_nxyz_interface, &
                  ibm_ptree_patch => self%ibm_ptree_patch, ibm_body_dist => self%ibm_body_dist, &
                  ibm_type_bc => self%ibm_type_bc, ibm_sbody => self%ibm_sbody, ibm_ptree => self%ibm_ptree)
!
        call mpi_allreduce(ibm_num_interface,num_interface_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        if (masterproc) then
         open(127,file='report_ibm.dat')
         write(127,*) 'ibm_num_body =', ibm_num_body
         write(127,*) 'ibm_num_bc =', ibm_num_bc
         write(127,*) 'Total number of interface nodes =', num_interface_tot
         close(127)
        endif
!
        n = 0
        do k=1,nz
         do j=1,ny
          do i=1,nx
           if (ibm_is_interface_node(i,j,k)==1) then ! Found interface node (solid)
            n = n+1
            query_x = x(i) ; query_y = y(j) ; query_z = z(k) ! coordinates of interface node
            rnx = ibm_body_dist(i+1,j,k)-ibm_body_dist(i-1,j,k)
            rny = ibm_body_dist(i,j+1,k)-ibm_body_dist(i,j-1,k)
            rnz = ibm_body_dist(i,j,k+1)-ibm_body_dist(i,j,k-1)
            rnx = -rnx/(x(i+1)-x(i-1))
            rny = -rny/(y(j+1)-y(j-1))
            rnz = -rnz/(z(k+1)-z(k-1))
!
            ibm_ijk_interface(1,n) = i
            ibm_ijk_interface(2,n) = j
            ibm_ijk_interface(3,n) = k
            ibm_nxyz_interface(1,n) = rnx
            ibm_nxyz_interface(2,n) = rny
            ibm_nxyz_interface(3,n) = rnz
!
            ib = ibm_sbody(i,j,k)
            call polyhedron_closest(ibm_ptree(ib),query_x,query_y,query_z,near_x,near_y,near_z)
            !dist   = ibm_body_dist(i,j,k)
            !near_x = query_x+rnx*dist
            !near_y = query_y+rny*dist
            !near_z = query_z+rnz*dist
!
            if (ibm_num_bc==1) then
             typebcloc  = ibm_type_bc(1) ! default bc
             indx_patch = 1              ! single patch
            else
             dminbc = huge(1._rkind)
             do l=1,ibm_num_bc
              query_x = near_x
              query_y = near_y
              query_z = near_z
              call polyhedron_closest(ibm_ptree_patch(l),query_x,query_y,query_z,near2_x,near2_y,near2_z)
              dbc = (near2_x-query_x)**2+(near2_y-query_y)**2+(near2_z-query_z)**2
              if (dbc<dminbc) then
               dminbc = dbc
               typebcloc  = ibm_type_bc(l)
               indx_patch = l
              endif
             enddo
            endif
!
            ibm_bc(1,n) = typebcloc
            ibm_bc(2,n) = indx_patch
!
           endif
          enddo
         enddo
        enddo
!
        write(chx,"(I3.3)") ncoords(1)
        write(chz,"(I3.3)") ncoords(3)
!
        open(444,file='ibm_geom_'//chx//'_'//chz//'.bin',form='unformatted')
        write(444) ibm_num_interface
        if (ibm_num_interface>0) then
         write(444) ibm_ijk_interface
         write(444) ibm_nxyz_interface
         write(444) ibm_bc
        endif
        close(444)
!
        if (masterproc) write(*,*) 'Done with ibm_compute_geo'
!
        endassociate
    endsubroutine ibm_setup_geo
!
    subroutine ibm_prepare_wm(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
        integer :: i,j,k,n,l,typebcloc,indx_patch,ib
        real(rkind) :: query_x,query_y,query_z,near_x,near_y,near_z,near2_x,near2_y,near2_z
        real(rkind) :: rnx,rny,rnz,dist,dbc,dminbc
        real(rkind) :: dxl,dyl,dzl,dlcell,refl_x,refl_y,refl_z
        real(rkind) :: dxloc,dyloc,dzloc,x0,y0,z0,xref,yref,zref,xyz1,xyz2,xyz3
        real(rkind), dimension(8,8) :: amat3d,amat3dtmp
        real(rkind), dimension(1,8) :: xtrasp3d,alftrasp3d
        integer :: sumnei,ii,jj,kk,iii,jjj,kkk,lll,istat
        character(3) :: chx,chz
        logical :: solidcell
!
        associate(masterproc => self%masterproc, iermpi => self%mpi_err, ibm_num_interface => self%ibm_num_interface, &
                  nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng,      &
                  ncoords => self%field%ncoords, x => self%field%x, y => self%field%y, z => self%field%z, &
                  ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc,                       &
                  ibm_is_interface_node => self%ibm_is_interface_node, ibm_ijk_interface => self%ibm_ijk_interface, &
                  ibm_bc => self%ibm_bc, ibm_nxyz_interface => self%ibm_nxyz_interface, &
                  ibm_ptree_patch => self%ibm_ptree_patch, ibm_body_dist => self%ibm_body_dist, &
                  ibm_type_bc => self%ibm_type_bc, ibm_sbody => self%ibm_sbody, ibm_ijk_hwm => self%ibm_ijk_hwm, &
                  ibm_coeff_hwm => self%ibm_coeff_hwm, ibm_dist_hwm => self%ibm_dist_hwm, ibm_ptree => self%ibm_ptree, &
                  ibm_xyz_hwm => self%ibm_xyz_hwm, ibm_hwm_dist => self%ibm_hwm_dist)
!
        n = 0
        do k=1,nz
         do j=1,ny
          do i=1,nx
!          
           if (ibm_is_interface_node(i,j,k)==1) then ! Found interface node (solid)
            n = n+1
            query_x = x(i) ; query_y = y(j) ; query_z = z(k) ! coordinates of interface node
!
            rnx = ibm_nxyz_interface(1,n)
            rny = ibm_nxyz_interface(2,n)
            rnz = ibm_nxyz_interface(3,n)
!
!           dist   = ibm_body_dist(i,j,k)
!           near_x = query_x+rnx*dist
!           near_y = query_y+rny*dist
!           near_z = query_z+rnz*dist
            ib = ibm_sbody(i,j,k)
            call polyhedron_closest(ibm_ptree(ib),query_x,query_y,query_z,near_x,near_y,near_z)
!
            typebcloc = ibm_bc(1,n)
!
            if (typebcloc==16.or.typebcloc==18) then ! wall model
             sumnei = ibm_sbody(i+1,j,k)+ibm_sbody(i-1,j,k)+ibm_sbody(i,j+1,k)+ &
                      ibm_sbody(i,j-1,k)+ibm_sbody(i,j,k+1)+ibm_sbody(i,j,k-1)
             if (sumnei<6) then ! found first level interface node
              dxl    = 0.5_rkind*(x(i+1)-x(i-1))
              dyl    = 0.5_rkind*(y(j+1)-y(j-1))
              dzl    = 0.5_rkind*(z(k+1)-z(k-1))
              dlcell = (dxl*dyl*dzl)**(1._rkind/3._rkind)
              dist   = ibm_hwm_dist*dlcell
              ibm_dist_hwm(n) = dist
              refl_x = near_x+rnx*dist
              refl_y = near_y+rny*dist
              refl_z = near_z+rnz*dist
!              
              if (refl_x<x(1 -ng)) call fail_input_any("Image point outside myblock (side 1), try to increase ng")
              if (refl_x>x(nx+ng)) call fail_input_any("Image point outside myblock (side 2), try to increase ng")
              if (refl_y<y(1 -ng)) call fail_input_any("Image point outside myblock (side 3), try to increase ng")
              if (refl_y>y(ny+ng)) call fail_input_any("Image point outside myblock (side 4), try to increase ng")
              if (refl_z<z(1 -ng)) call fail_input_any("Image point outside myblock (side 5), try to increase ng")
              if (refl_z>z(nz+ng)) call fail_input_any("Image point outside myblock (side 6), try to increase ng")
!
              call locateval(x(1-ng:nx+ng),nx+2*ng,refl_x,ii)
              call locateval(y(1-ng:ny+ng),ny+2*ng,refl_y,jj)
              call locateval(z(1-ng:nz+ng),nz+2*ng,refl_z,kk)
              ii = ii-ng ; jj = jj-ng ; kk = kk-ng
!
!             Check reflected nodes in solid cells (ii,ii+1,jj,jj+1,kk,kk+1)
!
              solidcell = .true.
              loopcheckscell: do kkk=0,1
               do jjj=0,1
                do iii=0,1
                 if (ibm_sbody(ii+iii,jj+jjj,kk+kkk)==0) then
                  solidcell = .false.
                  exit loopcheckscell
                 endif
                enddo
               enddo
              enddo loopcheckscell
              if (solidcell) call fail_input_any("found solid cell for wall function application")
!
              ibm_ijk_hwm(1,n) = ii
              ibm_ijk_hwm(2,n) = jj
              ibm_ijk_hwm(3,n) = kk
              ibm_xyz_hwm(1,n) = refl_x
              ibm_xyz_hwm(2,n) = refl_y
              ibm_xyz_hwm(3,n) = refl_z
!
!             Trilinear Dirichlet
!
              x0 = x(ii)
              y0 = y(jj)
              z0 = z(kk)
!
              dxloc = x(ii+1)-x(ii)
              dyloc = y(jj+1)-y(jj)
              dzloc = z(kk+1)-z(kk)
              xref = (refl_x-x0)/dxloc
              yref = (refl_y-y0)/dyloc
              zref = (refl_z-z0)/dzloc
!
              xtrasp3d(1,1) = xref*yref*zref
              xtrasp3d(1,2) = xref*yref
              xtrasp3d(1,3) = xref*zref
              xtrasp3d(1,4) = yref*zref
              xtrasp3d(1,5) = xref
              xtrasp3d(1,6) = yref
              xtrasp3d(1,7) = zref
              xtrasp3d(1,8) = 1._rkind

!             Dirichlet
              lll = 0
              do kkk=0,1
               do jjj=0,1
                do iii=0,1
                 lll = lll+1
                 xyz1 = x(ii+iii)
                 xyz2 = y(jj+jjj)
                 xyz3 = z(kk+kkk)
                 xyz1 = (xyz1-x0)/dxloc
                 xyz2 = (xyz2-y0)/dyloc
                 xyz3 = (xyz3-z0)/dzloc
                 amat3d(lll,:) = [xyz1*xyz2*xyz3,xyz1*xyz2,xyz1*xyz3,xyz2*xyz3,xyz1,xyz2,xyz3,1._rkind]
                enddo
               enddo
              enddo
!
              call invmat(amat3d,8)
              alftrasp3d = matmul(xtrasp3d,amat3d)
              ibm_coeff_hwm(1,1,1,n) = alftrasp3d(1,1)
              ibm_coeff_hwm(2,1,1,n) = alftrasp3d(1,2)
              ibm_coeff_hwm(1,2,1,n) = alftrasp3d(1,3)
              ibm_coeff_hwm(2,2,1,n) = alftrasp3d(1,4)
              ibm_coeff_hwm(1,1,2,n) = alftrasp3d(1,5)
              ibm_coeff_hwm(2,1,2,n) = alftrasp3d(1,6)
              ibm_coeff_hwm(1,2,2,n) = alftrasp3d(1,7)
              ibm_coeff_hwm(2,2,2,n) = alftrasp3d(1,8)
             else
              ibm_bc(1,n) = ibm_bc(1,n)-10
             endif
            endif
!
           endif
          enddo
         enddo
        enddo
!
        write(chx,"(I3.3)") ncoords(1)
        write(chz,"(I3.3)") ncoords(3)
!
        open(444,file='ibm_wm_'//chx//'_'//chz//'.bin',form='unformatted')
        write(444) ibm_num_interface
        if (ibm_num_interface>0) then
         write(444) ibm_dist_hwm
         write(444) ibm_xyz_hwm
         write(444) ibm_ijk_hwm
         write(444) ibm_coeff_hwm
        endif
        close(444)
!
        if (masterproc) write(*,*) 'Done with ibm_prepare_wm'
!
        endassociate
    endsubroutine ibm_prepare_wm
!
    subroutine ibm_read_geo(self)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        integer, allocatable, dimension(:) :: ibm_type_bc_tmp
        character(3) :: chx,chz
!
        call self%cfg%get("ibmpar","ibm_type_bc",ibm_type_bc_tmp)
        if (size(ibm_type_bc_tmp)==self%ibm_num_bc) then
         self%ibm_type_bc = ibm_type_bc_tmp
        else
         call fail_input_any("Error! Check number of bc for IBM")
        endif
        if (any(self%ibm_type_bc == 16)) self%ibm_wm = 1
        if (any(self%ibm_type_bc == 18)) self%ibm_wm = 1
!
        write(chx,"(I3.3)") self%field%ncoords(1)
        write(chz,"(I3.3)") self%field%ncoords(3)
!
        open(444,file='ibm_geom_'//chx//'_'//chz//'.bin',form='unformatted')
        read(444) self%ibm_num_interface
        call self%ibm_alloc(step=2)
        if (self%ibm_num_interface>0) then
         read(444) self%ibm_ijk_interface
         read(444) self%ibm_nxyz_interface
         read(444) self%ibm_bc
        endif
        close(444)
!
!        if (self%ibm_wm==1) then
!         open(444,file='ibm_wm_'//chx//'_'//chz//'.bin',form='unformatted')
!         read(444) self%ibm_num_interface
!         if (self%ibm_num_interface>0) then
!          read(444) self%ibm_dist_hwm
!          read(444) self%ibm_xyz_hwm
!          read(444) self%ibm_ijk_hwm
!          read(444) self%ibm_coeff_hwm
!         endif
!         close(444)
!        endif
!
        if (self%masterproc) write(*,*) 'Done with ibm_read_geo'
!
    endsubroutine ibm_read_geo
    subroutine ibm_write_wm_wallprop(self)
        class(equation_multideal_object), intent(inout) :: self
        character(4) :: chx,chz
        character(6) :: chstore
        integer :: l,istat,iermpi
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat => self%nv_stat, &
                  ncoords => self%field%ncoords, itslice_vtr => self%itslice_vtr, ibm_xyz_hwm => self%ibm_xyz_hwm, &
                  ibm_wm_wallprop => self%ibm_wm_wallprop, ibm_num_interface => self%ibm_num_interface)

        if (self%masterproc) write(*,*) 'Writing ibm_wm_wallprop'
        if (self%masterproc) istat = create_folder("WM_WALLPROP")
        call mpi_barrier(mpi_comm_world, iermpi)

        write(chx,'(I4.4)') ncoords(1)
        write(chz,'(I4.4)') ncoords(3)
        write(chstore,'(I6.6)') itslice_vtr

        if (ibm_num_interface>0) then
         open(444,file='WM_WALLPROP/ibm_wm_wallprop_'//chx//'_'//chz//'_'//chstore//'.dat',form='formatted')
         do l=1,ibm_num_interface
          write(444,100) ibm_xyz_hwm(1,l),ibm_xyz_hwm(2,l),ibm_xyz_hwm(3,l),ibm_wm_wallprop(1,l),ibm_wm_wallprop(2,l)
         enddo
         close(444)
        endif
 100    format(20ES20.10)
!
        endassociate
    endsubroutine ibm_write_wm_wallprop

    subroutine ibm_write_wm_stat(self)
        class(equation_multideal_object), intent(inout) :: self
        character(4) :: chx,chz
        integer :: l
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat => self%nv_stat, &
                  ncoords => self%field%ncoords, ibm_wm_stat => self%ibm_wm_stat, &
                  ibm_xyz_hwm => self%ibm_xyz_hwm, ibm_num_interface => self%ibm_num_interface)

        if (self%masterproc) write(*,*) 'Writing ibm_wm_stat1_XXX_XXX.bin'

        write(chx,'(I4.4)') ncoords(1)
        write(chz,'(I4.4)') ncoords(3)

        if (ibm_num_interface>0) then
         open(444,file='ibm_wm_wallprop_stat1_'//chx//'_'//chz//'.bin',form='unformatted')
         write(444) ibm_xyz_hwm,ibm_wm_stat
         close(444)
         open(444,file='ibm_wm_wallprop_stat1_'//chx//'_'//chz//'.dat',form='formatted')
         do l=1,ibm_num_interface
          write(444,100) ibm_xyz_hwm(1,l),ibm_xyz_hwm(2,l),ibm_xyz_hwm(3,l),ibm_wm_stat(1,l),ibm_wm_stat(2,l),&
                                                                            ibm_wm_stat(3,l),ibm_wm_stat(4,l)
         enddo
         close(444)
        endif
 100    format(20ES20.10)
!
        endassociate
    endsubroutine ibm_write_wm_stat
    subroutine ibm_read_wm_stat(self)
        class(equation_multideal_object), intent(inout) :: self
        character(4) :: chx,chz
        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, nv_stat => self%nv_stat, &
                  ncoords => self%field%ncoords, ibm_wm_stat => self%ibm_wm_stat, &
                  ibm_xyz_hwm => self%ibm_xyz_hwm, ibm_num_interface => self%ibm_num_interface)

        if (self%masterproc) write(*,*) "Reading ibm_wm_stat0_XXX_XXX.bin"


        write(chx,'(I4.4)') ncoords(1)
        write(chz,'(I4.4)') ncoords(3)

        if (ibm_num_interface>0) then
         open(444,file='ibm_wm_wallprop_stat0_'//chx//'_'//chz//'.bin',form='unformatted')
         read(444) ibm_xyz_hwm,ibm_wm_stat
         close(444)
        endif
!
        endassociate
    endsubroutine ibm_read_wm_stat
!
    subroutine jcf_initialize(self)
     class(equation_multideal_object), intent(inout) :: self              !< The equation.
     type(phase_t) :: jet
     integer :: l,lsp
     character(2) :: chbcnum
     real(rkind), allocatable, dimension(:) :: tmp_arr1,tmp_arr2
     real(rkind) :: vv,pp,tt,rho,ee
     real(rkind), dimension(N_S) :: yy
     
     call self%cfg%get("jcfpar","jcf_jet_num",self%jcf_jet_num)
     call self%cfg%get("jcfpar","jcf_jet_rad",self%jcf_jet_rad)
     call self%cfg%get("jcfpar","jcf_relax_factor",self%jcf_relax_factor)

     jet = importphase('input_cantera.yaml')

     allocate(self%jcf_parbc(self%jcf_jet_num,self%nv))
     allocate(self%jcf_coords(self%jcf_jet_num,3))
     do l=1,self%jcf_jet_num
      write(chbcnum,'(I2.2)') l
      call self%cfg%get("jcfpar","jcf_bc_var_"//chbcnum,tmp_arr1)
      call self%cfg%get("jcfpar","jcf_coords_"//chbcnum,tmp_arr2)
      self%jcf_coords(l,1:3) = tmp_arr2
      pp = tmp_arr1(1)
      tt = tmp_arr1(2)
      vv = tmp_arr1(3)
      yy = tmp_arr1(4:3+N_S)
      deallocate(tmp_arr1)
      deallocate(tmp_arr2)
      call setState_TPY(jet,tt,pp,yy)
      rho = pp/tt/(R_univ/meanMolecularWeight(jet))
      do lsp=1,N_S
       self%jcf_parbc(l,lsp) = rho*yy(lsp) 
      enddo
      self%jcf_parbc(l,I_U) = 0._rkind
      self%jcf_parbc(l,I_V) = rho*vv 
      self%jcf_parbc(l,I_W) = 0._rkind
      ee = get_e_from_temperature(tt,self%indx_cp_l,self%indx_cp_r,self%cv_coeff,self%nsetcv,self%trange,yy)
      self%jcf_parbc(l,I_E) = rho*(ee+0.5_rkind*vv**2)
      open(10,file='jcf_report.dat')
      if (self%masterproc) then
       write(10,*) 'Jet '//chbcnum//' state:'
       write(10,*) 'gas constant   : ', R_univ/meanMolecularWeight(jet)
       write(10,*) 'cp0            : ', cp_mass(jet)
       write(10,*) 'cv0            : ', cv_mass(jet)
       write(10,*) 'Gamma          : ', cp_mass(jet)/cv_mass(jet)
       write(10,*) 'pressure       : ', pp
       write(10,*) 'density        : ', rho
       write(10,*) 'temperature    : ', tt
       write(10,*) 'viscosity      : ', viscosity(jet)
       write(10,*) 'speed of sound : ', sqrt(cp_mass(jet)/cv_mass(jet)*R_univ/meanMolecularWeight(jet)*tt)
       write(10,*) 'Mach           : ', vv/sqrt(cp_mass(jet)/cv_mass(jet)*R_univ/meanMolecularWeight(jet)*tt)
      endif
     enddo
     close(10)
200    format(100ES20.10)
    endsubroutine jcf_initialize
!    
    subroutine recyc_prepare(self)
        class(equation_multideal_object), intent(inout) :: self
        integer :: ig_recyc, j, k
        real(rkind) :: bexprecyc_base, my_eta0, my_deltablend
        real(rkind) :: alfa_blend, beta_blend, c_blend, yy
        
        call self%cfg%get("bc","x_recyc",self%x_recyc)
        self%x_recyc = self%x_recyc*self%delta0

        associate(ng => self%grid%ng, ny => self%field%ny, nz => self%field%nz)
        allocate(self%yplus_inflow(1-ng:ny+ng))
        allocate(self%eta_inflow(1-ng:ny+ng))
        allocate(self%yplus_recyc(1-ng:ny+ng))
        allocate(self%eta_recyc(1-ng:ny+ng))
        allocate(self%eta_recyc_blend(1-ng:ny+ng))
        allocate(self%map_j_inn(1:ny))
        allocate(self%map_j_out(1:ny))
        allocate(self%map_j_out_blend(1:ny))
        allocate(self%weta_inflow(1:ny))
        if (self%double_bl_case) then
         allocate(self%yplus_inflow2(1-ng:ny+ng))
         allocate(self%eta_inflow2(1-ng:ny+ng))
         allocate(self%yplus_recyc2(1-ng:ny+ng))
         allocate(self%eta_recyc2(1-ng:ny+ng))
         allocate(self%eta_recyc_blend2(1-ng:ny+ng))
         allocate(self%map_j_inn2(1:ny))
         allocate(self%map_j_out2(1:ny))
         allocate(self%map_j_out_blend2(1:ny))
         allocate(self%weta_inflow2(1:ny))
        endif
        allocate(self%inflow_random_plane(1:ny,1:nz,3))
        endassociate

        associate(xg => self%grid%xg, nxmax => self%grid%nxmax, nx => self%field%nx, ny => self%field%ny, &
                  ng => self%grid%ng, xrecyc => self%x_recyc, i_recyc => self%i_recyc, ib_recyc => self%ib_recyc, &
                  deltavec => self%deltavec, deltavvec => self%deltavvec, betarecyc => self%betarecyc, &
                  y => self%field%y, yplus_inflow => self%yplus_inflow, eta_inflow => self%eta_inflow, &
                  yplus_recyc  => self%yplus_recyc,  eta_recyc  => self%eta_recyc, delta0 => self%delta0, &
                  eta_recyc_blend => self%eta_recyc_blend, map_j_out_blend => self%map_j_out_blend, &
                  map_j_inn => self%map_j_inn, map_j_out => self%map_j_out, weta_inflow => self%weta_inflow, &
                  inflow_random_plane => self%inflow_random_plane, glund1 => self%glund1, &
                  nv_recyc => self%nv_recyc, deltavec2 => self%deltavec2, deltavvec2 => self%deltavvec2, &
                  delta02 => self%delta02, betarecyc2 => self%betarecyc2, glund12 => self%glund12, &
                  yplus_inflow2 => self%yplus_inflow2, eta_inflow2 => self%eta_inflow2, &
                  yplus_recyc2  => self%yplus_recyc2,  eta_recyc2  => self%eta_recyc2, &
                  eta_recyc_blend2 => self%eta_recyc_blend2, map_j_out_blend2 => self%map_j_out_blend2, &
                  map_j_inn2 => self%map_j_inn2, map_j_out2 => self%map_j_out2, weta_inflow2 => self%weta_inflow2)

        nv_recyc = self%nv+1-N_S

        inflow_random_plane = 0.5_rkind

        bexprecyc_base = 0.13_rkind
        my_eta0        = 0.08_rkind
        my_deltablend  = 1.10_rkind

        call locateval(xg(1:nxmax),nxmax,xrecyc,ig_recyc) ! xrecyc is between xg(ii) and xg(ii+1), ii is between 0 and nxmax
        ib_recyc = (ig_recyc-1)/nx
        i_recyc  = ig_recyc-nx*ib_recyc
        if (i_recyc<ng) then
         i_recyc  = ng
         ig_recyc = i_recyc+nx*ib_recyc
        endif
        if (self%masterproc) write(*,*) 'Recycling station exactly at = ', xg(ig_recyc)

        betarecyc = deltavvec(ig_recyc)/deltavvec(1)
        glund1    = (deltavec(ig_recyc)/deltavec(1))**bexprecyc_base
        if (self%masterproc) write(*,*) 'Recycling beta factor = ', betarecyc
        if (self%masterproc) write(*,*) 'Pirozzoli-Ceci beta factor = ', glund1
        ! check which Reynolds is the next in the formula
        ! if (self%masterproc) write(*,*) 'xrecyc: ',xrecyc, self%Reynolds
        if (self%masterproc) then
            write(*,*) 'Urbin-Knight beta factor = ', &
                 ((1._rkind+(xrecyc/delta0)*0.27_rkind**1.2_rkind/self%Reynolds**0.2_rkind)**(5._rkind/6._rkind))**0.1_rkind
        endif

        do j=1-ng,ny+ng
         yy = y(j)
         yplus_inflow(j)    = yy/deltavvec(1)
         eta_inflow(j)      = yy/deltavec (1)
         yplus_recyc(j)     = yy/deltavvec(ig_recyc)
         eta_recyc(j)       = yy/deltavec (ig_recyc)
         eta_recyc_blend(j) = yy*((deltavec(1)/deltavec(ig_recyc))**bexprecyc_base+&
                              0.5_rkind*(1._rkind+tanh(log(abs(eta_recyc(j))/my_eta0)/my_deltablend))*&
                              ((deltavec(1)/deltavec(ig_recyc))-&
                              (deltavec(1)/deltavec(ig_recyc))**bexprecyc_base))
        enddo

        alfa_blend = 4._rkind
        beta_blend = .2_rkind
        c_blend = 1._rkind-2._rkind*beta_blend
        do j=1,ny
         call locateval(yplus_recyc(1:ny),ny,yplus_inflow(j),map_j_inn(j))
         call locateval(eta_recyc(1:ny),ny,eta_inflow(j),map_j_out(j))
         call locateval(eta_recyc_blend(1:ny),ny,eta_inflow(j),map_j_out_blend(j))
         weta_inflow(j) = 0.5_rkind*(1._rkind+tanh((alfa_blend*(eta_inflow(j)-beta_blend))&
                          /(beta_blend+eta_inflow(j)*c_blend))/tanh(alfa_blend))
        enddo
 

        if (self%double_bl_case) then
         betarecyc2 = deltavvec2(ig_recyc)/deltavvec2(1)
         glund12    = (deltavec2(ig_recyc)/deltavec2(1))**bexprecyc_base
         if (self%masterproc) write(*,*) 'Recycling beta factor (upper wall) = ', betarecyc2
         if (self%masterproc) write(*,*) 'Pirozzoli-Ceci beta factor (upper wall) = ', glund12
         if (self%masterproc) then
             write(*,*) 'Urbin-Knight beta factor (upper_wall) = ', &
                  ((1._rkind+(xrecyc/delta02)*0.27_rkind**1.2_rkind/self%Reynolds2**0.2_rkind)**(5._rkind/6._rkind))**0.1_rkind
         endif

         do j=1-ng,ny+ng
          yy = y(j)
          yplus_inflow2(j)    = yy/deltavvec2(1)
          eta_inflow2(j)      = yy/deltavec2 (1)
          yplus_recyc2(j)     = yy/deltavvec2(ig_recyc)
          eta_recyc2(j)       = yy/deltavec2 (ig_recyc)
          eta_recyc_blend2(j) = yy*((deltavec2(1)/deltavec2(ig_recyc))**bexprecyc_base+&
                                0.5_rkind*(1._rkind+tanh(log(abs(eta_recyc2(j))/my_eta0)/my_deltablend))*&
                                ((deltavec2(1)/deltavec2(ig_recyc))-&
                                (deltavec2(1)/deltavec2(ig_recyc))**bexprecyc_base))
         enddo
         alfa_blend = 4._rkind
         beta_blend = .2_rkind
         c_blend = 1._rkind-2._rkind*beta_blend
         do j=1,ny
          call locateval(yplus_recyc2(1:ny),ny,yplus_inflow2(j),map_j_inn2(j))
          call locateval(eta_recyc2(1:ny),ny,eta_inflow2(j),map_j_out2(j))
          call locateval(eta_recyc_blend2(1:ny),ny,eta_inflow2(j),map_j_out_blend2(j))
          weta_inflow2(j) = 0.5_rkind*(1._rkind+tanh((alfa_blend*(eta_inflow2(j)-beta_blend))&
                           /(beta_blend+eta_inflow2(j)*c_blend))/tanh(alfa_blend))
         enddo
        endif
        open(10,file='recyc1.dat')
        if (self%double_bl_case) open(11,file='recyc2.dat')
        do j=1-ng,ny+ng
         write(10,'(6ES20.10,3I10,ES20.10)') y(j),yplus_inflow(j),eta_inflow(j),yplus_recyc(j),eta_recyc(j),eta_recyc_blend(j),map_j_inn(j),&
                       map_j_out(j),map_j_out_blend(j),weta_inflow(j)
         if (self%double_bl_case) write(11,'(6ES20.10,3I10,ES20.10)') y(j),yplus_inflow2(j),eta_inflow2(j),yplus_recyc2(j),eta_recyc2(j),eta_recyc_blend2(j),map_j_inn2(j),&
                       map_j_out2(j),map_j_out_blend2(j),weta_inflow2(j)
        enddo
        close(10)
        if (self%double_bl_case) close(11)
!       
!       do j=1,ny
!        call locateval(yplus_recyc(1:ny),ny,yplus_inflow(j),map_j_inn(j))
!        call locateval(eta_recyc(1:ny),ny,eta_inflow(j),map_j_out(j))
!        call locateval(eta_recyc_blend(1:ny),ny,eta_inflow(j),map_j_out_blend(j))
!        weta_inflow(j) = 0.5_rkind*(1._rkind+tanh((4._rkind*(eta_inflow(j)-0.2_rkind))&
!                       /(0.2_rkind+eta_inflow(j)*0.6_rkind))/tanh(4._rkind))
!       enddo
       endassociate
    endsubroutine recyc_prepare
!
    subroutine write_slice_vtr(self)
     class(equation_multideal_object), intent(inout) :: self
     integer :: i, j, k, ni, nj, nk, nimax, njmax, nkmax, it_vtr, size_real, size_integer
     integer :: ii, jj, kk, i_slice, j_slice, k_slice, isize, jsize, ksize, istat
     character(len=14) :: i_folder, j_folder, k_folder
     character(len=14) :: file_prefix
     logical :: my_master
     
     associate(masterproc => self%masterproc, nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, &
               nxmax => self%grid%nxmax, nymax => self%grid%nymax, nzmax => self%grid%nzmax, &
               mp_cart => self%field%mp_cart, mp_cartx => self%field%mp_cartx, mp_cartz => self%field%mp_cartz, &
               nblocks => self%field%nblocks, ncoords => self%field%ncoords, iermpi => self%mpi_err, &
               time => self%time, itslice_vtr => self%itslice_vtr, flow_init => self%flow_init, &
               nv_aux => self%nv_aux, x => self%field%x, y => self%field%y, z => self%field%z, &
               w_aux => self%w_aux, num_aux_slice => self%num_aux_slice, list_aux_slice => self%list_aux_slice, &
               enable_les => self%enable_les, enable_chemistry => self%enable_chemistry)

     if (masterproc) write(*,*) 'Storing slices at time', time

     size_real = storage_size(1._rkind)/8
     size_integer = storage_size(1)/8

     it_vtr      = itslice_vtr
!
     if (allocated(self%islice_vtr)) then
        my_master   = .FALSE.
        if (ncoords(3)==0) my_master = .TRUE.
        i_folder    = "SLICEYZ_??????"
        isize       = size(self%islice_vtr)
        file_prefix = "sliceyz_??????"
        nimax       = 1
        njmax       = nymax
        nkmax       = nzmax
        ni          = 1
        nj          = ny
        nk          = nz
        do ii=1,isize
          i_slice   = self%islice_vtr(ii)
          write(i_folder(9:14),'(I6.6)') i_slice + ncoords(1)*nx
          write(file_prefix(9:14),'(I6.6)') i_slice + ncoords(1)*nx
          if (ncoords(3)==0) istat = create_folder(i_folder)
          call self%field%write_vtk_general(it_vtr, file_prefix, i_folder, &
               nimax, njmax, nkmax, ni, nj, nk, i_slice, i_slice, 1, nj, 1, nk, mp_cartz, my_master, &
               num_aux_slice, list_aux_slice, w_aux(1:nx,1:ny,1:nz,1:nv_aux), self%aux_names)
        enddo
     endif
!          
     if (allocated(self%jslice_vtr)) then
        j_folder    = "SLICEXZ_??????"
        jsize       = size(self%jslice_vtr)
        file_prefix = "slicexz_??????"
        nimax       = nxmax
        njmax       = 1
        nkmax       = nzmax
        ni          = nx
        nj          = 1
        nk          = nz
        do jj=1,jsize
          j_slice   = self%jslice_vtr(jj)
          write(j_folder(9:14),'(I6.6)') j_slice + ncoords(2)*ny
          write(file_prefix(9:14),'(I6.6)') j_slice + ncoords(2)*ny
          if (masterproc) istat = create_folder(j_folder)
          call self%field%write_vtk_general(it_vtr, file_prefix, j_folder, &
               nimax, njmax, nkmax, ni, nj, nk, 1, ni, j_slice, j_slice, 1, nk, mp_cart, masterproc, &
               num_aux_slice, list_aux_slice, w_aux(1:nx,1:ny,1:nz,1:nv_aux), self%aux_names)
        enddo
     endif
!          
     if (allocated(self%kslice_vtr)) then
        my_master   = .FALSE.
        if (ncoords(1)==0) my_master = .TRUE.
        k_folder    = "SLICEXY_??????"
        ksize       = size(self%kslice_vtr)
        file_prefix = "slicexy_??????"
        nimax       = nxmax
        njmax       = nymax
        nkmax       = 1
        ni          = nx
        nj          = ny
        nk          = 1
        do kk=1,ksize
          k_slice   = self%kslice_vtr(kk)
          write(k_folder(9:14),'(I6.6)') k_slice + ncoords(3)*nz
          write(file_prefix(9:14),'(I6.6)') k_slice + ncoords(3)*nz
          if (ncoords(1)==0) istat = create_folder(k_folder)
          call self%field%write_vtk_general(it_vtr, file_prefix, k_folder, &
               nimax, njmax, nkmax, ni, nj, nk, 1, ni, 1, nj, k_slice, k_slice, mp_cartx, my_master, &
               num_aux_slice, list_aux_slice, w_aux(1:nx,1:ny,1:nz,1:nv_aux), self%aux_names)
        enddo
     endif

     endassociate
    endsubroutine write_slice_vtr
!
    subroutine ibm_initialize_old(self)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        integer :: n
        character(3) :: chx,chz
        integer, allocatable, dimension(:) :: ibm_type_bc_tmp
        logical :: file_exists,file_exists_glob
!
!       ibm_num_body =  1 ==> Single-body (multi patch)
!       ibm_num_body >  1 ==> Multi-body (single patch)
!
        call self%cfg%get("ibmpar","ibm_type",self%ibm_type)
        if (self%ibm_num_body>1) self%ibm_type   = 0 ! External flow
!
        call self%cfg%get("ibmpar","ibm_num_bc",self%ibm_num_bc)
!
        call self%cfg%get("ibmpar","ibm_stencil_size",self%ibm_stencil_size)
!
        if (self%cfg%has_key("ibmpar","ibm_bc_relax_factor")) then
         call self%cfg%get("ibmpar","ibm_bc_relax_factor",self%ibm_bc_relax_factor)
        else
         self%ibm_bc_relax_factor = 1._rkind
        endif
!
        if (self%cfg%has_key("ibmpar","ibm_order_reduce")) then
         call self%cfg%get("ibmpar","ibm_order_reduce",self%ibm_order_reduce)
        else
         self%ibm_order_reduce = 0
        endif
!
        if (self%cfg%has_key("ibmpar","ibm_vega_moving")) then
         call self%cfg%get("ibmpar","ibm_vega_moving",self%ibm_vega_moving)
        else
         self%ibm_vega_moving = 0
         self%ibm_trajectory_points = 0
        endif
        if (self%cfg%has_key("ibmpar","ibm_vega_species")) then
         call self%cfg%get("ibmpar","ibm_vega_species",self%ibm_vega_species)
        else
         self%ibm_vega_species = 1
        endif
!
        if (self%cfg%has_key("ibmpar","ibm_eikonal")) then
         call self%cfg%get("ibmpar","ibm_eikonal",self%ibm_eikonal)
        else
         self%ibm_eikonal = 0
        endif
!
        if (self%ibm_vega_moving>0) then
         if (self%restart_type>0) then
          call self%field_info_cfg%get("field_info","ibm_vega_displacement",self%ibm_vega_displacement)
          call self%field_info_cfg%get("field_info","ibm_vega_ymin",self%ibm_vega_ymin)
          call self%field_info_cfg%get("field_info","ibm_vega_ymax",self%ibm_vega_ymax)
          do n=1,self%num_probe
           if (self%moving_probe(n)>0) then
            self%probe_coord(2,n) = self%probe_coord(2,n)+self%ibm_vega_displacement
           endif
          enddo
         endif
        endif
!
        call self%ibm_alloc_old(step=1)
!
        call self%cfg%get("ibmpar","ibm_type_bc",ibm_type_bc_tmp)
        if (size(ibm_type_bc_tmp)==self%ibm_num_bc) then
         self%ibm_type_bc = ibm_type_bc_tmp
        else
         call fail_input_any("Error! Check number of bc for IBM")
        endif
!
        call self%ibm_readoff_old() ! Read geometry in off format
!
        write(chx,"(I3.3)") self%field%ncoords(1)
        write(chz,"(I3.3)") self%field%ncoords(3)
!
        inquire(file='ibm_raytracing_'//chx//'_'//chz//'.bin',exist=file_exists)
        call mpi_allreduce(file_exists,file_exists_glob,1,mpi_logical,mpi_land,mpi_comm_world,self%mpi_err)
        if (file_exists_glob) then
         call self%ibm_raytracing_read_old()
        else
         call self%ibm_raytracing_old()
         call self%ibm_raytracing_write_old()
        endif
!
!       if (self%ibm_eikonal<=0) then
        inquire(file='ibm_geom_'//chx//'_'//chz//'.bin',exist=file_exists)
        call mpi_allreduce(file_exists,file_exists_glob,1,mpi_logical,mpi_land,mpi_comm_world,self%mpi_err)
        if (file_exists_glob) then
         call self%ibm_read_geo_old()
        else
         call self%ibm_compute_geo_old()
        endif
        call self%ibm_coeff_setup_old()
!       endif
!
        call self%ibm_correct_fields_old()
        call self%ibm_setup_computation_old()
!
    end subroutine ibm_initialize_old
!
    subroutine ibm_alloc_old(self,step)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
        integer, intent(in) :: step

        associate(nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng, &
                  ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc, &
                  ibm_eikonal => self%ibm_eikonal, &
                  ibm_vega_moving => self%ibm_vega_moving)

        select case(step)
        case(1)
         allocate(self%ibm_sbody(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
         allocate(self%ibm_is_interface_node(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
         if (ibm_num_body>0) then
          allocate(self%ibm_ptree(ibm_num_body))
          allocate(self%ibm_bbox(6,ibm_num_body))
          allocate(self%ibm_bbox_vega(6))
         endif
         if (ibm_vega_moving>0) then
          allocate(self%ibm_vega_ptree(1))
          allocate(self%ibm_vega_dist(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
         endif
         if (ibm_eikonal>0) then
          allocate(self%ibm_body_dist(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
         endif
         if (ibm_num_bc>1) allocate(self%ibm_ptree_patch(ibm_num_bc)) ! Patch needed only when ibm_num_bc > 1
         if (ibm_num_bc>0) then
          allocate(self%ibm_parbc(ibm_num_bc,IBM_MAX_PARBC))
          allocate(self%ibm_type_bc(ibm_num_bc))
         endif
        case(2)
         if (self%ibm_num_interface>0) then
          allocate(self%ibm_ijk_interface (3,self%ibm_num_interface))  ! Local values of i,j,k for the interface node
          allocate(self%ibm_ijk_refl      (3,self%ibm_num_interface))  ! Reflected node bwtween i,i+1 and j,j+1 and k,k+1
          allocate(self%ibm_ijk_wall      (3,self%ibm_num_interface))  ! Wall node between i,i+1 and j,j+1 and k,k+1
          allocate(self%ibm_nxyz_interface(3,self%ibm_num_interface))  ! Wall-normal components
          allocate(self%ibm_bc            (2,self%ibm_num_interface))  ! Bc tag (1,:) and patch index (2,:) for interface nodes
          allocate(self%ibm_refl_type     (  self%ibm_num_interface))  ! Type of the reflected node
          ! Distance between interface node and wall point (1) and reflected point and wall point (2)
          allocate(self%ibm_dist          (2,self%ibm_num_interface))
          allocate(self%ibm_coeff_d     (2,2,2,self%ibm_num_interface))  ! Coefficients for trilin interpolation (Dirichlet)
          allocate(self%ibm_coeff_n     (2,2,2,self%ibm_num_interface))  ! Coefficients for trilin interpolation (Neumann)
!         The following do not have a corresponding GPU version
          allocate(self%ibm_coeff_tril_d(2,2,2,self%ibm_num_interface))
          allocate(self%ibm_coeff_tril_n(2,2,2,self%ibm_num_interface))
          allocate(self%ibm_coeff_idf   (2,2,2,self%ibm_num_interface))
          allocate(self%ibm_coeff_idfw  (2,2,2,self%ibm_num_interface))
          allocate(self%ibm_dets        (    2,self%ibm_num_interface))
          allocate(self%ibm_refl_insolid(self%ibm_num_interface))
         endif
        end select
!
        end associate
    endsubroutine ibm_alloc_old

    subroutine ibm_raytracing_old(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.

        integer     :: i,j,k
        integer     :: ii,jj,kk
        integer     :: iii,jjj,kkk
        integer     :: ib
        integer     :: imin,imax,jmin,jmax,kmin,kmax
        integer     :: stencil_size
        real(rkind) :: query_x, query_y, query_z
        real(rkind) :: near_x,near_y,near_z,ib_dist
        real(rkind) :: ib_dist_min
        logical     :: is_inside
        logical, dimension(:,:,:), allocatable :: is_body_node
        integer, dimension(:,:,:), allocatable :: sbody_extended
        integer :: num_in_solid, num_in_solid_tot
        integer :: num_interface_tot
!
        associate(masterproc => self%masterproc, ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc, &
                  xg => self%grid%xg, nxmax => self%grid%nxmax, nx => self%field%nx,              &
                  yg => self%grid%yg, nymax => self%grid%nymax, ny => self%field%ny,              &
                  zg => self%grid%zg, nzmax => self%grid%nzmax, nz => self%field%nz,              &
                  x => self%field%x, y => self%field%y, z => self%field%z,                        &
                  ng => self%grid%ng, ncoords => self%field%ncoords,                              &
                  ibm_sbody => self%ibm_sbody, is_xyz_periodic => self%grid%is_xyz_periodic,      &
                  ibm_is_interface_node => self%ibm_is_interface_node, ibm_type => self%ibm_type, &
                  nblocks => self%field%nblocks, ibm_ptree => self%ibm_ptree, &
                  ibm_bbox => self%ibm_bbox, &
                  ibm_ptree_patch => self%ibm_ptree_patch, iermpi => self%mpi_err, &
                  ibm_num_interface => self%ibm_num_interface, &
                  ibm_vega_ptree => self%ibm_vega_ptree, &
                  ibm_vega_dist => self%ibm_vega_dist, &
                  ibm_body_dist => self%ibm_body_dist, &
                  ibm_eikonal => self%ibm_eikonal, &
                  ibm_vega_moving => self%ibm_vega_moving)
!
        stencil_size = self%ibm_stencil_size
        imin =  1-ng-stencil_size
        imax = nx+ng+stencil_size
        jmin =  1-ng-stencil_size
        jmax = ny+ng+stencil_size
        kmin =  1-ng-stencil_size
        kmax = nz+ng+stencil_size
!
        if (ncoords(1)==0.and.(.not.is_xyz_periodic(1)))  imin = 1-ng
        if (ncoords(2)==0.and.(.not.is_xyz_periodic(2)))  jmin = 1-ng
        if (ncoords(3)==0.and.(.not.is_xyz_periodic(3)))  kmin = 1-ng
        if (ncoords(1)==(nblocks(1)-1).and.(.not.is_xyz_periodic(1))) imax = nx+ng
        if (ncoords(2)==(nblocks(2)-1).and.(.not.is_xyz_periodic(2))) jmax = ny+ng
        if (ncoords(3)==(nblocks(3)-1).and.(.not.is_xyz_periodic(3))) kmax = nz+ng

        allocate(is_body_node(imin:imax,jmin:jmax,kmin:kmax))
        allocate(sbody_extended(imin:imax,jmin:jmax,kmin:kmax))
        ! Default fluid
        is_body_node = .false.
        ibm_sbody    = 0
!
        if (masterproc) write(*,*) 'Start raytracing with CGAL'
!
        do k=kmin,kmax
         kk = ncoords(3)*nz+k
!        if (is_xyz_periodic(3)) kk = modulo(kk-1,nzmax)+1
         query_z = zg(kk)
         do j=jmin,jmax
          jj = ncoords(2)*ny+j
          query_y = yg(jj)
          do i=imin,imax
           ii = ncoords(1)*nx+i
!          if (is_xyz_periodic(1)) ii = modulo(ii-1,nxmax)+1
           query_x = xg(ii)
           loopib: do ib=1,ibm_num_body
            if (query_x>ibm_bbox(1,ib).and.query_x<ibm_bbox(4,ib).and.&
                query_y>ibm_bbox(2,ib).and.query_y<ibm_bbox(5,ib).and.&
                query_z>ibm_bbox(3,ib).and.query_z<ibm_bbox(6,ib)) then
             is_inside = cgal_polyhedron_inside(ibm_ptree(ib),query_x,query_y,query_z)
             if (is_inside) then
              is_body_node(i,j,k) = .true.
              sbody_extended(i,j,k) = ib
              exit loopib
             endif
            endif
           enddo loopib
          enddo
         enddo
        enddo
!
        if (ibm_type==1) then
         if (ibm_num_body==1) then
          is_body_node   = .not.is_body_node ! Fluid is inside stl
          sbody_extended = 1-sbody_extended
         else
          if (masterproc) write(*,*) 'Error, ibm_type=1 not supported for multi body'
         endif
        endif
!
        ibm_sbody = sbody_extended(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng)
        deallocate(sbody_extended)
!
        ibm_is_interface_node = 0
        do k=1-ng,nz+ng
         do j=1-ng,ny+ng
          found: do i=1-ng,nx+ng
!
           if (is_body_node(i,j,k)) then ! Solid node
!
            do kk=-stencil_size,stencil_size
             kkk = k+kk
             if (kkk<kmin.or.kkk>kmax) cycle
             if (.not.is_body_node(i,j,kkk)) then
              ibm_is_interface_node(i,j,k) = 1
              cycle found
             endif
            enddo
            do jj=-stencil_size,stencil_size
             jjj = j+jj
             if (jjj<jmin.or.jjj>jmax) cycle
             if (.not.is_body_node(i,jjj,k)) then
              ibm_is_interface_node(i,j,k) = 1
              cycle found
             endif
            enddo
            do ii=-stencil_size,stencil_size
             iii = i+ii
             if (iii<imin.or.iii>imax) cycle
             if (.not.is_body_node(iii,j,k)) then
              ibm_is_interface_node(i,j,k) = 1
              cycle found
             endif
            enddo
!
           endif
          enddo found
         enddo
        enddo
!
!       Count the number of local interface nodes (ghost are excluded)
!
        ibm_num_interface = 0
        num_in_solid = 0
        do k=1,nz
         do j=1,ny
          do i=1,nx
           if (ibm_is_interface_node(i,j,k)==1) ibm_num_interface = ibm_num_interface+1
           if (ibm_sbody(i,j,k) > 0) num_in_solid = num_in_solid+1
          enddo
         enddo
        enddo
!
        call mpi_allreduce(ibm_num_interface,num_interface_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        if (masterproc) write(*,*) 'Total number of interface nodes = ', num_interface_tot
        call mpi_allreduce(num_in_solid,num_in_solid_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        if (masterproc) write(*,*) 'Total number of nodes in the solid body = ', num_in_solid_tot
!
        if (masterproc) write(*,*) 'Done with ibm_raytracing'
!
        deallocate(is_body_node)
!
        if (ibm_vega_moving>0) then
         do k=1-ng,nz+ng
          do j=1-ng,ny+ng
           do i=1-ng,nx+ng
            query_x = x(i)
            query_y = y(j)
            query_z = z(k)
            call polyhedron_closest(ibm_vega_ptree(1),query_x,query_y,query_z,near_x,near_y,near_z)
            ib_dist = sqrt((near_x-query_x)**2+(near_y-query_y)**2+(near_z-query_z)**2)
            is_inside = cgal_polyhedron_inside(ibm_vega_ptree(1),query_x,query_y,query_z)
            if (is_inside) then
             ibm_vega_dist(i,j,k) = ib_dist
            else
             ibm_vega_dist(i,j,k) = -ib_dist
            endif
           enddo
          enddo
         enddo
        endif
!
        if (ibm_eikonal>0) then
         do k=1-ng,nz+ng
          do j=1-ng,ny+ng
           do i=1-ng,nx+ng
            query_x = x(i)
            query_y = y(j)
            query_z = z(k)
            if (ibm_sbody(i,j,k) > 0) then
             ib = ibm_sbody(i,j,k)
             call polyhedron_closest(ibm_ptree(ib),query_x,query_y,query_z,near_x,near_y,near_z)
             ib_dist = sqrt((near_x-query_x)**2+(near_y-query_y)**2+(near_z-query_z)**2)
             ibm_body_dist(i,j,k) = ib_dist
            else
             ib_dist_min = huge(1._rkind)
             do ib=1,ibm_num_body ! Inefficient, to be improved in the future
!                                 (compute distance from bbox, sort and then compute distance)
              call polyhedron_closest(ibm_ptree(ib),query_x,query_y,query_z,near_x,near_y,near_z)
              ib_dist = sqrt((near_x-query_x)**2+(near_y-query_y)**2+(near_z-query_z)**2)
              ib_dist_min = min(ib_dist,ib_dist_min)
             enddo
             ibm_body_dist(i,j,k) = -ib_dist
            endif
           enddo
          enddo
         enddo
        endif
!
        endassociate
    endsubroutine ibm_raytracing_old
   
    subroutine ibm_readoff_old(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.

        integer     :: ib,ip
        real(rkind) :: db_x, db_y, db_z
        real(rkind) :: c_x, c_y, c_z
        real(rkind) :: bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z
        character(128) :: filename
!
        associate(masterproc => self%masterproc, ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc, &
                  ibm_ptree => self%ibm_ptree, ibm_ptree_patch => self%ibm_ptree_patch, ibm_bbox => self%ibm_bbox, &
                  ibm_bbox_vega => self%ibm_bbox_vega, &
                  ibm_vega_moving => self%ibm_vega_moving, ibm_vega_ptree => self%ibm_vega_ptree)
!
        ! Read geometry (ibm_num_body is the number of solid objects)
        do ib=1,ibm_num_body
         filename = "X_YYYYY.off"
         write(filename(3:7), "(I5.5)") ib
         if (masterproc) write(*,*) 'Reading geometry: ', filename
         call cgal_polyhedron_read(ibm_ptree(ib),filename)
        enddo

        if (ibm_vega_moving>0) then
         ! Read Vega C geometry
         filename = "vegac.off"
         call cgal_polyhedron_read(ibm_vega_ptree(1),filename)
         call polyhedron_bbox(ibm_vega_ptree(1),bmin_x,bmin_y,bmin_z,bmax_x,bmax_y,bmax_z)
         ibm_bbox_vega(1:6) = [bmin_x,bmin_y,bmin_z,bmax_x,bmax_y,bmax_z]
         if (masterproc) write(*,*) 'Reading Vega geometry: ', filename
        endif
!
        if (ibm_num_bc>1) then
        ! Read patches (ibm_num_bc, active only for single body)
         do ip=1,ibm_num_bc
          if (masterproc) write(*,*) 'Reading patch #', ip
          filename = "patchxx.off"
          write(filename(6:7),"(I2.2)") ip
          call cgal_polyhedron_read(ibm_ptree_patch(ip),filename)
         enddo
        endif

        ! Compute bounding box (only for ib = 1)
        do ib=1,ibm_num_body
         call polyhedron_bbox(ibm_ptree(ib),bmin_x,bmin_y,bmin_z,bmax_x,bmax_y,bmax_z)
         ibm_bbox(1:6,ib) = [bmin_x,bmin_y,bmin_z,bmax_x,bmax_y,bmax_z]
         db_x = bmax_x - bmin_x ; db_y = bmax_y - bmin_y ; db_z = bmax_z - bmin_z
         c_x = 0.5*(bmax_x + bmin_x) ; c_y = 0.5*(bmax_y + bmin_y) ; c_z = 0.5*(bmax_z + bmin_z)
         if (ib==1) then
          ! if (masterproc) write(*,*) 'Solid center: ', c_x, c_y, c_z
          ! if (masterproc) write(*,*) 'Solid bounding box: ', db_x, db_y, db_z
          if (masterproc) write(*,*) 'Solid limits x: ', c_x-0.5_rkind*db_x,c_x+0.5_rkind*db_x
          if (masterproc) write(*,*) 'Solid limits y: ', c_y-0.5_rkind*db_y,c_y+0.5_rkind*db_y
          if (masterproc) write(*,*) 'Solid limits z: ', c_z-0.5_rkind*db_z,c_z+0.5_rkind*db_z
         endif
        enddo

        if (masterproc) write(*,*) 'Done with ibm_readoff'
!
        endassociate
    endsubroutine ibm_readoff_old

    subroutine ibm_raytracing_write_old(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        character(3) :: chx,chz
!
        associate(ncoords => self%field%ncoords,   &
                  ibm_sbody => self%ibm_sbody,     &
                  ibm_is_interface_node => self%ibm_is_interface_node, &
                  ibm_vega_dist => self%ibm_vega_dist, &
                  ibm_eikonal => self%ibm_eikonal, &
                  ibm_body_dist => self%ibm_body_dist, &
                  ibm_num_interface => self%ibm_num_interface, &
                  ibm_vega_moving => self%ibm_vega_moving)
!
         write(chx,"(I3.3)") ncoords(1)
         write(chz,"(I3.3)") ncoords(3)
!
         open(444,file='ibm_raytracing_'//chx//'_'//chz//'.bin',form='unformatted')
         write(444) ibm_num_interface
         write(444) ibm_sbody
         write(444) ibm_is_interface_node
         if (ibm_vega_moving>0) write(444) ibm_vega_dist
         if (ibm_eikonal>0) write(444) ibm_body_dist
         close(444)
!
        endassociate

    endsubroutine ibm_raytracing_write_old

    subroutine ibm_raytracing_read_old(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        character(3) :: chx,chz
!
        associate(ncoords => self%field%ncoords,   &
                  ibm_sbody => self%ibm_sbody,     &
                  ibm_is_interface_node => self%ibm_is_interface_node, &
                  ibm_vega_dist => self%ibm_vega_dist, &
                  ibm_eikonal => self%ibm_eikonal, &
                  ibm_body_dist => self%ibm_body_dist, &
                  ibm_num_interface => self%ibm_num_interface, &
                  ibm_vega_moving => self%ibm_vega_moving)
!
         write(chx,"(I3.3)") ncoords(1)
         write(chz,"(I3.3)") ncoords(3)
!
         open(444,file='ibm_raytracing_'//chx//'_'//chz//'.bin',form='unformatted')
         read(444) ibm_num_interface
         read(444) ibm_sbody
         read(444) ibm_is_interface_node
         if (ibm_vega_moving>0) read(444) ibm_vega_dist
         if (ibm_eikonal>0) read(444) ibm_body_dist
         close(444)
!
        endassociate

    endsubroutine ibm_raytracing_read_old    

    subroutine ibm_compute_geo_old(self)
        !< Preprocessing for application of the immersed boundary method (IBM)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        integer :: i,j,k,l,n,nout,nout_tot,nrefl_insolidcell,nrefl_insolidcell_tot
        integer :: ii,jj,kk
        integer :: iii,jjj,kkk,lll
!       integer :: dirgood,inters_type,ll
        integer :: ib
        integer :: num_interface_tot
        integer :: typebcloc
        integer :: stencil_size,ishift,jshift,kshift
        integer :: ibm_fix_solid_refl
        integer :: indx_patch
        logical :: in_k,in_j,in_i
        logical :: instl,insolid
        logical :: solidcell, solidcell_bad
        real(rkind) :: query_x, query_y, query_z
!       real(rkind) :: query2_x, query2_y, query2_z
        real(rkind) :: near_x, near_y, near_z
!       real(rkind) :: near_x_new, near_y_new, near_z_new
        real(rkind) :: near2_x, near2_y, near2_z
        real(rkind) :: dist,dbc,dminbc,distmin
        real(rkind) :: rnx,rny,rnz,rnw1,rnw2,rnw3
        real(rkind) :: xrefl,yrefl,zrefl
        real(rkind) :: dxshift,dyshift,dzshift
        real(rkind) :: xref,yref,zref,x0,y0,z0,dxloc,dyloc,dzloc
        real(rkind) :: xyz1,xyz2,xyz3
        real(rkind) :: dinvf,dinvfw,sumidf,sumidfw,dinv2
        real(rkind) :: det_tril_d,det_tril_n
        real(rkind), dimension(8,8) :: amat3d,amat3dtmp
        real(rkind), dimension(1,8) :: xtrasp3d,alftrasp3d
!
        real(rkind), allocatable, dimension(:,:,:,:) :: xyz_wall
        real(rkind), allocatable, dimension(:,:,:,:) :: xyz_refl
        real(rkind), allocatable, dimension(:,:,:,:) :: xyz_n
        real(rkind) :: detmax_d,detmin_d,detmax_n,detmin_n
!       integer, allocatable, dimension(:) :: ibm_type_bc_tmp
        character(3) :: chx,chz
!
        call self%cfg%get("ibmpar","ibm_tol_distance",self%ibm_tol_distance)
!       call self%cfg%get("ibmpar","ibm_type_bc",ibm_type_bc_tmp)
        call self%cfg%get("ibmpar","ibm_fix_solid_refl",ibm_fix_solid_refl)
!       if (size(ibm_type_bc_tmp)==self%ibm_num_bc) then
!        self%ibm_type_bc = ibm_type_bc_tmp
!       else
!        call fail_input_any("Error! Check number of bc for IBM")
!       endif
        call self%ibm_alloc_old(step=2)
!
        associate(masterproc => self%masterproc, ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc, &
                  nx => self%field%nx, ny => self%field%ny, nz => self%field%nz, ng => self%grid%ng,      &
                  ncoords => self%field%ncoords, x => self%field%x, y => self%field%y, z => self%field%z, &
                  ibm_sbody => self%ibm_sbody, is_xyz_periodic => self%grid%is_xyz_periodic,              &
                  ibm_is_interface_node => self%ibm_is_interface_node, ibm_type => self%ibm_type,         &
                  nblocks => self%field%nblocks, domain_size => self%grid%domain_size,                    &
                  iermpi => self%mpi_err, ibm_num_interface => self%ibm_num_interface,                    &
                  ibm_ptree => self%ibm_ptree, ibm_ptree_patch => self%ibm_ptree_patch,                   &
                  ibm_ijk_interface => self%ibm_ijk_interface,   &
                  ibm_ijk_refl => self%ibm_ijk_refl,             &
                  ibm_ijk_wall => self%ibm_ijk_wall,             &
                  ibm_bc       => self%ibm_bc,                   &
                  ibm_nxyz_interface => self%ibm_nxyz_interface, &
                  ibm_dist => self%ibm_dist,                     &
                  ibm_refl_type => self%ibm_refl_type,           &
                  ibm_coeff_d => self%ibm_coeff_d,               &
                  ibm_coeff_n => self%ibm_coeff_n,               &
                  ibm_tol_distance => self%ibm_tol_distance,     &
                  ibm_type_bc => self%ibm_type_bc,               &
                  ibm_coeff_tril_d => self%ibm_coeff_tril_d,     &
                  ibm_coeff_tril_n => self%ibm_coeff_tril_n,     &
                  ibm_coeff_idfw => self%ibm_coeff_idfw,         &
                  ibm_coeff_idf => self%ibm_coeff_idf,           &
                  ibm_refl_insolid => self%ibm_refl_insolid,     &
                  ibm_dets => self%ibm_dets,                     &
                  dcsidx => self%field%dcsidx, detady => self%field%detady, dzitdz => self%field%dzitdz)
!
        stencil_size = self%ibm_stencil_size
!        
        write(*,*) 'xcoord, zcoord and number of interface nodes =', ncoords(1), ncoords(3), ibm_num_interface
!
        call mpi_allreduce(ibm_num_interface,num_interface_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        if (masterproc) then
         open(127,file='report_ibm.dat')
         write(127,*) 'ibm_num_body =', ibm_num_body
         write(127,*) 'ibm_num_bc =', ibm_num_bc
         write(127,*) 'ibm_tol_distance =', ibm_tol_distance
         write(127,*) 'ibm_fix_solid_refl =', ibm_fix_solid_refl
         write(127,*) 'Total number of interface nodes =', num_interface_tot
        endif
        allocate(xyz_wall(3,1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng)) ! Coordinates of wall point (meaningful only for interface nodes)
        allocate(xyz_refl(3,1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng)) ! Coordinates of reflected point (meaningful only for interface nodes)
        ! Components  of normal to the wall point (meaningful only for interface nodes)
        allocate(xyz_n   (3,1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng))
!
        if (ibm_num_interface>0) ibm_refl_insolid = .false.
        xyz_wall = 0._rkind
        xyz_refl = 0._rkind
        xyz_n    = 0._rkind
!
!       Loop over grid to define relevant quantities associated to interface nodes
!
        n    = 0 ! Local counter for interface nodes
        nout = 0 ! Local counter for reflected nodes in the solid
        nrefl_insolidcell = 0 ! Local counter for reflected nodes in solid cell
        if (ibm_num_interface>0) ibm_refl_type = 0
!
        do k=1-ng,nz+ng
         in_k = .true.
         if (k<1.or.k>nz) in_k = .false.
         do j=1-ng,ny+ng
          in_j = .true.
          if (j<1.or.j>ny) in_j = .false.
          do i=1-ng,nx+ng
           in_i = .true.
           if (i<1.or.i>nx) in_i = .false.
!
           if (ibm_is_interface_node(i,j,k)==1) then ! Found interface node (solid)
!
            query_x = x(i) ; query_y = y(j) ; query_z = z(k)
!           if (is_xyz_periodic(1)) query_x = modulo(query_x,domain_size(1))
!           if (is_xyz_periodic(3)) query_z = modulo(query_z,domain_size(3))
!
            ib = ibm_sbody(i,j,k)
            call polyhedron_closest(ibm_ptree(ib),query_x,query_y,query_z,near_x,near_y,near_z)
!
            xyz_wall(1,i,j,k) = near_x
            xyz_wall(2,i,j,k) = near_y
            xyz_wall(3,i,j,k) = near_z
            dist = sqrt((near_x-query_x)**2+(near_y-query_y)**2+(near_z-query_z)**2)
!
            if (dist<ibm_tol_distance) then
             ! recompute distance
            else
             rnx = (near_x-query_x)/dist
             rny = (near_y-query_y)/dist
             rnz = (near_z-query_z)/dist
            endif
!
            xyz_n(1,i,j,k) = rnx
            xyz_n(2,i,j,k) = rny
            xyz_n(3,i,j,k) = rnz
!
            if (in_i.and.in_j.and.in_k) then
!
             n = n+1

             ibm_dist(1,n) = dist
             ibm_dist(2,n) = dist

             xrefl = near_x+rnx*dist
             yrefl = near_y+rny*dist
             zrefl = near_z+rnz*dist
             if (xrefl<x(1 -ng)) call fail_input_any("Image point outside myblock (side 1), try to increase ng")
             if (xrefl>x(nx+ng)) call fail_input_any("Image point outside myblock (side 2), try to increase ng")
             if (yrefl<y(1 -ng)) call fail_input_any("Image point outside myblock (side 3), try to increase ng")
             if (yrefl>y(ny+ng)) call fail_input_any("Image point outside myblock (side 4), try to increase ng")
             if (zrefl<z(1 -ng)) call fail_input_any("Image point outside myblock (side 5), try to increase ng")
             if (zrefl>z(nz+ng)) call fail_input_any("Image point outside myblock (side 6), try to increase ng")
             call locateval(x(1-ng:nx+ng),nx+2*ng,xrefl,ii)
             call locateval(y(1-ng:ny+ng),ny+2*ng,yrefl,jj)
             call locateval(z(1-ng:nz+ng),nz+2*ng,zrefl,kk)
             ii = ii-ng ; jj = jj-ng ; kk = kk-ng
!
!            Check reflected nodes in solid cells (ii,ii+1,jj,jj+1,kk,kk+1)
!
             solidcell = .true.
             loopcheckscell: do kkk=0,1
              do jjj=0,1
               do iii=0,1
                if (ibm_sbody(ii+iii,jj+jjj,kk+kkk)==0) then
                 solidcell = .false.
                 exit loopcheckscell
                endif
               enddo
              enddo
             enddo loopcheckscell
!
             if (solidcell) then
              nrefl_insolidcell = nrefl_insolidcell+1
!
              if (ibm_fix_solid_refl>0) then
!
!              Try to fix reflected
!
               distmin = huge(1._rkind)
               do kk=1,stencil_size
                if (ibm_sbody(i,j,k-kk)==0) then
                 dist = abs(z(k-kk)-z(k))
                 if (dist<distmin) then
                  ishift = 0; jshift = 0; kshift = -kk
                  rnx = 0._rkind; rny = 0._rkind; rnz = -1._rkind
                  distmin = dist
                 endif
                endif
                if (ibm_sbody(i,j,k+kk)==0) then
                 dist = abs(z(k+kk)-z(k))
                 if (dist<distmin) then
                  ishift = 0; jshift = 0; kshift =  kk
                  rnx = 0._rkind; rny = 0._rkind; rnz = 1._rkind
                  distmin = dist
                 endif
                endif
               enddo
               do jj=1,stencil_size
                if (ibm_sbody(i,j-jj,k)==0) then
                 dist = abs(y(j-jj)-y(j))
                 if (dist<distmin) then
                  ishift = 0; jshift = -jj; kshift = 0
                  rnx = 0._rkind; rny = -1._rkind; rnz = 0._rkind
                  distmin = dist
                 endif
                endif
                if (ibm_sbody(i,j+jj,k)==0) then
                 dist = abs(y(j+jj)-y(j))
                 if (dist<distmin) then
                  ishift = 0; jshift = jj; kshift =  0
                  rnx = 0._rkind; rny = 1._rkind; rnz = 0._rkind
                  distmin = dist
                 endif
                endif
               enddo
               do ii=1,stencil_size
                if (ibm_sbody(i-ii,j,k)==0) then
                 dist = abs(x(i-ii)-x(i))
                 if (dist<distmin) then
                  ishift = -ii; jshift = 0; kshift = 0
                  rnx = -1._rkind; rny = 0._rkind; rnz = 0._rkind
                  distmin = dist
                 endif
                endif
                if (ibm_sbody(i+ii,j,k)==0) then
                 dist = abs(x(i+ii)-x(i))
                 if (dist<distmin) then
                  ishift = ii; jshift = 0; kshift =  0
                  rnx = 1._rkind; rny = 0._rkind; rnz = 0._rkind
                  distmin = dist
                 endif
                endif
               enddo
               dxshift = 0._rkind
               dyshift = 0._rkind
               dzshift = 0._rkind
               if (ishift<0) dxshift = -0.5_rkind/dcsidx(i+ishift)
               if (ishift>0) dxshift =  0.5_rkind/dcsidx(i+ishift)
               if (jshift<0) dyshift = -0.5_rkind/detady(j+jshift)
               if (jshift>0) dyshift =  0.5_rkind/detady(j+jshift)
               if (kshift<0) dzshift = -0.5_rkind/dzitdz(k+kshift)
               if (kshift>0) dzshift =  0.5_rkind/dzitdz(k+kshift)
               xrefl = x(i+ishift)+dxshift
               yrefl = y(j+jshift)+dyshift
               zrefl = z(k+kshift)+dzshift
               dist = sqrt((xrefl-x(i))**2+(yrefl-y(j))**2+(zrefl-z(k))**2)
               ibm_dist(1,n) = 0.5_rkind*dist
               ibm_dist(2,n) = 0.5_rkind*dist
               xyz_wall(1,i,j,k) = 0.5_rkind*(xrefl+x(i))
               xyz_wall(2,i,j,k) = 0.5_rkind*(yrefl+y(j))
               xyz_wall(3,i,j,k) = 0.5_rkind*(zrefl+z(k))
               xyz_n(1,i,j,k) = rnx
               xyz_n(2,i,j,k) = rny
               xyz_n(3,i,j,k) = rnz
!              distmin = huge(1._rkind)
!              do ll=1,6
!               ishift = 0
!               jshift = 0
!               kshift = 0
!               select case(ll)
!               case(1)
!                ishift = -stencil_size
!               case(2)
!                ishift =  stencil_size
!               case(3)
!                jshift = -stencil_size
!               case(4)
!                jshift =  stencil_size
!               case(5)
!                kshift = -stencil_size
!               case(6)
!                kshift =  stencil_size
!               end select
!               query2_x = x(i+ishift) ; query2_y = y(j+jshift) ; query2_z = z(k+kshift)
!               call cgal_polyhedron_intersection(ibm_ptree(ib),query_x,query_y,query_z, &
!                                                 query2_x,query2_y,query2_z,near_x_new,near_y_new,near_z_new,inters_type)
!               dist = sqrt((near_x_new-query_x)**2+(near_y_new-query_y)**2+(near_z_new-query_z)**2)
!               if (inters_type==1) then
!                if (dist<distmin) then
!                 distmin = dist
!                 dirgood = ll
!                endif
!               endif
!              enddo
! 
!              ishift = 0
!              jshift = 0
!              kshift = 0
!              rnx = 0._rkind
!              rny = 0._rkind
!              rnz = 0._rkind
!              select case(dirgood)
!              case(1)
!               ishift = -stencil_size
!               rnx    = -1._rkind
!              case(2)
!               ishift =  stencil_size
!               rnx    =  1._rkind
!              case(3)
!               jshift = -stencil_size
!               rny    = -1._rkind
!              case(4)
!               jshift =  stencil_size
!               rny    =  1._rkind
!              case(5)
!               kshift = -stencil_size
!               rnz    = -1._rkind
!              case(6)
!               kshift =  stencil_size
!               rnz    =  1._rkind
!              end select
!              query2_x = x(i+ishift) ; query2_y = y(j+jshift) ; query2_z = z(k+kshift)
!              call cgal_polyhedron_intersection(ibm_ptree(ib),query_x,query_y,query_z, &
!                                                query2_x,query2_y,query2_z,near_x_new,near_y_new,near_z_new,inters_type)
!              dist = sqrt((near_x_new-query_x)**2+(near_y_new-query_y)**2+(near_z_new-query_z)**2)
!              xrefl = near_x_new+rnx*dist
!              yrefl = near_y_new+rny*dist
!              zrefl = near_z_new+rnz*dist
!              ibm_dist(1,n) = dist
!              ibm_dist(2,n) = dist
!              xyz_wall(1,i,j,k) = near_x_new
!              xyz_wall(2,i,j,k) = near_y_new
!              xyz_wall(3,i,j,k) = near_z_new
!              xyz_n(1,i,j,k) = rnx
!              xyz_n(2,i,j,k) = rny
!              xyz_n(3,i,j,k) = rnz
              endif
             endif
!
             instl   = cgal_polyhedron_inside(ibm_ptree(ib),xrefl,yrefl,zrefl)
             insolid = instl
             if (ibm_type==1) insolid = .not.instl
             if (insolid) then
              nout = nout+1 ! Image point in the solid
              ibm_refl_insolid(n) = .true.
             endif
!
             xyz_refl(1,i,j,k) = xrefl
             xyz_refl(2,i,j,k) = yrefl
             xyz_refl(3,i,j,k) = zrefl
!
             if (xrefl<x(1 -ng)) call fail_input_any("Image point outside myblock (side 1), try to increase ng")
             if (xrefl>x(nx+ng)) call fail_input_any("Image point outside myblock (side 2), try to increase ng")
             if (yrefl<y(1 -ng)) call fail_input_any("Image point outside myblock (side 3), try to increase ng")
             if (yrefl>y(ny+ng)) call fail_input_any("Image point outside myblock (side 4), try to increase ng")
             if (zrefl<z(1 -ng)) call fail_input_any("Image point outside myblock (side 5), try to increase ng")
             if (zrefl>z(nz+ng)) call fail_input_any("Image point outside myblock (side 6), try to increase ng")
!
             call locateval(x(1-ng:nx+ng),nx+2*ng,xrefl,ii)
             call locateval(y(1-ng:ny+ng),ny+2*ng,yrefl,jj)
             call locateval(z(1-ng:nz+ng),nz+2*ng,zrefl,kk)
             ii = ii-ng ; jj = jj-ng ; kk = kk-ng
!
             ibm_ijk_interface (1,n) = i
             ibm_ijk_interface (2,n) = j
             ibm_ijk_interface (3,n) = k
             ibm_ijk_refl      (1,n) = ii
             ibm_ijk_refl      (2,n) = jj
             ibm_ijk_refl      (3,n) = kk
             ibm_nxyz_interface(1,n) = rnx
             ibm_nxyz_interface(2,n) = rny
             ibm_nxyz_interface(3,n) = rnz
!
             if (near_x<x(1 -ng)) call fail_input_any("Wall point outside myblock (side 1)")
             if (near_x>x(nx+ng)) call fail_input_any("Wall point outside myblock (side 2)")
             if (near_y<y(1 -ng)) call fail_input_any("Wall point outside myblock (side 3)")
             if (near_y>y(ny+ng)) call fail_input_any("Wall point outside myblock (side 4)")
             if (near_z<z(1 -ng)) call fail_input_any("Wall point outside myblock (side 5)")
             if (near_z>z(nz+ng)) call fail_input_any("Wall point outside myblock (side 6)")
             call locateval(x(1-ng:nx+ng),nx+2*ng,near_x,ii)
             call locateval(y(1-ng:ny+ng),ny+2*ng,near_y,jj)
             call locateval(z(1-ng:nz+ng),nz+2*ng,near_z,kk)
             ii = ii-ng ; jj = jj-ng ; kk = kk-ng
!
             ibm_ijk_wall(1,n) = ii
             ibm_ijk_wall(2,n) = jj
             ibm_ijk_wall(3,n) = kk
!
            endif
!
           endif
!
          enddo
         enddo
        enddo
!
        call mpi_allreduce(nout,nout_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(nrefl_insolidcell,nrefl_insolidcell_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        if (masterproc) write(*,*) 'Total number of reflected nodes in the solid = ', nout_tot
        if (masterproc) write(127,*) 'Total number of reflected nodes in the solid = ', nout_tot
        if (masterproc) write(*,*) 'Total number of reflected nodes in a solid cell = ', nrefl_insolidcell_tot
        if (masterproc) write(127,*) 'Total number of reflected nodes in a solid cell = ', nrefl_insolidcell_tot
!
        detmax_d  = 0._rkind
        detmin_d  = huge(1._rkind)
        detmax_n  = 0._rkind
        detmin_n  = huge(1._rkind)
        write(chx,"(I3.3)") ncoords(1)
        write(chz,"(I3.3)") ncoords(3)
        open(444,file='ibm_det_d_'//chx//'_'//chz//'.dat',form='formatted')
        open(445,file='ibm_det_n_'//chx//'_'//chz//'.dat',form='formatted')
!
        do n=1,ibm_num_interface
!
         ii = ibm_ijk_refl(1,n)
         jj = ibm_ijk_refl(2,n)
         kk = ibm_ijk_refl(3,n)
!
!        Check reflected nodes in solid cells
!
         solidcell = .true.
         loopcheckcell: do kkk=0,1
          do jjj=0,1
           do iii=0,1
            if (ibm_sbody(ii+iii,jj+jjj,kk+kkk)==0) then
             solidcell = .false.
             exit loopcheckcell
            endif
           enddo
          enddo
         enddo loopcheckcell
!
         if (solidcell) then
          solidcell_bad = .true.
          secondloopcheckcell: do kkk=0,1
           do jjj=0,1
            do iii=0,1
             if (ibm_is_interface_node(ii+iii,jj+jjj,kk+kkk)==1) then
              solidcell_bad = .false.
              exit secondloopcheckcell
             endif
            enddo
           enddo
          enddo secondloopcheckcell
          if (solidcell_bad) then
           call fail_input_any("Reflected node in solid cell with all non-interface nodes")
          else
           ibm_refl_type(n) = 3
          endif
         endif
!
!        Identify reflected node type (0 ==> Fully fluid, 
!                                      1 ==> NFF with at least 1 solid interface, 
!                                      2 ==> NFF with at least 1 solid not interface, 
!                                      3 ==> Fully solid but at least 1 interface)
!
         if (ibm_refl_type(n)/=3) then
          looptype: do kkk=0,1
           do jjj=0,1
            do iii=0,1
             if (ibm_sbody(ii+iii,jj+jjj,kk+kkk)>0) then
              if (ibm_is_interface_node(ii+iii,jj+jjj,kk+kkk)==1) then
               ibm_refl_type(n) = 1
              else
               ibm_refl_type(n) = 2
               exit looptype
              endif
             endif
            enddo
           enddo
          enddo looptype
         endif
!
         i = ibm_ijk_interface(1,n)
         j = ibm_ijk_interface(2,n)
         k = ibm_ijk_interface(3,n)
         xrefl = xyz_refl(1,i,j,k)
         yrefl = xyz_refl(2,i,j,k)
         zrefl = xyz_refl(3,i,j,k)
!
!        Trilinear Dirichlet and Neumann
!
         x0 = x(ii)
         y0 = y(jj)
         z0 = z(kk)
!
         dxloc = x(ii+1)-x(ii)
         dyloc = y(jj+1)-y(jj)
         dzloc = z(kk+1)-z(kk)
         xref = (xrefl-x0)/dxloc
         yref = (yrefl-y0)/dyloc
         zref = (zrefl-z0)/dzloc
!
         xtrasp3d(1,1) = xref*yref*zref
         xtrasp3d(1,2) = xref*yref
         xtrasp3d(1,3) = xref*zref
         xtrasp3d(1,4) = yref*zref
         xtrasp3d(1,5) = xref
         xtrasp3d(1,6) = yref
         xtrasp3d(1,7) = zref
         xtrasp3d(1,8) = 1._rkind
!
!        Dirichlet
         lll = 0
         do kkk=0,1
          do jjj=0,1
           do iii=0,1
            lll = lll+1
            if (ibm_is_interface_node(ii+iii,jj+jjj,kk+kkk)==1) then
             xyz1 = xyz_wall(1,ii+iii,jj+jjj,kk+kkk)
             xyz2 = xyz_wall(2,ii+iii,jj+jjj,kk+kkk)
             xyz3 = xyz_wall(3,ii+iii,jj+jjj,kk+kkk)
            else
             xyz1 = x(ii+iii)
             xyz2 = y(jj+jjj)
             xyz3 = z(kk+kkk)
            endif
            xyz1 = (xyz1-x0)/dxloc
            xyz2 = (xyz2-y0)/dyloc
            xyz3 = (xyz3-z0)/dzloc
            amat3d(lll,:) = [xyz1*xyz2*xyz3,xyz1*xyz2,xyz1*xyz3,xyz2*xyz3,xyz1,xyz2,xyz3,1._rkind]
           enddo
          enddo
         enddo
!
         amat3dtmp = amat3d
         call detmat(amat3dtmp,8,det_tril_d)
         ibm_dets(1,n) = det_tril_d
         if (ibm_refl_type(n)/=0) write(444,*) det_tril_d
         detmax_d = max(abs(det_tril_d),detmax_d)
         detmin_d = min(abs(det_tril_d),detmin_d)
!
         call invmat(amat3d,8)
         alftrasp3d = matmul(xtrasp3d,amat3d)
         ibm_coeff_tril_d(1,1,1,n) = alftrasp3d(1,1)
         ibm_coeff_tril_d(2,1,1,n) = alftrasp3d(1,2)
         ibm_coeff_tril_d(1,2,1,n) = alftrasp3d(1,3)
         ibm_coeff_tril_d(2,2,1,n) = alftrasp3d(1,4)
         ibm_coeff_tril_d(1,1,2,n) = alftrasp3d(1,5)
         ibm_coeff_tril_d(2,1,2,n) = alftrasp3d(1,6)
         ibm_coeff_tril_d(1,2,2,n) = alftrasp3d(1,7)
         ibm_coeff_tril_d(2,2,2,n) = alftrasp3d(1,8)
!
!        Neumann
!        
         lll = 0
         do kkk=0,1
          do jjj=0,1
           do iii=0,1
            lll = lll+1
            if (ibm_is_interface_node(ii+iii,jj+jjj,kk+kkk)==1) then
             xyz1 = xyz_wall(1,ii+iii,jj+jjj,kk+kkk)
             xyz2 = xyz_wall(2,ii+iii,jj+jjj,kk+kkk)
             xyz3 = xyz_wall(3,ii+iii,jj+jjj,kk+kkk)
             xyz1 = (xyz1-x0)/dxloc
             xyz2 = (xyz2-y0)/dyloc
             xyz3 = (xyz3-z0)/dzloc
             rnw1 = xyz_n (1,ii+iii,jj+jjj,kk+kkk)
             rnw2 = xyz_n (2,ii+iii,jj+jjj,kk+kkk)
             rnw3 = xyz_n (3,ii+iii,jj+jjj,kk+kkk)
             amat3d(lll,:) = [xyz2*xyz3*rnw1+xyz1*xyz3*rnw2+xyz1*xyz2*rnw3,&
                              xyz2*rnw1+xyz1*rnw2,&
                              xyz3*rnw1+xyz1*rnw3,&
                              xyz2*rnw3+xyz3*rnw2,&
                              rnw1,rnw2,rnw3,0._rkind]
            else
             xyz1 = x(ii+iii)
             xyz2 = y(jj+jjj)
             xyz3 = z(kk+kkk)
             xyz1 = (xyz1-x0)/dxloc
             xyz2 = (xyz2-y0)/dyloc
             xyz3 = (xyz3-z0)/dzloc
             amat3d(lll,:) = [xyz1*xyz2*xyz3,xyz1*xyz2,xyz1*xyz3,xyz2*xyz3,xyz1,xyz2,xyz3,1._rkind]
            endif
           enddo
          enddo
         enddo
!
         amat3dtmp = amat3d
         call detmat(amat3dtmp,8,det_tril_n)
         ibm_dets(2,n) = det_tril_n
         if (ibm_refl_type(n)/=0) write(445,*) det_tril_n
         detmax_n = max(abs(det_tril_n),detmax_n)
         detmin_n = min(abs(det_tril_n),detmin_n)
!
         call invmat(amat3d,8)
         alftrasp3d = matmul(xtrasp3d,amat3d)
         ibm_coeff_tril_n(1,1,1,n) = alftrasp3d(1,1)
         ibm_coeff_tril_n(2,1,1,n) = alftrasp3d(1,2)
         ibm_coeff_tril_n(1,2,1,n) = alftrasp3d(1,3)
         ibm_coeff_tril_n(2,2,1,n) = alftrasp3d(1,4)
         ibm_coeff_tril_n(1,1,2,n) = alftrasp3d(1,5)
         ibm_coeff_tril_n(2,1,2,n) = alftrasp3d(1,6)
         ibm_coeff_tril_n(1,2,2,n) = alftrasp3d(1,7)
         ibm_coeff_tril_n(2,2,2,n) = alftrasp3d(1,8)
!
!        Inverse distance
!
         sumidf  = 0._rkind
         sumidfw = 0._rkind
         do kkk=0,1
          do jjj=0,1
           do iii=0,1
            dinv2 = (xrefl-x(ii+iii))**2+(yrefl-y(jj+jjj))**2+(zrefl-z(kk+kkk))**2
            if (ibm_sbody(ii+iii,jj+jjj,kk+kkk)==0) then ! Fluid
             if (dinv2<tol_iter) dinv2 = tol_iter
             dinvf  = 1._rkind/dinv2
             dinvfw = 1._rkind/dinv2
            else
             dinvf = 0._rkind
             if (ibm_is_interface_node(ii+iii,jj+jjj,kk+kkk)==1) then
              xyz1 = xyz_wall(1,ii+iii,jj+jjj,kk+kkk)
              xyz2 = xyz_wall(2,ii+iii,jj+jjj,kk+kkk)
              xyz3 = xyz_wall(3,ii+iii,jj+jjj,kk+kkk)
              dinv2 = (xrefl-xyz1)**2+(yrefl-xyz2)**2+(zrefl-xyz3)**2
              if (dinv2<tol_iter) dinv2 = tol_iter
              dinvfw = 1._rkind/dinv2
             else
              dinvfw = 0._rkind
             endif
            endif
            ibm_coeff_idf (iii+1,jjj+1,kkk+1,n) = dinvf
            ibm_coeff_idfw(iii+1,jjj+1,kkk+1,n) = dinvfw
            sumidf  = sumidf +dinvf
            sumidfw = sumidfw+dinvfw
           enddo
          enddo
         enddo
         ibm_coeff_idf (:,:,:,n) = ibm_coeff_idf (:,:,:,n)/sumidf
         ibm_coeff_idfw(:,:,:,n) = ibm_coeff_idfw(:,:,:,n)/sumidfw
!
        enddo
        close(444)
        close(445)
!
        call mpi_allreduce(MPI_IN_PLACE,detmax_d,1,mpi_prec,mpi_max,mpi_comm_world,iermpi)
        call mpi_allreduce(MPI_IN_PLACE,detmin_d,1,mpi_prec,mpi_min,mpi_comm_world,iermpi)
        call mpi_allreduce(MPI_IN_PLACE,detmax_n,1,mpi_prec,mpi_max,mpi_comm_world,iermpi)
        call mpi_allreduce(MPI_IN_PLACE,detmin_n,1,mpi_prec,mpi_min,mpi_comm_world,iermpi)
!
        if (masterproc) write(*  ,*) 'Max det = ', detmax_d, detmax_n
        if (masterproc) write(*  ,*) 'Min det = ', detmin_d, detmin_n
        if (masterproc) write(127,*) 'Max det = ', detmax_d, detmax_n
        if (masterproc) write(127,*) 'Min det = ', detmin_d, detmin_n
        if (masterproc) close(127)
!
        if (ibm_num_interface>0) then
         if (ibm_num_bc==1) then
          ibm_bc(1,:) = ibm_type_bc(1) ! Default bc
          ibm_bc(2,:) = 1
         else
          do n=1,ibm_num_interface
           i = ibm_ijk_interface(1,n)
           j = ibm_ijk_interface(2,n)
           k = ibm_ijk_interface(3,n)
           dminbc = huge(1._rkind)
           near_x = xyz_wall(1,i,j,k)
           near_y = xyz_wall(2,i,j,k)
           near_z = xyz_wall(3,i,j,k)
           do l=1,ibm_num_bc
            query_x = near_x
            query_y = near_y
            query_z = near_z
            call polyhedron_closest(ibm_ptree_patch(l),query_x,query_y,query_z,near2_x,near2_y,near2_z)
            dbc = (near2_x-query_x)**2+(near2_y-query_y)**2+(near2_z-query_z)**2
            if (dbc<dminbc) then
             dminbc = dbc
             typebcloc  = ibm_type_bc(l)
             indx_patch = l
            endif
           enddo
           ibm_bc(1,n) = typebcloc
           ibm_bc(2,n) = indx_patch
          enddo
         endif
        endif
!
        write(chx,"(I3.3)") ncoords(1)
        write(chz,"(I3.3)") ncoords(3)
!
        open(444,file='ibm_geom_'//chx//'_'//chz//'.bin',form='unformatted')
        write(444) ibm_num_interface
        if (ibm_num_interface>0) then
         write(444) ibm_ijk_interface
         write(444) ibm_ijk_refl
         write(444) ibm_ijk_wall
         write(444) ibm_nxyz_interface
         write(444) ibm_bc
         write(444) ibm_refl_type
         write(444) ibm_dist
!        write(444) ibm_coeff_d
!        write(444) ibm_coeff_n
         write(444) ibm_coeff_tril_d
         write(444) ibm_coeff_tril_n
         write(444) ibm_coeff_idfw
         write(444) ibm_coeff_idf
         write(444) ibm_dets
         write(444) ibm_refl_insolid
        endif
        close(444)
!
        if (masterproc) write(*,*) 'Done with ibm_compute_geo'
!
        endassociate
    endsubroutine ibm_compute_geo_old

    subroutine ibm_read_geo_old(self)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        integer, allocatable, dimension(:) :: ibm_type_bc_tmp
        character(3) :: chx,chz
!
        call self%cfg%get("ibmpar","ibm_type_bc",ibm_type_bc_tmp)
        if (size(ibm_type_bc_tmp)==self%ibm_num_bc) then
         self%ibm_type_bc = ibm_type_bc_tmp
        else
         call fail_input_any("Error! Check number of bc for IBM")
        endif
!
        write(chx,"(I3.3)") self%field%ncoords(1)
        write(chz,"(I3.3)") self%field%ncoords(3)
!
        open(444,file='ibm_geom_'//chx//'_'//chz//'.bin',form='unformatted')
        read(444) self%ibm_num_interface
        call self%ibm_alloc_old(step=2)
        if (self%ibm_num_interface>0) then
         read(444) self%ibm_ijk_interface
         read(444) self%ibm_ijk_refl
         read(444) self%ibm_ijk_wall
         read(444) self%ibm_nxyz_interface
         read(444) self%ibm_bc
         read(444) self%ibm_refl_type
         read(444) self%ibm_dist
!        read(444) self%ibm_coeff_d
!        read(444) self%ibm_coeff_n
         read(444) self%ibm_coeff_tril_d
         read(444) self%ibm_coeff_tril_n
         read(444) self%ibm_coeff_idfw
         read(444) self%ibm_coeff_idf
         read(444) self%ibm_dets
         read(444) self%ibm_refl_insolid
        endif
        close(444)
!
        if (self%masterproc) write(*,*) 'Done with ibm_read_geo'
!
    endsubroutine ibm_read_geo_old

    subroutine ibm_correct_fields_old(self)
        class(equation_multideal_object), intent(inout) :: self
!
        integer :: i,j,k,l
        integer :: stencil_size
        real(rkind) :: rho,rhou,rhov,rhow,rhoe,uu,vv,ww,qq
!
        associate(nx => self%field%nx,ny => self%field%ny,nz => self%field%nz, &
                  ng => self%grid%ng)
!
        do k=1-ng,nz+ng
         do j=1-ng,ny+ng
          do i=1-ng,nx+ng
           if (self%ibm_sbody(i,j,k)>0) then ! solid
            self%fluid_mask(i,j,k) = 1
            if (self%ibm_is_interface_node(i,j,k) == 0) then ! no interface
             rho  = findrho(N_S,self%field%w(1:N_S,i,j,k))
             rhou = self%field%w(I_U,i,j,k)
             rhov = self%field%w(I_V,i,j,k)
             rhow = self%field%w(I_W,i,j,k)
             rhoe = self%field%w(I_E,i,j,k)
             uu   = rhou/rho
             vv   = rhov/rho
             ww   = rhow/rho
             qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
             self%field%w(I_U,i,j,k) = 0._rkind
             self%field%w(I_V,i,j,k) = 0._rkind
             self%field%w(I_W,i,j,k) = 0._rkind
             self%field%w(I_E,i,j,k) = rhoe-rho*qq
            else
            endif
           endif
          enddo
         enddo
        enddo
!
!       i direction
        stencil_size = self%ibm_stencil_size+1
        do k=1,nz
         do j=1,ny
          idir: do i=0,nx
           do l=1,stencil_size-1
            if (self%ibm_sbody(i+l,j,k)>0) then
             self%ep_ord_change(i,j,k,1) = self%ibm_order_reduce ! -1 !-(stencil_size-l) 
             cycle idir
            endif
            if (self%ibm_sbody(i-l,j,k)>0) then
             self%ep_ord_change(i,j,k,1) = self%ibm_order_reduce ! -1 !-(stencil_size-l)
             cycle idir
            endif
           enddo
          enddo idir
         enddo
        enddo
!
!       j direction
        do k=1,nz
         do i=1,nx
          jdir: do j=0,ny
           do l=1,stencil_size-1
            if (self%ibm_sbody(i,j+l,k)>0) then
             self%ep_ord_change(i,j,k,2) = self%ibm_order_reduce ! -1 !-(stencil_size-l)
             cycle jdir
            endif
            if (self%ibm_sbody(i,j-l,k)>0) then
             self%ep_ord_change(i,j,k,2) = self%ibm_order_reduce ! -1 !-(stencil_size-l)
             cycle jdir
            endif
           enddo
          enddo jdir
         enddo
        enddo
!       k direction
        do j=1,ny
         do i=1,nx
          kdir: do k=0,nz
           do l=1,stencil_size-1
            if (self%ibm_sbody(i,j,k+l)>0) then
             self%ep_ord_change(i,j,k,3) = self%ibm_order_reduce ! -1 !-(stencil_size-l)
             cycle kdir
            endif
            if (self%ibm_sbody(i,j,k-l)>0) then
             self%ep_ord_change(i,j,k,3) = self%ibm_order_reduce ! -1 !-(stencil_size-l)
             cycle kdir
            endif
           enddo
          enddo kdir
         enddo
        enddo

        endassociate
!
    end subroutine ibm_correct_fields_old

    subroutine ibm_setup_computation_old(self)
        class(equation_multideal_object), intent(inout) :: self
!
        integer :: i,j,k,l,ierr,lefbil,lsp
        real(rkind) :: volchanl,dyl,yatm
        logical :: ramp_exists
        logical :: done,done_global
        real(rkind), allocatable, dimension(:) :: ibm_aero_par
        real(rkind), allocatable, dimension(:) :: tmp_arr1,tmp_arr2,tmp_arr3,tmp_arr4,tmp_arr5
        !real(rkind), parameter :: rintinv = 436.681222707_rkind, nexp = 0.151515151515_rkind, xp = 0.005_rkind,yp = 0.005_rkind, zp = 0.005_rkind
        !real(rkind) :: rad,ur,sum_sq_local,sum_sq_global,rms
        !integer :: counter_local,counter_global
        character(2) :: chbcnum
!
        associate(nx => self%field%nx,ny => self%field%ny,nz => self%field%nz, &
                  ng => self%grid%ng, volchan => self%volchan, yn => self%grid%yn, &
                  ibm_bbox_vega => self%ibm_bbox_vega, &
                  masterproc => self%masterproc, iermpi => self%mpi_err, &
                  ibm_num_bc => self%ibm_num_bc, ibm_type_bc => self%ibm_type_bc, ibm_aeroacoustics => self%ibm_aeroacoustics, x => self%field%x, y => self%field%y, z => self%field%z)
!
        if (self%ibm_vega_moving>0) then
!        open(12,file='vegac.dat')
!        read(12,*) self%ibm_vega_ny, self%ibm_vega_dy
!        allocate(self%ibm_vega_y(1:self%ibm_vega_ny+1))
!        allocate(self%ibm_vega_r(1:self%ibm_vega_ny+1))
!        do j=1,self%ibm_vega_ny+1
!         read(12,*) self%ibm_vega_y(j), self%ibm_vega_r(j)
!         self%ibm_vega_y(j) = self%ibm_vega_y(j)+self%ibm_vega_displacement
!        enddo
!        close(12)
         if (self%restart_type==0) then
          self%ibm_vega_ymin = ibm_bbox_vega(2)
          self%ibm_vega_ymax = ibm_bbox_vega(5)
          if (masterproc) write(*,*) 'Min and max y Vega: ', self%ibm_vega_ymin, self%ibm_vega_ymax
         endif
         open(12,file='motion_law.dat')
         read(12,*) self%ibm_trajectory_points
         self%ibm_vega_vel = 0._rkind
         allocate(self%ibm_trajectory(2,self%ibm_trajectory_points))
         do j=1,self%ibm_trajectory_points
          read(12,*) self%ibm_trajectory(1,j), self%ibm_trajectory(2,j)
         enddo
         close(12)
        endif
!
        ibm_aeroacoustics = .false.
        ramp_exists   = .false.
        do l=1,ibm_num_bc
         write(chbcnum,'(I2.2)') l
         if (ibm_type_bc(l)==1) then ! Supersonic inflow
          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,tmp_arr3)
          self%ibm_parbc(l,1:3+N_S) = tmp_arr3
          deallocate(tmp_arr3)
         elseif (ibm_type_bc(l)==2) then ! Supersonic Turbulent Inflow
          self%turinf = 1
          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,tmp_arr5)
          self%ibm_parbc(l,1:5+N_S) = tmp_arr5
          if (self%masterproc) then
           call get_crandom_f(self%randvar_a)
           call get_crandom_f(self%randvar_p)
           self%randvar_a = self%randvar_a*8._rkind*datan(1._rkind)
           self%randvar_p = self%randvar_p*1.E5_rkind!*1.E5_rkind
          endif
!
          call mpi_bcast(self%randvar_a, 8, mpi_prec, 0, MPI_COMM_WORLD, ierr)
          call mpi_bcast(self%randvar_p, 8, mpi_prec, 0, MPI_COMM_WORLD, ierr)

          deallocate(tmp_arr5)
         elseif (ibm_type_bc(l)==3) then ! Subsonic inflow
          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,tmp_arr2)
          self%ibm_parbc(l,1:2+N_S) = tmp_arr2
          deallocate(tmp_arr2)
        elseif (ibm_type_bc(l)==4) then ! Subsonic Turbulent Inflow
          self%turinf = 1
          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,tmp_arr4)
          self%ibm_parbc(l,1:4+N_S) = tmp_arr4
          if (self%masterproc) then
           call get_crandom_f(self%randvar_a)
           call get_crandom_f(self%randvar_p)
           self%randvar_a = self%randvar_a*8._rkind*datan(1._rkind)
           self%randvar_p = self%randvar_p*1.E5_rkind!*1.E5_rkind
          endif
!
          call mpi_bcast(self%randvar_a, 8, mpi_prec, 0, MPI_COMM_WORLD, ierr)
          call mpi_bcast(self%randvar_p, 8, mpi_prec, 0, MPI_COMM_WORLD, ierr)

          deallocate(tmp_arr4)
          elseif (ibm_type_bc(l)==6) then
          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,tmp_arr1)
          self%ibm_parbc(l,1:1) = tmp_arr1
          deallocate(tmp_arr1)
         elseif (ibm_type_bc(l)==9) then
          ibm_aeroacoustics = .true.
          call self%cfg%get("ibmpar","ibm_bc_var_"//chbcnum,ibm_aero_par)
          if (size(ibm_aero_par)/=3) then
           call fail_input_any("Error! Check number of bc for IBM type 9")
          else
           self%ibm_npr       = ibm_aero_par(1)
           self%ibm_ntr       = ibm_aero_par(2)
           self%ibm_timeshift = ibm_aero_par(3)
           if (self%restart_type==0) then
            self%time0         = self%ibm_timeshift
            self%time          = self%time0
            self%istore        = int(self%time0/self%dtsave)+1
           endif
          endif
          inquire(file="nozzle_ramp.dat",exist=ramp_exists)
          if (ramp_exists) then
           open(12,file='nozzle_ramp.dat')
           read(12,*) self%ibm_aero_nramp, self%ibm_aero_rthroat, self%ibm_aero_rexit
           allocate(self%ibm_aero_ramp(3,self%ibm_aero_nramp))
           do i=1,self%ibm_aero_nramp
            read(12,*) (self%ibm_aero_ramp(j,i),j=1,3)
           enddo
           close(12)
          else
           call fail_input_any("Error! File nozzle_ramp.dat not found")
          endif
         endif
        enddo
!
        endassociate
!
    end subroutine ibm_setup_computation_old

    subroutine ibm_bc_prepare_old(self,advance_moving)
     class(equation_multideal_object), intent(inout) :: self
     integer, intent(in), optional :: advance_moving
     integer :: advance_moving_
     integer :: n,nn,nnn,m
     real(rkind) :: rmf,ttot,gamloc,arat,cploc,rmixt
     associate(ibm_aeroacoustics => self%ibm_aeroacoustics, &
               nramp => self%ibm_aero_nramp, &
               ibm_aero_ramp => self%ibm_aero_ramp, &
               time => self%time, &
               ibm_ramp_ptot => self%ibm_ramp_ptot, &
               ibm_ramp_Mach => self%ibm_ramp_Mach, &
               ibm_ntr => self%ibm_ntr, &
               ibm_npr => self%ibm_npr, &
               t0 => self%t0, &
               p0 => self%p0, &
               indx_cp_l => self%indx_cp_l, &
               indx_cp_r => self%indx_cp_r, &
               rmixt0 => self%rmixt0, &
               init_mf => self%init_mf, &
               cp_coeff => self%cp_coeff, &
               cv_coeff => self%cv_coeff, &
               ibm_aero_rthroat => self%ibm_aero_rthroat, &
               ibm_aero_rad => self%ibm_aero_rad, &
               ibm_aero_pp => self%ibm_aero_pp, &
               ibm_aero_tt => self%ibm_aero_tt, &
               ibm_aero_modvel => self%ibm_aero_modvel, &
               ibm_trajectory_points => self%ibm_trajectory_points, &
               ibm_trajectory => self%ibm_trajectory, &
               nsetcv => self%nsetcv, trange => self%trange, &
               ibm_vega_vel => self%ibm_vega_vel)

     advance_moving_ = 0 ; if (present(advance_moving)) advance_moving_ = advance_moving

     if (ibm_aeroacoustics) then
      call locateval(ibm_aero_ramp(1,1:nramp),nramp,time,nn)
      m = 2
      nnn = min(max(nn-(m-1)/2,1),nramp+1-m)
      call pol_int(ibm_aero_ramp(1,nnn:nnn+m-1),ibm_aero_ramp(2,nnn:nnn+m-1),m,time,ibm_ramp_ptot) ! Current total pressure
      call pol_int(ibm_aero_ramp(1,nnn:nnn+m-1),ibm_aero_ramp(3,nnn:nnn+m-1),m,time,ibm_ramp_Mach) ! Current Mach number
      ttot    = t0*ibm_ntr ! Total temperature
      cploc   = get_cp(ttot,indx_cp_l,indx_cp_r,cp_coeff,nsetcv,trange,init_mf) ! Local value of gamma at ttot
      rmixt   = get_rmixture(N_S,self%rgas,init_mf)
      gamloc  = cploc/(cploc-rmixt)
!
      rmf     = 1._rkind+0.5_rkind*(gamloc-1._rkind)*ibm_ramp_Mach**2 ! 1+delta*Mach**2
      arat    = 2._rkind*rmf/(gamloc+1._rkind)
      arat    = arat**((gamloc+1._rkind)/(2._rkind*(gamloc-1._rkind)))
      arat    = arat/ibm_ramp_Mach      ! Aexit_eff/A*
!     athroat = pi*ibm_aero_rthroat**2
!     radnew  = sqrt(arat*athroat/pi)
      ibm_aero_rad     = sqrt(arat*ibm_aero_rthroat**2)
      ibm_aero_pp      = ibm_npr*rmf**(-gamloc/(gamloc-1._rkind))*ibm_ramp_ptot*p0
      ibm_aero_tt      = ibm_ntr/rmf*t0
      ibm_aero_modvel  = ibm_ramp_Mach*sqrt(gamloc*rmixt0*ibm_aero_tt)
     endif
!
     if (advance_moving_>0) then
      if (ibm_trajectory_points>0) then
       call locateval(ibm_trajectory(1,1:ibm_trajectory_points),ibm_trajectory_points,time,nn)
       m = 2
       nnn = min(max(nn-(m-1)/2,1),ibm_trajectory_points+1-m)
       call pol_int(ibm_trajectory(1,nnn:nnn+m-1),ibm_trajectory(2,nnn:nnn+m-1),m,time,ibm_vega_vel)
!      self%ibm_vega_y(1:self%ibm_vega_ny+1) = self%ibm_vega_y(1:self%ibm_vega_ny+1)+ibm_vega_vel*self%dt
       self%ibm_vega_displacement = self%ibm_vega_displacement+ibm_vega_vel*self%dt
       self%ibm_vega_ymin = self%ibm_vega_ymin+ibm_vega_vel*self%dt
       self%ibm_vega_ymax = self%ibm_vega_ymax+ibm_vega_vel*self%dt
       do n=1,self%num_probe
        if (self%moving_probe(n)>0) then
         self%probe_coord(2,n) = self%probe_coord(2,n)+ibm_vega_vel*self%dt
        endif
       enddo
       call self%probe_compute_coeff
      endif
     endif
!
     endassociate
    end subroutine ibm_bc_prepare_old

    subroutine ibm_coeff_setup_old(self)
        class(equation_multideal_object), intent(inout) :: self              !< The equation.
!
        integer :: n
        integer :: ntype0,ntype1,ntype2,ntype3,nfixdetd,nfixdetn
        integer :: ntype0_tot,ntype1_tot,ntype2_tot,ntype3_tot,nfixdetd_tot,nfixdetn_tot
        integer :: ntype0s,ntype1s,ntype2s,ntype3s
        integer :: ntype0s_tot,ntype1s_tot,ntype2s_tot,ntype3s_tot
        real(rkind) :: det_tril_d,det_tril_n
!
        associate(masterproc => self%masterproc, ibm_num_body => self%ibm_num_body, ibm_num_bc => self%ibm_num_bc, &
                  ibm_is_interface_node => self%ibm_is_interface_node, ibm_type => self%ibm_type,         &
                  iermpi => self%mpi_err, ibm_num_interface => self%ibm_num_interface,                    &
                  ibm_interpolation_id_d => self%ibm_interpolation_id_d,                                  &
                  ibm_interpolation_id_n => self%ibm_interpolation_id_n,                                  &
                  ibm_ijk_interface => self%ibm_ijk_interface,   &
                  ibm_ijk_refl => self%ibm_ijk_refl,             &
                  ibm_ijk_wall => self%ibm_ijk_wall,             &
                  ibm_bc       => self%ibm_bc,                   &
                  ibm_nxyz_interface => self%ibm_nxyz_interface, &
                  ibm_dist => self%ibm_dist,                     &
                  ibm_refl_type => self%ibm_refl_type,           &
                  ibm_coeff_d => self%ibm_coeff_d,               &
                  ibm_coeff_n => self%ibm_coeff_n,               &
                  ibm_tol_det_D => self%ibm_tol_det_D,           &
                  ibm_tol_det_N => self%ibm_tol_det_N,           &
                  ibm_tol_distance => self%ibm_tol_distance,     &
                  ibm_type_bc => self%ibm_type_bc,               &
                  ibm_coeff_tril_d => self%ibm_coeff_tril_d,     &
                  ibm_coeff_tril_n => self%ibm_coeff_tril_n,     &
                  ibm_coeff_idfw => self%ibm_coeff_idfw,         &
                  ibm_coeff_idf => self%ibm_coeff_idf,           &
                  ibm_dets => self%ibm_dets,                     &
                  ibm_refl_insolid => self%ibm_refl_insolid)
!
        call self%cfg%get("ibmpar","ibm_tol_det_D",self%ibm_tol_det_D)
        call self%cfg%get("ibmpar","ibm_tol_det_N",self%ibm_tol_det_N)
        call self%cfg%get("ibmpar","ibm_interpolation_id_d",self%ibm_interpolation_id_d)
        call self%cfg%get("ibmpar","ibm_interpolation_id_n",self%ibm_interpolation_id_n)
!
        ntype0   = 0
        ntype1   = 0
        ntype2   = 0
        ntype3   = 0
        ntype0s  = 0
        ntype1s  = 0
        ntype2s  = 0
        ntype3s  = 0
        nfixdetd = 0
        nfixdetn = 0
        do n=1,ibm_num_interface
         det_tril_d = ibm_dets(1,n)
         det_tril_n = ibm_dets(2,n)
         select case(ibm_refl_type(n))
         case(0) !FF
          ntype0 = ntype0+1
          if (ibm_refl_insolid(n)) ntype0s = ntype0s+1
          ibm_coeff_d(:,:,:,n) = ibm_coeff_tril_d(:,:,:,n) ! Assuming trilinear interpolation for FF reflected node
         case(1) !NFF but with interface
          ntype1 = ntype1+1
          if (ibm_refl_insolid(n)) ntype1s = ntype1s+1
!         Dirichlet
          if (ibm_interpolation_id_d==1) then
                  ibm_coeff_d(:,:,:,n) = ibm_coeff_idfw(:,:,:,n)  ! Selecting inverse distance interpolation with wall point included
          else
           ibm_coeff_d(:,:,:,n) = ibm_coeff_tril_d(:,:,:,n)       ! Selecting trilinear interpolation
           if (det_tril_d>-ibm_tol_det_D.or.det_tril_d<-1.0000001_rkind) then
            nfixdetd = nfixdetd+1
            ibm_coeff_d(:,:,:,n) = ibm_coeff_idfw(:,:,:,n)
           endif
          endif
!         Neumann
          if (ibm_interpolation_id_n==1) then
           ibm_coeff_n(:,:,:,n) = ibm_coeff_idf(:,:,:,n)           ! Selecting inverse distance interpolation without wall point
          else
           ibm_coeff_n(:,:,:,n) = ibm_coeff_tril_n(:,:,:,n)        ! Selecting trilinear interpolation
           if (abs(det_tril_n)<ibm_tol_det_N) then
            nfixdetn = nfixdetn+1
            ibm_coeff_n(:,:,:,n) = ibm_coeff_idf(:,:,:,n)
           endif
          endif
         case(2) !NFF with solid point
          ntype2 = ntype2+1
          if (ibm_refl_insolid(n)) ntype2s = ntype2s+1
          ibm_coeff_d(:,:,:,n) = ibm_coeff_idfw(:,:,:,n)
          ibm_coeff_n(:,:,:,n) = ibm_coeff_idf(:,:,:,n)
         case(3)
          ntype3 = ntype3+1
          if (ibm_refl_insolid(n)) ntype3s = ntype3s+1
          ibm_coeff_d(:,:,:,n) = ibm_coeff_idfw(:,:,:,n)
          ibm_coeff_n(:,:,:,n) = ibm_coeff_idfw(:,:,:,n)
         end select
        enddo
!
        call mpi_allreduce(ntype0 ,ntype0_tot ,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(ntype1 ,ntype1_tot ,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(ntype2 ,ntype2_tot ,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(ntype3 ,ntype3_tot ,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(ntype0s,ntype0s_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(ntype1s,ntype1s_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(ntype2s,ntype2s_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(ntype3s,ntype3s_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(nfixdetd,nfixdetd_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        call mpi_allreduce(nfixdetn,nfixdetn_tot,1,mpi_integer,mpi_sum,mpi_comm_world,iermpi)
        if (masterproc) write(*  ,*) 'Total num refl nodes type 0 (all nodes fluid) = ', ntype0_tot,ntype0s_tot
        if (masterproc) write(*  ,*) 'Total num refl nodes type 1 (at least 1 interface) = ', ntype1_tot,ntype1s_tot
        if (masterproc) write(*  ,*) 'Total num refl nodes type 2 (at least 1 solid not interface) = ', ntype2_tot,ntype2s_tot
        if (masterproc) write(*  ,*) 'Total num refl nodes type 3 (all solids, at least 1 interface )= ', ntype3_tot,ntype3s_tot
        if (masterproc) write(*  ,*) 'Total num refl nodes type 1 with fixing D = ', nfixdetd_tot
        if (masterproc) write(*  ,*) 'Total num refl nodes type 1 with fixing N = ', nfixdetn_tot
!
        if (masterproc) open(127,file='report_ibm_2.dat')
        if (masterproc) write(127,*) 'ibm_tol_det_D =', ibm_tol_det_D
        if (masterproc) write(127,*) 'ibm_tol_det_N =', ibm_tol_det_N
        if (masterproc) write(127,*) 'ibm_interpolation_id_d =', ibm_interpolation_id_d
        if (masterproc) write(127,*) 'ibm_interpolation_id_n =', ibm_interpolation_id_n
        if (masterproc) write(127,*) 'Total num refl nodes type 0 (all nodes fluid) = ', ntype0_tot,ntype0s_tot
        if (masterproc) write(127,*) 'Total num refl nodes type 1 (at least 1 interface) = ', ntype1_tot,ntype1s_tot
        if (masterproc) write(127,*) 'Total num refl nodes type 2 (at least 1 solid not interface) = ', ntype2_tot,ntype2s_tot
        if (masterproc) write(127,*) 'Total num refl nodes type 3 (all solids, at least 1 interface )= ', ntype3_tot,ntype3s_tot
        if (masterproc) write(127,*) 'Total num refl nodes type 1 with fixing D = ', nfixdetd_tot
        if (masterproc) write(127,*) 'Total num refl nodes type 1 with fixing N = ', nfixdetn_tot
        if (masterproc) close(127)
!
        if (self%masterproc) write(*,*) 'Done with ibm_coeff_setup'
!
        endassociate
    endsubroutine ibm_coeff_setup_old

endmodule streams_equation_multideal_object

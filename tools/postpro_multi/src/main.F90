program main

    use reader
    use postpro_cha
    use postpro_bl
    use postpro_chacurv
    use postpro_airfoil
    use postpro_ramp
    use budget_TKE
    implicit none
    logical :: dir_exist

    call execute_command_line('mkdir -p POSTPRO/')

    call read_input
    call read_stat

!    if (flow_init==0) call stats1d !channel flow statistics

    if (flow_init==1 .or. flow_init==4) then
      call stats2d !boundary layer flow statistics
    endif

    if (save_budget_plot3d > 0) call compute_budget

    if(save_plot3d > 0) call save_plot3d_fields()
end program main

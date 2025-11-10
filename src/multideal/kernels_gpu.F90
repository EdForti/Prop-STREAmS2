#include 'index_define.h'
module streams_kernels_gpu

    use streams_parameters, only : rkind, ikind, REAL64
    use CUDAFOR
    use ieee_arithmetic 
    implicit none

contains

    subroutine zero_flux_cuf(nx, ny, nz, nv, fl_gpu)
        integer :: nx, ny, nz, nv
        real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fl_gpu
        integer :: i,j,k,m,iercuda
        !$cuf kernel do(3) <<<*,*>>> 
         do k=1,nz
          do j=1,ny
           do i=1,nx
            do m=1,nv
             fl_gpu(i,j,k,m)  = 0._rkind
            enddo
           enddo
          enddo
         enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine zero_flux_cuf

    subroutine init_flux_cuf(nx, ny, nz, nv, fl_gpu, fln_gpu, rhodt) 
        integer :: nx, ny, nz, nv
        real(rkind) :: rhodt
        real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fl_gpu, fln_gpu
        integer :: i,j,k,m,iercuda
        !$cuf kernel do(3) <<<*,*>>> 
         do k=1,nz
          do j=1,ny
           do i=1,nx
            do m=1,nv
             fln_gpu(i,j,k,m) = - rhodt * fl_gpu(i,j,k,m)
             fl_gpu(i,j,k,m)  = 0._rkind
            enddo
           enddo
          enddo
         enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine init_flux_cuf

    subroutine init_flux_simpler_cuf(nx, ny, nz, nv, fl_gpu, fln_gpu, fl_sav_gpu, rhodt)
        integer :: nx, ny, nz, nv
        real(rkind) :: rhodt
        real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fl_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(out), device :: fln_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: fl_sav_gpu
        integer :: i,j,k,m,iercuda
        !$cuf kernel do(3) <<<*,*>>> 
         do k=1,nz
          do j=1,ny
           do i=1,nx
            do m=1,nv
             fln_gpu(i,j,k,m) = - rhodt*(fl_gpu(i,j,k,m)-fl_sav_gpu(i,j,k,m))
             fl_gpu(i,j,k,m)  = 0._rkind
            enddo 
           enddo
          enddo 
         enddo  
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine init_flux_simpler_cuf

    subroutine sav_flx_cuf(nx, ny, nz, nv, fl_gpu, fl_sav_gpu)
        integer :: nx, ny, nz, nv
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: fl_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(out), device :: fl_sav_gpu
        integer :: i,j,k,m,iercuda

        !$cuf kernel do(3) <<<*,*>>>
         do k=1,nz
          do j=1,ny
           do i=1,nx
            do m=1,nv
             fl_sav_gpu(i,j,k,m) = fl_gpu(i,j,k,m)
            enddo
           enddo
          enddo
         enddo
        !@cuf iercuda=cudaDeviceSynchronize()

    endsubroutine sav_flx_cuf

    attributes(global) launch_bounds(256) subroutine euler_x_fluxes_hybrid_kernel(nv, nv_aux, nx, ny, nz, ng, &
        istart_face, iend_face, lmax_base, nkeep, rgas_gpu, w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, fhat_gpu, &
        force_zero_flux_min, force_zero_flux_max, &
        weno_scheme, weno_version, sensor_threshold, weno_size, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, &
        indx_cp_l, indx_cp_r, ep_ord_change_gpu, tol_iter_nr,rho0,u0)

        implicit none
        ! Passed arguments
        integer, value :: nv, nx, ny, nz, ng, nv_aux, nsetcv
        integer, value :: istart_face, iend_face, lmax_base, nkeep, indx_cp_l, indx_cp_r
        integer, value :: weno_scheme, weno_size, weno_version
        integer, dimension(0:nx,0:ny,0:nz,1:3) :: ep_ord_change_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv) :: fhat_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind), dimension(N_S) :: rgas_gpu
        real(rkind), dimension(4,4) :: coeff_deriv1_gpu
        real(rkind), dimension(nx) :: dcsidx_gpu
        integer, value :: force_zero_flux_min, force_zero_flux_max
        real(rkind), value :: sensor_threshold, tol_iter_nr,rho0,u0
        ! Local variables
        integer :: i, j, k, m, l, lsp
        real(rkind) :: fh1, fh2, fh3, fh4, fh5
        real(rkind) :: rhom, uui, vvi, wwi, ppi, enti, rhoi, tti
        real(rkind) :: uuip, vvip, wwip, ppip, entip, rhoip, ttip
        real(rkind) :: ft1, ft2, ft3, ft4, ft5, ft6
        real(rkind) :: uvs1, uvs2, uvs3, uvs4, uvs6, uv_part
        real(rkind) :: uvs5
        integer :: ii, lmax, wenorec_ord
        integer :: ishk
        real(rkind) :: b1, b2, b3, c, ci, h, uu, vv, ww
        real(rkind), dimension(4+N_S,4+N_S) :: el, er
        real(rkind), dimension(4+N_S,8) :: gp,gm
        real(rkind), dimension(4+N_S) :: evmax, fi
        real(rkind), dimension(N_S) :: yyroe, prhoi
        integer :: ll, mm
        real(rkind) :: rho, pp, wc, gc, rhou
        real(rkind) :: tt
        real(rkind) :: uvs5_i,uvs5_k,uvs5_p,eei,eeip,yyi,yyip
        real(rkind) :: drho, dee, eem
        real(rkind) :: drhof, deef
        real(rkind) :: sumnumrho,sumnumee,sumdenrho,sumdenee
        integer :: n,n2
        real(rkind) :: t_sumdenrho,  t_sumdenee,  t2_sumdenrho, t2_sumdenee

        !j = blockDim%x * (blockIdx%x - 1) + threadIdx%x 
        !k = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        !if (j > ny .or. k > nz) return
        !do i=eul_imin-1,eul_imax

        i = blockDim%x * (blockIdx%x - 1) + threadIdx%x + istart_face - 1
        j = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        if (j > ny .or. i > iend_face) return
        do k=1,nz

            ishk = 0
            do ii=i-weno_scheme+1,i+weno_scheme
                if (w_aux_gpu(ii,j,k,J_DUC) > sensor_threshold) ishk = 1
            enddo

            if (ishk == 0) then
                lmax = max(lmax_base+ep_ord_change_gpu(i,j,k,1),1)
!
                do lsp=1,N_S
                    ft1  = 0._rkind
                    do l=1,lmax
                        uvs1 = 0._rkind
                        do m=0,l-1
 
                            rhoi  = w_aux_gpu(i-m,j,k,J_R)
                            uui   = w_aux_gpu(i-m,j,k,J_U)
                            yyi   = w_aux_gpu(i-m,j,k,lsp)
 
                            rhoip = w_aux_gpu(i-m+l,j,k,J_R)
                            uuip  = w_aux_gpu(i-m+l,j,k,J_U)
                            yyip  = w_aux_gpu(i-m+l,j,k,lsp)
 
                            rhom  = rhoi+rhoip
 
                            if (nkeep <= 0) then
                                drhof = 1._rkind
                            else
                                sumnumrho = 1._rkind
                                drho   = 2._rkind*(rhoip-rhoi)/rhom
                                t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                                t2_sumdenrho = t_sumdenrho
                                sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                                do n = 2, nkeep
                                   n2 = 2*n
                                   t_sumdenrho = t2_sumdenrho * t_sumdenrho
                                   sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                                enddo
                                drhof = sumnumrho/sumdenrho
                            endif
 
                            uv_part = (uui+uuip) * rhom * drhof
                            uvs1 = uvs1 + uv_part * (yyi+yyip)
                        enddo
                        ft1  = ft1  + coeff_deriv1_gpu(l,lmax)*uvs1
                    enddo
 
                    fh1 = 0.25_rkind*ft1
 
                    if ((i==0 .and. force_zero_flux_min == 1).or.(i==nx .and. force_zero_flux_max == 1)) then
                        fh1 = 0._rkind
                    endif
 
                    fhat_gpu(i,j,k,lsp) = fh1
                enddo
!
                ft2  = 0._rkind
                ft3  = 0._rkind
                ft4  = 0._rkind
                ft5  = 0._rkind
                ft6  = 0._rkind
!               lmax = max(lmax_base+ep_ord_change_gpu(i,j,k,1),1)
                if (nkeep>=0) then
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5_i = 0._rkind
                     uvs5_k = 0._rkind
                     uvs5_p = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1
 
                         rhoi  = w_aux_gpu(i-m,j,k,J_R)
                         uui   = w_aux_gpu(i-m,j,k,J_U)
                         vvi   = w_aux_gpu(i-m,j,k,J_V)
                         wwi   = w_aux_gpu(i-m,j,k,J_W)
                         enti  = w_aux_gpu(i-m,j,k,J_H)
                         tti   = w_aux_gpu(i-m,j,k,J_T)
                         ppi   = w_aux_gpu(i-m,j,k,J_P)
                         eei   = enti-ppi/rhoi-0.5_rkind*(uui*uui+vvi*vvi+wwi*wwi)
 
                         rhoip = w_aux_gpu(i-m+l,j,k,J_R)
                         uuip  = w_aux_gpu(i-m+l,j,k,J_U)
                         vvip  = w_aux_gpu(i-m+l,j,k,J_V)
                         wwip  = w_aux_gpu(i-m+l,j,k,J_W)
                         entip = w_aux_gpu(i-m+l,j,k,J_H)
                         ttip  = w_aux_gpu(i-m+l,j,k,J_T)
                         ppip  = w_aux_gpu(i-m+l,j,k,J_P)
                         eeip  = entip-ppip/rhoip-0.5_rkind*(uuip*uuip+vvip*vvip+wwip*wwip)

                         rhom  = rhoi+rhoip
                         eem   = eei + eeip
 
                         if(nkeep == 0) then
                             drhof = 1._rkind
                             deef  = 1._rkind
                         else
                            sumnumrho = 1._rkind
                            drho   = 2._rkind*(rhoip-rhoi)/rhom
                            dee    = 2._rkind*(eeip - eei)/eem
                            t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                            t_sumdenee  = (0.5_rkind*dee )*(0.5_rkind*dee )
                            t2_sumdenrho = t_sumdenrho
                            t2_sumdenee  = t_sumdenee
                            sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                            sumdenee  = 1._rkind + t_sumdenee
                            sumnumee  = 1._rkind + t_sumdenee  / (3._rkind)
                            do n = 2, nkeep
                               n2 = 2*n
                               t_sumdenrho = t2_sumdenrho * t_sumdenrho
                               t_sumdenee  = t2_sumdenee  * t_sumdenee
                               sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                               sumdenee  = sumdenee  + t_sumdenee
                               sumnumee  = sumnumee  + t_sumdenee  / (1._rkind+n2)
                            enddo
                            drhof = sumnumrho/sumdenrho
                            deef  = sumnumee /sumdenee
                         endif
                         !NUOVO

                         uv_part = (uui+uuip) * rhom * drhof
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5_i = uvs5_i + uv_part * eem * deef
                         uvs5_k = uvs5_k + uv_part * (uui*uuip+vvi*vvip+wwi*wwip)
                         uvs5_p = uvs5_p + 4._rkind*(uui*ppip+uuip*ppi)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2  = ft2  + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3  = ft3  + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4  = ft4  + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5  = ft5  + coeff_deriv1_gpu(l,lmax)*(uvs5_i+uvs5_k+uvs5_p)
                     ft6  = ft6  + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                else
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5 = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1
 
                         rhoi  = w_aux_gpu(i-m,j,k,J_R)
                         uui   = w_aux_gpu(i-m,j,k,J_U)
                         vvi   = w_aux_gpu(i-m,j,k,J_V)
                         wwi   = w_aux_gpu(i-m,j,k,J_W)
                         enti  = w_aux_gpu(i-m,j,k,J_H)
                         tti   = w_aux_gpu(i-m,j,k,J_T)
                         ppi   = w_aux_gpu(i-m,j,k,J_P)
 
                         rhoip = w_aux_gpu(i-m+l,j,k,J_R)
                         uuip  = w_aux_gpu(i-m+l,j,k,J_U)
                         vvip  = w_aux_gpu(i-m+l,j,k,J_V)
                         wwip  = w_aux_gpu(i-m+l,j,k,J_W)
                         entip = w_aux_gpu(i-m+l,j,k,J_H)
                         ttip  = w_aux_gpu(i-m+l,j,k,J_T)
                         ppip  = w_aux_gpu(i-m+l,j,k,J_P)

                         rhom  = rhoi+rhoip
                         uv_part = (uui+uuip) * rhom
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5 = uvs5 + uv_part * (enti+entip)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2 = ft2 + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3 = ft3 + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4 = ft4 + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5 = ft5 + coeff_deriv1_gpu(l,lmax)*uvs5
                     ft6 = ft6 + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                endif

                fh2 = 0.25_rkind*ft2 
                fh3 = 0.25_rkind*ft3
                fh4 = 0.25_rkind*ft4
                fh5 = 0.25_rkind*ft5

                if ((i==0 .and. force_zero_flux_min == 1).or.(i==nx .and. force_zero_flux_max == 1)) then
                   fh2 = 0._rkind
                   fh3 = 0._rkind
                   fh4 = 0._rkind
                   fh5 = 0._rkind
                endif
                fh2 = fh2 + 0.5_rkind*ft6

                fhat_gpu(i,j,k,I_U) = fh2
                fhat_gpu(i,j,k,I_V) = fh3
                fhat_gpu(i,j,k,I_W) = fh4
                fhat_gpu(i,j,k,I_E) = fh5
            else
                call compute_roe_average(nx, ny, nz, ng, nv_aux, i, i+1, j, j, k, k, w_aux_gpu, rgas_gpu, &
                     b1, b2, b3, rho, c, ci, h, uu, vv, ww, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
                     tol_iter_nr, yyroe, prhoi)

                call eigenvectors_x(nv, b1, b2, b3, rho, uu, vv, ww, c, ci, h, el, er, yyroe, prhoi)
!
!               evmax(1) = abs(uu-c)
!               evmax(2) = abs(uu)
!               do lsp=3,N_S+3
!                evmax(lsp) = evmax(2)
!               enddo
!               evmax(N_S+4) = abs(uu+c)
                do m=1,N_S+4 ! loop on characteristic fields
                    evmax(m) = -1._rkind
                enddo
                do l=1,weno_size ! LLF
                    ll   = i + l - weno_scheme
                    uu   = w_aux_gpu(ll,j,k,J_U)
                    tt   = w_aux_gpu(ll,j,k,J_T)
                    c    = w_aux_gpu(ll,j,k,J_C)
                    evmax(1) = max(abs(uu-c),evmax(1))
                    evmax(2) = max(abs(uu  ),evmax(2))
                    do lsp=3,N_S+3
                     evmax(lsp) = evmax(2)
                    enddo
                    evmax(N_S+4) = max(abs(uu+c),evmax(N_S+4))
                enddo
                do l=1,weno_size ! loop over the stencil centered at face i
                    ll = i + l - weno_scheme

                    rho    = w_aux_gpu(ll,j,k,J_R)
                    uu     = w_aux_gpu(ll,j,k,J_U)
                    vv     = w_aux_gpu(ll,j,k,J_V)
                    ww     = w_aux_gpu(ll,j,k,J_W)
                    h      = w_aux_gpu(ll,j,k,J_H) 
                    rhou   = rho*uu
                    pp     = w_aux_gpu(ll,j,k,J_P)
                    do lsp=1,N_S
                     fi(lsp)  = rhou*w_aux_gpu(ll,j,k,lsp)
                    enddo
                    fi(I_U)  = uu * rhou + pp
                    fi(I_V)  = vv * rhou
                    fi(I_W)  = ww * rhou
                    fi(I_E)  = h  * rhou
                    do m=1,N_S+4
                        wc = 0._rkind
                        gc = 0._rkind

                        do lsp=1,N_S
                         wc = wc + el(lsp,m) * rho*w_aux_gpu(ll,j,k,lsp)
                         gc = gc + el(lsp,m) * fi(lsp)
                        enddo
                        wc = wc + el(I_U,m) * rho*uu
                        gc = gc + el(I_U,m) * fi(I_U)
                        wc = wc + el(I_V,m) * rho*vv
                        gc = gc + el(I_V,m) * fi(I_V)
                        wc = wc + el(I_W,m) * rho*ww
                        gc = gc + el(I_W,m) * fi(I_W)
                        wc = wc + el(I_E,m) * (rho*h-pp)
                        gc = gc + el(I_E,m) * fi(I_E)

                        c = 0.5_rkind * (gc + evmax(m) * wc)
                        !gplus_x_gpu (i,j,m,l) = c
                        !gminus_x_gpu(i,j,m,l) = gc - c
                        gp(m,l) = c
                        gm(m,l) = gc - c
                    enddo
                enddo
!
!               Reconstruction of the '+' and '-' fluxes
!
                wenorec_ord = max(weno_scheme+ep_ord_change_gpu(i,j,k,1),1)
                !call wenorec_x(i,j,nv,nx,ny,gplus_x_gpu,gminus_x_gpu,fi,weno_scheme,wenorec_ord,weno_version)
                call wenorec_1d(nv,gp,gm,fi,weno_scheme,wenorec_ord,weno_version,rho0,u0)
!
!               !Return to conservative fluxes
                do m=1,4+N_S
                    fhat_gpu(i,j,k,m) = 0._rkind
                    do mm=1,4+N_S
                       fhat_gpu(i,j,k,m) = fhat_gpu(i,j,k,m) + er(mm,m) * fi(mm)
                    enddo
                enddo

            endif
        enddo

    endsubroutine euler_x_fluxes_hybrid_kernel

    attributes(global) launch_bounds(256) subroutine euler_x_fluxes_hybrid_rusanov_kernel(nv, nv_aux, nx, ny, nz, ng, &
        istart_face, iend_face, lmax_base, nkeep, w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, fhat_gpu, &
        force_zero_flux_min, force_zero_flux_max, &
        weno_scheme, weno_version, sensor_threshold, weno_size, cp_coeff_gpu, nsetcv, trange_gpu, &
        indx_cp_l, indx_cp_r, ep_ord_change_gpu, tol_iter_nr,rho0,u0)

        implicit none
        ! Passed arguments
        integer, value :: nv, nx, ny, nz, ng, nv_aux, nsetcv
        integer, value :: istart_face, iend_face, lmax_base, nkeep, indx_cp_l, indx_cp_r
        integer, value :: weno_scheme, weno_size, weno_version
        integer, dimension(0:nx,0:ny,0:nz,1:3) :: ep_ord_change_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv) :: fhat_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind), dimension(4,4) :: coeff_deriv1_gpu
        real(rkind), dimension(nx) :: dcsidx_gpu
        integer, value :: force_zero_flux_min, force_zero_flux_max
        real(rkind), value :: sensor_threshold, tol_iter_nr,rho0,u0
        ! Local variables
        integer :: i, j, k, m, l, lsp
        real(rkind) :: fh1, fh2, fh3, fh4, fh5
        real(rkind) :: rhom, uui, vvi, wwi, ppi, enti, rhoi, tti
        real(rkind) :: uuip, vvip, wwip, ppip, entip, rhoip, ttip
        real(rkind) :: ft1, ft2, ft3, ft4, ft5, ft6
        real(rkind) :: uvs1, uvs2, uvs3, uvs4, uvs6, uv_part
        real(rkind) :: uvs5
        integer :: ii, lmax, wenorec_ord
        integer :: ishk
        real(rkind) :: b1, b2, b3, c, ci, h, uu, vv, ww
        real(rkind), dimension(4+N_S)   :: fi
        real(rkind), dimension(4+N_S,8) :: gp,gm
        integer :: ll, mm
        real(rkind) :: evm,evmax,rhoevm
        real(rkind) :: rho, pp, wc, gc, rhou
        real(rkind) :: tt
        real(rkind) :: uvs5_i,uvs5_k,uvs5_p,eei,eeip,yyi,yyip
        real(rkind) :: drho, dee, eem
        real(rkind) :: drhof, deef
        real(rkind) :: sumnumrho,sumnumee,sumdenrho,sumdenee
        integer :: n,n2
        real(rkind) :: t_sumdenrho,  t_sumdenee,  t2_sumdenrho, t2_sumdenee 
        real(rkind) :: yy

        i = blockDim%x * (blockIdx%x - 1) + threadIdx%x + istart_face - 1
        j = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        if (j > ny .or. i > iend_face) return
        do k=1,nz

            ishk = 0
            do ii=i-weno_scheme+1,i+weno_scheme
                if (w_aux_gpu(ii,j,k,J_DUC) > sensor_threshold) ishk = 1
            enddo

            if (ishk == 0) then
!
                lmax = max(lmax_base+ep_ord_change_gpu(i,j,k,1),1)
!
                do lsp=1,N_S
                    ft1  = 0._rkind
                    do l=1,lmax
                        uvs1 = 0._rkind
                        do m=0,l-1
 
                            rhoi  = w_aux_gpu(i-m,j,k,J_R)
                            uui   = w_aux_gpu(i-m,j,k,J_U)
                            yyi   = w_aux_gpu(i-m,j,k,lsp)
 
                            rhoip = w_aux_gpu(i-m+l,j,k,J_R)
                            uuip  = w_aux_gpu(i-m+l,j,k,J_U)
                            yyip  = w_aux_gpu(i-m+l,j,k,lsp)
 
                            rhom  = rhoi+rhoip
 
                            if (nkeep <= 0) then
                                drhof = 1._rkind
                            else
                                sumnumrho = 1._rkind
                                drho   = 2._rkind*(rhoip-rhoi)/rhom
                                t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                                t2_sumdenrho = t_sumdenrho
                                sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                                do n = 2, nkeep
                                   n2 = 2*n
                                   t_sumdenrho = t2_sumdenrho * t_sumdenrho
                                   sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                                enddo
                                drhof = sumnumrho/sumdenrho
                            endif
 
                            uv_part = (uui+uuip) * rhom * drhof
                            uvs1 = uvs1 + uv_part * (yyi+yyip)
                        enddo
                        ft1  = ft1  + coeff_deriv1_gpu(l,lmax)*uvs1
                    enddo
 
                    fh1 = 0.25_rkind*ft1
 
                    if ((i==0 .and. force_zero_flux_min == 1).or.(i==nx .and. force_zero_flux_max == 1)) then
                        fh1 = 0._rkind
                    endif
 
                    fhat_gpu(i,j,k,lsp) = fh1
                enddo
!
                ft2  = 0._rkind
                ft3  = 0._rkind
                ft4  = 0._rkind
                ft5  = 0._rkind
                ft6  = 0._rkind
!               lmax = max(lmax_base+ep_ord_change_gpu(i,j,k,1),1)
                if (nkeep>=0) then
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5_i = 0._rkind
                     uvs5_k = 0._rkind
                     uvs5_p = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1
 
                         rhoi  = w_aux_gpu(i-m,j,k,J_R)
                         uui   = w_aux_gpu(i-m,j,k,J_U)
                         vvi   = w_aux_gpu(i-m,j,k,J_V)
                         wwi   = w_aux_gpu(i-m,j,k,J_W)
                         enti  = w_aux_gpu(i-m,j,k,J_H)
                         tti   = w_aux_gpu(i-m,j,k,J_T)
                         ppi   = w_aux_gpu(i-m,j,k,J_P)
                         eei   = enti-ppi/rhoi-0.5_rkind*(uui*uui+vvi*vvi+wwi*wwi)
 
                         rhoip = w_aux_gpu(i-m+l,j,k,J_R)
                         uuip  = w_aux_gpu(i-m+l,j,k,J_U)
                         vvip  = w_aux_gpu(i-m+l,j,k,J_V)
                         wwip  = w_aux_gpu(i-m+l,j,k,J_W)
                         entip = w_aux_gpu(i-m+l,j,k,J_H)
                         ttip  = w_aux_gpu(i-m+l,j,k,J_T)
                         ppip  = w_aux_gpu(i-m+l,j,k,J_P)
                         eeip  = entip-ppip/rhoip-0.5_rkind*(uuip*uuip+vvip*vvip+wwip*wwip)

                         rhom  = rhoi+rhoip
                         eem   = eei + eeip
 
                         if(nkeep == 0) then
                             drhof = 1._rkind
                             deef  = 1._rkind
                         else
                            sumnumrho = 1._rkind
                            drho   = 2._rkind*(rhoip-rhoi)/rhom
                            dee    = 2._rkind*(eeip - eei)/eem
                            t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                            t_sumdenee  = (0.5_rkind*dee )*(0.5_rkind*dee )
                            t2_sumdenrho = t_sumdenrho
                            t2_sumdenee  = t_sumdenee
                            sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                            sumdenee  = 1._rkind + t_sumdenee
                            sumnumee  = 1._rkind + t_sumdenee  / (3._rkind)
                            do n = 2, nkeep
                               n2 = 2*n
                               t_sumdenrho = t2_sumdenrho * t_sumdenrho
                               t_sumdenee  = t2_sumdenee  * t_sumdenee
                               sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                               sumdenee  = sumdenee  + t_sumdenee
                               sumnumee  = sumnumee  + t_sumdenee  / (1._rkind+n2)
                            enddo
                            drhof = sumnumrho/sumdenrho
                            deef  = sumnumee /sumdenee
                         endif

                         uv_part = (uui+uuip) * rhom * drhof
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5_i = uvs5_i + uv_part * eem * deef
                         uvs5_k = uvs5_k + uv_part * (uui*uuip+vvi*vvip+wwi*wwip)
                         uvs5_p = uvs5_p + 4._rkind*(uui*ppip+uuip*ppi)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2  = ft2  + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3  = ft3  + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4  = ft4  + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5  = ft5  + coeff_deriv1_gpu(l,lmax)*(uvs5_i+uvs5_k+uvs5_p)
                     ft6  = ft6  + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                else
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5 = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1
 
                         rhoi  = w_aux_gpu(i-m,j,k,J_R)
                         uui   = w_aux_gpu(i-m,j,k,J_U)
                         vvi   = w_aux_gpu(i-m,j,k,J_V)
                         wwi   = w_aux_gpu(i-m,j,k,J_W)
                         enti  = w_aux_gpu(i-m,j,k,J_H)
                         tti   = w_aux_gpu(i-m,j,k,J_T)
                         ppi   = w_aux_gpu(i-m,j,k,J_P)
 
                         rhoip = w_aux_gpu(i-m+l,j,k,J_R)
                         uuip  = w_aux_gpu(i-m+l,j,k,J_U)
                         vvip  = w_aux_gpu(i-m+l,j,k,J_V)
                         wwip  = w_aux_gpu(i-m+l,j,k,J_W)
                         entip = w_aux_gpu(i-m+l,j,k,J_H)
                         ttip  = w_aux_gpu(i-m+l,j,k,J_T)
                         ppip  = w_aux_gpu(i-m+l,j,k,J_P)

                         rhom  = rhoi+rhoip
                         uv_part = (uui+uuip) * rhom
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5 = uvs5 + uv_part * (enti+entip)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2 = ft2 + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3 = ft3 + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4 = ft4 + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5 = ft5 + coeff_deriv1_gpu(l,lmax)*uvs5
                     ft6 = ft6 + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                endif

                fh2 = 0.25_rkind*ft2 
                fh3 = 0.25_rkind*ft3
                fh4 = 0.25_rkind*ft4
                fh5 = 0.25_rkind*ft5

                if ((i==0 .and. force_zero_flux_min == 1).or.(i==nx .and. force_zero_flux_max == 1)) then
                   fh2 = 0._rkind
                   fh3 = 0._rkind
                   fh4 = 0._rkind
                   fh5 = 0._rkind
                endif
                fh2 = fh2 + 0.5_rkind*ft6

                fhat_gpu(i,j,k,I_U) = fh2
                fhat_gpu(i,j,k,I_V) = fh3
                fhat_gpu(i,j,k,I_W) = fh4
                fhat_gpu(i,j,k,I_E) = fh5
            else
                evmax = -1._rkind
                do l=1,weno_size ! LLF
                    ll   = i + l - weno_scheme
                    uu   = w_aux_gpu(ll,j,k,J_U)
                    tt   = w_aux_gpu(ll,j,k,J_T)
                    c    = w_aux_gpu(ll,j,k,J_C)
                    evm    = max(abs(uu-c),abs(uu+c))
                    evmax  = max(evm,evmax)
                enddo
                do l=1,weno_size ! loop over the stencil centered at face i
                    ll = i + l - weno_scheme

                    rho    = w_aux_gpu(ll,j,k,J_R)
                    uu     = w_aux_gpu(ll,j,k,J_U)
                    vv     = w_aux_gpu(ll,j,k,J_V)
                    ww     = w_aux_gpu(ll,j,k,J_W)
                    h      = w_aux_gpu(ll,j,k,J_H) 
                    rhou   = rho*uu
                    pp     = w_aux_gpu(ll,j,k,J_P)

                    rhoevm  = rho*evmax 
                    do lsp=1,N_S
                     yy = w_aux_gpu(ll,j,k,lsp)
                     evm = rhou*yy
                     c = 0.5_rkind * (evm + rhoevm * yy)
                     gp(lsp,l) = c
                     gm(lsp,l) = evm - c
                    enddo
                    evm    = uu * rhou + pp
                    c = 0.5_rkind * (evm + rhoevm * uu)
                    gp(I_U,l) = c
                    gm(I_U,l) = evm - c
                    evm    = vv * rhou
                    c = 0.5_rkind * (evm + rhoevm * vv)
                    gp(I_V,l) = c
                    gm(I_V,l) = evm - c
                    evm    = ww * rhou
                    c = 0.5_rkind * (evm + rhoevm * ww)
                    gp(I_W,l) = c
                    gm(I_W,l) = evm - c
                    evm    = h  * rhou
                    c = 0.5_rkind * (evm + evmax * (rho*h-pp))
                    gp(I_E,l) = c
                    gm(I_E,l) = evm - c
                enddo
!
!               Reconstruction of the '+' and '-' fluxes
!
                wenorec_ord = max(weno_scheme+ep_ord_change_gpu(i,j,k,1),1)
                call wenorec_1d_rusanov(nv,gp,gm,fi,weno_scheme,wenorec_ord,weno_version,rho0,u0)
!
!               !Return to conservative fluxes
                do m=1,4+N_S
                   fhat_gpu(i,j,k,m) = fi(m)
                enddo

            endif
        enddo

    endsubroutine euler_x_fluxes_hybrid_rusanov_kernel

    subroutine euler_x_update_cuf(nx, ny, nz, ng, nv, eul_imin, eul_imax, &
            fhat_gpu,fl_gpu,dcsidx_gpu,stream_id)
        integer :: nx,ny,nz,ng,nv,eul_imin,eul_imax
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(in), device :: fhat_gpu
        real(rkind), dimension(1:nx,1:ny,1:nz,1:nv), intent(inout), device :: fl_gpu
        real(rkind), dimension(1:nx), intent(in), device :: dcsidx_gpu  
        integer(kind=cuda_stream_kind) :: stream_id
        integer :: i,j,k,m,iv,iercuda

        !$cuf kernel do(3) <<<*,*,stream=stream_id>>>
        do k=1,nz
         do j=1,ny
          do i=eul_imin,eul_imax
           do iv=1,nv
            fl_gpu(i,j,k,iv) = fl_gpu(i,j,k,iv) + (fhat_gpu(i,j,k,iv)-fhat_gpu(i-1,j,k,iv))*dcsidx_gpu(i)
           enddo
          enddo
         enddo
        enddo
    endsubroutine euler_x_update_cuf

    subroutine euler_y_update_cuf(nx, ny, nz, ng, nv, eul_jmin, eul_jmax, &
            fhat_gpu,fl_gpu,detady_gpu,stream_id)
        integer :: nx,ny,nz,ng,nv,eul_jmin,eul_jmax
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(in), device :: fhat_gpu
        real(rkind), dimension(1:nx,1:ny,1:nz,1:nv), intent(inout), device :: fl_gpu
        real(rkind), dimension(1:ny), intent(in), device :: detady_gpu  
        integer(kind=cuda_stream_kind) :: stream_id
        integer :: i,j,k,m,iv,iercuda

        !$cuf kernel do(3) <<<*,*,stream=stream_id>>>
        do k=1,nz
         do j=eul_jmin,eul_jmax
          do i=1,nx
           do iv=1,nv
            fl_gpu(i,j,k,iv) = fl_gpu(i,j,k,iv) + (fhat_gpu(i,j,k,iv)-fhat_gpu(i,j-1,k,iv))*detady_gpu(j)
           enddo
          enddo
         enddo
        enddo
    endsubroutine euler_y_update_cuf

    subroutine euler_z_update_cuf(nx, ny, nz, ng, nv, eul_kmin, eul_kmax, &
            fhat_gpu,fl_gpu,dzitdz_gpu,stream_id)
        integer :: nx,ny,nz,ng,nv,eul_kmin,eul_kmax
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(in), device :: fhat_gpu
        real(rkind), dimension(1:nx,1:ny,1:nz,1:nv), intent(inout), device :: fl_gpu
        real(rkind), dimension(1:nz), intent(in), device :: dzitdz_gpu  
        integer(kind=cuda_stream_kind) :: stream_id
        integer :: i,j,k,m,iv,iercuda

        !$cuf kernel do(3) <<<*,*,stream=stream_id>>>
        do k=eul_kmin,eul_kmax
         do j=1,ny
          do i=1,nx
           do iv=1,nv
            fl_gpu(i,j,k,iv) = fl_gpu(i,j,k,iv) + (fhat_gpu(i,j,k,iv)-fhat_gpu(i,j,k-1,iv))*dzitdz_gpu(k)
           enddo
          enddo
         enddo
        enddo
    endsubroutine euler_z_update_cuf

    attributes(global) launch_bounds(256) subroutine euler_z_hybrid_kernel(nv, nv_aux, nx, ny, nz, ng, &
        eul_kmin, eul_kmax, lmax_base, nkeep, rgas_gpu, w_aux_gpu, fl_gpu, coeff_deriv1_gpu, dzitdz_gpu, fhat_gpu, &
        force_zero_flux_min, force_zero_flux_max, &
        weno_scheme, weno_version, sensor_threshold, weno_size, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, &
        indx_cp_l, indx_cp_r, ep_ord_change_gpu, tol_iter_nr,rho0,u0)
        implicit none
        ! Passed arguments
        integer, value :: nv, nx, ny, nz, ng, nv_aux, nsetcv
        integer, value :: eul_kmin, eul_kmax, lmax_base, nkeep, indx_cp_l, indx_cp_r
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv) :: fhat_gpu
        integer, dimension(0:nx,0:ny,0:nz,1:3) :: ep_ord_change_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind), dimension(N_S) :: rgas_gpu
        real(rkind), dimension(nx,ny,nz,nv) :: fl_gpu
        real(rkind), dimension(4,4) :: coeff_deriv1_gpu
        real(rkind), dimension(nz) :: dzitdz_gpu
        integer, value :: force_zero_flux_min, force_zero_flux_max
        real(rkind), value :: sensor_threshold, tol_iter_nr ,rho0,u0
        integer, value :: weno_scheme, weno_size, weno_version
        ! Local variables
        integer :: i, j, k, m, l, lsp
        real(rkind) :: fh1, fh2, fh3, fh4, fh5
        real(rkind) :: rhom, uui, vvi, wwi, ppi, enti, rhoi, tti
        real(rkind) :: uuip, vvip, wwip, ppip, entip, rhoip, ttip
        real(rkind) :: ft1, ft2, ft3, ft4, ft5, ft6
        real(rkind) :: uvs1, uvs2, uvs3, uvs4, uvs6, uv_part
        real(rkind) :: uvs5
        integer :: kk, lmax, wenorec_ord
        integer :: ishk
        real(rkind) :: b1, b2, b3, c, ci, h, uu, vv, ww
        real(rkind), dimension(4+N_S,4+N_S) :: el, er
        real(rkind), dimension(4+N_S) :: evmax, fk
        real(rkind), dimension(4+N_S,8) :: gp, gm
        real(rkind), dimension(N_S) :: yyroe, prhoi
        integer :: ll, mm
        real(rkind) :: rho, pp, wc, gc, rhow
        real(rkind) :: tt
        real(rkind) :: uvs5_i,uvs5_k,uvs5_p,eei,eeip,yyi,yyip
        real(rkind) :: drho, dee, eem
        real(rkind) :: drhof, deef
        real(rkind) :: sumnumrho,sumnumee,sumdenrho,sumdenee
        integer :: n,n2
        real(rkind) :: t_sumdenrho,  t_sumdenee,  t2_sumdenrho, t2_sumdenee 
 
        i = blockDim%x * (blockIdx%x - 1) + threadIdx%x 
        j = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        if(i > nx .or. j > ny) return

        do k=eul_kmin-1,eul_kmax
            ishk = 0
            do kk=k-weno_scheme+1,k+weno_scheme
                if (w_aux_gpu(i,j,kk,J_DUC) > sensor_threshold) ishk = 1
            enddo

            if (ishk == 0) then
!
                lmax = max(lmax_base+ep_ord_change_gpu(i,j,k,3),1)
!
                do lsp=1,N_S
                    ft1  = 0._rkind
                    do l=1,lmax
                        uvs1 = 0._rkind
                        do m=0,l-1
 
                            rhoi  = w_aux_gpu(i,j,k-m,J_R)
                            wwi   = w_aux_gpu(i,j,k-m,J_W)
                            yyi   = w_aux_gpu(i,j,k-m,lsp)
 
                            rhoip = w_aux_gpu(i,j,k-m+l,J_R)
                            wwip  = w_aux_gpu(i,j,k-m+l,J_W)
                            yyip  = w_aux_gpu(i,j,k-m+l,lsp)
 
                            rhom  = rhoi+rhoip
 
                            if (nkeep <= 0) then
                                drhof = 1._rkind
                            else
                                sumnumrho = 1._rkind
                                drho   = 2._rkind*(rhoip-rhoi)/rhom
                                t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                                t2_sumdenrho = t_sumdenrho
                                sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                                do n = 2, nkeep
                                   n2 = 2*n
                                   t_sumdenrho = t2_sumdenrho * t_sumdenrho
                                   sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                                enddo
                                drhof = sumnumrho/sumdenrho
                            endif
 
                            uv_part = (wwi+wwip) * rhom * drhof
                            uvs1 = uvs1 + uv_part * (yyi+yyip)
                        enddo
                        ft1  = ft1  + coeff_deriv1_gpu(l,lmax)*uvs1
                    enddo
 
                    fh1 = 0.25_rkind*ft1
 
                    if ((k==0 .and. force_zero_flux_min == 1).or.(k==nz .and. force_zero_flux_max == 1)) then
                        fh1 = 0._rkind
                    endif
 
                    fhat_gpu(i,j,k,lsp) = fh1
                enddo
!
                ft2  = 0._rkind
                ft3  = 0._rkind
                ft4  = 0._rkind
                ft5  = 0._rkind
                ft6  = 0._rkind
                if (nkeep>=0) then
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5_i = 0._rkind
                     uvs5_k = 0._rkind
                     uvs5_p = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1

                         rhoi  = w_aux_gpu(i,j,k-m,J_R)
                         uui   = w_aux_gpu(i,j,k-m,J_U)
                         vvi   = w_aux_gpu(i,j,k-m,J_V)
                         wwi   = w_aux_gpu(i,j,k-m,J_W)
                         enti  = w_aux_gpu(i,j,k-m,J_H)
                         tti   = w_aux_gpu(i,j,k-m,J_T)
                         ppi   = w_aux_gpu(i,j,k-m,J_P)
                         eei   = enti-ppi/rhoi-0.5_rkind*(uui*uui+vvi*vvi+wwi*wwi)
 
                         rhoip = w_aux_gpu(i,j,k-m+l,J_R)
                         uuip  = w_aux_gpu(i,j,k-m+l,J_U)
                         vvip  = w_aux_gpu(i,j,k-m+l,J_V)
                         wwip  = w_aux_gpu(i,j,k-m+l,J_W)
                         entip = w_aux_gpu(i,j,k-m+l,J_H)
                         ttip  = w_aux_gpu(i,j,k-m+l,J_T)
                         ppip  = w_aux_gpu(i,j,k-m+l,J_P)
                         eeip  = entip-ppip/rhoip-0.5_rkind*(uuip*uuip+vvip*vvip+wwip*wwip)

                         rhom  = rhoi + rhoip
                         eem   = eei  + eeip

                         if(nkeep == 0) then
                             drhof = 1._rkind
                             deef  = 1._rkind
                         else
                            sumnumrho = 1._rkind
                            drho   = 2._rkind*(rhoip-rhoi)/rhom
                            dee    = 2._rkind*(eeip - eei)/eem
                            t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                            t_sumdenee  = (0.5_rkind*dee )*(0.5_rkind*dee )
                            t2_sumdenrho = t_sumdenrho
                            t2_sumdenee  = t_sumdenee
                            sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                            sumdenee  = 1._rkind + t_sumdenee
                            sumnumee  = 1._rkind + t_sumdenee  / (3._rkind)
                            do n = 2, nkeep
                               n2 = 2*n
                               t_sumdenrho = t2_sumdenrho * t_sumdenrho
                               t_sumdenee  = t2_sumdenee  * t_sumdenee
                               sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                               sumdenee  = sumdenee  + t_sumdenee
                               sumnumee  = sumnumee  + t_sumdenee  / (1._rkind+n2)
                            enddo
                            drhof = sumnumrho/sumdenrho
                            deef  = sumnumee /sumdenee
                         endif

                         uv_part = (wwi+wwip) * rhom * drhof
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5_i = uvs5_i + uv_part * eem * deef
                         uvs5_k = uvs5_k + uv_part * (uui*uuip+vvi*vvip+wwi*wwip)
                         uvs5_p = uvs5_p + 4._rkind*(wwi*ppip+wwip*ppi)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2  = ft2  + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3  = ft3  + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4  = ft4  + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5  = ft5  + coeff_deriv1_gpu(l,lmax)*(uvs5_i+uvs5_k+uvs5_p)
                     ft6  = ft6  + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                else
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5 = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1

                         rhoi  = w_aux_gpu(i,j,k-m,J_R)
                         uui   = w_aux_gpu(i,j,k-m,J_U)
                         vvi   = w_aux_gpu(i,j,k-m,J_V)
                         wwi   = w_aux_gpu(i,j,k-m,J_W)
                         enti  = w_aux_gpu(i,j,k-m,J_H)
                         tti   = w_aux_gpu(i,j,k-m,J_T)
                         ppi   = w_aux_gpu(i,j,k-m,J_P)
 
                         rhoip = w_aux_gpu(i,j,k-m+l,J_R)
                         uuip  = w_aux_gpu(i,j,k-m+l,J_U)
                         vvip  = w_aux_gpu(i,j,k-m+l,J_V)
                         wwip  = w_aux_gpu(i,j,k-m+l,J_W)
                         entip = w_aux_gpu(i,j,k-m+l,J_H)
                         ttip  = w_aux_gpu(i,j,k-m+l,J_T)
                         ppip  = w_aux_gpu(i,j,k-m+l,J_P)

                         rhom  = rhoi+rhoip
                         uv_part = (wwi+wwip) * rhom
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5 = uvs5 + uv_part * (enti+entip)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2 = ft2 + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3 = ft3 + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4 = ft4 + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5 = ft5 + coeff_deriv1_gpu(l,lmax)*uvs5
                     ft6 = ft6 + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                endif
                fh2 = 0.25_rkind*ft2
                fh3 = 0.25_rkind*ft3
                fh4 = 0.25_rkind*ft4
                fh5 = 0.25_rkind*ft5
                if ((k==0 .and. force_zero_flux_min == 1).or.(k==nz .and. force_zero_flux_max == 1)) then
                   fh2 = 0._rkind
                   fh3 = 0._rkind
                   fh4 = 0._rkind
                   fh5 = 0._rkind
                endif
                fh4 = fh4 + 0.5_rkind*ft6

                fhat_gpu(i,j,k,I_U) = fh2
                fhat_gpu(i,j,k,I_V) = fh3
                fhat_gpu(i,j,k,I_W) = fh4
                fhat_gpu(i,j,k,I_E) = fh5
            else
                call compute_roe_average(nx, ny, nz, ng, nv_aux, i, i, j, j, k, k+1, w_aux_gpu, rgas_gpu, &
                     b1, b2, b3, rho, c, ci, h, uu, vv, ww, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
                     tol_iter_nr, yyroe, prhoi)

                call eigenvectors_z(nv, b1, b2, b3, rho, uu, vv, ww, c, ci, h, el, er, yyroe, prhoi)
!
                do m=1,4+N_S ! loop on characteristic fields
                    evmax(m) = -1._rkind
                enddo
                do l=1,weno_size ! LLF
                    ll = k + l - weno_scheme
                    ww   = w_aux_gpu(i,j,ll,J_W)
                    tt   = w_aux_gpu(i,j,ll,J_T)
                    c    = w_aux_gpu(i,j,ll,J_C)
                    evmax(1) = max(abs(ww-c),evmax(1))
                    evmax(2) = max(abs(ww  ),evmax(2))
                    do lsp=3,N_S+3
                     evmax(lsp) = evmax(2)
                    enddo
                    evmax(4+N_S) = max(abs(ww+c),evmax(4+N_S))
                enddo
                do l=1,weno_size ! loop over the stencil centered at face i
                    ll = k + l - weno_scheme

                    rho    = w_aux_gpu(i,j,ll,J_R)
                    uu     = w_aux_gpu(i,j,ll,J_U)
                    vv     = w_aux_gpu(i,j,ll,J_V)
                    ww     = w_aux_gpu(i,j,ll,J_W)
                    h      = w_aux_gpu(i,j,ll,J_H) 
                    rhow   = rho*ww
                    pp     = w_aux_gpu(i,j,ll,J_P)
                    do lsp=1,N_S
                     fk(lsp)  = rhow*w_aux_gpu(i,j,ll,lsp)
                    enddo
                    fk(I_U)  = uu * rhow
                    fk(I_V)  = vv * rhow 
                    fk(I_W)  = ww * rhow + pp
                    fk(I_E)  = h  * rhow
                    do m=1,4+N_S
                        wc = 0._rkind
                        gc = 0._rkind

                        do lsp=1,N_S
                         wc = wc + el(lsp,m) * rho*w_aux_gpu(i,j,ll,lsp)
                         gc = gc + el(lsp,m) * fk(lsp)
                        enddo
                        wc = wc + el(I_U,m) * rho*uu
                        gc = gc + el(I_U,m) * fk(I_U)
                        wc = wc + el(I_V,m) * rho*vv
                        gc = gc + el(I_V,m) * fk(I_V)
                        wc = wc + el(I_W,m) * rho*ww
                        gc = gc + el(I_W,m) * fk(I_W)
                        wc = wc + el(I_E,m) * (rho*h-pp)
                        gc = gc + el(I_E,m) * fk(I_E)

                        c = 0.5_rkind * (gc + evmax(m) * wc)
                        gp(m,l) = c
                        gm(m,l) = gc - c
                    enddo
                enddo
!
!               Reconstruction of the '+' and '-' fluxes
!
                wenorec_ord = max(weno_scheme+ep_ord_change_gpu(i,j,k,3),1)
                !call wenorec(i,j,nv,nx,ny,gplus_z_gpu,gminus_z_gpu,fk,weno_scheme,wenorec_ord,weno_version)
                call wenorec_1d(nv,gp,gm,fk,weno_scheme,wenorec_ord,weno_version,rho0,u0)
!
!               !Return to conservative fluxes
                do m=1,4+N_S
                    fhat_gpu(i,j,k,m) = 0._rkind
                    do mm=1,4+N_S
                       fhat_gpu(i,j,k,m) = fhat_gpu(i,j,k,m) + er(mm,m) * fk(mm)
                    enddo
                enddo

            endif
        enddo

!       Update net flux 
        !do k=eul_kmin,eul_kmax ! loop on the inner nodes
        !    do m=1,5
        !        fl_gpu(i,j,k,m) = fl_gpu(i,j,k,m) + (fhat_gpu(i,j,k,m)-fhat_gpu(i,j,k-1,m))*dzitdz_gpu(k)
        !    enddo
        !enddo

    endsubroutine euler_z_hybrid_kernel

    attributes(global) launch_bounds(256) subroutine euler_z_hybrid_rusanov_kernel(nv, nv_aux, nx, ny, nz, ng, &
        eul_kmin, eul_kmax, lmax_base, nkeep, w_aux_gpu, fl_gpu, coeff_deriv1_gpu, dzitdz_gpu, fhat_gpu, &
        force_zero_flux_min, force_zero_flux_max, &
        weno_scheme, weno_version, sensor_threshold, weno_size, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
        ep_ord_change_gpu, tol_iter_nr,rho0,u0)
        implicit none
        ! Passed arguments
        integer, value :: nv, nx, ny, nz, ng, nv_aux, nsetcv
        integer, value :: eul_kmin, eul_kmax, lmax_base, nkeep, indx_cp_l, indx_cp_r
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv) :: fhat_gpu
        integer, dimension(0:nx,0:ny,0:nz,1:3) :: ep_ord_change_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind), dimension(nx,ny,nz,nv) :: fl_gpu
        real(rkind), dimension(4,4) :: coeff_deriv1_gpu
        real(rkind), dimension(nz) :: dzitdz_gpu
        integer, value :: force_zero_flux_min, force_zero_flux_max
        real(rkind), value :: sensor_threshold, tol_iter_nr,rho0,u0
        integer, value :: weno_scheme, weno_size, weno_version
        ! Local variables
        integer :: i, j, k, m, l, lsp
        real(rkind) :: fh1, fh2, fh3, fh4, fh5
        real(rkind) :: rhom, uui, vvi, wwi, ppi, enti, rhoi, tti
        real(rkind) :: uuip, vvip, wwip, ppip, entip, rhoip, ttip
        real(rkind) :: ft1, ft2, ft3, ft4, ft5, ft6
        real(rkind) :: uvs1, uvs2, uvs3, uvs4, uvs6, uv_part
        real(rkind) :: uvs5
        integer :: kk, lmax, wenorec_ord
        integer :: ishk
        real(rkind) :: b1, b2, b3, c, ci, h, uu, vv, ww
        real(rkind), dimension(4+N_S) :: fk
        real(rkind), dimension(4+N_S,8) :: gp, gm
        integer :: ll, mm
        real(rkind) :: evm, evmax, rhoevm
        real(rkind) :: rho, pp, wc, gc, rhow
        real(rkind) :: tt
        real(rkind) :: uvs5_i,uvs5_k,uvs5_p,eei,eeip,yyi,yyip
        real(rkind) :: drho, dee, eem
        real(rkind) :: drhof, deef
        real(rkind) :: sumnumrho,sumnumee,sumdenrho,sumdenee
        integer :: n,n2
        real(rkind) :: t_sumdenrho,  t_sumdenee,  t2_sumdenrho, t2_sumdenee 
        real(rkind) :: yy
 
        i = blockDim%x * (blockIdx%x - 1) + threadIdx%x 
        j = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        if(i > nx .or. j > ny) return

        do k=eul_kmin-1,eul_kmax
            ishk = 0
            do kk=k-weno_scheme+1,k+weno_scheme
                if (w_aux_gpu(i,j,kk,J_DUC) > sensor_threshold) ishk = 1
            enddo

            if (ishk == 0) then
!
                lmax = max(lmax_base+ep_ord_change_gpu(i,j,k,3),1)
!
                do lsp=1,N_S
                    ft1  = 0._rkind
                    do l=1,lmax
                        uvs1 = 0._rkind
                        do m=0,l-1
 
                            rhoi  = w_aux_gpu(i,j,k-m,J_R)
                            wwi   = w_aux_gpu(i,j,k-m,J_W)
                            yyi   = w_aux_gpu(i,j,k-m,lsp)
 
                            rhoip = w_aux_gpu(i,j,k-m+l,J_R)
                            wwip  = w_aux_gpu(i,j,k-m+l,J_W)
                            yyip  = w_aux_gpu(i,j,k-m+l,lsp)
 
                            rhom  = rhoi+rhoip
 
                            if (nkeep <= 0) then
                                drhof = 1._rkind
                            else
                                sumnumrho = 1._rkind
                                drho   = 2._rkind*(rhoip-rhoi)/rhom
                                t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                                t2_sumdenrho = t_sumdenrho
                                sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                                do n = 2, nkeep
                                   n2 = 2*n
                                   t_sumdenrho = t2_sumdenrho * t_sumdenrho
                                   sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                                enddo
                                drhof = sumnumrho/sumdenrho
                            endif
 
                            uv_part = (wwi+wwip) * rhom * drhof
                            uvs1 = uvs1 + uv_part * (yyi+yyip)
                        enddo
                        ft1  = ft1  + coeff_deriv1_gpu(l,lmax)*uvs1
                    enddo
 
                    fh1 = 0.25_rkind*ft1
 
                    if ((k==0 .and. force_zero_flux_min == 1).or.(k==nz .and. force_zero_flux_max == 1)) then
                        fh1 = 0._rkind
                    endif
 
                    fhat_gpu(i,j,k,lsp) = fh1
                enddo
!
                ft2  = 0._rkind
                ft3  = 0._rkind
                ft4  = 0._rkind
                ft5  = 0._rkind
                ft6  = 0._rkind
                if (nkeep>=0) then
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5_i = 0._rkind
                     uvs5_k = 0._rkind
                     uvs5_p = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1

                         rhoi  = w_aux_gpu(i,j,k-m,J_R)
                         uui   = w_aux_gpu(i,j,k-m,J_U)
                         vvi   = w_aux_gpu(i,j,k-m,J_V)
                         wwi   = w_aux_gpu(i,j,k-m,J_W)
                         enti  = w_aux_gpu(i,j,k-m,J_H)
                         tti   = w_aux_gpu(i,j,k-m,J_T)
                         ppi   = w_aux_gpu(i,j,k-m,J_P)
                         eei   = enti-ppi/rhoi-0.5_rkind*(uui*uui+vvi*vvi+wwi*wwi)
 
                         rhoip = w_aux_gpu(i,j,k-m+l,J_R)
                         uuip  = w_aux_gpu(i,j,k-m+l,J_U)
                         vvip  = w_aux_gpu(i,j,k-m+l,J_V)
                         wwip  = w_aux_gpu(i,j,k-m+l,J_W)
                         entip = w_aux_gpu(i,j,k-m+l,J_H)
                         ttip  = w_aux_gpu(i,j,k-m+l,J_T)
                         ppip  = w_aux_gpu(i,j,k-m+l,J_P)
                         eeip  = entip-ppip/rhoip-0.5_rkind*(uuip*uuip+vvip*vvip+wwip*wwip)

                         rhom  = rhoi + rhoip
                         eem   = eei  + eeip

                         if(nkeep == 0) then
                             drhof = 1._rkind
                             deef  = 1._rkind
                         else
                            sumnumrho = 1._rkind
                            drho   = 2._rkind*(rhoip-rhoi)/rhom
                            dee    = 2._rkind*(eeip - eei)/eem
                            t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                            t_sumdenee  = (0.5_rkind*dee )*(0.5_rkind*dee )
                            t2_sumdenrho = t_sumdenrho
                            t2_sumdenee  = t_sumdenee
                            sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                            sumdenee  = 1._rkind + t_sumdenee
                            sumnumee  = 1._rkind + t_sumdenee  / (3._rkind)
                            do n = 2, nkeep
                               n2 = 2*n
                               t_sumdenrho = t2_sumdenrho * t_sumdenrho
                               t_sumdenee  = t2_sumdenee  * t_sumdenee
                               sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                               sumdenee  = sumdenee  + t_sumdenee
                               sumnumee  = sumnumee  + t_sumdenee  / (1._rkind+n2)
                            enddo
                            drhof = sumnumrho/sumdenrho
                            deef  = sumnumee /sumdenee
                         endif

                         uv_part = (wwi+wwip) * rhom * drhof
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5_i = uvs5_i + uv_part * eem * deef
                         uvs5_k = uvs5_k + uv_part * (uui*uuip+vvi*vvip+wwi*wwip)
                         uvs5_p = uvs5_p + 4._rkind*(wwi*ppip+wwip*ppi)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2  = ft2  + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3  = ft3  + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4  = ft4  + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5  = ft5  + coeff_deriv1_gpu(l,lmax)*(uvs5_i+uvs5_k+uvs5_p)
                     ft6  = ft6  + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                else
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5 = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1

                         rhoi  = w_aux_gpu(i,j,k-m,J_R)
                         uui   = w_aux_gpu(i,j,k-m,J_U)
                         vvi   = w_aux_gpu(i,j,k-m,J_V)
                         wwi   = w_aux_gpu(i,j,k-m,J_W)
                         enti  = w_aux_gpu(i,j,k-m,J_H)
                         tti   = w_aux_gpu(i,j,k-m,J_T)
                         ppi   = w_aux_gpu(i,j,k-m,J_P)
 
                         rhoip = w_aux_gpu(i,j,k-m+l,J_R)
                         uuip  = w_aux_gpu(i,j,k-m+l,J_U)
                         vvip  = w_aux_gpu(i,j,k-m+l,J_V)
                         wwip  = w_aux_gpu(i,j,k-m+l,J_W)
                         entip = w_aux_gpu(i,j,k-m+l,J_H)
                         ttip  = w_aux_gpu(i,j,k-m+l,J_T)
                         ppip  = w_aux_gpu(i,j,k-m+l,J_P)

                         rhom  = rhoi+rhoip
                         uv_part = (wwi+wwip) * rhom
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5 = uvs5 + uv_part * (enti+entip)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2 = ft2 + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3 = ft3 + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4 = ft4 + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5 = ft5 + coeff_deriv1_gpu(l,lmax)*uvs5
                     ft6 = ft6 + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                endif
                fh2 = 0.25_rkind*ft2
                fh3 = 0.25_rkind*ft3
                fh4 = 0.25_rkind*ft4
                fh5 = 0.25_rkind*ft5
                if ((k==0 .and. force_zero_flux_min == 1).or.(k==nz .and. force_zero_flux_max == 1)) then
                   fh2 = 0._rkind
                   fh3 = 0._rkind
                   fh4 = 0._rkind
                   fh5 = 0._rkind
                endif
                fh4 = fh4 + 0.5_rkind*ft6

                fhat_gpu(i,j,k,I_U) = fh2
                fhat_gpu(i,j,k,I_V) = fh3
                fhat_gpu(i,j,k,I_W) = fh4
                fhat_gpu(i,j,k,I_E) = fh5
            else
                evmax = -1._rkind
                do l=1,weno_size ! LLF
                    ll = k + l - weno_scheme
                    ww   = w_aux_gpu(i,j,ll,J_W)
                    tt   = w_aux_gpu(i,j,ll,J_T)
                    c    = w_aux_gpu(i,j,ll,J_C)
                    evm   = max(abs(ww-c),abs(ww+c))
                    evmax = max(evm,evmax)
                enddo
                do l=1,weno_size ! loop over the stencil centered at face i
                    ll = k + l - weno_scheme

                    rho    = w_aux_gpu(i,j,ll,J_R)
                    uu     = w_aux_gpu(i,j,ll,J_U)
                    vv     = w_aux_gpu(i,j,ll,J_V)
                    ww     = w_aux_gpu(i,j,ll,J_W)
                    h      = w_aux_gpu(i,j,ll,J_H) 
                    rhow   = rho*ww
                    pp     = w_aux_gpu(i,j,ll,J_P)

                    rhoevm = rho*evmax
                    do lsp=1,N_S
                     yy = w_aux_gpu(i,j,ll,lsp)
                     evm = rhow*yy
                     c = 0.5_rkind * (evm + rhoevm*yy)
                     gp(lsp,l) = c
                     gm(lsp,l) = evm-c
                    enddo
                    evm  = uu * rhow
                    c = 0.5_rkind * (evm + rhoevm * uu)
                    gp(I_U,l) = c
                    gm(I_U,l) = evm-c
                    evm = vv * rhow
                    c = 0.5_rkind * (evm + rhoevm * vv)
                    gp(I_V,l) = c
                    gm(I_V,l) = evm-c
                    evm = ww * rhow + pp
                    c = 0.5_rkind * (evm + rhoevm * ww)
                    gp(I_W,l) = c
                    gm(I_W,l) = evm-c
                    evm = h  * rhow
                    c = 0.5_rkind * (evm + evmax * (rho*h-pp))
                    gp(I_E,l) = c
                    gm(I_E,l) = evm-c
                enddo
!
!               Reconstruction of the '+' and '-' fluxes
!
                wenorec_ord = max(weno_scheme+ep_ord_change_gpu(i,j,k,3),1)
                call wenorec_1d_rusanov(nv,gp,gm,fk,weno_scheme,wenorec_ord,weno_version,rho0,u0)
!
!               !Return to conservative fluxes
                do m=1,4+N_S
                   fhat_gpu(i,j,k,m) = fk(m)
                enddo

            endif
        enddo

    endsubroutine euler_z_hybrid_rusanov_kernel

    attributes(global) launch_bounds(256) subroutine euler_y_hybrid_kernel(nv, nv_aux, nx, ny, nz, ng, &
        eul_jmin, eul_jmax, lmax_base, nkeep, rgas_gpu, w_aux_gpu, fl_gpu, coeff_deriv1_gpu, detady_gpu, fhat_gpu, &
        force_zero_flux_min, force_zero_flux_max, &
        weno_scheme, weno_version, sensor_threshold, weno_size, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, &
        indx_cp_l, indx_cp_r, ep_ord_change_gpu, tol_iter_nr, rho0, u0)
        implicit none
        ! Passed arguments
        integer, value :: nv, nx, ny, nz, ng, nv_aux, nsetcv
        integer, value :: eul_jmin, eul_jmax, lmax_base, nkeep, indx_cp_l, indx_cp_r
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv) :: fhat_gpu
        integer, dimension(0:nx,0:ny,0:nz,1:3) :: ep_ord_change_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind), dimension(N_S) :: rgas_gpu
        real(rkind), dimension(nx,ny,nz,nv) :: fl_gpu
        real(rkind), dimension(4,4) :: coeff_deriv1_gpu
        real(rkind), dimension(ny) :: detady_gpu
        integer, value :: force_zero_flux_min, force_zero_flux_max
        real(rkind), value :: sensor_threshold, tol_iter_nr, rho0, u0
        integer, value :: weno_scheme, weno_size, weno_version
        ! Local variables
        integer :: i, j, k, m, l, lsp
        real(rkind) :: fh1, fh2, fh3, fh4, fh5
        real(rkind) :: rhom, uui, vvi, wwi, ppi, enti, rhoi, tti
        real(rkind) :: uuip, vvip, wwip, ppip, entip, rhoip, ttip
        real(rkind) :: ft1, ft2, ft3, ft4, ft5, ft6
        real(rkind) :: uvs1, uvs2, uvs3, uvs4, uvs6, uv_part
        real(rkind) :: uvs5
        integer :: jj
        integer :: ishk
        real(rkind) :: b1, b2, b3, c, ci, h, uu, vv, ww
        real(rkind), dimension(4+N_S,4+N_S) :: el, er
        real(rkind), dimension(4+N_S) :: evmax, fj
        real(rkind), dimension(4+N_S,8) :: gp, gm
        real(rkind), dimension(N_S) :: yyroe, prhoi
        integer :: ll, mm, lmax, wenorec_ord
        real(rkind) :: rho, pp, wc, gc, rhov
        real(rkind) :: tt
        real(rkind) :: uvs5_i,uvs5_k,uvs5_p,eei,eeip,yyi,yyip
        real(rkind) :: drho, dee, eem
        real(rkind) :: drhof, deef
        real(rkind) :: sumnumrho,sumnumee,sumdenrho,sumdenee
        integer :: n,n2
        real(rkind) :: t_sumdenrho,  t_sumdenee,  t2_sumdenrho, t2_sumdenee 
 
        i = blockDim%x * (blockIdx%x - 1) + threadIdx%x 
        k = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        if (i > nx .or. k > nz) return

        do j=eul_jmin-1,eul_jmax

            ishk = 0
            do jj=j-weno_scheme+1,j+weno_scheme
                if (w_aux_gpu(i,jj,k,J_DUC) > sensor_threshold) ishk = 1
            enddo

            if (ishk == 0) then
!
                lmax = max(lmax_base+ep_ord_change_gpu(i,j,k,2),1)
!
                do lsp=1,N_S
                    ft1  = 0._rkind
                    do l=1,lmax
                        uvs1 = 0._rkind
                        do m=0,l-1
 
                            rhoi  = w_aux_gpu(i,j-m,k,J_R)
                            vvi   = w_aux_gpu(i,j-m,k,J_V)
                            yyi   = w_aux_gpu(i,j-m,k,lsp)
 
                            rhoip = w_aux_gpu(i,j-m+l,k,J_R)
                            vvip  = w_aux_gpu(i,j-m+l,k,J_V)
                            yyip  = w_aux_gpu(i,j-m+l,k,lsp)
 
                            rhom  = rhoi+rhoip
 
                            if (nkeep <= 0) then
                                drhof = 1._rkind
                            else
                                sumnumrho = 1._rkind
                                drho   = 2._rkind*(rhoip-rhoi)/rhom
                                t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                                t2_sumdenrho = t_sumdenrho
                                sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                                do n = 2, nkeep
                                   n2 = 2*n
                                   t_sumdenrho = t2_sumdenrho * t_sumdenrho
                                   sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                                enddo
                                drhof = sumnumrho/sumdenrho
                            endif
 
                            uv_part = (vvi+vvip) * rhom * drhof
                            uvs1 = uvs1 + uv_part * (yyi+yyip)
                        enddo
                        ft1  = ft1  + coeff_deriv1_gpu(l,lmax)*uvs1
                    enddo
 
                    fh1 = 0.25_rkind*ft1
 
                    if ((j==0 .and. force_zero_flux_min == 1).or.(j==ny .and. force_zero_flux_max == 1)) then
                        fh1 = 0._rkind
                    endif
 
                    fhat_gpu(i,j,k,lsp) = fh1
                enddo
!
                ft2  = 0._rkind
                ft3  = 0._rkind
                ft4  = 0._rkind
                ft5  = 0._rkind
                ft6  = 0._rkind
                if (nkeep>=0) then 
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5_i = 0._rkind
                     uvs5_k = 0._rkind
                     uvs5_p = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1

                         rhoi  = w_aux_gpu(i,j-m,k,J_R)
                         uui   = w_aux_gpu(i,j-m,k,J_U)
                         vvi   = w_aux_gpu(i,j-m,k,J_V)
                         wwi   = w_aux_gpu(i,j-m,k,J_W)
                         enti  = w_aux_gpu(i,j-m,k,J_H)
                         tti   = w_aux_gpu(i,j-m,k,J_T)
                         ppi   = w_aux_gpu(i,j-m,k,J_P)
                         eei   = enti-ppi/rhoi-0.5_rkind*(uui*uui+vvi*vvi+wwi*wwi)

                         rhoip = w_aux_gpu(i,j-m+l,k,J_R)
                         uuip  = w_aux_gpu(i,j-m+l,k,J_U)
                         vvip  = w_aux_gpu(i,j-m+l,k,J_V)
                         wwip  = w_aux_gpu(i,j-m+l,k,J_W)
                         entip = w_aux_gpu(i,j-m+l,k,J_H)
                         ttip  = w_aux_gpu(i,j-m+l,k,J_T)
                         ppip  = w_aux_gpu(i,j-m+l,k,J_P)
                         eeip  = entip-ppip/rhoip-0.5_rkind*(uuip*uuip+vvip*vvip+wwip*wwip)

                         rhom  = rhoi + rhoip
                         eem   = eei  + eeip

                         if(nkeep == 0) then
                             drhof = 1._rkind
                             deef  = 1._rkind
                         else
                            sumnumrho = 1._rkind
                            drho   = 2._rkind*(rhoip-rhoi)/rhom
                            dee    = 2._rkind*(eeip - eei)/eem
                            t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                            t_sumdenee  = (0.5_rkind*dee )*(0.5_rkind*dee )
                            t2_sumdenrho = t_sumdenrho
                            t2_sumdenee  = t_sumdenee
                            sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                            sumdenee  = 1._rkind + t_sumdenee
                            sumnumee  = 1._rkind + t_sumdenee  / (3._rkind)
                            do n = 2, nkeep
                               n2 = 2*n
                               t_sumdenrho = t2_sumdenrho * t_sumdenrho
                               t_sumdenee  = t2_sumdenee  * t_sumdenee
                               sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                               sumdenee  = sumdenee  + t_sumdenee
                               sumnumee  = sumnumee  + t_sumdenee  / (1._rkind+n2)
                            enddo
                            drhof = sumnumrho/sumdenrho
                            deef  = sumnumee /sumdenee
                         endif

                         uv_part = (vvi+vvip) * rhom * drhof
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5_i = uvs5_i + uv_part * eem * deef
                         uvs5_k = uvs5_k + uv_part * (uui*uuip+vvi*vvip+wwi*wwip)
                         uvs5_p = uvs5_p + 4._rkind*(vvi*ppip+vvip*ppi)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2  = ft2  + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3  = ft3  + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4  = ft4  + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5  = ft5  + coeff_deriv1_gpu(l,lmax)*(uvs5_i+uvs5_k+uvs5_p)
                     ft6  = ft6  + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                else
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5 = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1

                         rhoi  = w_aux_gpu(i,j-m,k,J_R)
                         uui   = w_aux_gpu(i,j-m,k,J_U)
                         vvi   = w_aux_gpu(i,j-m,k,J_V)
                         wwi   = w_aux_gpu(i,j-m,k,J_W)
                         enti  = w_aux_gpu(i,j-m,k,J_H)
                         tti   = w_aux_gpu(i,j-m,k,J_T)
                         ppi   = w_aux_gpu(i,j-m,k,J_P)

                         rhoip = w_aux_gpu(i,j-m+l,k,J_R)
                         uuip  = w_aux_gpu(i,j-m+l,k,J_U)
                         vvip  = w_aux_gpu(i,j-m+l,k,J_V)
                         wwip  = w_aux_gpu(i,j-m+l,k,J_W)
                         entip = w_aux_gpu(i,j-m+l,k,J_H)
                         ttip  = w_aux_gpu(i,j-m+l,k,J_T)
                         ppip  = w_aux_gpu(i,j-m+l,k,J_P)

                         rhom  = rhoi + rhoip
                         uv_part = (vvi+vvip) * rhom
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5 = uvs5 + uv_part * (enti+entip)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2 = ft2 + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3 = ft3 + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4 = ft4 + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5 = ft5 + coeff_deriv1_gpu(l,lmax)*uvs5
                     ft6 = ft6 + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                endif
                fh2 = 0.25_rkind*ft2
                fh3 = 0.25_rkind*ft3
                fh4 = 0.25_rkind*ft4
                fh5 = 0.25_rkind*ft5
                if ((j==0 .and. force_zero_flux_min == 1).or.(j==ny .and. force_zero_flux_max == 1)) then
                   fh2 = 0._rkind
                   fh3 = 0._rkind
                   fh4 = 0._rkind
                   fh5 = 0._rkind
                endif

                fh3 = fh3 + 0.5_rkind*ft6

                fhat_gpu(i,j,k,I_U) = fh2
                fhat_gpu(i,j,k,I_V) = fh3
                fhat_gpu(i,j,k,I_W) = fh4
                fhat_gpu(i,j,k,I_E) = fh5
            else
                call compute_roe_average(nx, ny, nz, ng, nv_aux, i, i, j, j+1, k, k, w_aux_gpu, rgas_gpu, &
                     b1, b2, b3, rho, c, ci, h, uu, vv, ww, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
                     tol_iter_nr, yyroe, prhoi)

                call eigenvectors_y(nv, b1, b2, b3, rho, uu, vv, ww, c, ci, h, el, er, yyroe, prhoi)
!
                do m=1,4+N_S ! loop on characteristic fields
                    evmax(m) = -1._rkind
                enddo
                do l=1,weno_size ! LLF
                    ll = j + l - weno_scheme
                    vv   = w_aux_gpu(i,ll,k,J_V)
                    tt   = w_aux_gpu(i,ll,k,J_T)
                    c    = w_aux_gpu(i,ll,k,J_C)
                    evmax(1) = max(abs(vv-c),evmax(1))
                    evmax(2) = max(abs(vv  ),evmax(2))
                    do lsp=3,N_S+3
                     evmax(lsp) = evmax(2)
                    enddo
                    evmax(4+N_S) = max(abs(vv+c),evmax(4+N_S))
                enddo
                do l=1,weno_size ! loop over the stencil centered at face i
                    ll = j + l - weno_scheme

                    rho    = w_aux_gpu(i,ll,k,J_R)
                    uu     = w_aux_gpu(i,ll,k,J_U)
                    vv     = w_aux_gpu(i,ll,k,J_V)
                    ww     = w_aux_gpu(i,ll,k,J_W)
                    h      = w_aux_gpu(i,ll,k,J_H) 
                    rhov   = rho*vv
                    pp     = w_aux_gpu(i,ll,k,J_P)
                    do lsp=1,N_S
                     fj(lsp)  = rhov*w_aux_gpu(i,ll,k,lsp)
                    enddo
                    fj(I_U)  = uu * rhov
                    fj(I_V)  = vv * rhov + pp
                    fj(I_W)  = ww * rhov
                    fj(I_E)  = h  * rhov
                    do m=1,4+N_S
                        wc = 0._rkind
                        gc = 0._rkind

                        do lsp=1,N_S
                         wc = wc + el(lsp,m) * rho*w_aux_gpu(i,ll,k,lsp)
                         gc = gc + el(lsp,m) * fj(lsp)
                        enddo
                        wc = wc + el(I_U,m) * rho*uu
                        gc = gc + el(I_U,m) * fj(I_U)
                        wc = wc + el(I_V,m) * rho*vv
                        gc = gc + el(I_V,m) * fj(I_V)
                        wc = wc + el(I_W,m) * rho*ww
                        gc = gc + el(I_W,m) * fj(I_W)
                        wc = wc + el(I_E,m) * (rho*h-pp)
                        gc = gc + el(I_E,m) * fj(I_E)

                        c = 0.5_rkind * (gc + evmax(m) * wc)
                        !gplus_y_gpu (i,k,m,l) = c
                        !gminus_y_gpu(i,k,m,l) = gc - c
                        gp(m,l) = c
                        gm(m,l) = gc - c
                    enddo
                enddo
!
!               Reconstruction of the '+' and '-' fluxes
!
                wenorec_ord = max(weno_scheme+ep_ord_change_gpu(i,j,k,2),1)
                call wenorec_1d(nv,gp,gm,fj,weno_scheme,wenorec_ord,weno_version,rho0,u0)
                !call wenorec(i,k,nv,nx,nz,gplus_y_gpu,gminus_y_gpu,fj,weno_scheme,wenorec_ord,weno_version)
!
!               !Return to conservative fluxes
                do m=1,4+N_S
                    fhat_gpu(i,j,k,m) = 0._rkind
                    do mm=1,4+N_S
                       fhat_gpu(i,j,k,m) = fhat_gpu(i,j,k,m) + er(mm,m) * fj(mm)
                    enddo
                enddo

            endif
        enddo

!       Update net flux 
        !do j=eul_jmin,eul_jmax ! loop on the inner nodes
        !   do m=1,5
        !       fl_gpu(i,j,k,m) = fl_gpu(i,j,k,m) + (fhat_gpu(i,j,k,m)-fhat_gpu(i,j-1,k,m))*detady_gpu(j)
        !   enddo
        !enddo

    endsubroutine euler_y_hybrid_kernel

    attributes(global) launch_bounds(256) subroutine euler_y_hybrid_rusanov_kernel(nv, nv_aux, nx, ny, nz, ng, &
        eul_jmin, eul_jmax, lmax_base, nkeep, w_aux_gpu, fl_gpu, coeff_deriv1_gpu, detady_gpu, fhat_gpu, &
        force_zero_flux_min, force_zero_flux_max, &
        weno_scheme, weno_version, sensor_threshold, weno_size, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, &
        ep_ord_change_gpu, tol_iter_nr,rho0,u0)
        implicit none
        ! Passed arguments
        integer, value :: nv, nx, ny, nz, ng, nv_aux, nsetcv
        integer, value :: eul_jmin, eul_jmax, lmax_base, nkeep, indx_cp_l, indx_cp_r
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv) :: fhat_gpu
        integer, dimension(0:nx,0:ny,0:nz,1:3) :: ep_ord_change_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind), dimension(nx,ny,nz,nv) :: fl_gpu
        real(rkind), dimension(4,4) :: coeff_deriv1_gpu
        real(rkind), dimension(ny) :: detady_gpu
        integer, value :: force_zero_flux_min, force_zero_flux_max
        real(rkind), value :: sensor_threshold, tol_iter_nr, rho0, u0
        integer, value :: weno_scheme, weno_size, weno_version
        ! Local variables
        integer :: i, j, k, m, l, lsp
        real(rkind) :: fh1, fh2, fh3, fh4, fh5
        real(rkind) :: rhom, uui, vvi, wwi, ppi, enti, rhoi, tti
        real(rkind) :: uuip, vvip, wwip, ppip, entip, rhoip, ttip
        real(rkind) :: ft1, ft2, ft3, ft4, ft5, ft6
        real(rkind) :: uvs1, uvs2, uvs3, uvs4, uvs6, uv_part
        real(rkind) :: uvs5
        integer :: jj
        integer :: ishk
        real(rkind) :: b1, b2, b3, c, ci, h, uu, vv, ww
        real(rkind), dimension(4+N_S) :: fj
        real(rkind), dimension(4+N_S,8) :: gp, gm
        integer :: ll, mm, lmax, wenorec_ord
        real(rkind) :: evm,evmax,rhoevm
        real(rkind) :: rho, pp, wc, gc, rhov
        real(rkind) :: tt
        real(rkind) :: uvs5_i,uvs5_k,uvs5_p,eei,eeip,yyi,yyip
        real(rkind) :: drho, dee, eem
        real(rkind) :: drhof, deef
        real(rkind) :: sumnumrho,sumnumee,sumdenrho,sumdenee
        integer :: n,n2
        real(rkind) :: t_sumdenrho,  t_sumdenee,  t2_sumdenrho, t2_sumdenee 
        real(rkind) :: yy
 
        i = blockDim%x * (blockIdx%x - 1) + threadIdx%x 
        k = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        if (i > nx .or. k > nz) return

        do j=eul_jmin-1,eul_jmax

            ishk = 0
            do jj=j-weno_scheme+1,j+weno_scheme
                if (w_aux_gpu(i,jj,k,J_DUC) > sensor_threshold) ishk = 1
            enddo

            if (ishk == 0) then
!
                lmax = max(lmax_base+ep_ord_change_gpu(i,j,k,2),1)
!
                do lsp=1,N_S
                    ft1  = 0._rkind
                    do l=1,lmax
                        uvs1 = 0._rkind
                        do m=0,l-1
 
                            rhoi  = w_aux_gpu(i,j-m,k,J_R)
                            vvi   = w_aux_gpu(i,j-m,k,J_V)
                            yyi   = w_aux_gpu(i,j-m,k,lsp)
 
                            rhoip = w_aux_gpu(i,j-m+l,k,J_R)
                            vvip  = w_aux_gpu(i,j-m+l,k,J_V)
                            yyip  = w_aux_gpu(i,j-m+l,k,lsp)
 
                            rhom  = rhoi+rhoip
 
                            if (nkeep <= 0) then
                                drhof = 1._rkind
                            else
                                sumnumrho = 1._rkind
                                drho   = 2._rkind*(rhoip-rhoi)/rhom
                                t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                                t2_sumdenrho = t_sumdenrho
                                sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                                do n = 2, nkeep
                                   n2 = 2*n
                                   t_sumdenrho = t2_sumdenrho * t_sumdenrho
                                   sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                                enddo
                                drhof = sumnumrho/sumdenrho
                            endif
 
                            uv_part = (vvi+vvip) * rhom * drhof
                            uvs1 = uvs1 + uv_part * (yyi+yyip)
                        enddo
                        ft1  = ft1  + coeff_deriv1_gpu(l,lmax)*uvs1
                    enddo
 
                    fh1 = 0.25_rkind*ft1
 
                    if ((j==0 .and. force_zero_flux_min == 1).or.(j==ny .and. force_zero_flux_max == 1)) then
                        fh1 = 0._rkind
                    endif
 
                    fhat_gpu(i,j,k,lsp) = fh1
                enddo
!
                ft2  = 0._rkind
                ft3  = 0._rkind
                ft4  = 0._rkind
                ft5  = 0._rkind
                ft6  = 0._rkind
                if (nkeep>=0) then 
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5_i = 0._rkind
                     uvs5_k = 0._rkind
                     uvs5_p = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1

                         rhoi  = w_aux_gpu(i,j-m,k,J_R)
                         uui   = w_aux_gpu(i,j-m,k,J_U)
                         vvi   = w_aux_gpu(i,j-m,k,J_V)
                         wwi   = w_aux_gpu(i,j-m,k,J_W)
                         enti  = w_aux_gpu(i,j-m,k,J_H)
                         tti   = w_aux_gpu(i,j-m,k,J_T)
                         ppi   = w_aux_gpu(i,j-m,k,J_P)
                         eei   = enti-ppi/rhoi-0.5_rkind*(uui*uui+vvi*vvi+wwi*wwi)

                         rhoip = w_aux_gpu(i,j-m+l,k,J_R)
                         uuip  = w_aux_gpu(i,j-m+l,k,J_U)
                         vvip  = w_aux_gpu(i,j-m+l,k,J_V)
                         wwip  = w_aux_gpu(i,j-m+l,k,J_W)
                         entip = w_aux_gpu(i,j-m+l,k,J_H)
                         ttip  = w_aux_gpu(i,j-m+l,k,J_T)
                         ppip  = w_aux_gpu(i,j-m+l,k,J_P)
                         eeip  = entip-ppip/rhoip-0.5_rkind*(uuip*uuip+vvip*vvip+wwip*wwip)

                         rhom  = rhoi + rhoip
                         eem   = eei  + eeip

                         if(nkeep == 0) then
                             drhof = 1._rkind
                             deef  = 1._rkind
                         else
                            sumnumrho = 1._rkind
                            drho   = 2._rkind*(rhoip-rhoi)/rhom
                            dee    = 2._rkind*(eeip - eei)/eem
                            t_sumdenrho = (0.5_rkind*drho)*(0.5_rkind*drho)
                            t_sumdenee  = (0.5_rkind*dee )*(0.5_rkind*dee )
                            t2_sumdenrho = t_sumdenrho
                            t2_sumdenee  = t_sumdenee
                            sumdenrho = 1._rkind + t_sumdenrho / (3._rkind)
                            sumdenee  = 1._rkind + t_sumdenee
                            sumnumee  = 1._rkind + t_sumdenee  / (3._rkind)
                            do n = 2, nkeep
                               n2 = 2*n
                               t_sumdenrho = t2_sumdenrho * t_sumdenrho
                               t_sumdenee  = t2_sumdenee  * t_sumdenee
                               sumdenrho = sumdenrho + t_sumdenrho / (1._rkind+n2)
                               sumdenee  = sumdenee  + t_sumdenee
                               sumnumee  = sumnumee  + t_sumdenee  / (1._rkind+n2)
                            enddo
                            drhof = sumnumrho/sumdenrho
                            deef  = sumnumee /sumdenee
                         endif

                         uv_part = (vvi+vvip) * rhom * drhof
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5_i = uvs5_i + uv_part * eem * deef
                         uvs5_k = uvs5_k + uv_part * (uui*uuip+vvi*vvip+wwi*wwip)
                         uvs5_p = uvs5_p + 4._rkind*(vvi*ppip+vvip*ppi)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2  = ft2  + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3  = ft3  + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4  = ft4  + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5  = ft5  + coeff_deriv1_gpu(l,lmax)*(uvs5_i+uvs5_k+uvs5_p)
                     ft6  = ft6  + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                else
                 do l=1,lmax
                     uvs2 = 0._rkind
                     uvs3 = 0._rkind
                     uvs4 = 0._rkind
                     uvs5 = 0._rkind
                     uvs6 = 0._rkind
                     do m=0,l-1

                         rhoi  = w_aux_gpu(i,j-m,k,J_R)
                         uui   = w_aux_gpu(i,j-m,k,J_U)
                         vvi   = w_aux_gpu(i,j-m,k,J_V)
                         wwi   = w_aux_gpu(i,j-m,k,J_W)
                         enti  = w_aux_gpu(i,j-m,k,J_H)
                         tti   = w_aux_gpu(i,j-m,k,J_T)
                         ppi   = w_aux_gpu(i,j-m,k,J_P)

                         rhoip = w_aux_gpu(i,j-m+l,k,J_R)
                         uuip  = w_aux_gpu(i,j-m+l,k,J_U)
                         vvip  = w_aux_gpu(i,j-m+l,k,J_V)
                         wwip  = w_aux_gpu(i,j-m+l,k,J_W)
                         entip = w_aux_gpu(i,j-m+l,k,J_H)
                         ttip  = w_aux_gpu(i,j-m+l,k,J_T)
                         ppip  = w_aux_gpu(i,j-m+l,k,J_P)

                         rhom  = rhoi + rhoip
                         uv_part = (vvi+vvip) * rhom
                         uvs2 = uvs2 + uv_part * (uui+uuip)
                         uvs3 = uvs3 + uv_part * (vvi+vvip)
                         uvs4 = uvs4 + uv_part * (wwi+wwip)
                         uvs5 = uvs5 + uv_part * (enti+entip)
                         uvs6 = uvs6 + (2._rkind)*(ppi+ppip)
                     enddo
                     ft2 = ft2 + coeff_deriv1_gpu(l,lmax)*uvs2
                     ft3 = ft3 + coeff_deriv1_gpu(l,lmax)*uvs3
                     ft4 = ft4 + coeff_deriv1_gpu(l,lmax)*uvs4
                     ft5 = ft5 + coeff_deriv1_gpu(l,lmax)*uvs5
                     ft6 = ft6 + coeff_deriv1_gpu(l,lmax)*uvs6
                 enddo
                endif
                fh2 = 0.25_rkind*ft2
                fh3 = 0.25_rkind*ft3
                fh4 = 0.25_rkind*ft4
                fh5 = 0.25_rkind*ft5
                if ((j==0 .and. force_zero_flux_min == 1).or.(j==ny .and. force_zero_flux_max == 1)) then
                   fh2 = 0._rkind
                   fh3 = 0._rkind
                   fh4 = 0._rkind
                   fh5 = 0._rkind
                endif

                fh3 = fh3 + 0.5_rkind*ft6

                fhat_gpu(i,j,k,I_U) = fh2
                fhat_gpu(i,j,k,I_V) = fh3
                fhat_gpu(i,j,k,I_W) = fh4
                fhat_gpu(i,j,k,I_E) = fh5
            else
                evmax = -1._rkind
                do l=1,weno_size ! LLF
                    ll = j + l - weno_scheme
                    vv   = w_aux_gpu(i,ll,k,J_V)
                    tt   = w_aux_gpu(i,ll,k,J_T)
                    c    = w_aux_gpu(i,ll,k,J_C)
                    evm   = max(abs(vv-c),abs(vv+c))
                    evmax = max(evm,evmax)
                enddo
                do l=1,weno_size ! loop over the stencil centered at face i
                    ll = j + l - weno_scheme

                    rho    = w_aux_gpu(i,ll,k,J_R)
                    uu     = w_aux_gpu(i,ll,k,J_U)
                    vv     = w_aux_gpu(i,ll,k,J_V)
                    ww     = w_aux_gpu(i,ll,k,J_W)
                    h      = w_aux_gpu(i,ll,k,J_H) 
                    rhov   = rho*vv
                    pp     = w_aux_gpu(i,ll,k,J_P)

                    rhoevm = rho*evmax
                    do lsp=1,N_S
                     yy = w_aux_gpu(i,ll,k,lsp)
                     evm = rhov*yy
                     c = 0.5_rkind * (evm + rhoevm * yy)
                     gp(lsp,l) = c
                     gm(lsp,l) = evm-c
                    enddo
                    evm = uu * rhov
                    c = 0.5_rkind * (evm + rhoevm * uu)
                    gp(I_U,l) = c
                    gm(I_U,l) = evm-c
                    evm = vv * rhov + pp
                    c = 0.5_rkind * (evm + rhoevm * vv)
                    gp(I_V,l) = c
                    gm(I_V,l) = evm-c
                    evm = ww * rhov
                    c = 0.5_rkind * (evm + rhoevm * ww)
                    gp(I_W,l) = c
                    gm(I_W,l) = evm-c
                    evm =  h  * rhov
                    c = 0.5_rkind * (evm + evmax * (rho*h-pp))
                    gp(I_E,l) = c
                    gm(I_E,l) = evm-c
                enddo
!
!               Reconstruction of the '+' and '-' fluxes
!
                wenorec_ord = max(weno_scheme+ep_ord_change_gpu(i,j,k,2),1)
                call wenorec_1d_rusanov(nv,gp,gm,fj,weno_scheme,wenorec_ord,weno_version,rho0,u0)
!
!               !Return to conservative fluxes
                do m=1,4+N_S
                   fhat_gpu(i,j,k,m) = fj(m)
                enddo

            endif
        enddo

    endsubroutine euler_y_hybrid_rusanov_kernel

    attributes(device) subroutine wenorec_1d_rusanov(nvar,vp,vm,vhat,iweno,wenorec_ord,weno_version,rho0,u0)
    
    !    Passed arguments
         integer :: nvar, iweno, wenorec_ord, weno_version
         real(rkind), dimension(nvar,8) :: vm,vp
         real(rkind), dimension(nvar) :: vhat
         real(rkind) :: rho0, u0
    
    !    Local variables
         real(rkind), dimension(-1:4) :: dwe           ! linear weights
         real(rkind), dimension(-1:4) :: betap,betam   ! beta_l
         real(rkind), dimension(N_S+4) :: betascale
         real(rkind) :: vminus, vplus
    !    
         integer :: i,l,m,lsp
         real(rkind) :: c0,c1,c2,c3,c4,d0,d1,d2,d3,summ,sump
         real(rkind) :: tau5p,tau5m,eps40
         real(rkind) :: u0_2, rho0_2u0_2, rho0_2u0_4, rho0_2u0_4i

         u0_2       = u0*u0
         rho0_2u0_2 = rho0*rho0*u0_2
         rho0_2u0_4 = rho0_2u0_2*u0_2
         rho0_2u0_4i = 1._rkind/rho0_2u0_4
         betascale(1) = 1._rkind/rho0_2u0_2
         do lsp=2,N_S
          betascale(lsp) = betascale(1)
         enddo
         betascale(I_U) = rho0_2u0_4i
         betascale(I_V) = rho0_2u0_4i
         betascale(I_W) = rho0_2u0_4i
         betascale(I_E) = rho0_2u0_4i/u0_2
    !
         if (wenorec_ord==1) then ! Godunov
    !    
             i = iweno ! index of intermediate node to perform reconstruction
    !    
             do m=1,nvar
              vminus  = vp(m,i)
              vplus   = vm(m,i+1)
              vhat(m) = vminus+vplus
             enddo
    !    
         elseif (wenorec_ord==2) then ! WENO-3
    !    
             i = iweno ! index of intermediate node to perform reconstruction
    !    
             dwe(1)   = 2._rkind/3._rkind
             dwe(0)   = 1._rkind/3._rkind
    !    
             do m=1,nvar
    !    
                 betap(0)  = (vp(m,i  )-vp(m,i-1))**2
                 betap(1)  = (vp(m,i+1)-vp(m,i  ))**2
                 betap(1) = betascale(m)*betap(1)
                 betap(0) = betascale(m)*betap(0)

                 betam(0)  = (vm(m,i+2)-vm(m,i+1))**2
                 betam(1)  = (vm(m,i+1)-vm(m,i  ))**2
                 betam(1) = betascale(m)*betam(1)
                 betam(0) = betascale(m)*betam(0)
    !    
                 sump = 0._rkind
                 summ = 0._rkind
                 do l=0,1
                     betap(l) = dwe(l)/(0.000001_rkind+betap(l))**2
                     betam(l) = dwe(l)/(0.000001_rkind+betam(l))**2
                     sump = sump + betap(l)
                     summ = summ + betam(l)
                 enddo
                 do l=0,1
                     betap(l) = betap(l)/sump
                     betam(l) = betam(l)/summ
                 enddo
    !    
                 vminus = betap(0) *(-vp(m,i-1)+3*vp(m,i  )) + betap(1) *( vp(m,i  )+ vp(m,i+1))
                 vplus  = betam(0) *(-vm(m,i+2)+3*vm(m,i+1)) + betam(1) *( vm(m,i  )+ vm(m,i+1))
                 vhat(m) = 0.5_rkind*(vminus+vplus)
    !    
             enddo ! end of m-loop
    !    
         elseif (wenorec_ord==3) then ! WENO-5
    !    
          i = iweno ! index of intermediate node to perform reconstruction
    !    
          dwe( 0) = 1._rkind/10._rkind
          dwe( 1) = 6._rkind/10._rkind
          dwe( 2) = 3._rkind/10._rkind
    !      
    !     JS
          d0 = 13._rkind/12._rkind
          d1 = 1._rkind/4._rkind
    !     Weights for polynomial reconstructions
          c0 = 1._rkind/3._rkind
          c1 = 5._rkind/6._rkind
          c2 =-1._rkind/6._rkind
          c3 =-7._rkind/6._rkind
          c4 =11._rkind/6._rkind
    !
          if (weno_version==0) then ! Standard JS WENO 5
    !
           do m=1,nvar
    !    
            betap(2) = d0*(vp(m,i)-2._rkind*vp(m,i+1)+vp(m,i+2))**2+&
                       d1*(3._rkind*vp(m,i)-4._rkind*vp(m,i+1)+vp(m,i+2))**2
            betap(1) = d0*(vp(m,i-1)-2._rkind*vp(m,i)+vp(m,i+1))**2+&
                       d1*(     vp(m,i-1)-vp(m,i+1) )**2
            betap(0) = d0*(vp(m,i)-2._rkind*vp(m,i-1)+vp(m,i-2))**2+&
                       d1*(3._rkind*vp(m,i)-4._rkind*vp(m,i-1)+vp(m,i-2))**2
            betap(2) = betascale(m)*betap(2)
            betap(1) = betascale(m)*betap(1)
            betap(0) = betascale(m)*betap(0)
    !    
            betam(2) = d0*(vm(m,i+1)-2._rkind*vm(m,i)+vm(m,i-1))**2+&
                       d1*(3._rkind*vm(m,i+1)-4._rkind*vm(m,i)+vm(m,i-1))**2
            betam(1) = d0*(vm(m,i+2)-2._rkind*vm(m,i+1)+vm(m,i))**2+&
                       d1*(     vm(m,i+2)-vm(m,i) )**2
            betam(0) = d0*(vm(m,i+1)-2._rkind*vm(m,i+2)+vm(m,i+3))**2+&
                       d1*(3._rkind*vm(m,i+1)-4._rkind*vm(m,i+2)+vm(m,i+3))**2
            betam(2) = betascale(m)*betam(2)
            betam(1) = betascale(m)*betam(1)
            betam(0) = betascale(m)*betam(0)
    !    
            sump = 0._rkind
            summ = 0._rkind
            do l=0,2
             betap(l) = dwe(  l)/(0.000001_rkind+betap(l))**2
             betam(l) = dwe(  l)/(0.000001_rkind+betam(l))**2
             sump = sump + betap(l)
             summ = summ + betam(l)
            enddo
            do l=0,2
             betap(l) = betap(l)/sump
             betam(l) = betam(l)/summ
            enddo
    !
            vminus = betap(2)*(c0*vp(m,i  )+c1*vp(m,i+1)+c2*vp(m,i+2)) + &
                     betap(1)*(c2*vp(m,i-1)+c1*vp(m,i  )+c0*vp(m,i+1)) + &
                     betap(0)*(c0*vp(m,i-2)+c3*vp(m,i-1)+c4*vp(m,i  ))
            vplus  = betam(2)*(c0*vm(m,i+1)+c1*vm(m,i  )+c2*vm(m,i-1)) + &
                     betam(1)*(c2*vm(m,i+2)+c1*vm(m,i+1)+c0*vm(m,i  )) + &
                     betam(0)*(c0*vm(m,i+3)+c3*vm(m,i+2)+c4*vm(m,i+1))
    !    
            vhat(m) = vminus+vplus
    !
           enddo ! end of m-loop 
    !
          else ! WENO 5Z 
    !              
           do m=1,nvar
    !
            betap(2) = d0*(vp(m,i)-2._rkind*vp(m,i+1)+vp(m,i+2))**2+&
                       d1*(3._rkind*vp(m,i)-4._rkind*vp(m,i+1)+vp(m,i+2))**2
            betap(1) = d0*(vp(m,i-1)-2._rkind*vp(m,i)+vp(m,i+1))**2+&
                       d1*(     vp(m,i-1)-vp(m,i+1) )**2
            betap(0) = d0*(vp(m,i)-2._rkind*vp(m,i-1)+vp(m,i-2))**2+&
                       d1*(3._rkind*vp(m,i)-4._rkind*vp(m,i-1)+vp(m,i-2))**2
            betap(2) = betascale(m)*betap(2)
            betap(1) = betascale(m)*betap(1)
            betap(0) = betascale(m)*betap(0)
    !
            betam(2) = d0*(vm(m,i+1)-2._rkind*vm(m,i)+vm(m,i-1))**2+&
                       d1*(3._rkind*vm(m,i+1)-4._rkind*vm(m,i)+vm(m,i-1))**2
            betam(1) = d0*(vm(m,i+2)-2._rkind*vm(m,i+1)+vm(m,i))**2+&
                       d1*(     vm(m,i+2)-vm(m,i) )**2
            betam(0) = d0*(vm(m,i+1)-2._rkind*vm(m,i+2)+vm(m,i+3))**2+&
                       d1*(3._rkind*vm(m,i+1)-4._rkind*vm(m,i+2)+vm(m,i+3))**2
            betam(2) = betascale(m)*betam(2)
            betam(1) = betascale(m)*betam(1)
            betam(0) = betascale(m)*betam(0)
    !
            eps40 = 1.D-40
            tau5p = abs(betap(0)-betap(2))+eps40
            tau5m = abs(betam(0)-betam(2))+eps40
    !
            do l=0,2
             betap(l) = (betap(l)+eps40)/(betap(l)+tau5p)
             betam(l) = (betam(l)+eps40)/(betam(l)+tau5m)
            enddo
    !
            sump = 0._rkind
            summ = 0._rkind
            do l=0,2
             betap(l) = dwe(l)/betap(l)
             betam(l) = dwe(l)/betam(l)
             sump = sump + betap(l)
             summ = summ + betam(l)
            enddo
            do l=0,2
             betap(l) = betap(l)/sump
             betam(l) = betam(l)/summ
            enddo
    !
            vminus = betap(2)*(c0*vp(m,i  )+c1*vp(m,i+1)+c2*vp(m,i+2)) + &
                     betap(1)*(c2*vp(m,i-1)+c1*vp(m,i  )+c0*vp(m,i+1)) + &
                     betap(0)*(c0*vp(m,i-2)+c3*vp(m,i-1)+c4*vp(m,i  ))
            vplus  = betam(2)*(c0*vm(m,i+1)+c1*vm(m,i  )+c2*vm(m,i-1)) + &
                     betam(1)*(c2*vm(m,i+2)+c1*vm(m,i+1)+c0*vm(m,i  )) + &
                     betam(0)*(c0*vm(m,i+3)+c3*vm(m,i+2)+c4*vm(m,i+1))
    !
            vhat(m) = vminus+vplus
    !
           enddo ! end of m-loop
    
          endif
    !    
         elseif (wenorec_ord==4) then ! WENO-7
    !    
          i = iweno ! index of intermediate node to perform reconstruction
    !    
          dwe( 0) = 1._rkind/35._rkind
          dwe( 1) = 12._rkind/35._rkind
          dwe( 2) = 18._rkind/35._rkind
          dwe( 3) = 4._rkind/35._rkind
    !    
    !     JS weights
          d1 = 1._rkind/36._rkind
          d2 = 13._rkind/12._rkind
          d3 = 781._rkind/720._rkind
    !    
          do m=1,nvar
    !    
           betap(3)= d1*(-11*vp(m,  i)+18*vp(m,i+1)- 9*vp(m,i+2)+ 2*vp(m,i+3))**2+&
              d2*(  2*vp(m,  i)- 5*vp(m,i+1)+ 4*vp(m,i+2)-   vp(m,i+3))**2+ &
             d3*(   -vp(m,  i)+ 3*vp(m,i+1)- 3*vp(m,i+2)+   vp(m,i+3))**2
           betap(2)= d1*(- 2*vp(m,i-1)- 3*vp(m,i  )+ 6*vp(m,i+1)-   vp(m,i+2))**2+&
              d2*(    vp(m,i-1)- 2*vp(m,i  )+   vp(m,i+1)             )**2+&
              d3*(   -vp(m,i-1)+ 3*vp(m,i  )- 3*vp(m,i+1)+   vp(m,i+2))**2
           betap(1)= d1*(    vp(m,i-2)- 6*vp(m,i-1)+ 3*vp(m,i  )+ 2*vp(m,i+1))**2+&
              d2*( vp(m,i-1)- 2*vp(m,i  )+   vp(m,i+1))**2+ &
              d3*(   -vp(m,i-2)+ 3*vp(m,i-1)- 3*vp(m,i  )+   vp(m,i+1))**2
           betap(0)= d1*(- 2*vp(m,i-3)+ 9*vp(m,i-2)-18*vp(m,i-1)+11*vp(m,i  ))**2+&
              d2*(-   vp(m,i-3)+ 4*vp(m,i-2)- 5*vp(m,i-1)+ 2*vp(m,i  ))**2+&
              d3*(   -vp(m,i-3)+ 3*vp(m,i-2)- 3*vp(m,i-1)+   vp(m,i  ))**2
           betap(3) = betascale(m)*betap(3)
           betap(2) = betascale(m)*betap(2)
           betap(1) = betascale(m)*betap(1)
           betap(0) = betascale(m)*betap(0)
    !    
           betam(3)= d1*(-11*vm(m,i+1)+18*vm(m,i  )- 9*vm(m,i-1)+ 2*vm(m,i-2))**2+&
              d2*(  2*vm(m,i+1)- 5*vm(m,i  )+ 4*vm(m,i-1)-   vm(m,i-2))**2+&
              d3*(   -vm(m,i+1)+ 3*vm(m,i  )- 3*vm(m,i-1)+   vm(m,i-2))**2
           betam(2)= d1*(- 2*vm(m,i+2)- 3*vm(m,i+1)+ 6*vm(m,i  )-   vm(m,i-1))**2+&
              d2*(    vm(m,i+2)- 2*vm(m,i+1)+   vm(m,i  )             )**2+&
              d3*(   -vm(m,i+2)+ 3*vm(m,i+1)- 3*vm(m,i  )+   vm(m,i-1))**2
           betam(1)= d1*(    vm(m,i+3)- 6*vm(m,i+2)+ 3*vm(m,i+1)+ 2*vm(m,i  ))**2+&
              d2*(                 vm(m,i+2)- 2*vm(m,i+1)+   vm(m,i  ))**2+&
              d3*(   -vm(m,i+3)+ 3*vm(m,i+2)- 3*vm(m,i+1)+   vm(m,i  ))**2
           betam(0)= d1*(- 2*vm(m,i+4)+ 9*vm(m,i+3)-18*vm(m,i+2)+11*vm(m,i+1))**2+&
              d2*(-   vm(m,i+4)+ 4*vm(m,i+3)- 5*vm(m,i+2)+ 2*vm(m,i+1))**2+&
              d3*(   -vm(m,i+4)+ 3*vm(m,i+3)- 3*vm(m,i+2)+   vm(m,i+1))**2 
           betam(3) = betascale(m)*betam(3)
           betam(2) = betascale(m)*betam(2)
           betam(1) = betascale(m)*betam(1)
           betam(0) = betascale(m)*betam(0)
    !    
           sump = 0._rkind
           summ = 0._rkind
           do l=0,3
            betap(l) = dwe(  l)/(0.000001_rkind+betap(l))**2
            betam(l) = dwe(  l)/(0.000001_rkind+betam(l))**2
            sump = sump + betap(l)
            summ = summ + betam(l)
           enddo
           do l=0,3
            betap(l) = betap(l)/sump
            betam(l) = betam(l)/summ
           enddo
    !    
           vminus = betap(3)*( 6*vp(m,i  )+26*vp(m,i+1)-10*vp(m,i+2)+ 2*vp(m,i+3))+&
            betap(2)*(-2*vp(m,i-1)+14*vp(m,i  )+14*vp(m,i+1)- 2*vp(m,i+2))+&
            betap(1)*( 2*vp(m,i-2)-10*vp(m,i-1)+26*vp(m,i  )+ 6*vp(m,i+1))+&
            betap(0)*(-6*vp(m,i-3)+26*vp(m,i-2)-46*vp(m,i-1)+50*vp(m,i  ))
           vplus  =  betam(3)*( 6*vm(m,i+1)+26*vm(m,i  )-10*vm(m,i-1)+ 2*vm(m,i-2))+&
            betam(2)*(-2*vm(m,i+2)+14*vm(m,i+1)+14*vm(m,i  )- 2*vm(m,i-1))+&
            betam(1)*( 2*vm(m,i+3)-10*vm(m,i+2)+26*vm(m,i+1)+ 6*vm(m,i  ))+&
            betam(0)*(-6*vm(m,i+4)+26*vm(m,i+3)-46*vm(m,i+2)+50*vm(m,i+1))
    !    
            vhat(m) = (vminus+vplus)/24._rkind
    !
          enddo ! end of m-loop 
    !    
         !old else
         !old write(*,*) 'Error! WENO scheme not implemented'
         !old stop
         endif
    
    endsubroutine wenorec_1d_rusanov

    attributes(device) subroutine wenorec_1d(nvar,vp,vm,vhat,iweno,wenorec_ord,weno_version,rho0,u0)
    
    !    Passed arguments
         integer :: nvar, iweno, wenorec_ord, weno_version
         real(rkind), dimension(nvar,8) :: vm,vp
         real(rkind), dimension(nvar) :: vhat
         real(rkind) :: rho0,u0
    
    !    Local variables
         real(rkind), dimension(-1:4) :: dwe           ! linear weights
         real(rkind), dimension(-1:4) :: betap,betam   ! beta_l
         real(rkind), dimension(N_S+4) :: betascale
         real(rkind) :: vminus, vplus
    !    
         integer :: i,l,m,lsp
         real(rkind) :: c0,c1,c2,c3,c4,d0,d1,d2,d3,summ,sump
         real(rkind) :: tau5p,tau5m,eps40
         real(rkind) :: u0_2, rho0_2u0_2, rho0_2u0_4
    !
         u0_2       = u0*u0
         rho0_2u0_2 = rho0*rho0*u0_2
         rho0_2u0_4 = rho0_2u0_2*u0_2
         betascale(1) = 1._rkind/rho0_2u0_2
         do lsp=1,N_S+1
          betascale(lsp) = betascale(1)
         enddo
         betascale(N_S+2) = 1._rkind/rho0_2u0_4
         betascale(N_S+3) = betascale(N_S+2)
         betascale(N_S+4) = betascale(1)

         if (wenorec_ord==1) then ! Godunov
    !    
             i = iweno ! index of intermediate node to perform reconstruction
    !    
             do m=1,nvar
              vminus  = vp(m,i)
              vplus   = vm(m,i+1)
              vhat(m) = vminus+vplus
             enddo
    !    
         elseif (wenorec_ord==2) then ! WENO-3
    !    
             i = iweno ! index of intermediate node to perform reconstruction
    !    
             dwe(1)   = 2._rkind/3._rkind
             dwe(0)   = 1._rkind/3._rkind
    !    
             do m=1,nvar
    !    
                 betap(0)  = (vp(m,i  )-vp(m,i-1))**2
                 betap(1)  = (vp(m,i+1)-vp(m,i  ))**2
                 betap(1) = betascale(m)*betap(1)
                 betap(0) = betascale(m)*betap(0)

                 betam(0)  = (vm(m,i+2)-vm(m,i+1))**2
                 betam(1)  = (vm(m,i+1)-vm(m,i  ))**2
                 betam(1) = betascale(m)*betam(1)
                 betam(0) = betascale(m)*betam(0)
    !    
                 sump = 0._rkind
                 summ = 0._rkind
                 do l=0,1
                     betap(l) = dwe(l)/(0.000001_rkind+betap(l))**2
                     betam(l) = dwe(l)/(0.000001_rkind+betam(l))**2
                     sump = sump + betap(l)
                     summ = summ + betam(l)
                 enddo
                 do l=0,1
                     betap(l) = betap(l)/sump
                     betam(l) = betam(l)/summ
                 enddo
    !    
                 vminus = betap(0) *(-vp(m,i-1)+3*vp(m,i  )) + betap(1) *( vp(m,i  )+ vp(m,i+1))
                 vplus  = betam(0) *(-vm(m,i+2)+3*vm(m,i+1)) + betam(1) *( vm(m,i  )+ vm(m,i+1))
                 vhat(m) = 0.5_rkind*(vminus+vplus)
    !    
             enddo ! end of m-loop
    !    
         elseif (wenorec_ord==3) then ! WENO-5
    !    
          i = iweno ! index of intermediate node to perform reconstruction
    !    
          dwe( 0) = 1._rkind/10._rkind
          dwe( 1) = 6._rkind/10._rkind
          dwe( 2) = 3._rkind/10._rkind
    !      
    !     JS
          d0 = 13._rkind/12._rkind
          d1 = 1._rkind/4._rkind
    !     Weights for polynomial reconstructions
          c0 = 1._rkind/3._rkind
          c1 = 5._rkind/6._rkind
          c2 =-1._rkind/6._rkind
          c3 =-7._rkind/6._rkind
          c4 =11._rkind/6._rkind
    !
          if (weno_version==0) then ! Standard JS WENO 5
    !
           do m=1,nvar
    !    
            betap(2) = d0*(vp(m,i)-2._rkind*vp(m,i+1)+vp(m,i+2))**2+&
                       d1*(3._rkind*vp(m,i)-4._rkind*vp(m,i+1)+vp(m,i+2))**2
            betap(1) = d0*(vp(m,i-1)-2._rkind*vp(m,i)+vp(m,i+1))**2+&
                       d1*(     vp(m,i-1)-vp(m,i+1) )**2
            betap(0) = d0*(vp(m,i)-2._rkind*vp(m,i-1)+vp(m,i-2))**2+&
                       d1*(3._rkind*vp(m,i)-4._rkind*vp(m,i-1)+vp(m,i-2))**2
            betap(2) = betascale(m)*betap(2)
            betap(1) = betascale(m)*betap(1)
            betap(0) = betascale(m)*betap(0)
    !    
            betam(2) = d0*(vm(m,i+1)-2._rkind*vm(m,i)+vm(m,i-1))**2+&
                       d1*(3._rkind*vm(m,i+1)-4._rkind*vm(m,i)+vm(m,i-1))**2
            betam(1) = d0*(vm(m,i+2)-2._rkind*vm(m,i+1)+vm(m,i))**2+&
                       d1*(     vm(m,i+2)-vm(m,i) )**2
            betam(0) = d0*(vm(m,i+1)-2._rkind*vm(m,i+2)+vm(m,i+3))**2+&
                       d1*(3._rkind*vm(m,i+1)-4._rkind*vm(m,i+2)+vm(m,i+3))**2
            betam(2) = betascale(m)*betam(2)
            betam(1) = betascale(m)*betam(1)
            betam(0) = betascale(m)*betam(0)
    !    
            sump = 0._rkind
            summ = 0._rkind
            do l=0,2
             betap(l) = dwe(  l)/(0.000001_rkind+betap(l))**2
             betam(l) = dwe(  l)/(0.000001_rkind+betam(l))**2
             sump = sump + betap(l)
             summ = summ + betam(l)
            enddo
            do l=0,2
             betap(l) = betap(l)/sump
             betam(l) = betam(l)/summ
            enddo
    !
            vminus = betap(2)*(c0*vp(m,i  )+c1*vp(m,i+1)+c2*vp(m,i+2)) + &
                     betap(1)*(c2*vp(m,i-1)+c1*vp(m,i  )+c0*vp(m,i+1)) + &
                     betap(0)*(c0*vp(m,i-2)+c3*vp(m,i-1)+c4*vp(m,i  ))
            vplus  = betam(2)*(c0*vm(m,i+1)+c1*vm(m,i  )+c2*vm(m,i-1)) + &
                     betam(1)*(c2*vm(m,i+2)+c1*vm(m,i+1)+c0*vm(m,i  )) + &
                     betam(0)*(c0*vm(m,i+3)+c3*vm(m,i+2)+c4*vm(m,i+1))
    !    
            vhat(m) = vminus+vplus
    !
           enddo ! end of m-loop 
    !
          else ! WENO 5Z 
    !              
           do m=1,nvar
    !
            betap(2) = d0*(vp(m,i)-2._rkind*vp(m,i+1)+vp(m,i+2))**2+&
                       d1*(3._rkind*vp(m,i)-4._rkind*vp(m,i+1)+vp(m,i+2))**2
            betap(1) = d0*(vp(m,i-1)-2._rkind*vp(m,i)+vp(m,i+1))**2+&
                       d1*(     vp(m,i-1)-vp(m,i+1) )**2
            betap(0) = d0*(vp(m,i)-2._rkind*vp(m,i-1)+vp(m,i-2))**2+&
                       d1*(3._rkind*vp(m,i)-4._rkind*vp(m,i-1)+vp(m,i-2))**2
            betap(2) = betascale(m)*betap(2)
            betap(1) = betascale(m)*betap(1)
            betap(0) = betascale(m)*betap(0)
    !
            betam(2) = d0*(vm(m,i+1)-2._rkind*vm(m,i)+vm(m,i-1))**2+&
                       d1*(3._rkind*vm(m,i+1)-4._rkind*vm(m,i)+vm(m,i-1))**2
            betam(1) = d0*(vm(m,i+2)-2._rkind*vm(m,i+1)+vm(m,i))**2+&
                       d1*(     vm(m,i+2)-vm(m,i) )**2
            betam(0) = d0*(vm(m,i+1)-2._rkind*vm(m,i+2)+vm(m,i+3))**2+&
                       d1*(3._rkind*vm(m,i+1)-4._rkind*vm(m,i+2)+vm(m,i+3))**2
            betam(2) = betascale(m)*betam(2)
            betam(1) = betascale(m)*betam(1)
            betam(0) = betascale(m)*betam(0)
    !
            eps40 = 1.D-40
            tau5p = abs(betap(0)-betap(2))+eps40
            tau5m = abs(betam(0)-betam(2))+eps40
    !
            do l=0,2
             betap(l) = (betap(l)+eps40)/(betap(l)+tau5p)
             betam(l) = (betam(l)+eps40)/(betam(l)+tau5m)
            enddo
    !
            sump = 0._rkind
            summ = 0._rkind
            do l=0,2
             betap(l) = dwe(l)/betap(l)
             betam(l) = dwe(l)/betam(l)
             sump = sump + betap(l)
             summ = summ + betam(l)
            enddo
            do l=0,2
             betap(l) = betap(l)/sump
             betam(l) = betam(l)/summ
            enddo
    !
            vminus = betap(2)*(c0*vp(m,i  )+c1*vp(m,i+1)+c2*vp(m,i+2)) + &
                     betap(1)*(c2*vp(m,i-1)+c1*vp(m,i  )+c0*vp(m,i+1)) + &
                     betap(0)*(c0*vp(m,i-2)+c3*vp(m,i-1)+c4*vp(m,i  ))
            vplus  = betam(2)*(c0*vm(m,i+1)+c1*vm(m,i  )+c2*vm(m,i-1)) + &
                     betam(1)*(c2*vm(m,i+2)+c1*vm(m,i+1)+c0*vm(m,i  )) + &
                     betam(0)*(c0*vm(m,i+3)+c3*vm(m,i+2)+c4*vm(m,i+1))
    !
            vhat(m) = vminus+vplus
    !
           enddo ! end of m-loop
    
          endif
    !    
         elseif (wenorec_ord==4) then ! WENO-7
    !    
          i = iweno ! index of intermediate node to perform reconstruction
    !    
          dwe( 0) = 1._rkind/35._rkind
          dwe( 1) = 12._rkind/35._rkind
          dwe( 2) = 18._rkind/35._rkind
          dwe( 3) = 4._rkind/35._rkind
    !    
    !     JS weights
          d1 = 1._rkind/36._rkind
          d2 = 13._rkind/12._rkind
          d3 = 781._rkind/720._rkind
    !    
          do m=1,nvar
    !    
           betap(3)= d1*(-11*vp(m,  i)+18*vp(m,i+1)- 9*vp(m,i+2)+ 2*vp(m,i+3))**2+&
              d2*(  2*vp(m,  i)- 5*vp(m,i+1)+ 4*vp(m,i+2)-   vp(m,i+3))**2+ &
             d3*(   -vp(m,  i)+ 3*vp(m,i+1)- 3*vp(m,i+2)+   vp(m,i+3))**2
           betap(2)= d1*(- 2*vp(m,i-1)- 3*vp(m,i  )+ 6*vp(m,i+1)-   vp(m,i+2))**2+&
              d2*(    vp(m,i-1)- 2*vp(m,i  )+   vp(m,i+1)             )**2+&
              d3*(   -vp(m,i-1)+ 3*vp(m,i  )- 3*vp(m,i+1)+   vp(m,i+2))**2
           betap(1)= d1*(    vp(m,i-2)- 6*vp(m,i-1)+ 3*vp(m,i  )+ 2*vp(m,i+1))**2+&
              d2*( vp(m,i-1)- 2*vp(m,i  )+   vp(m,i+1))**2+ &
              d3*(   -vp(m,i-2)+ 3*vp(m,i-1)- 3*vp(m,i  )+   vp(m,i+1))**2
           betap(0)= d1*(- 2*vp(m,i-3)+ 9*vp(m,i-2)-18*vp(m,i-1)+11*vp(m,i  ))**2+&
              d2*(-   vp(m,i-3)+ 4*vp(m,i-2)- 5*vp(m,i-1)+ 2*vp(m,i  ))**2+&
              d3*(   -vp(m,i-3)+ 3*vp(m,i-2)- 3*vp(m,i-1)+   vp(m,i  ))**2
           betap(3) = betascale(m)*betap(3)
           betap(2) = betascale(m)*betap(2)
           betap(1) = betascale(m)*betap(1)
           betap(0) = betascale(m)*betap(0)
    !    
           betam(3)= d1*(-11*vm(m,i+1)+18*vm(m,i  )- 9*vm(m,i-1)+ 2*vm(m,i-2))**2+&
              d2*(  2*vm(m,i+1)- 5*vm(m,i  )+ 4*vm(m,i-1)-   vm(m,i-2))**2+&
              d3*(   -vm(m,i+1)+ 3*vm(m,i  )- 3*vm(m,i-1)+   vm(m,i-2))**2
           betam(2)= d1*(- 2*vm(m,i+2)- 3*vm(m,i+1)+ 6*vm(m,i  )-   vm(m,i-1))**2+&
              d2*(    vm(m,i+2)- 2*vm(m,i+1)+   vm(m,i  )             )**2+&
              d3*(   -vm(m,i+2)+ 3*vm(m,i+1)- 3*vm(m,i  )+   vm(m,i-1))**2
           betam(1)= d1*(    vm(m,i+3)- 6*vm(m,i+2)+ 3*vm(m,i+1)+ 2*vm(m,i  ))**2+&
              d2*(                 vm(m,i+2)- 2*vm(m,i+1)+   vm(m,i  ))**2+&
              d3*(   -vm(m,i+3)+ 3*vm(m,i+2)- 3*vm(m,i+1)+   vm(m,i  ))**2
           betam(0)= d1*(- 2*vm(m,i+4)+ 9*vm(m,i+3)-18*vm(m,i+2)+11*vm(m,i+1))**2+&
              d2*(-   vm(m,i+4)+ 4*vm(m,i+3)- 5*vm(m,i+2)+ 2*vm(m,i+1))**2+&
              d3*(   -vm(m,i+4)+ 3*vm(m,i+3)- 3*vm(m,i+2)+   vm(m,i+1))**2 
           betam(3) = betascale(m)*betam(3)
           betam(2) = betascale(m)*betam(2)
           betam(1) = betascale(m)*betam(1)
           betam(0) = betascale(m)*betam(0)
    !    
           sump = 0._rkind
           summ = 0._rkind
           do l=0,3
            betap(l) = dwe(  l)/(0.000001_rkind+betap(l))**2
            betam(l) = dwe(  l)/(0.000001_rkind+betam(l))**2
            sump = sump + betap(l)
            summ = summ + betam(l)
           enddo
           do l=0,3
            betap(l) = betap(l)/sump
            betam(l) = betam(l)/summ
           enddo
    !    
           vminus = betap(3)*( 6*vp(m,i  )+26*vp(m,i+1)-10*vp(m,i+2)+ 2*vp(m,i+3))+&
            betap(2)*(-2*vp(m,i-1)+14*vp(m,i  )+14*vp(m,i+1)- 2*vp(m,i+2))+&
            betap(1)*( 2*vp(m,i-2)-10*vp(m,i-1)+26*vp(m,i  )+ 6*vp(m,i+1))+&
            betap(0)*(-6*vp(m,i-3)+26*vp(m,i-2)-46*vp(m,i-1)+50*vp(m,i  ))
           vplus  =  betam(3)*( 6*vm(m,i+1)+26*vm(m,i  )-10*vm(m,i-1)+ 2*vm(m,i-2))+&
            betam(2)*(-2*vm(m,i+2)+14*vm(m,i+1)+14*vm(m,i  )- 2*vm(m,i-1))+&
            betam(1)*( 2*vm(m,i+3)-10*vm(m,i+2)+26*vm(m,i+1)+ 6*vm(m,i  ))+&
            betam(0)*(-6*vm(m,i+4)+26*vm(m,i+3)-46*vm(m,i+2)+50*vm(m,i+1))
    !    
            vhat(m) = (vminus+vplus)/24._rkind
    !
          enddo ! end of m-loop 
    !    
         !old else
         !old write(*,*) 'Error! WENO scheme not implemented'
         !old stop
         endif
    
    endsubroutine wenorec_1d

!    attributes(device) subroutine eigenvectors_x(b1, b2, b3, uu, vv, ww, c, ci, h, el, er)
!        real(rkind), intent(in) :: b1, b2, b3, uu, vv, ww, c, ci, h
!        real(rkind), dimension(:,:), intent(out) ::  el, er
!
!!       left eigenvectors matrix (at Roe state)
!!       matrix L-1 AIAA2001-2609 - Rohde after transposition and change in the sign of the fourth row (here is the fourth column)
!!       matrix reported in Notebook_Summary.nb pag 5 after transposition and order change of the rows (5=>3, 3=>4, 4=>5)
!        el(1,1)   =   0.5_rkind * (b1     + uu * ci)
!        el(2,1)   =  -0.5_rkind * (b2 * uu +     ci)
!        el(3,1)   =  -0.5_rkind * (b2 * vv         )
!        el(4,1)   =  -0.5_rkind * (b2 * ww         )
!        el(5,1)   =   0.5_rkind * b2
!        el(1,2)   =   1._rkind - b1
!        el(2,2)   =   b2*uu
!        el(3,2)   =   b2*vv
!        el(4,2)   =   b2*ww
!        el(5,2)   =  -b2
!        el(1,3)   =   0.5_rkind * (b1     - uu * ci)
!        el(2,3)   =  -0.5_rkind * (b2 * uu -     ci)
!        el(3,3)   =  -0.5_rkind * (b2 * vv         )
!        el(4,3)   =  -0.5_rkind * (b2 * ww         )
!        el(5,3)   =   0.5_rkind * b2
!        el(1,4)   =   -vv ! vv
!        el(2,4)   =   0._rkind
!        el(3,4)   =   1._rkind ! -1._rkind
!        el(4,4)   =   0._rkind
!        el(5,4)   =   0._rkind
!        el(1,5)   =  -ww
!        el(2,5)   =   0._rkind
!        el(3,5)   =   0._rkind
!        el(4,5)   =   1._rkind
!        el(5,5)   =   0._rkind
!
!!       right eigenvectors matrix (at Roe state)
!!       matrix R-1 AIAA2001-2609 - Rohde after transposition and change in the sign of the fourth column (here is the fourth row)
!!       matrix reported in Notebook_Summary.nb pag 5 after transposition and order change of the columns (5=>3, 3=>4, 4=>5)
!        er(1,1)   =  1._rkind
!        er(2,1)   =  1._rkind
!        er(3,1)   =  1._rkind
!        er(4,1)   =  0._rkind
!        er(5,1)   =  0._rkind
!        er(1,2)   =  uu -  c
!        er(2,2)   =  uu
!        er(3,2)   =  uu +  c
!        er(4,2)   =  0._rkind
!        er(5,2)   =  0._rkind
!        er(1,3)   =  vv
!        er(2,3)   =  vv
!        er(3,3)   =  vv
!        er(4,3)   =  1._rkind ! -1._rkind
!        er(5,3)   =   0._rkind
!        er(1,4)   =  ww
!        er(2,4)   =  ww
!        er(3,4)   =  ww
!        er(4,4)   =  0._rkind
!        er(5,4)   =  1._rkind
!        er(1,5)   =  h  - uu * c
!        er(2,5)   =  b3 ! etot - rho * p_rho/p_e
!        er(3,5)   =  h  + uu * c
!        er(4,5)   =  vv ! -vv
!        er(5,5)   =  ww
!
!    endsubroutine eigenvectors_x

    attributes(device) subroutine eigenvectors_x(nv, b1, b2, b3, rho, uu, vv, ww, c, ci, h, el, er, yyroe, prhoi)
        integer, intent(in) :: nv
        real(rkind), intent(in) :: b1, b2, b3, uu, vv, ww, c, ci, h, rho
        real(rkind), dimension(N_S), intent(in) :: yyroe, prhoi
        real(rkind), dimension(I_E,I_E), intent(out) :: el, er
        real(rkind) :: cci,b1i,b3i
        integer :: lsp,msp

!       left eigenvectors matrix (at Roe state)
!       matrix reported in Notebook_Summary.nb pag 5 after transposition
!
!       b2  = p_e/(rho*cc)
!       b3  = etot
!       b1  = - b2*(etot - 2._rkind*qq)
!       b3i = etot - rho * p_rho/p_e = b3-prhoi/b2/cc
!       b1i = p_rho/cc - b2*(etot - 2._rkind*qq) = b1+p_rho/cc
!
        cci = ci*ci
!
        do lsp=1,4+N_S
         do msp=1,4+N_S
          el(lsp,msp) = 0._rkind
         enddo
        enddo
        do lsp=1,N_S
         b1i = b1+prhoi(lsp)*cci
         el(lsp,1) = 0.5_rkind*(b1i+uu*ci)
         el(lsp,N_S+2) = -vv
         el(lsp,N_S+3) = -ww
         el(lsp,N_S+4) = 0.5_rkind * (b1i-uu*ci)
         el(lsp,lsp+1) = 1._rkind
         el(I_U,1+lsp) =  b2*uu*yyroe(lsp)
         el(I_V,1+lsp) =  b2*vv*yyroe(lsp)
         el(I_W,1+lsp) =  b2*ww*yyroe(lsp)
         el(I_E,1+lsp) = -b2   *yyroe(lsp)
         do msp=1,N_S
          b1i = b1+prhoi(msp)*cci
          el(msp,1+lsp) = el(msp,1+lsp)-b1i*yyroe(lsp)
         enddo
        enddo
! 
        el(I_U,1)     =  -0.5_rkind * (b2 * uu +     ci)
        el(I_V,1)     =  -0.5_rkind * (b2 * vv         )
        el(I_W,1)     =  -0.5_rkind * (b2 * ww         )
        el(I_E,1)     =   0.5_rkind *  b2
        el(I_U,N_S+2) =   0._rkind
        el(I_V,N_S+2) =   1._rkind ! -1._rkind
        el(I_W,N_S+2) =   0._rkind
        el(I_E,N_S+2) =   0._rkind
        el(I_U,N_S+3) =   0._rkind
        el(I_V,N_S+3) =   0._rkind
        el(I_W,N_S+3) =   1._rkind
        el(I_E,N_S+3) =   0._rkind
        el(I_U,N_S+4) =  -0.5_rkind * (b2 * uu -     ci)
        el(I_V,N_S+4) =  -0.5_rkind * (b2 * vv         )
        el(I_W,N_S+4) =  -0.5_rkind * (b2 * ww         )
        el(I_E,N_S+4) =   0.5_rkind *  b2
!
!       right eigenvectors matrix (at Roe state)
!       matrix reported in Notebook_Summary.nb pag 5 after transposition
!
        do lsp=1,4+N_S
         do msp=1,4+N_S
          er(lsp,msp) = 0._rkind
         enddo
        enddo
        do lsp=1,N_S
         b3i = b3-prhoi(lsp)/b2*cci
         er(1    ,lsp) = yyroe(lsp)
         er(1+lsp,lsp) = 1._rkind
         er(4+N_S,lsp) = yyroe(lsp)
         er(1+lsp,I_U) =  uu
         er(1+lsp,I_V) =  vv
         er(1+lsp,I_W) =  ww
         er(1+lsp,I_E) = b3i
        enddo
        er(1,I_U)   =  uu -  c
        er(1,I_V)   =  vv
        er(1,I_W)   =  ww
        er(1,I_E)   =  h  - uu * c
        er(N_S+2,I_U) =  0._rkind
        er(N_S+3,I_U) =  0._rkind
        er(N_S+4,I_U) =  uu +  c
        er(N_S+2,I_V) =  1._rkind ! -1._rkind
        er(N_S+3,I_V) =  0._rkind
        er(N_S+4,I_V) =  vv
        er(N_S+2,I_W) =  0._rkind
        er(N_S+3,I_W) =  1._rkind
        er(N_S+4,I_W) =  ww
        er(N_S+2,I_E) =  vv ! -vv
        er(N_S+3,I_E) =  ww
        er(N_S+4,I_E) =  h  + uu * c
!
    endsubroutine eigenvectors_x

    attributes(device) subroutine eigenvectors_y(nv, b1, b2, b3, rho, uu, vv, ww, c, ci, h, el, er, yyroe, prhoi)
        integer, intent(in) :: nv
        real(rkind), intent(in) :: b1, b2, b3, rho, uu, vv, ww, c, ci, h
        real(rkind), dimension(N_S), intent(in) :: yyroe, prhoi
        real(rkind), dimension(I_E,I_E), intent(out) :: el, er
        real(rkind) :: cci,b1i,b3i
        integer :: lsp,msp
!
!       b2  = p_e/(rho*cc)
!       b3  = etot
!       b1  = - b2*(etot - 2._rkind*qq)
!       b3i = etot - rho * p_rho/p_e = b3-prhoi/b2/cc
!       b1i = p_rho/cc - b2*(etot - 2._rkind*qq) = b1+p_rho/cc
!
!       left eigenvectors matrix (at Roe state)
!       matrix reported in Notebook_Summary.nb pag 7 after transposition
!
        cci = ci*ci
! 
        do lsp=1,4+N_S
         do msp=1,4+N_S
          el(lsp,msp) = 0._rkind
         enddo
        enddo
        do lsp=1,N_S
         b1i = b1+prhoi(lsp)*cci
         el(lsp,1)     = 0.5_rkind*(b1i+vv*ci)
         el(lsp,N_S+2) = -uu
         el(lsp,N_S+3) = -ww
         el(lsp,N_S+4) = 0.5_rkind*(b1i-vv*ci)
         el(lsp,lsp+1) = 1._rkind
         el(I_U,1+lsp) =  b2*uu*yyroe(lsp)
         el(I_V,1+lsp) =  b2*vv*yyroe(lsp)
         el(I_W,1+lsp) =  b2*ww*yyroe(lsp)
         el(I_E,1+lsp) = -b2   *yyroe(lsp)
         do msp=1,N_S
          b1i = b1+prhoi(msp)*cci
          el(msp,1+lsp) = el(msp,1+lsp)-b1i*yyroe(lsp)
         enddo
        enddo
        el(I_U,1)     =  -0.5_rkind * (b2 * uu         )
        el(I_V,1)     =  -0.5_rkind * (b2 * vv +     ci)
        el(I_W,1)     =  -0.5_rkind * (b2 * ww         )
        el(I_E,1)     =   0.5_rkind * b2
        el(I_U,N_S+2) = 1._rkind
        el(I_V,N_S+2) = 0._rkind
        el(I_W,N_S+2) = 0._rkind
        el(I_E,N_S+2) = 0._rkind
        el(I_U,N_S+3) =   0._rkind
        el(I_V,N_S+3) =   0._rkind
        el(I_W,N_S+3) =   1._rkind
        el(I_E,N_S+3) =   0._rkind
        el(I_U,N_S+4) =  -0.5_rkind * (b2 * uu         )
        el(I_V,N_S+4) =  -0.5_rkind * (b2 * vv -     ci)
        el(I_W,N_S+4) =  -0.5_rkind * (b2 * ww         )
        el(I_E,N_S+4) =   0.5_rkind * b2
!
!       right eigenvectors matrix (at Roe state)
!       matrix reported in Notebook_Summary.nb pag 7 after transposition
!
        do lsp=1,4+N_S
         do msp=1,4+N_S
          er(lsp,msp) = 0._rkind
         enddo
        enddo
        do lsp=1,N_S
         b3i = b3-prhoi(lsp)/b2*cci
         er(1    ,lsp) = yyroe(lsp)
         er(1+lsp,lsp) = 1._rkind
         er(4+N_S,lsp) = yyroe(lsp)
         er(1+lsp,I_U) =  uu
         er(1+lsp,I_V) =  vv
         er(1+lsp,I_W) =  ww
         er(1+lsp,I_E) = b3i
        enddo
        er(1,I_U)     =  uu
        er(1,I_V)     =  vv - c
        er(1,I_W)     =  ww
        er(1,I_E)     =  h  - vv * c
        er(N_S+2,I_U) =  1._rkind
        er(N_S+3,I_U) =  0._rkind
        er(N_S+4,I_U) =  uu
        er(N_S+2,I_V) =  0._rkind
        er(N_S+3,I_V) =  0._rkind
        er(N_S+4,I_V) =  vv + c
        er(N_S+2,I_W) =  0._rkind
        er(N_S+3,I_W) =  1._rkind
        er(N_S+4,I_W) =  ww
        er(N_S+2,I_E) =  uu
        er(N_S+3,I_E) =  ww
        er(N_S+4,I_E) =  h  + vv * c
!
    endsubroutine eigenvectors_y

    attributes(device) subroutine eigenvectors_z(nv,b1, b2, b3, rho, uu, vv, ww, c, ci, h, el, er, yyroe, prhoi)
        integer, intent(in) :: nv
        real(rkind), intent(in) :: b1, b2, b3, rho, uu, vv, ww, c, ci, h
        real(rkind), dimension(N_S), intent(in) :: yyroe, prhoi
        real(rkind), dimension(I_E,I_E), intent(out) :: el, er
        real(rkind) :: cci,b1i,b3i
        integer :: lsp,msp
!
!       b2  = p_e/(rho*cc)
!       b3  = etot
!       b1  = - b2*(etot - 2._rkind*qq)
!       b3i = etot - rho * p_rho/p_e = b3-prhoi/b2/cc
!       b1i = p_rho/cc - b2*(etot - 2._rkind*qq) = b1+p_rho/cc
!
!       left eigenvectors matrix (at Roe state)
!       matrix reported in Notebook_Summary.nb pag 7 after transposition
!
        cci = ci*ci
! 
        do lsp=1,4+N_S
         do msp=1,4+N_S
          el(lsp,msp) = 0._rkind
         enddo
        enddo
        do lsp=1,N_S
         b1i = b1+prhoi(lsp)*cci
         el(lsp,1)     = 0.5_rkind*(b1i+ww*ci)
         el(lsp,N_S+2) = -uu
         el(lsp,N_S+3) = -vv
         el(lsp,N_S+4) = 0.5_rkind*(b1i-ww*ci)
         el(lsp,lsp+1) = 1._rkind
         el(I_U,1+lsp) =  b2*uu*yyroe(lsp)
         el(I_V,1+lsp) =  b2*vv*yyroe(lsp)
         el(I_W,1+lsp) =  b2*ww*yyroe(lsp)
         el(I_E,1+lsp) = -b2   *yyroe(lsp)
         do msp=1,N_S
          b1i = b1+prhoi(msp)*cci
          el(msp,1+lsp) = el(msp,1+lsp)-b1i*yyroe(lsp)
         enddo
        enddo
        el(I_U,1)     =  -0.5_rkind * (b2 * uu         )
        el(I_V,1)     =  -0.5_rkind * (b2 * vv         )
        el(I_W,1)     =  -0.5_rkind * (b2 * ww +     ci)
        el(I_E,1)     =   0.5_rkind * b2
        el(I_U,N_S+2) = 1._rkind
        el(I_V,N_S+2) = 0._rkind
        el(I_W,N_S+2) = 0._rkind
        el(I_E,N_S+2) = 0._rkind
        el(I_U,N_S+3) = 0._rkind
        el(I_V,N_S+3) = 1._rkind
        el(I_W,N_S+3) = 0._rkind
        el(I_E,N_S+3) = 0._rkind
        el(I_U,N_S+4) = -0.5_rkind * (b2 * uu         )
        el(I_V,N_S+4) = -0.5_rkind * (b2 * vv         )
        el(I_W,N_S+4) = -0.5_rkind * (b2 * ww -     ci)
        el(I_E,N_S+4) =  0.5_rkind * b2
!
!       right eigenvectors matrix (at Roe state)
!       matrix reported in Notebook_Summary.nb pag 7 after transposition
!
        do lsp=1,4+N_S
         do msp=1,4+N_S
          er(lsp,msp) = 0._rkind
         enddo
        enddo
        do lsp=1,N_S
         b3i = b3-prhoi(lsp)/b2*cci
         er(1    ,lsp) = yyroe(lsp)
         er(1+lsp,lsp) = 1._rkind
         er(4+N_S,lsp) = yyroe(lsp)
         er(1+lsp,I_U) =  uu
         er(1+lsp,I_V) =  vv
         er(1+lsp,I_W) =  ww
         er(1+lsp,I_E) = b3i
        enddo
        er(1,I_U)     =  uu
        er(1,I_V)     =  vv
        er(1,I_W)     =  ww - c
        er(1,I_E)     =  h  - ww * c
        er(N_S+2,I_U) =  1._rkind ! -1._rkind 
        er(N_S+3,I_U) =  0._rkind
        er(N_S+4,I_U) =  uu
        er(N_S+2,I_V) =  0._rkind
        er(N_S+3,I_V) =  1._rkind
        er(N_S+4,I_V) =  vv
        er(N_S+2,I_W) =  0._rkind
        er(N_S+3,I_W) =  0._rkind
        er(N_S+4,I_W) =  ww + c
        er(N_S+2,I_E) =  uu ! -uu
        er(N_S+3,I_E) =  vv
        er(N_S+4,I_E) =  h  + ww * c
!
    endsubroutine eigenvectors_z

    attributes(device) subroutine compute_roe_average(nx, ny, nz, ng, nv_aux, i, ip, j, jp, k, kp, w_aux_gpu, rgas_gpu, &
                                  b1, b2, b3, rho, c, ci, h, uu, vv, ww, cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, &
                                  indx_cp_l, indx_cp_r, tol_iter_nr, yyroe, prhoi)
        integer, intent(in) :: ng,i,ip,j,jp,k,kp,indx_cp_l,indx_cp_r
        integer, intent(in) :: nx,ny,nz,nv_aux,nsetcv
        real(rkind), intent(in) :: tol_iter_nr
        real(rkind), intent(out) :: b1, b2, b3, uu, vv, ww, ci, h, c, rho
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(in) :: w_aux_gpu 
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in) :: cp_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in) :: trange_gpu
        real(rkind), dimension(N_S), intent(in) :: rgas_gpu
        real(rkind), dimension(N_S), intent(inout) :: yyroe, prhoi
        real(rkind), dimension(N_S) :: rhoi_roe
        real(rkind) :: up, vp, wp, qqp, hp, r, rp1, cc, qq, yy, yyp
        real(rkind) :: pp, ppp, rhop, ee, eep, rho_L, ee_L, dp_SLV, de_SLV
        real(rkind) :: pede, dp_err, den_correction, prhoidrho, dpfact, p_over_rho, ee_yroe
        integer :: l,iter,lsp,llsp
        real(rkind) :: tt,hbar,ttp,told,num,den,gamloc,tpow,p_e,etot,rhoi,rhoip
        real(rkind) :: cploc,gm1loc,rmixtloc,cp_l,eei
        real(rkind) :: tt_L, tt_R, eeip, tti, ee_tti, gamloc_i, gm1loc_i
!       real(rkind) :: uui,vvi,wwi,hi
        real(rkind), dimension(N_S) :: drho_SLV
        integer, dimension(N_S) :: nrange
        integer :: max_iter,nmax,jl,ju,jm,nrangeloc
        real(rkind) :: T_start,sumb,ebar,T_pow,T_powp,T_old,tden,tnum,cv_l
!
!       rhoi @ Roe state of single species
! 
!       rho = 0._rkind
!       do lsp=1,N_S
!        rhoi  = w_aux_gpu(i,j,k,lsp)*w_aux_gpu(i,j,k,J_R)
!        rhoip = w_aux_gpu(ip,jp,kp,lsp)*w_aux_gpu(ip,jp,kp,J_R)
!        rhoi_roe(lsp) = sqrt(rhoi*rhoip)
!        rho = rho + rhoi_roe(lsp)
!       enddo
!        
!       Compute Roe average
!       Left state (node i)
        rho        =  w_aux_gpu(i,j,k,J_R)
        uu         =  w_aux_gpu(i,j,k,J_U)
        vv         =  w_aux_gpu(i,j,k,J_V)
        ww         =  w_aux_gpu(i,j,k,J_W)
        qq         =  0.5_rkind * (uu*uu  +vv*vv + ww*ww)
        h          =  w_aux_gpu(i,j,k,J_H) ! Total enthalpy
        pp         =  w_aux_gpu(i,j,k,J_P)
        ee         =  h-pp/rho-qq        ! Internal energy
        rho_L      =  rho
        ee_L       =  ee
        tt_L       =  w_aux_gpu(i,j,k,J_T)
!       Right state (node i+1)
        rhop       =  w_aux_gpu(ip,jp,kp,J_R)
        up         =  w_aux_gpu(ip,jp,kp,J_U)
        vp         =  w_aux_gpu(ip,jp,kp,J_V)
        wp         =  w_aux_gpu(ip,jp,kp,J_W)
        qqp        =  0.5_rkind * (up*up  +vp*vp +wp*wp)
        hp         =  w_aux_gpu(ip,jp,kp,J_H)
        ppp        =  w_aux_gpu(ip,jp,kp,J_P)
        eep        =  hp-ppp/rhop-qqp
        tt_R       =  w_aux_gpu(ip,jp,kp,J_T)
!       average state
        r          =  sqrt(rhop/rho)
        rho        =  r*rho_L
        rp1        =  1._rkind/(r+1._rkind)
        uu         =  (r*up+uu)*rp1
        vv         =  (r*vp+vv)*rp1
        ww         =  (r*wp+ww)*rp1
        h          =  (r*hp+ h)*rp1
        qq         =  0.5_rkind * (uu*uu  +vv*vv + ww*ww)
        ee         =  (r*eep  +ee)*rp1
        p_over_rho = h-ee-qq

!       average state for species
        rmixtloc = 0._rkind
        do lsp=1,N_S
         yy  = w_aux_gpu(i,j,k,lsp)
         yyp = w_aux_gpu(ip,jp,kp,lsp)
         drho_SLV(lsp) = rhop*yyp - rho_L*yy
         yy  = (r*yyp+yy)*rp1
         rmixtloc = rmixtloc+rgas_gpu(lsp)*yy
         yyroe(lsp) = yy
!        yyroe(lsp) = rhoi_roe(lsp)/rho                ! Y based on rhoi and rhoi @ Roe state
!        rmixtloc = rmixtloc+rgas_gpu(lsp)*yyroe(lsp)  ! R mixture based on rhoi @ Roe state only 
        enddo
!
!       tt        = w_aux_gpu(i ,j ,k ,J_T)
!       ttp       = w_aux_gpu(ip,jp,kp,J_T)
!       tt        = (r*ttp +tt)*rp1
!       told      = tt ! First attempt computed as Roe average of temperature
        told      = p_over_rho/rmixtloc ! First attempt as in HTR
!       tt        = get_mixture_temperature_from_h_roe_dev(h-qq,told,cp_coeff_gpu,nsetcv,trange_gpu,indx_cp_l,indx_cp_r,&
!                   tol_iter_nr,yyroe)
!-------------------------------------------------------------------------------------------------------------------------
! inlining get_mixture_temperature_from_e_roe_dev
!        tt        = get_mixture_temperature_from_e_roe_dev(ee,told,cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l,indx_cp_r,&
!                    tol_iter_nr,yyroe)
        T_start   = told
        max_iter = 10
!
        nmax = 100000
        do lsp=1,N_S
         nrange(lsp) = 1
        enddo
        if (nsetcv>1) then ! Replicate locate function of numerical recipes
         do lsp=1,N_S
          jl = 0
          ju = nsetcv+1+1
          do l=1,nmax
           if (ju-jl <= 1) exit
           jm = (ju+jl)/2
           if (T_start >= trange_gpu(lsp,jm)) then
            jl=jm
           else
            ju=jm
           endif
          enddo
          nrange(lsp) = jl
          nrange(lsp) = max(nrange(lsp),1)
          nrange(lsp) = min(nrange(lsp),nsetcv)
         enddo
        endif
!
        T_old = T_start
        do iter=1,max_iter
!
         if (nsetcv>1) then
          do lsp=1,N_S
           nrangeloc = nrange(lsp)
!          if (T_old>trange_gpu(lsp,nrangeloc).and.T_old<trange_gpu(lsp,nrangeloc+1)) then
!           else
!          endif
!          Assume maximum range jump in the iterative process equal to 1
           if (T_old<trange_gpu(lsp,nrangeloc)) then
            nrange(lsp) = nrange(lsp)-1
           elseif(T_old>trange_gpu(lsp,nrangeloc+1)) then
            nrange(lsp) = nrange(lsp)+1
           endif
           nrange(lsp) = max(nrange(lsp),1)
           nrange(lsp) = min(nrange(lsp),nsetcv)
          enddo
         endif
    !
         sumb = 0._rkind
         do lsp=1,N_S
          nrangeloc = nrange(lsp)
          sumb = sumb+cv_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)*yyroe(lsp)
         enddo
         ebar = ee-sumb
!
         den = 0._rkind
         num = 0._rkind
         do l=indx_cp_l,indx_cp_r
          T_pow  = T_old**l
          tden   = T_pow
          if (l==-1) then
           tnum   = log(T_old)
          else
           T_powp = T_old*T_pow
           tnum   = T_powp/(l+1._rkind)
          endif
          cv_l = 0._rkind
          do lsp=1,N_S
           nrangeloc = nrange(lsp)
           cv_l = cv_l+cv_coeff_gpu(l,lsp,nrangeloc)*yyroe(lsp)
          enddo
          num = num+cv_l*tnum
          den = den+cv_l*tden
         enddo
         tt = T_old+(ebar-num)/den
         if (abs(tt-T_old) < tol_iter_nr) exit
         T_old = tt
        enddo
!-------------------------------------------------------------------------------------------------------------------------
!
!-------------------------------------------------------------------------------------------------------------------------
! inlining get_cp_roe_dev
!        cploc = get_cp_roe_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,yyroe)
        nmax = 100000
        do lsp=1,N_S
         nrange(lsp) = 1
        enddo
        if (nsetcv>1) then ! Replicate locate function of numerical recipes
         do lsp=1,N_S
          jl = 0
          ju = nsetcv+1+1
          do l=1,nmax
           if (ju-jl <= 1) exit
           jm = (ju+jl)/2
           if (tt>= trange_gpu(lsp,jm)) then
            jl=jm
           else
            ju=jm
           endif
          enddo
          nrange(lsp) = jl
          nrange(lsp) = max(nrange(lsp),1)
          nrange(lsp) = min(nrange(lsp),nsetcv)
         enddo
        endif
!           
        cploc = 0._rkind
        do l=indx_cp_l,indx_cp_r
         T_pow = tt**l
         cp_l = 0._rkind
         do lsp=1,N_S
          nrangeloc = nrange(lsp)
          cp_l = cp_l+cp_coeff_gpu(l,lsp,nrangeloc)*yyroe(lsp)
         enddo
         cploc = cploc+cp_l*T_pow
        enddo
!-------------------------------------------------------------------------------------------------------------------------
        gamloc    = cploc/(cploc-rmixtloc)
        gm1loc    = gamloc-1._rkind
!
!       p_rho     = tt*rmixtloc
        p_e       = rho*gm1loc   ! p_e   = rho * R_gas / Cv
!
!       ee_yroe = 0._rkind
        do lsp=1,N_S
!-------------------------------------------------------------------------------------------------------------------------
! inlining get_species_e_from_temperature_dev
!         eei = get_species_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,lsp)
         nmax = 100000
         nrangeloc = 1
         if (nsetcv>1) then
          jl = 0
          ju = nsetcv+1+1
          do l=1,nmax
           if (ju-jl <= 1) exit
           jm = (ju+jl)/2
           if (tt >= trange_gpu(lsp,jm)) then
            jl=jm
           else
            ju=jm
           endif
          enddo
          nrangeloc = jl
          nrangeloc = max(nrangeloc,1)
          nrangeloc = min(nrangeloc,nsetcv)
         endif
         eei = cv_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)
         do l=indx_cp_l,indx_cp_r
          if (l==-1) then
           eei = eei+cv_coeff_gpu(l,lsp,nrangeloc)*log(tt)
          else
           eei = eei+cv_coeff_gpu(l,lsp,nrangeloc)/(l+1._rkind)*(tt)**(l+1)
          endif
         enddo
!-------------------------------------------------------------------------------------------------------------------------
!        eei  = get_species_e_from_temperature_dev(tt_L,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,lsp)
!        eeip = get_species_e_from_temperature_dev(tt_R,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,lsp)
!        eei =  (r*eeip+eei)*rp1
!        ee_yroe = ee_yroe+eei*yyroe(lsp)
         prhoi(lsp) = eei
!        prhoi(lsp) = rgas_gpu(lsp)*tt+gm1loc*(etot-qq-eei)
!        prhoi(lsp) = rgas_gpu(lsp)*tt+gm1loc*(ee_yroe-eei)
        enddo
        do lsp=1,N_S
         eei = prhoi(lsp)
!        prhoi(lsp) = rgas_gpu(lsp)*tt+gm1loc*(ee_yroe-eei)
         prhoi(lsp) = rgas_gpu(lsp)*tt+gm1loc*(ee-eei)
!        prhoi(lsp) = rgas_gpu(lsp)*tti+gm1loc*(ee-eei)
!        prhoi(lsp) = rgas_gpu(lsp)*tti+gm1loc*(ee_tti-eei)
!        prhoi(lsp) = rgas_gpu(lsp)*tti+gm1loc_i*(ee_tti-eei)
        enddo
!
!       etot = h - tt*rmixtloc
!       etot = ee_yroe+qq
        etot = ee+qq
!
        dp_SLV = ppp-pp
        de_SLV = eep-ee_L
!
        pede = p_e*de_SLV
        dp_err = dp_SLV-pede
        den_correction = pede*pede
        do lsp=1,N_S
         prhoidrho = prhoi(lsp)*drho_SLV(lsp)
         dp_err = dp_err-prhoidrho
         den_correction = den_correction+prhoidrho*prhoidrho
        enddo
        dpfact = dp_err/max(den_correction,0.000001_rkind)
        p_e = p_e*(1._rkind+pede*dpfact)
        do lsp=1,N_S
         prhoi(lsp) = prhoi(lsp)*(1._rkind+prhoi(lsp)*drho_SLV(lsp)*dpfact)
        enddo
!
!       cc = gamloc*rmixtloc*tt
!
        cc = p_over_rho/rho*p_e
        do lsp=1,N_S
         cc = cc+yyroe(lsp)*prhoi(lsp)
        enddo
!
        c         =  sqrt(cc)
        ci        =  1._rkind/c
!
!       b3        = etot - rho * p_rho/p_e ! qq in the case of calorically perfect
        b3        = etot
        b2        = p_e/(rho*cc)
!       b1        = p_rho/cc - b2*(etot - 2._rkind*qq)
        b1        = - b2*(etot - 2._rkind*qq)
!
!       b2        = gm1/cc  ! 1/(cp*T)
!       b1        = b2 * qq
!       b3        = qq 
!       b2        = gm1/cc
!       b1        = b2 * b3

    endsubroutine compute_roe_average

    subroutine update_flux_cuf(nx, ny, nz, nv, fl_gpu, fln_gpu, gamdt) 
        integer :: nx, ny, nz, nv
        real(rkind) :: gamdt
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: fl_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fln_gpu
        integer :: i,j,k,m,iercuda
        !$cuf kernel do(3) <<<*,*>>> 
         do k=1,nz
          do j=1,ny
           do i=1,nx
            do m=1,nv
             fln_gpu(i,j,k,m) = fln_gpu(i,j,k,m)-gamdt*fl_gpu(i,j,k,m)
            enddo
           enddo
          enddo
         enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine update_flux_cuf

    subroutine update_simpler_flux_cuf(nx, ny, nz, nv, fl_gpu, fln_gpu, fl_sav_gpu, gamdt)
        integer :: nx, ny, nz, nv
        real(rkind) :: gamdt
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: fl_gpu,fl_sav_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fln_gpu
        integer :: i,j,k,m,iercuda

        !$cuf kernel do(3) <<<*,*>>> 
         do k=1,nz
          do j=1,ny
           do i=1,nx
            do m=1,nv
             fln_gpu(i,j,k,m) = fln_gpu(i,j,k,m)-gamdt*(fl_gpu(i,j,k,m)-fl_sav_gpu(i,j,k,m))
            enddo
           enddo
          enddo
         enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine update_simpler_flux_cuf

    subroutine update_field_cuf(nx, ny, nz, ng, nv, w_gpu, fln_gpu, fluid_mask_gpu)
        integer :: nx, ny, nz, nv, ng
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: fln_gpu
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: fluid_mask_gpu
        integer :: i,j,k,m,iercuda
        !$cuf kernel do(3) <<<*,*>>> 
         do k=1,nz
          do j=1,ny
           do i=1,nx
            if (fluid_mask_gpu(i,j,k)==0) then
             do m=1,nv
              w_gpu(i,j,k,m) = w_gpu(i,j,k,m)+fln_gpu(i,j,k,m)
             enddo
            endif
           enddo
          enddo
         enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine update_field_cuf

    subroutine visflx_cuf(nx, ny, nz, ng, visc_order, &
            w_aux_gpu, fl_gpu, &
            coeff_deriv1_gpu, coeff_deriv2_gpu, &
            dcsidx_gpu, detady_gpu, dzitdz_gpu,  &
            dcsidxs_gpu, detadys_gpu, dzitdzs_gpu, & 
            dcsidx2_gpu, detady2_gpu, dzitdz2_gpu)

        integer, intent(in) :: nx, ny, nz, ng, visc_order
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: w_aux_gpu 
        real(rkind), dimension(1:,1:,1:, 1:), intent(inout), device :: fl_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: coeff_deriv1_gpu
        real(rkind), dimension(0:,1:), intent(in), device  :: coeff_deriv2_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu, detady_gpu, dzitdz_gpu 
        real(rkind), dimension(1:), intent(in), device :: dcsidxs_gpu, detadys_gpu, dzitdzs_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx2_gpu, detady2_gpu, dzitdz2_gpu
        integer     :: i,j,k,l,ll,iercuda
        real(rkind) :: ccl,clapl
        real(rkind) :: sig11,sig12,sig13
        real(rkind) :: sig22,sig23
        real(rkind) :: sig33
        real(rkind) :: uu,vv,ww,tt,mu,k_cond
        real(rkind) :: ux,uy,uz
        real(rkind) :: vx,vy,vz
        real(rkind) :: wx,wy,wz
        real(rkind) :: tx,ty,tz
        real(rkind) :: mux,muy,muz
        real(rkind) :: divx3l,divy3l,divz3l
        real(rkind) :: ulap,ulapx,ulapy,ulapz
        real(rkind) :: vlap,vlapx,vlapy,vlapz
        real(rkind) :: wlap,wlapx,wlapy,wlapz
        real(rkind) :: tlap,tlapx,tlapy,tlapz
        real(rkind) :: sigq,sigx,sigy,sigz,sigqt,sigah
        real(rkind) :: div3l
        integer     :: lmax

        lmax = visc_order/2
     
        !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          do k=1,nz
      
           uu     = w_aux_gpu(i,j,k,J_U)
           vv     = w_aux_gpu(i,j,k,J_V)
           ww     = w_aux_gpu(i,j,k,J_W)
           tt     = w_aux_gpu(i,j,k,J_T)
           mu     = w_aux_gpu(i,j,k,J_MU)
           k_cond = w_aux_gpu(i,j,k,J_K_COND)
           div3l  = w_aux_gpu(i,j,k,J_DIV)

           ux = 0._rkind
           vx = 0._rkind
           wx = 0._rkind
           tx = 0._rkind
           mux = 0._rkind
           divx3l = 0._rkind
           uy = 0._rkind
           vy = 0._rkind
           wy = 0._rkind
           ty = 0._rkind
           muy = 0._rkind
           divy3l = 0._rkind
           uz = 0._rkind
           vz = 0._rkind
           wz = 0._rkind
           tz = 0._rkind
           muz = 0._rkind
           divz3l = 0._rkind
!           
           ulapx = coeff_deriv2_gpu(0,lmax)*uu
           ulapy = ulapx
           ulapz = ulapx
           vlapx = coeff_deriv2_gpu(0,lmax)*vv
           vlapy = vlapx
           vlapz = vlapx
           wlapx = coeff_deriv2_gpu(0,lmax)*ww
           wlapy = wlapx
           wlapz = wlapx
           tlapx = coeff_deriv2_gpu(0,lmax)*tt
           tlapy = tlapx
           tlapz = tlapx
!
           do l=1,lmax
            clapl = coeff_deriv2_gpu(l,lmax)
            ulapx = ulapx + clapl*(w_aux_gpu(i+l,j,k,J_U)+w_aux_gpu(i-l,j,k,J_U))
            ulapy = ulapy + clapl*(w_aux_gpu(i,j+l,k,J_U)+w_aux_gpu(i,j-l,k,J_U))
            ulapz = ulapz + clapl*(w_aux_gpu(i,j,k+l,J_U)+w_aux_gpu(i,j,k-l,J_U))
            vlapx = vlapx + clapl*(w_aux_gpu(i+l,j,k,J_V)+w_aux_gpu(i-l,j,k,J_V))
            vlapy = vlapy + clapl*(w_aux_gpu(i,j+l,k,J_V)+w_aux_gpu(i,j-l,k,J_V))
            vlapz = vlapz + clapl*(w_aux_gpu(i,j,k+l,J_V)+w_aux_gpu(i,j,k-l,J_V))
            wlapx = wlapx + clapl*(w_aux_gpu(i+l,j,k,J_W)+w_aux_gpu(i-l,j,k,J_W))
            wlapy = wlapy + clapl*(w_aux_gpu(i,j+l,k,J_W)+w_aux_gpu(i,j-l,k,J_W))
            wlapz = wlapz + clapl*(w_aux_gpu(i,j,k+l,J_W)+w_aux_gpu(i,j,k-l,J_W))
            tlapx = tlapx + clapl*(w_aux_gpu(i+l,j,k,J_T)+w_aux_gpu(i-l,j,k,J_T))
            tlapy = tlapy + clapl*(w_aux_gpu(i,j+l,k,J_T)+w_aux_gpu(i,j-l,k,J_T))
            tlapz = tlapz + clapl*(w_aux_gpu(i,j,k+l,J_T)+w_aux_gpu(i,j,k-l,J_T))
           enddo
!
           do l=1,lmax
            ccl = coeff_deriv1_gpu(l,lmax)
            ux = ux+ccl*(w_aux_gpu(i+l,j,k,J_U)-w_aux_gpu(i-l,j,k,J_U))
            vx = vx+ccl*(w_aux_gpu(i+l,j,k,J_V)-w_aux_gpu(i-l,j,k,J_V))
            wx = wx+ccl*(w_aux_gpu(i+l,j,k,J_W)-w_aux_gpu(i-l,j,k,J_W))
            tx = tx+ccl*(w_aux_gpu(i+l,j,k,J_T)-w_aux_gpu(i-l,j,k,J_T))
            mux = mux+ccl*(w_aux_gpu(i+l,j,k,J_MU)-w_aux_gpu(i-l,j,k,J_MU))
            divx3l = divx3l+ccl*(w_aux_gpu(i+l,j,k,J_DIV)-w_aux_gpu(i-l,j,k,J_DIV))
            uy = uy+ccl*(w_aux_gpu(i,j+l,k,J_U)-w_aux_gpu(i,j-l,k,J_U))
            vy = vy+ccl*(w_aux_gpu(i,j+l,k,J_V)-w_aux_gpu(i,j-l,k,J_V))
            wy = wy+ccl*(w_aux_gpu(i,j+l,k,J_W)-w_aux_gpu(i,j-l,k,J_W))
            ty = ty+ccl*(w_aux_gpu(i,j+l,k,J_T)-w_aux_gpu(i,j-l,k,J_T))
            muy = muy+ccl*(w_aux_gpu(i,j+l,k,J_MU)-w_aux_gpu(i,j-l,k,J_MU))
            divy3l = divy3l+ccl*(w_aux_gpu(i,j+l,k,J_DIV)-w_aux_gpu(i,j-l,k,J_DIV))
            uz = uz+ccl*(w_aux_gpu(i,j,k+l,J_U)-w_aux_gpu(i,j,k-l,J_U))
            vz = vz+ccl*(w_aux_gpu(i,j,k+l,J_V)-w_aux_gpu(i,j,k-l,J_V))
            wz = wz+ccl*(w_aux_gpu(i,j,k+l,J_W)-w_aux_gpu(i,j,k-l,J_W))
            tz = tz+ccl*(w_aux_gpu(i,j,k+l,J_T)-w_aux_gpu(i,j,k-l,J_T))
            muz = muz+ccl*(w_aux_gpu(i,j,k+l,J_MU)-w_aux_gpu(i,j,k-l,J_MU))
            divz3l = divz3l+ccl*(w_aux_gpu(i,j,k+l,J_DIV)-w_aux_gpu(i,j,k-l,J_DIV))
           enddo
           ux = ux*dcsidx_gpu(i)
           vx = vx*dcsidx_gpu(i)
           wx = wx*dcsidx_gpu(i)
           tx = tx*dcsidx_gpu(i)
           mux = mux*dcsidx_gpu(i)
           divx3l = divx3l*dcsidx_gpu(i)
           uy = uy*detady_gpu(j)
           vy = vy*detady_gpu(j)
           wy = wy*detady_gpu(j)
           ty = ty*detady_gpu(j)
           muy = muy*detady_gpu(j)
           divy3l = divy3l*detady_gpu(j)
           uz = uz*dzitdz_gpu(k)
           vz = vz*dzitdz_gpu(k)
           wz = wz*dzitdz_gpu(k)
           tz = tz*dzitdz_gpu(k)
           muz = muz*dzitdz_gpu(k)
           divz3l = divz3l*dzitdz_gpu(k)
!
           ulapx = ulapx*dcsidxs_gpu(i)+ux*dcsidx2_gpu(i)
           vlapx = vlapx*dcsidxs_gpu(i)+vx*dcsidx2_gpu(i)
           wlapx = wlapx*dcsidxs_gpu(i)+wx*dcsidx2_gpu(i)
           tlapx = tlapx*dcsidxs_gpu(i)+tx*dcsidx2_gpu(i)
           ulapy = ulapy*detadys_gpu(j)+uy*detady2_gpu(j)
           vlapy = vlapy*detadys_gpu(j)+vy*detady2_gpu(j)
           wlapy = wlapy*detadys_gpu(j)+wy*detady2_gpu(j)
           tlapy = tlapy*detadys_gpu(j)+ty*detady2_gpu(j)
           ulapz = ulapz*dzitdzs_gpu(k)+uz*dzitdz2_gpu(k)
           vlapz = vlapz*dzitdzs_gpu(k)+vz*dzitdz2_gpu(k)
           wlapz = wlapz*dzitdzs_gpu(k)+wz*dzitdz2_gpu(k)
           tlapz = tlapz*dzitdzs_gpu(k)+tz*dzitdz2_gpu(k)
!      
           ulap  = ulapx+ulapy+ulapz
           vlap  = vlapx+vlapy+vlapz
           wlap  = wlapx+wlapy+wlapz
           tlap  = tlapx+tlapy+tlapz
!     
           sig11 = 2._rkind*(ux-div3l)
           sig12 = uy+vx 
           sig13 = uz+wx
           sig22 = 2._rkind*(vy-div3l)
           sig23 = vz+wy
           sig33 = 2._rkind*(wz-div3l)
           sigx  = mux*sig11 + muy*sig12 + muz*sig13 + mu*(ulap+divx3l)
           sigy  = mux*sig12 + muy*sig22 + muz*sig23 + mu*(vlap+divy3l)
           sigz  = mux*sig13 + muy*sig23 + muz*sig33 + mu*(wlap+divz3l)
           sigqt = (mux*tx+muy*ty+muz*tz+mu*tlap)*k_cond/mu ! Conduction
!           
           sigah = (sig11*ux+sig12*uy+sig13*uz+sig12*vx+sig22*vy+sig23*vz+sig13*wx+sig23*wy+sig33*wz)*mu ! Aerodynamic heating
!       
           sigq  = sigx*uu+sigy*vv+sigz*ww+sigah+sigqt
!      
           fl_gpu(i,j,k,I_U) = fl_gpu(i,j,k,I_U) - sigx
           fl_gpu(i,j,k,I_V) = fl_gpu(i,j,k,I_V) - sigy
           fl_gpu(i,j,k,I_W) = fl_gpu(i,j,k,I_W) - sigz
           fl_gpu(i,j,k,I_E) = fl_gpu(i,j,k,I_E) - sigq 
!           
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine visflx_cuf

    subroutine visflx_reduced_cuf(nx, ny, nz, ng, visc_order, &
            w_aux_gpu, fl_gpu, &
            coeff_deriv1_gpu, &
            dcsidx_gpu, detady_gpu, dzitdz_gpu)

        integer, intent(in) :: nx, ny, nz, ng, visc_order
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: w_aux_gpu 
        real(rkind), dimension(1:,1:,1:, 1:), intent(inout), device :: fl_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: coeff_deriv1_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu, detady_gpu, dzitdz_gpu 
        integer     :: i,j,k,l,ll,iercuda
        real(rkind) :: ccl
        real(rkind) :: sig11,sig12,sig13
        real(rkind) :: sig21,sig22,sig23
        real(rkind) :: sig31,sig32,sig33
        real(rkind) :: uu,vv,ww,tt,mu,k_cond,div3l
        real(rkind) :: ux,uy,uz
        real(rkind) :: vx,vy,vz
        real(rkind) :: wx,wy,wz
        real(rkind) :: mux,muy,muz
        real(rkind) :: divx3l,divy3l,divz3l
        real(rkind) :: sigq,sigx,sigy,sigz,sigah
        integer     :: lmax

        lmax = visc_order/2
     
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do i=1,nx
! 
           uu = w_aux_gpu(i,j,k,J_U)
           vv = w_aux_gpu(i,j,k,J_V)
           ww = w_aux_gpu(i,j,k,J_W)
           tt = w_aux_gpu(i,j,k,J_T)
           mu = w_aux_gpu(i,j,k,J_MU)
           k_cond = w_aux_gpu(i,j,k,J_K_COND)
           div3l  = w_aux_gpu(i,j,k,J_DIV)
!
           ux = 0._rkind
           vx = 0._rkind
           wx = 0._rkind
           mux = 0._rkind
           divx3l = 0._rkind
           uy = 0._rkind
           vy = 0._rkind
           wy = 0._rkind
           muy = 0._rkind
           divy3l = 0._rkind
           uz = 0._rkind
           vz = 0._rkind
           wz = 0._rkind
           muz = 0._rkind
           divz3l = 0._rkind

           do l=1,lmax
            ccl = coeff_deriv1_gpu(l,lmax)
            ux = ux+ccl*(w_aux_gpu(i+l,j,k,J_U)-w_aux_gpu(i-l,j,k,J_U))
            vx = vx+ccl*(w_aux_gpu(i+l,j,k,J_V)-w_aux_gpu(i-l,j,k,J_V))
            wx = wx+ccl*(w_aux_gpu(i+l,j,k,J_W)-w_aux_gpu(i-l,j,k,J_W))
            mux = mux+ccl*(w_aux_gpu(i+l,j,k,J_MU)-w_aux_gpu(i-l,j,k,J_MU))
            divx3l = divx3l+ccl*(w_aux_gpu(i+l,j,k,J_DIV)-w_aux_gpu(i-l,j,k,J_DIV))
      
            uy = uy+ccl*(w_aux_gpu(i,j+l,k,J_U)-w_aux_gpu(i,j-l,k,J_U))
            vy = vy+ccl*(w_aux_gpu(i,j+l,k,J_V)-w_aux_gpu(i,j-l,k,J_V))
            wy = wy+ccl*(w_aux_gpu(i,j+l,k,J_W)-w_aux_gpu(i,j-l,k,J_W))
            muy = muy+ccl*(w_aux_gpu(i,j+l,k,J_MU)-w_aux_gpu(i,j-l,k,J_MU))
            divy3l = divy3l+ccl*(w_aux_gpu(i,j+l,k,J_DIV)-w_aux_gpu(i,j-l,k,J_DIV))
     
            uz = uz+ccl*(w_aux_gpu(i,j,k+l,J_U)-w_aux_gpu(i,j,k-l,J_U))
            vz = vz+ccl*(w_aux_gpu(i,j,k+l,J_V)-w_aux_gpu(i,j,k-l,J_V))
            wz = wz+ccl*(w_aux_gpu(i,j,k+l,J_W)-w_aux_gpu(i,j,k-l,J_W))
            muz = muz+ccl*(w_aux_gpu(i,j,k+l,J_MU)-w_aux_gpu(i,j,k-l,J_MU))
            divz3l = divz3l+ccl*(w_aux_gpu(i,j,k+l,J_DIV)-w_aux_gpu(i,j,k-l,J_DIV))
           enddo
           ux = ux*dcsidx_gpu(i)
           vx = vx*dcsidx_gpu(i)
           wx = wx*dcsidx_gpu(i)
           mux = mux*dcsidx_gpu(i)
           divx3l = divx3l*dcsidx_gpu(i)
           uy = uy*detady_gpu(j)
           vy = vy*detady_gpu(j)
           wy = wy*detady_gpu(j)
           muy = muy*detady_gpu(j)
           divy3l = divy3l*detady_gpu(j)
           uz = uz*dzitdz_gpu(k)
           vz = vz*dzitdz_gpu(k)
           wz = wz*dzitdz_gpu(k)
           muz = muz*dzitdz_gpu(k)
           divz3l = divz3l*dzitdz_gpu(k)
!
           sig11 = ux-2._rkind*div3l
           sig12 = vx
           sig13 = wx
           sig21 = uy
           sig22 = vy-2._rkind*div3l
           sig23 = wy
           sig31 = uz
           sig32 = vz
           sig33 = wz-2._rkind*div3l
           sigx  = mux*sig11 + muy*sig12 + muz*sig13 + mu*divx3l
           sigy  = mux*sig21 + muy*sig22 + muz*sig23 + mu*divy3l
           sigz  = mux*sig31 + muy*sig32 + muz*sig33 + mu*divz3l
!
           sigah = (sig11*ux+sig12*uy+sig13*uz+sig21*vx+sig22*vy+sig23*vz+sig31*wx+sig32*wy+sig33*wz)*mu ! Aerodynamic heating
!       
           sigq  = sigx*uu+sigy*vv+sigz*ww+sigah
!
           fl_gpu(i,j,k,I_U) = fl_gpu(i,j,k,I_U) - sigx
           fl_gpu(i,j,k,I_V) = fl_gpu(i,j,k,I_V) - sigy
           fl_gpu(i,j,k,I_W) = fl_gpu(i,j,k,I_W) - sigz
           fl_gpu(i,j,k,I_E) = fl_gpu(i,j,k,I_E) - sigq
!     
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine visflx_reduced_cuf

    subroutine visflx_x_cuf(nx, ny, nz, nv, nv_aux, ng, &
            x_gpu, w_aux_gpu, fl_gpu, fhat_gpu, &
            indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,rgas_gpu,R_univ)

        integer, intent(in) :: nx, ny, nz, nv, nv_aux, ng
        integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
        real(rkind), intent(in) :: R_univ
        real(rkind), dimension(1:nx,1:ny,1:nz,1:nv), intent(inout), device :: fl_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(in), device :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(out), device :: fhat_gpu
        real(rkind), dimension(1-ng:), intent(in), device :: x_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
        real(rkind), dimension(nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cp_coeff_gpu
        integer     :: i,j,k,iv,ll,lsp,iercuda,msp
        real(rkind) :: uu,vv,ww,tt,mu,qq,k_cond
        real(rkind) :: uup,vvp,wwp,ttp,mup,qqp,k_cond_p
        real(rkind) :: sigq,sigx,sigy,sigz,sigq_tt,sigq_qq
        real(rkind) :: muf,k_cond_f, fhat_tmp
        real(rkind) :: xxp,xx,mwmif,mwm,mwmp,yy,yyp
        real(rkind) :: fhat_tmp_et,hh,hhp,dxhl,hhf,yyf
        real(rkind) :: siglsp,rmixtloc,rhodiff
!
!
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do i=0,nx
           uu  = w_aux_gpu(i  ,j,k,J_U) 
           uup = w_aux_gpu(i+1,j,k,J_U) 
           vv  = w_aux_gpu(i  ,j,k,J_V) 
           vvp = w_aux_gpu(i+1,j,k,J_V) 
           ww  = w_aux_gpu(i  ,j,k,J_W) 
           wwp = w_aux_gpu(i+1,j,k,J_W) 
           tt  = w_aux_gpu(i  ,j,k,J_T) 
           ttp = w_aux_gpu(i+1,j,k,J_T) 
           mu  = w_aux_gpu(i  ,j,k,J_MU) 
           mup = w_aux_gpu(i+1,j,k,J_MU) 
           k_cond   = w_aux_gpu(i  ,j,k,J_K_COND) 
           k_cond_p = w_aux_gpu(i+1,j,k,J_K_COND) 
           qq  = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
           qqp = 0.5_rkind*(uup*uup+vvp*vvp+wwp*wwp)
!
!          cpm  = get_cp_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,&
!                i,j,k,nx,ny,nz,ng,nv_aux,w_aux_gpu)
!          cpmp = get_cp_dev(ttp,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,&
!                i+1,j,k,nx,ny,nz,ng,nv_aux,w_aux_gpu)
!
           sigx     = uup-uu
           sigy     = vvp-vv
           sigz     = wwp-ww
           sigq_tt  = ttp-tt
           sigq_qq  = qqp-qq
           muf      = mu+mup
           muf      = 0.5_rkind*muf/(x_gpu(i+1)-x_gpu(i))
           k_cond_f = k_cond+k_cond_p
           k_cond_f = 0.5_rkind*k_cond_f/(x_gpu(i+1)-x_gpu(i))
!          k_cond_fcp = 0.5_rkind*(k_cond/cpm+k_cond_p/cpmp)/(x_gpu(i+1)-x_gpu(i))
!
           sigx = sigx*muf
           sigy = sigy*muf
           sigz = sigz*muf
           sigq = sigq_tt*k_cond_f+sigq_qq*muf
!
           !if (i>0) then
           ! fl_trans_gpu(j,i,k,2) = fl2o-sigx*dxhl
           ! fl_trans_gpu(j,i,k,3) = fl3o-sigy*dxhl
           ! fl_trans_gpu(j,i,k,4) = fl4o-sigz*dxhl
           ! fl_trans_gpu(j,i,k,5) = fl5o-sigq*dxhl
           !endif
           !if (i<nx) then
            !fl2o = sigx*dxhl
            !fl3o = sigy*dxhl
            !fl4o = sigz*dxhl
            !fl5o = sigq*dxhl
!
!           mwm   = w_aux_gpu(i  ,j,k,J_MW)
            rmixtloc = get_rmixture(i  ,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
            mwm      = R_univ/rmixtloc
!           mwmp  = w_aux_gpu(i+1,j,k,J_MW)
            rmixtloc = get_rmixture(i+1,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
            mwmp     = R_univ/rmixtloc
!            
            mwmif = 0.5_rkind*(mwmp+mwm)/(mwmp*mwm)
            mwmif = mwmif/(x_gpu(i+1)-x_gpu(i))
            fhat_tmp = 0._rkind
            do msp=1,N_S
             rhodiff = 0.5_rkind*(w_aux_gpu(i,j,k,J_D_START+msp)+w_aux_gpu(i+1,j,k,J_D_START+msp))
             xxp       = w_aux_gpu(i+1,j,k,msp)*mwmp
             xx        = w_aux_gpu(i  ,j,k,msp)*mwm
             fhat_tmp = fhat_tmp + (xxp-xx)*rhodiff
            enddo
            fhat_tmp = fhat_tmp*mwmif
!
            fhat_tmp_et = 0._rkind
            do lsp=1,N_S
             rhodiff = 0.5_rkind*(w_aux_gpu(i,j,k,J_D_START+lsp)+w_aux_gpu(i+1,j,k,J_D_START+lsp))
             yyp       = w_aux_gpu(i+1,j,k,lsp)
             yy        = w_aux_gpu(i  ,j,k,lsp)
             yyf       = 0.5_rkind*(yyp+yy)
             xxp   = yyp*mwmp
             xx    = yy *mwm
             siglsp    = -rhodiff*(xxp-xx)*mwmif + fhat_tmp*yyf
             fhat_gpu(i,j,k,lsp) = siglsp
             hh  = get_species_h_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu, &
                                nsetcv,trange_gpu,lsp)
             hhp = get_species_h_from_temperature_dev(ttp,indx_cp_l,indx_cp_r,cp_coeff_gpu,&
                                nsetcv,trange_gpu,lsp)
             hhf = 0.5_rkind*(hh+hhp)
             fhat_tmp_et = fhat_tmp_et+siglsp*hhf
            enddo
            fhat_gpu(i,j,k,I_U) = - sigx
            fhat_gpu(i,j,k,I_V) = - sigy
            fhat_gpu(i,j,k,I_W) = - sigz
            fhat_gpu(i,j,k,I_E) = - sigq + fhat_tmp_et
!         
           enddo
          enddo
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()
!
         !$cuf kernel do(3) <<<*,*>>>
         do k=1,nz
          do j=1,ny
           do i=1,nx
            dxhl = 2._rkind/(x_gpu(i+1)-x_gpu(i-1))
            do iv=1,nv
             fl_gpu(i,j,k,iv) = fl_gpu(i,j,k,iv) + dxhl*(fhat_gpu(i,j,k,iv)-fhat_gpu(i-1,j,k,iv))
            enddo
           enddo
          enddo
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()

    endsubroutine visflx_x_cuf

    subroutine visflx_y_cuf(nx, ny, nz, nv, nv_aux, ng, &
            y_gpu, w_aux_gpu, fl_gpu, fhat_gpu,         &
            indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,rgas_gpu,R_univ)

        integer, intent(in) :: nx, ny, nz, nv, nv_aux, ng
        integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
        real(rkind), intent(in) :: R_univ
        real(rkind), dimension(1:nx,1:ny,1:nz,1:nv), intent(inout), device :: fl_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(in), device :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(out), device :: fhat_gpu
        real(rkind), dimension(1-ng:), intent(in), device :: y_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
        real(rkind), dimension(nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cp_coeff_gpu
        integer     :: i,j,k,iv,ll,lsp,iercuda,msp
        real(rkind) :: uu,vv,ww,tt,mu,qq,k_cond
        real(rkind) :: uup,vvp,wwp,ttp,mup,qqp,k_cond_p
        real(rkind) :: sigq,sigx,sigy,sigz,sigq_tt,sigq_qq
        real(rkind) :: dyhl,fl2o,fl3o,fl4o,fl5o
        real(rkind) :: muf,k_cond_f,fhat_tmp
        real(rkind) :: xxp,xx,mwmif,mwm,mwmp,yy,yyp
        real(rkind) :: fhat_tmp_et,hh,hhp,dxhl,hhf,yyf
        real(rkind) :: rhodiff,siglsp,rmixtloc
!
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=0,ny
          do i=1,nx
!
           uu  = w_aux_gpu(i,j  ,k,J_U) 
           uup = w_aux_gpu(i,j+1,k,J_U) 
           vv  = w_aux_gpu(i,j  ,k,J_V) 
           vvp = w_aux_gpu(i,j+1,k,J_V) 
           ww  = w_aux_gpu(i,j  ,k,J_W) 
           wwp = w_aux_gpu(i,j+1,k,J_W) 
           tt  = w_aux_gpu(i,j  ,k,J_T) 
           ttp = w_aux_gpu(i,j+1,k,J_T) 
           mu  = w_aux_gpu(i,j  ,k,J_MU) 
           mup = w_aux_gpu(i,j+1,k,J_MU) 
           k_cond   = w_aux_gpu(i,j  ,k,J_K_COND) 
           k_cond_p = w_aux_gpu(i,j+1,k,J_K_COND) 
           qq  = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
           qqp = 0.5_rkind*(uup*uup+vvp*vvp+wwp*wwp)
!
           sigx    = uup-uu
           sigy    = vvp-vv
           sigz    = wwp-ww
           sigq_tt = ttp-tt
           sigq_qq = qqp-qq
           muf     = mu+mup
           muf     = 0.5_rkind*muf/(y_gpu(j+1)-y_gpu(j))
           k_cond_f = k_cond+k_cond_p
           k_cond_f = 0.5_rkind*k_cond_f/(y_gpu(j+1)-y_gpu(j))
!
           sigx = sigx*muf
           sigy = sigy*muf
           sigz = sigz*muf
           sigq = sigq_tt*k_cond_f+sigq_qq*muf
!
!          mwm   = w_aux_gpu(i,j  ,k,J_MW)
           rmixtloc = get_rmixture(i,j   ,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           mwm      = R_univ/rmixtloc
!          mwmp  = w_aux_gpu(i,j+1,k,J_MW)
           rmixtloc = get_rmixture(i,j+1,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           mwmp     = R_univ/rmixtloc
!           
           mwmif = 0.5_rkind*(mwmp+mwm)/(mwmp*mwm)
           mwmif = mwmif/(y_gpu(j+1)-y_gpu(j))
           fhat_tmp = 0._rkind
           do msp=1,N_S
            rhodiff = 0.5_rkind*(w_aux_gpu(i,j,k,J_D_START+msp)+w_aux_gpu(i,j+1,k,J_D_START+msp))
            xxp = w_aux_gpu(i,j+1,k,msp)*mwmp
            xx  = w_aux_gpu(i,j  ,k,msp)*mwm
            fhat_tmp = fhat_tmp + (xxp-xx)*rhodiff
           enddo
           fhat_tmp = fhat_tmp*mwmif
!           
           fhat_tmp_et = 0._rkind
           do lsp=1,N_S
            rhodiff = 0.5_rkind*(w_aux_gpu(i,j,k,J_D_START+lsp)+w_aux_gpu(i,j+1,k,J_D_START+lsp))
            yyp   = w_aux_gpu(i,j+1,k,lsp)
            yy    = w_aux_gpu(i,j  ,k,lsp)
            yyf   = 0.5_rkind*(yyp+yy)
            xxp   = yyp*mwmp
            xx    = yy *mwm
            siglsp = -rhodiff*(xxp-xx)*mwmif + fhat_tmp*yyf
            fhat_gpu(i,j,k,lsp) = siglsp
            hh  = get_species_h_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu, &
                               nsetcv,trange_gpu,lsp)
            hhp = get_species_h_from_temperature_dev(ttp,indx_cp_l,indx_cp_r,cp_coeff_gpu,&
                               nsetcv,trange_gpu,lsp)
            hhf = 0.5_rkind*(hh+hhp)
            fhat_tmp_et = fhat_tmp_et+siglsp*hhf
           enddo           
           fhat_gpu(i,j,k,I_U) = - sigx
           fhat_gpu(i,j,k,I_V) = - sigy
           fhat_gpu(i,j,k,I_W) = - sigz
           fhat_gpu(i,j,k,I_E) = - sigq + fhat_tmp_et
!
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
!
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do i=1,nx
           dyhl = 2._rkind/(y_gpu(j+1)-y_gpu(j-1))
           do iv=1,nv
            fl_gpu(i,j,k,iv) = fl_gpu(i,j,k,iv) + dyhl*(fhat_gpu(i,j,k,iv)-fhat_gpu(i,j-1,k,iv))
           enddo
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()


    endsubroutine visflx_y_cuf
!    
    subroutine visflx_z_cuf(nx, ny, nz, nv, nv_aux, ng, &
            z_gpu, w_aux_gpu, fl_gpu, fhat_gpu,         &
            indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,rgas_gpu,R_univ)

        integer, intent(in) :: nx,ny,nz,nv,nv_aux,ng
        integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
        real(rkind), intent(in) :: R_univ
        real(rkind), dimension(1:nx,1:ny,1:nz,1:nv), intent(inout), device :: fl_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(in), device :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(out), device :: fhat_gpu
        real(rkind), dimension(1-ng:), intent(in), device :: z_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
        real(rkind), dimension(nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cp_coeff_gpu
        integer     :: i,j,k,iv,ll,lsp,iercuda,msp
        real(rkind) :: uu,vv,ww,tt,mu,qq,k_cond
        real(rkind) :: uup,vvp,wwp,ttp,mup,qqp,k_cond_p
        real(rkind) :: sigq,sigx,sigy,sigz,sigq_tt,sigq_qq
        real(rkind) :: dzhl,fl2o,fl3o,fl4o,fl5o
        real(rkind) :: muf,k_cond_f,fhat_tmp
        real(rkind) :: xxp,xx,mwmif,mwm,mwmp,yy,yyp
        real(rkind) :: fhat_tmp_et,hh,hhp,dxhl,hhf,yyf
        real(rkind) :: rhodiff,siglsp,rmixtloc
!
        !$cuf kernel do(3) <<<*,*>>>
        do k=0,nz
         do j=1,ny
          do i=1,nx
!
           uu  = w_aux_gpu(i,j,k  ,J_U) 
           uup = w_aux_gpu(i,j,k+1,J_U) 
           vv  = w_aux_gpu(i,j,k  ,J_V) 
           vvp = w_aux_gpu(i,j,k+1,J_V) 
           ww  = w_aux_gpu(i,j,k  ,J_W) 
           wwp = w_aux_gpu(i,j,k+1,J_W) 
           tt  = w_aux_gpu(i,j,k  ,J_T) 
           ttp = w_aux_gpu(i,j,k+1,J_T) 
           mu  = w_aux_gpu(i,j,k  ,J_MU) 
           mup = w_aux_gpu(i,j,k+1,J_MU) 
           k_cond   = w_aux_gpu(i,j,k  ,J_K_COND) 
           k_cond_p = w_aux_gpu(i,j,k+1,J_K_COND) 
           qq  = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
           qqp = 0.5_rkind*(uup*uup+vvp*vvp+wwp*wwp)
!
           sigx    = uup-uu
           sigy    = vvp-vv
           sigz    = wwp-ww
           sigq_tt = ttp-tt
           sigq_qq = qqp-qq
           muf     = mu+mup
           muf     = 0.5_rkind*muf/(z_gpu(k+1)-z_gpu(k))
           k_cond_f = k_cond+k_cond_p
           k_cond_f = 0.5_rkind*k_cond_f/(z_gpu(k+1)-z_gpu(k))
!
           sigx = sigx*muf
           sigy = sigy*muf
           sigz = sigz*muf
           sigq = sigq_tt*k_cond_f+sigq_qq*muf
!
!          mwm   = w_aux_gpu(i,j,k  ,J_MW)
           rmixtloc = get_rmixture(i,j,k  ,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           mwm      = R_univ/rmixtloc
!          mwmp  = w_aux_gpu(i,j,k+1,J_MW)
           rmixtloc = get_rmixture(i,j,k+1,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           mwmp     = R_univ/rmixtloc
!           
           mwmif = 0.5_rkind*(mwmp+mwm)/(mwmp*mwm)
           mwmif = mwmif/(z_gpu(k+1)-z_gpu(k))
           fhat_tmp = 0._rkind
           do msp=1,N_S
            rhodiff = 0.5_rkind*(w_aux_gpu(i,j,k,J_D_START+msp)+w_aux_gpu(i,j,k+1,J_D_START+msp))
            xxp = w_aux_gpu(i,j,k+1,msp)*mwmp
            xx  = w_aux_gpu(i,j,k  ,msp)*mwm
            fhat_tmp = fhat_tmp + (xxp-xx)*rhodiff
           enddo
           fhat_tmp = fhat_tmp*mwmif
!           
           fhat_tmp_et = 0._rkind
           do lsp=1,N_S
            rhodiff = 0.5_rkind*(w_aux_gpu(i,j,k,J_D_START+lsp)+w_aux_gpu(i,j,k+1,J_D_START+lsp))
            yyp   = w_aux_gpu(i,j,k+1,lsp)
            yy    = w_aux_gpu(i,j,k  ,lsp)
            yyf   = 0.5_rkind*(yyp+yy)
            xxp   = yyp*mwmp
            xx    = yy *mwm
            siglsp = -rhodiff*(xxp-xx)*mwmif + fhat_tmp*yyf
            fhat_gpu(i,j,k,lsp) = siglsp
            hh  = get_species_h_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu, &
                               nsetcv,trange_gpu,lsp)
            hhp = get_species_h_from_temperature_dev(ttp,indx_cp_l,indx_cp_r,cp_coeff_gpu,&
                               nsetcv,trange_gpu,lsp)
            hhf = 0.5_rkind*(hh+hhp)
            fhat_tmp_et = fhat_tmp_et+siglsp*hhf
           enddo
           fhat_gpu(i,j,k,I_U) = - sigx
           fhat_gpu(i,j,k,I_V) = - sigy
           fhat_gpu(i,j,k,I_W) = - sigz
           fhat_gpu(i,j,k,I_E) = - sigq + fhat_tmp_et
!
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
!
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do i=1,nx
           dzhl = 2._rkind/(z_gpu(k+1)-z_gpu(k-1))
           do iv=1,nv
            fl_gpu(i,j,k,iv) = fl_gpu(i,j,k,iv) + dzhl*(fhat_gpu(i,j,k,iv)-fhat_gpu(i,j,k-1,iv))
           enddo
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()

    endsubroutine visflx_z_cuf

    subroutine recyc_exchange_cuf_1(irecyc, w_gpu, wbuf1s_gpu, nx, ny, nz, ng)
      integer, intent(in) :: irecyc, nx, ny, nz, ng 
      real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(in), device :: w_gpu
      real(rkind), dimension(:,:,:,:), intent(inout), device :: wbuf1s_gpu
      real(rkind) :: rho
      integer :: i,j,k,m,iercuda,lsp
      !$cuf kernel do(3) <<<*,*>>>
      do k=1,nz
       do j=1,ny
        do i=1,ng
         rho = w_gpu(irecyc+1-i,j,k,1)
         do lsp=2,N_S
          rho = rho + w_gpu(irecyc+1-i,j,k,lsp)
         enddo
         wbuf1s_gpu(i,j,k,1) = rho
         wbuf1s_gpu(i,j,k,2) = w_gpu(irecyc+1-i,j,k,I_U)
         wbuf1s_gpu(i,j,k,3) = w_gpu(irecyc+1-i,j,k,I_V)
         wbuf1s_gpu(i,j,k,4) = w_gpu(irecyc+1-i,j,k,I_W)
         wbuf1s_gpu(i,j,k,5) = w_gpu(irecyc+1-i,j,k,I_E)
        enddo
       enddo
      enddo
      !@cuf iercuda=cudaDeviceSynchronize()
    end subroutine recyc_exchange_cuf_1
    subroutine recyc_exchange_cuf_2(n1_start_recv, n1_start_send, n1_end_recv, wrecyc_gpu, wbuf1r_gpu, nx, ny, nz, ng, nv_recyc)
        integer, intent(in) :: n1_start_recv, n1_start_send, n1_end_recv, nx, ny, nz, ng, nv_recyc
        real(rkind), dimension(:,:,:,:), intent(in), device :: wbuf1r_gpu
        real(rkind), dimension(:,:,:,:), intent(inout), device :: wrecyc_gpu
        integer :: i,j,k,m, iercuda
        !$cuf kernel do(3) <<<*,*>>>
         do k=n1_start_recv,n1_end_recv
          do j=1,ny
           do i=1,ng
            do m=1,nv_recyc
             wrecyc_gpu(i,j,k,m) = wbuf1r_gpu(i,j,k-n1_start_recv+n1_start_send,m)
            enddo
           enddo
          enddo
         enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    end subroutine recyc_exchange_cuf_2
    subroutine recyc_exchange_cuf_3(n2_start_recv, n2_start_send, n2_end_recv, wrecyc_gpu, wbuf2r_gpu, nx, ny, nz, ng, nv_recyc)
        integer, intent(in) :: n2_start_recv, n2_start_send, n2_end_recv, nx, ny, nz, ng, nv_recyc
        real(rkind), dimension(:,:,:,:), intent(in), device :: wbuf2r_gpu
        real(rkind), dimension(:,:,:,:), intent(inout), device :: wrecyc_gpu
        integer :: i,j,k,m, iercuda
        !$cuf kernel do(3) <<<*,*>>>
         do k=n2_start_recv,n2_end_recv
          do j=1,ny
           do i=1,ng
            do m=1,nv_recyc
             wrecyc_gpu(i,j,k,m) = wbuf2r_gpu(i,j,k-n2_start_recv+n2_start_send,m)
            enddo
           enddo
          enddo
         enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    end subroutine recyc_exchange_cuf_3

    subroutine bcextr_sub_cuf(ilat,nx,ny,nz,ng,nv,nv_aux,p0,rgas_gpu,w_gpu,w_aux_gpu, &
                              indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu)

       integer, intent(in) :: ilat,nx,ny,nz,ng,nv,nv_aux,indx_cp_l,indx_cp_r,nsetcv
       real(rkind), intent(in) :: p0
       real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(inout), device :: w_gpu
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
       real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
       real(rkind) :: rho,rhou,rhov,rhow,tt,ee,rmixtloc
       integer :: i,j,k,l,m,lsp,iercuda

       if (ilat==1) then     ! left side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          i = 1 
          rho = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
          do lsp=1,N_S
           w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           i    = 1 
           rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
           rhou = w_gpu(1,j,k,I_U)
           rhov = w_gpu(1,j,k,I_V)
           rhow = w_gpu(1,j,k,I_W)
           rmixtloc = get_rmixture(i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           tt   = p0/rho/rmixtloc
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu, &
                                                     i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           do lsp=1,N_S
            w_gpu(1-l,j,k,lsp) = w_gpu(1,j,k,lsp)
           enddo
           w_gpu(1-l,j,k,I_U) = rhou
           w_gpu(1-l,j,k,I_V) = rhov
           w_gpu(1-l,j,k,I_W) = rhow
           w_gpu(1-l,j,k,I_E) = rho*ee + 0.5_rkind*(rhou**2+rhov**2+rhow**2)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
!
       elseif (ilat==2) then ! right side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          i = nx
          rho = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
          do lsp=1,N_S
           w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           i    = nx
           rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
           rhou = w_gpu(nx,j,k,I_U)
           rhov = w_gpu(nx,j,k,I_V)
           rhow = w_gpu(nx,j,k,I_W)
           rmixtloc = get_rmixture(i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           tt   = p0/rho/rmixtloc
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu, &
                                                     i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           do lsp=1,N_S
            w_gpu(nx+l,j,k,lsp) = w_gpu(nx,j,k,lsp)
           enddo
           w_gpu(nx+l,j,k,I_U) = rhou
           w_gpu(nx+l,j,k,I_V) = rhov
           w_gpu(nx+l,j,k,I_W) = rhow
           w_gpu(nx+l,j,k,I_E) = rho*ee + 0.5_rkind*(rhou**2+rhov**2+rhow**2)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
!       
       elseif (ilat==3) then ! lower side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          j = 1
          rho = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
          do lsp=1,N_S
           w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           j    = 1
           rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
           rhou = w_gpu(i,1,k,I_U)
           rhov = w_gpu(i,1,k,I_V)
           rhow = w_gpu(i,1,k,I_W)
           rmixtloc = get_rmixture(i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           tt   = p0/rho/rmixtloc
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu, &
                                                     i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           do lsp=1,N_S
            w_gpu(i,1-l,k,lsp) = w_gpu(i,1,k,lsp)
           enddo
           w_gpu(i,1-l,k,I_U) = rhou
           w_gpu(i,1-l,k,I_V) = rhov
           w_gpu(i,1-l,k,I_W) = rhow
           w_gpu(i,1-l,k,I_E) = rho*ee + 0.5_rkind*(rhou**2+rhov**2+rhow**2)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
!
       elseif (ilat==4) then  ! upper side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          j = ny
          rho = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
          do lsp=1,N_S
           w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           j    = ny
           rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
           rhou = w_gpu(i,ny,k,I_U)
           rhov = w_gpu(i,ny,k,I_V)
           rhow = w_gpu(i,ny,k,I_W)
           rmixtloc = get_rmixture(i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           tt   = p0/rho/rmixtloc
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv, trange_gpu, &
                                                     i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           do lsp=1,N_S
            w_gpu(i,ny+l,k,lsp) = w_gpu(i,ny,k,lsp)
           enddo
           w_gpu(i,ny+l,k,I_U) = rhou
           w_gpu(i,ny+l,k,I_V) = rhov
           w_gpu(i,ny+l,k,I_W) = rhow
           w_gpu(i,ny+l,k,I_E) = rho*ee + 0.5_rkind*(rhou**2+rhov**2+rhow**2)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
!
       elseif (ilat==5) then  ! back side
       !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          k = 1
          rho = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
          do lsp=1,N_S
           w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(3) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          do l=1,ng
           k    = 1
           rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
           rhou = w_gpu(i,j,1,I_U)
           rhov = w_gpu(i,j,1,I_V)
           rhow = w_gpu(i,j,1,I_W)
           rmixtloc = get_rmixture(i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           tt   = p0/rho/rmixtloc
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu, &
                                                     i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           do lsp=1,N_S
            w_gpu(i,j,1-l,lsp) = w_gpu(i,j,1,lsp)
           enddo
           w_gpu(i,j,1-l,I_U) = rhou
           w_gpu(i,j,1-l,I_V) = rhov
           w_gpu(i,j,1-l,I_W) = rhow
           w_gpu(i,j,1-l,I_E) = rho*ee + 0.5_rkind*(rhou**2+rhov**2+rhow**2)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
!       
       elseif (ilat==6) then  ! fore side
       !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          k = nz
          rho = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
          do lsp=1,N_S
           w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(3) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          do l=1,ng
           k    = nz
           rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
           rhou = w_gpu(i,j,nz,I_U)
           rhov = w_gpu(i,j,nz,I_V)
           rhow = w_gpu(i,j,nz,I_W)
           rmixtloc = get_rmixture(i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           tt   = p0/rho/rmixtloc
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu, &
                                                     i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           do lsp=1,N_S
            w_gpu(i,j,nz+l,lsp) = w_gpu(i,j,nz,lsp)
           enddo
           w_gpu(i,j,nz+l,I_U) = rhou
           w_gpu(i,j,nz+l,I_V) = rhov
           w_gpu(i,j,nz+l,I_W) = rhow
           w_gpu(i,j,nz+l,I_E) = rho*ee + 0.5_rkind*(rhou**2+rhov**2+rhow**2)/rho
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       endif
    endsubroutine bcextr_sub_cuf


    attributes(global) subroutine bc_nr_lat_x_kernel(start_or_end, nr_type, &
                                  nx, ny, nz, ng, nv, w_aux_gpu, w_gpu, fl_gpu, &
                                  dcsidx_gpu, indx_cp_l, indx_cp_r, rgas_gpu, cv_coeff_gpu, &
                                  cp_coeff_gpu, nsetcv, trange_gpu, winf_gpu)
        integer, intent(in), value :: start_or_end, nr_type, nx, ny, nz, ng, nv, indx_cp_l, indx_cp_r, nsetcv
        real(rkind), dimension(3) :: c_one
        real(rkind), dimension(4+N_S) :: dw_dn, dwc_dn, ev, dw_dn_outer, dwc_dn_outer
        real(rkind), dimension(N_S) :: yyvec, prhoi
        real(rkind), dimension(4+N_S,4+N_S) :: el, er
        real(rkind) :: w_target
        integer :: i, j, k, l, m, mm, sgn_dw, lsp
        real(rkind) :: df, uu, vv, ww, h, qq, cc, c, ci, b2, b1
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind), dimension(N_S) :: rgas_gpu
        real(rkind), dimension(:,:,:,:) :: fl_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,:) :: w_aux_gpu, w_gpu
        real(rkind), dimension(:) :: dcsidx_gpu
        real(rkind), dimension(:) :: winf_gpu
        real(rkind) :: rho,tt,pp,rmix_tt,gamloc,p_rho,p_e,etot,b3,eei,gm1loc

        j = blockDim%x * (blockIdx%x - 1) + threadIdx%x 
        k = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        if (j > ny .or. k > nz) return

        ! Setup min or max boundary
        c_one = [-1.5_rkind, 2._rkind, -0.5_rkind]
        if(start_or_end == 1) then
            i      = 1
            sgn_dw = 1
        elseif(start_or_end == 2) then
            i      = nx
            sgn_dw = -1
        endif

        ! Compute d(U_cons)/dx: inner, and outer for relaxation
        do m=1,N_S+4
            dw_dn(m) = 0._rkind
            do l=1,3
                dw_dn(m) = dw_dn(m) + sgn_dw * c_one(l)*w_gpu(i+sgn_dw*(l-1),j,k,m)
            enddo
            ! Relax to w_gpu which if imposed by recycing works. Problems
            ! could arise at the exit, but we do not relax there.
            ! Another possible choice would be relaxing to w_inf.
            if (nr_type == 2) then
             w_target       = w_gpu(i-sgn_dw,j,k,m)
            elseif (nr_type == 3) then
             w_target       = winf_gpu(m)
            endif
            dw_dn_outer(m) = sgn_dw * (w_gpu(i,j,k,m)-w_target)
        enddo

        ! Compute eigenvectors
        do lsp=1,N_S
         yyvec(lsp) = w_aux_gpu(i,j,k,lsp)
        enddo
        rho       = w_aux_gpu(i,j,k,J_R)
        uu        = w_aux_gpu(i,j,k,J_U)
        vv        = w_aux_gpu(i,j,k,J_V)
        ww        = w_aux_gpu(i,j,k,J_W)
        h         = w_aux_gpu(i,j,k,J_H)
        tt        = w_aux_gpu(i,j,k,J_T)
        pp        = w_aux_gpu(i,j,k,J_P)
        c         = w_aux_gpu(i,j,k,J_C)
        qq        = 0.5_rkind * (uu*uu  +vv*vv + ww*ww)
!
        cc        = c*c
        rmix_tt   = pp/rho
        gamloc    = cc/rmix_tt
        gm1loc    = gamloc-1._rkind
        ci        = 1._rkind/c 
!
        p_rho     = rmix_tt
        p_e       = rho*gm1loc
        etot      = h - rmix_tt
!
        b3        = etot
        b2        = p_e/(rho*cc)
        b1        = - b2*(etot - 2._rkind*qq)
!
        do lsp=1,N_S
         eei = get_species_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,lsp)
         prhoi(lsp) = rgas_gpu(lsp)*tt+gm1loc*(etot-qq-eei)
        enddo
!
!       b2        =  gm1/cc  ! 1/(cp*T)
!       b1        =  b2 * qq
!
        call eigenvectors_x(nv, b1, b2, b3, rho, uu, vv, ww, c, ci, h, el, er, yyvec, prhoi)

        ! Pre-multiply to L to get derivative of characteristic variables
        do m=1,N_S+4
            dwc_dn(m)       = 0._rkind
            dwc_dn_outer(m) = 0._rkind
            do mm=1,N_S+4
                dwc_dn(m)       = dwc_dn(m)       + el(mm,m) * dw_dn(mm)
                dwc_dn_outer(m) = dwc_dn_outer(m) + el(mm,m) * dw_dn_outer(mm)
            enddo
        enddo

        ! Compute eigenvalues
        ev(1) = uu-c 
        do lsp=2,N_S+3
         ev(lsp) = uu   
        enddo
        ev(N_S+4) = uu+c 

        ! If nr_type=1, kill acousting ingoing waves
        if (nr_type == 1) then
            do l=1,N_S+4
                ev(l) = sgn_dw*min(sgn_dw*ev(l) ,0._rkind)
            enddo
        endif

        ! If nr_type=2, exiting waves are kept, incoming waves are assigned from outer derivatives
        if (nr_type == 2 .or. nr_type == 3) then
            do m=1,N_S+4
                if(sgn_dw*ev(m) > 0._rkind) then
                    dwc_dn(m) = dwc_dn_outer(m)
                endif
            enddo
        endif

        ! Compute wave amplitude vector (lambda*L*dw)
        do m=1,N_S+4
            dwc_dn(m) = ev(m) * dwc_dn(m)
        enddo

        ! If nr_type=6, enforce wave reflection to simulate purely reflective wall (LODI relations)
        if (nr_type == 6) then
            do lsp=2,N_S+3
             dwc_dn(lsp) = 0._rkind
            enddo
            if (start_or_end == 1) then
                dwc_dn(N_S+4) = dwc_dn(1) ! exiting wave value is used to impose entering wave
            elseif(start_or_end == 2) then
                dwc_dn(1) = dwc_dn(N_S+4) ! exiting wave value is used to impose entering wave
            endif
        endif

        ! Pre-multiply to R to return to conservative variables and assign result to fl
        do m=1,N_S+4
            df = 0._rkind
            do mm=1,N_S+4
                df = df + er(mm,m) * dwc_dn(mm)
            enddo
            fl_gpu(i,j,k,m) = fl_gpu(i,j,k,m) + df * dcsidx_gpu(i)
        enddo
    endsubroutine bc_nr_lat_x_kernel

    attributes(global) subroutine bc_nr_lat_y_kernel(start_or_end, nr_type, &
                                  nx, ny, nz, ng, nv, w_aux_gpu, w_gpu, fl_gpu, detady_gpu, &
                                  indx_cp_l, indx_cp_r, rgas_gpu, cv_coeff_gpu, &
                                  cp_coeff_gpu, nsetcv, trange_gpu, winf_gpu)
        integer, intent(in), value :: start_or_end, nr_type, nx, ny, nz, ng, nv, indx_cp_l, indx_cp_r, nsetcv
        real(rkind), dimension(3) :: c_one
        real(rkind), dimension(4+N_S) :: dw_dn, dwc_dn, ev, dw_dn_outer, dwc_dn_outer
        real(rkind), dimension(N_S) :: yyvec, prhoi
        real(rkind), dimension(4+N_S,4+N_S) :: el, er
        real(rkind) :: w_target
        integer :: i, j, k, l, m, mm, sgn_dw, lsp
        real(rkind) :: df, uu, vv, ww, h, qq, cc, c, ci, b2, b1
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind), dimension(N_S) :: rgas_gpu
        real(rkind), dimension(:,:,:,:) :: fl_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:) :: w_aux_gpu, w_gpu
        real(rkind), dimension(:) :: detady_gpu
        real(rkind), dimension(:) :: winf_gpu
        real(rkind) :: rho,tt,pp,rmix_tt,gamloc,p_rho,p_e,etot,b3,eei,gm1loc

        i = blockDim%x * (blockIdx%x - 1) + threadIdx%x 
        k = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        if (i > nx .or. k > nz) return

        ! Setup min or max boundary
        c_one = [-1.5_rkind, 2._rkind, -0.5_rkind]
        if(start_or_end == 1) then
            j      = 1
            sgn_dw = 1
        elseif(start_or_end == 2) then
            j      = ny
            sgn_dw = -1
        endif

        ! Compute d(U_cons)/dx
        do m=1,N_S+4
            dw_dn(m) = 0._rkind
            do l=1,3
                dw_dn(m) = dw_dn(m) + sgn_dw * c_one(l)*w_gpu(i,j+sgn_dw*(l-1),k,m)
            enddo
            ! Relax to w_gpu which if imposed by recycing works. Problems
            ! could arise at the exit, but we do not relax there.
            ! Another possible choice would be relaxing to w_inf.
            if (nr_type == 2) then
               w_target = w_gpu(i,j-sgn_dw,k,m)
            elseif(nr_type == 3) then
               w_target  = winf_gpu(m)
            endif
            dw_dn_outer(m) = sgn_dw * (w_gpu(i,j,k,m)-w_target)
        enddo

        ! Compute eigenvectors
        do lsp=1,N_S
         yyvec(lsp) = w_aux_gpu(i,j,k,lsp)
        enddo
        rho       =  w_aux_gpu(i,j,k,J_R)
        uu        =  w_aux_gpu(i,j,k,J_U)
        vv        =  w_aux_gpu(i,j,k,J_V)
        ww        =  w_aux_gpu(i,j,k,J_W)
        h         =  w_aux_gpu(i,j,k,J_H)
        tt        =  w_aux_gpu(i,j,k,J_T)
        pp        =  w_aux_gpu(i,j,k,J_P)
        c         =  w_aux_gpu(i,j,k,J_C)
        qq        =  0.5_rkind * (uu*uu  +vv*vv + ww*ww)
!
        cc        = c*c
        rmix_tt   = pp/rho
        gamloc    = cc/rmix_tt
        gm1loc    = gamloc-1._rkind
        ci        = 1._rkind/c        
!
        p_rho     = rmix_tt
        p_e       = rho*gm1loc
        etot      = h - rmix_tt
!
        b3        = etot
        b2        = p_e/(rho*cc)
        b1        = - b2*(etot - 2._rkind*qq)
!
        do lsp=1,N_S
         eei = get_species_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,lsp)
         prhoi(lsp) = rgas_gpu(lsp)*tt+gm1loc*(etot-qq-eei)
        enddo
!
!       b2        =  gm1/cc  ! 1/(cp*T)
!       b1        =  b2 * qq
!
        call eigenvectors_y(nv, b1, b2, b3, rho, uu, vv, ww, c, ci, h, el, er, yyvec, prhoi)

        ! Pre-multiply to L to get derivative of characteristic variables
        do m=1,N_S+4
            dwc_dn(m) = 0._rkind
            dwc_dn_outer(m) = 0._rkind
            do mm=1,N_S+4
                dwc_dn(m) = dwc_dn(m) + el(mm,m) * dw_dn(mm)
                dwc_dn_outer(m) = dwc_dn_outer(m) + el(mm,m) * dw_dn_outer(mm)
            enddo
        enddo

        ! Compute eigenvalues
        ev(1) = vv-c 
        do lsp=2,N_S+3
         ev(lsp) = vv   
        enddo
        ev(N_S+4) = vv+c 

        ! If nr_type=1, kill acousting ingoing waves
        if(nr_type == 1) then
            if(start_or_end == 1) then
                do l=1,N_S+4
                    ev(l) = min(ev(l) ,0._rkind)
                enddo
            elseif(start_or_end == 2) then
                do l=1,N_S+4
                    ev(l) = max(ev(l) ,0._rkind)
                enddo
            endif
        endif

        ! If nr_type=2, exiting waves are kept, incoming waves are assigned from outer derivatives
        if (nr_type == 2 .or. nr_type == 3) then
            do m=1,N_S+4
                if(sgn_dw*ev(m) > 0._rkind) then
                    dwc_dn(m) = dwc_dn_outer(m)
                endif
            enddo
        endif

        ! Compute wave amplitude vector (lambda*L*dw)
        do m=1,N_S+4
            dwc_dn(m) = ev(m) * dwc_dn(m)
        enddo

        ! If nr_type=6, enforce wave reflection to simulate purely reflective wall (LODI relations)
        if (nr_type == 6) then
             do lsp=2,N_S+3
              dwc_dn(lsp) = 0._rkind
             enddo
            if(start_or_end == 1) then
                dwc_dn(N_S+4) = dwc_dn(1) ! exiting wave value is used to impose entering wave
            elseif(start_or_end == 2) then
                dwc_dn(1) = dwc_dn(N_S+4) ! exiting wave value is used to impose entering wave
            endif
        endif

        ! Pre-multiply to R to return to conservative variables and assign result to fl
        do m=1,N_S+4
            df = 0._rkind
            do mm=1,N_S+4
                df = df + er(mm,m) * dwc_dn(mm)
            enddo
            fl_gpu(i,j,k,m) = fl_gpu(i,j,k,m) + df * detady_gpu(j)
        enddo
    endsubroutine bc_nr_lat_y_kernel

    attributes(global) subroutine bc_nr_lat_z_kernel(start_or_end, nr_type, &
                                  nx, ny, nz, ng, nv, w_aux_gpu, w_gpu, fl_gpu, dzitdz_gpu, &
                                  indx_cp_l, indx_cp_r, rgas_gpu, cv_coeff_gpu, &
                                  cp_coeff_gpu, nsetcv, trange_gpu, winf_gpu)
        integer, intent(in), value :: start_or_end, nr_type, nx, ny, nz, ng, nv, indx_cp_l, indx_cp_r, nsetcv
        real(rkind), dimension(3) :: c_one
        real(rkind), dimension(4+N_S) :: dw_dn, dwc_dn, ev, dw_dn_outer, dwc_dn_outer
        real(rkind), dimension(N_S) :: yyvec, prhoi
        real(rkind), dimension(4+N_S,4+N_S) :: el, er
        real(rkind) :: w_target
        integer :: i, j, k, l, m, mm, sgn_dw, lsp
        real(rkind) :: df, uu, vv, ww, h, qq, cc, c, ci, b2, b1
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind), dimension(N_S) :: rgas_gpu
        real(rkind), dimension(:,:,:,:) :: fl_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:) :: w_aux_gpu, w_gpu
        real(rkind), dimension(:) :: dzitdz_gpu
        real(rkind), dimension(:) :: winf_gpu
        real(rkind) :: rho,tt,pp,rmix_tt,gamloc,p_rho,p_e,etot,b3,eei,gm1loc

        i = blockDim%x * (blockIdx%x - 1) + threadIdx%x 
        j = blockDim%y * (blockIdx%y - 1) + threadIdx%y
        if (i > nx .or. j > ny) return

        ! Setup min or max boundary
        c_one = [-1.5_rkind, 2._rkind, -0.5_rkind]
        if(start_or_end == 1) then
            k      = 1
            sgn_dw = 1
        elseif(start_or_end == 2) then
            k      = nz
            sgn_dw = -1
        endif

        ! Compute d(U_cons)/dx
        do m=1,N_S+4
            dw_dn(m) = 0._rkind
            do l=1,3
                dw_dn(m) = dw_dn(m) + sgn_dw * c_one(l)*w_gpu(i,j,k+sgn_dw*(l-1),m)
            enddo
            ! Relax to w_gpu which if imposed by recycing works. Problems
            ! could arise at the exit, but we do not relax there.
            ! Another possible choice would be relaxing to w_inf.
            if (nr_type == 2) then
                w_target = w_gpu(i,j,k-sgn_dw,m)
            elseif(nr_type == 3) then
               w_target  = winf_gpu(m)
            endif
            dw_dn_outer(m) = sgn_dw * (w_gpu(i,j,k,m)-w_target)
        enddo

        ! Compute eigenvectors
        do lsp=1,N_S
         yyvec(lsp) = w_aux_gpu(i,j,k,lsp)
        enddo
        rho       =  w_aux_gpu(i,j,k,J_R)
        uu        =  w_aux_gpu(i,j,k,J_U)
        vv        =  w_aux_gpu(i,j,k,J_V)
        ww        =  w_aux_gpu(i,j,k,J_W)
        h         =  w_aux_gpu(i,j,k,J_H)
        tt        =  w_aux_gpu(i,j,k,J_T)
        pp        =  w_aux_gpu(i,j,k,J_P)
        c         =  w_aux_gpu(i,j,k,J_C)
        qq        =  0.5_rkind * (uu*uu  +vv*vv + ww*ww)
!
        cc        = c*c
        rmix_tt   = pp/rho
        gamloc    = cc/rmix_tt
        gm1loc    = gamloc-1._rkind
        ci        = 1._rkind/c
!
        p_rho     = rmix_tt
        p_e       = rho*gm1loc
        etot      = h - rmix_tt
!
        b3        = etot
        b2        = p_e/(rho*cc)
        b1        = - b2*(etot - 2._rkind*qq)
!
        do lsp=1,N_S
         eei = get_species_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,lsp)
         prhoi(lsp) = rgas_gpu(lsp)*tt+gm1loc*(etot-qq-eei)
        enddo
!
!       b2        =  gm1/cc  ! 1/(cp*T)
!       b1        =  b2 * qq
!
        call eigenvectors_z(nv, b1, b2, b3, rho, uu, vv, ww, c, ci, h, el, er, yyvec, prhoi)
!
        ! Pre-multiply to L to get derivative of characteristic variables
        do m=1,N_S+4
            dwc_dn(m) = 0._rkind
            dwc_dn_outer(m) = 0._rkind
            do mm=1,N_S+4
                dwc_dn(m) = dwc_dn(m) + el(mm,m) * dw_dn(mm)
                dwc_dn_outer(m) = dwc_dn_outer(m) + el(mm,m) * dw_dn_outer(mm)
            enddo
        enddo

        ! Compute eigenvalues
        ev(1) = ww-c 
        do lsp=2,N_S+3
         ev(lsp) = ww   
        enddo
        ev(N_S+4) = ww+c 

        ! If nr_type=1, kill acousting ingoing waves
        if(nr_type == 1) then
            if(start_or_end == 1) then
                do l=1,N_S+4
                    ev(l) = min(ev(l) ,0._rkind)
                enddo
            elseif(start_or_end == 2) then
                do l=1,N_S+4
                    ev(l) = max(ev(l) ,0._rkind)
                enddo
            endif
        endif

        ! If nr_type=2, exiting waves are kept, incoming waves are assigned from outer derivatives
        if (nr_type == 2 .or. nr_type == 3) then
            do m=1,N_S+4
                if(sgn_dw*ev(m) > 0._rkind) then
                    dwc_dn(m) = dwc_dn_outer(m)
                endif
            enddo
        endif

        ! Compute wave amplitude vector (lambda*L*dw)
        do m=1,N_S+4
            dwc_dn(m) = ev(m) * dwc_dn(m)
        enddo

        ! If nr_type=6, enforce wave reflection to simulate purely reflective wall (LODI relations)
        if (nr_type == 6) then
            do lsp=2,N_S+3
             dwc_dn(lsp) = 0._rkind
            enddo
            if(start_or_end == 1) then
                dwc_dn(N_S+4) = dwc_dn(1) ! exiting wave value is used to impose entering wave
            elseif(start_or_end == 2) then
                dwc_dn(1) = dwc_dn(N_S+4) ! exiting wave value is used to impose entering wave
            endif
        endif

        ! Pre-multiply to R to return to conservative variables and assign result to fl
        do m=1,N_S+4
            df = 0._rkind
            do mm=1,N_S+4
                df = df + er(mm,m) * dwc_dn(mm)
            enddo
            fl_gpu(i,j,k,m) = fl_gpu(i,j,k,m) + df * dzitdz_gpu(k)
        enddo
    endsubroutine bc_nr_lat_z_kernel

    subroutine  bcrecyc_cuf_1(nx, ny, nz, ng, nv_recyc, wrecycav_gpu, wrecyc_gpu)
        integer, intent(in) :: nx, ny, nz, ng, nv_recyc
        real(rkind), dimension(:,:,:), intent(inout), device :: wrecycav_gpu
        real(rkind), dimension(:,:,:,:), intent(in), device :: wrecyc_gpu
        integer :: i,j,k,m,iercuda
        !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,ng
          do m=1,nv_recyc
           wrecycav_gpu(i,j,m) = 0._rkind
           do k=1,nz
            wrecycav_gpu(i,j,m) = wrecycav_gpu(i,j,m)+wrecyc_gpu(i,j,k,m)
           enddo
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    end subroutine  bcrecyc_cuf_1

    subroutine  bcrecyc_cuf_2(nx, ny, nz, nzmax, ng, wrecycav_gpu, wrecyc_gpu)
        integer, intent(in) :: nx, ny, nz, nzmax, ng
        real(rkind), dimension(:,:,:), intent(in), device :: wrecycav_gpu
        real(rkind), dimension(:,:,:,:), intent(inout), device :: wrecyc_gpu
        real(rkind) :: ufav, vfav, wfav, rhom, rhofav, rho, ri
        integer :: i,j,k,iercuda,lsp

        !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,ng
          ufav = wrecycav_gpu(i,j,2)/wrecycav_gpu(i,j,1)
          vfav = wrecycav_gpu(i,j,3)/wrecycav_gpu(i,j,1)
          wfav = wrecycav_gpu(i,j,4)/wrecycav_gpu(i,j,1)
          rhom = wrecycav_gpu(i,j,1)/nzmax
          do k=1,nz
           wrecyc_gpu(i,j,k,2) = wrecyc_gpu(i,j,k,2)/wrecyc_gpu(i,j,k,1)-ufav ! Velocity fluctuations
           wrecyc_gpu(i,j,k,3) = wrecyc_gpu(i,j,k,3)/wrecyc_gpu(i,j,k,1)-vfav
           wrecyc_gpu(i,j,k,4) = wrecyc_gpu(i,j,k,4)/wrecyc_gpu(i,j,k,1)-wfav
           wrecyc_gpu(i,j,k,1) = wrecyc_gpu(i,j,k,1)-rhom                     ! Density fluctuations
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()

    end subroutine  bcrecyc_cuf_2

    subroutine bcrecyc_cuf_3(nx,ny,nz,nv,nv_aux,ng,p0,u0,rmixt0,w_gpu,w_aux_gpu,wmean_gpu,wrecyc_gpu, &
                             weta_inflow_gpu,map_j_inn_gpu,map_j_out_gpu,map_j_out_blend_gpu, &
                             yplus_inflow_gpu,eta_inflow_gpu,yplus_recyc_gpu,eta_recyc_gpu, &
                             eta_recyc_blend_gpu,betarecyc,glund1,inflow_random_plane_gpu, &
                             indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,rand_type)

     integer, intent(in) :: nx, ny, nz, nv, nv_aux, ng, indx_cp_l, indx_cp_r, nsetcv, rand_type
     real(rkind) :: p0, rmixt0, betarecyc, glund1, u0
     real(rkind), dimension(indx_cp_l:indx_cp_r+1), intent(in), device :: cv_coeff_gpu
     real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
     real(rkind), dimension(1-ng:,:,:), intent(in), device :: wmean_gpu
     real(rkind), dimension(:,:,:,:), intent(in), device :: wrecyc_gpu
     real(rkind), dimension(:,:,:), intent(in), device :: inflow_random_plane_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(inout), device :: w_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
     real(rkind), dimension(:), intent(in), device :: weta_inflow_gpu
     real(rkind), dimension(1-ng:), intent(in), device :: yplus_inflow_gpu, eta_inflow_gpu, yplus_recyc_gpu, eta_recyc_gpu
     real(rkind), dimension(1-ng:), intent(in), device :: eta_recyc_blend_gpu
     integer, dimension(:), intent(in), device :: map_j_inn_gpu, map_j_out_gpu, map_j_out_blend_gpu
     integer :: i,j,k,iercuda,lsp
     real(rkind) :: eta, weta, weta1, bdamp, disty_inn, disty_out, rhofluc, ufluc, vfluc, wfluc, rhof_inn, rhof_out
     real(rkind) :: uf_inn, uf_out, vf_inn, vf_out, wf_inn, wf_out
     real(rkind) :: rhomean, uumean , vvmean , wwmean , tmean  , rho , uu , vv , ww , rhou , rhov , rhow, tt, ee
     real(rkind) :: u0_05
     integer :: j_inn, j_out

     u0_05 = 0.05_rkind*u0
     if (rand_type==0) u0_05 = 0._rkind

     !$cuf kernel do(2) <<<*,*>>>
     do k=1,nz
      do j=1,ny
       do i=1,ng
        rho = get_rho_from_w_dev(1-i,j,k,nv,nx,ny,nz,ng,w_gpu)
        do lsp=1,N_S
         w_aux_gpu(1-i,j,k,lsp) = w_gpu(1-i,j,k,lsp)/rho
        enddo
       enddo
      enddo
     enddo
     !@cuf iercuda=cudaDeviceSynchronize()

     !$cuf kernel do(2) <<<*,*>>>
      do k=1,nz
       do j=1,ny
        eta   = eta_inflow_gpu(j)
        weta  = weta_inflow_gpu(j)
        weta1 = 1._rkind-weta
        bdamp = 0.5_rkind*(1._rkind-tanh(4._rkind*(eta_inflow_gpu(j)-2._rkind)))
        j_inn = map_j_inn_gpu(j)
        j_out = map_j_out_gpu(j)
        disty_inn = (yplus_inflow_gpu(j)-yplus_recyc_gpu(j_inn))/(yplus_recyc_gpu(j_inn+1)-yplus_recyc_gpu(j_inn))
        disty_out = (eta_inflow_gpu(j)-eta_recyc_gpu(j_out))/(eta_recyc_gpu(j_out+1)-eta_recyc_gpu(j_out))

        do i=1,ng

         if (j==1.or.j_inn>=ny.or.j_out>=ny) then
          rhofluc = 0._rkind
          ufluc   = 0._rkind
          vfluc   = 0._rkind
          wfluc   = 0._rkind
         else
          rhof_inn = wrecyc_gpu(i,j_inn,k,1)*(1._rkind-disty_inn)+wrecyc_gpu(i,j_inn+1,k,1)*disty_inn
          rhof_out = wrecyc_gpu(i,j_out,k,1)*(1._rkind-disty_out)+wrecyc_gpu(i,j_out+1,k,1)*disty_out
          uf_inn   = wrecyc_gpu(i,j_inn,k,2)*(1._rkind-disty_inn)+wrecyc_gpu(i,j_inn+1,k,2)*disty_inn
          uf_out   = wrecyc_gpu(i,j_out,k,2)*(1._rkind-disty_out)+wrecyc_gpu(i,j_out+1,k,2)*disty_out
          vf_inn   = wrecyc_gpu(i,j_inn,k,3)*(1._rkind-disty_inn)+wrecyc_gpu(i,j_inn+1,k,3)*disty_inn
          vf_out   = wrecyc_gpu(i,j_out,k,3)*(1._rkind-disty_out)+wrecyc_gpu(i,j_out+1,k,3)*disty_out
          wf_inn   = wrecyc_gpu(i,j_inn,k,4)*(1._rkind-disty_inn)+wrecyc_gpu(i,j_inn+1,k,4)*disty_inn
          wf_out   = wrecyc_gpu(i,j_out,k,4)*(1._rkind-disty_out)+wrecyc_gpu(i,j_out+1,k,4)*disty_out
     !
          rhofluc = rhof_inn*weta1+rhof_out*weta
          ufluc   =   uf_inn*weta1+  uf_out*weta
          vfluc   =   vf_inn*weta1+  vf_out*weta
          wfluc   =   wf_inn*weta1+  wf_out*weta
          rhofluc = rhofluc*bdamp
          ufluc   = ufluc  *bdamp*betarecyc
          vfluc   = vfluc  *bdamp*betarecyc
          wfluc   = wfluc  *bdamp*betarecyc
          ufluc   = ufluc+u0_05*(inflow_random_plane_gpu(j,k,1)-0.5_rkind)*eta
          vfluc   = vfluc+u0_05*(inflow_random_plane_gpu(j,k,2)-0.5_rkind)*eta
          wfluc   = wfluc+u0_05*(inflow_random_plane_gpu(j,k,3)-0.5_rkind)*eta
         endif

         rhomean = wmean_gpu(1-i,j,1)
         do lsp=2,N_S
          rhomean = rhomean + wmean_gpu(1-i,j,lsp)
         enddo
         uumean  = wmean_gpu(1-i,j,I_U)/rhomean
         vvmean  = wmean_gpu(1-i,j,I_V)/rhomean
         wwmean  = wmean_gpu(1-i,j,I_W)/rhomean
         tmean   = p0/rhomean/rmixt0
         rho     = rhomean + rhofluc
         uu      = uumean  + ufluc
         vv      = vvmean  + vfluc
         ww      = wwmean  + wfluc
         rhou    = rho*uu
         rhov    = rho*vv
         rhow    = rho*ww
         
         do lsp=1,N_S
          w_gpu(1-i,j,k,lsp) = rho*wmean_gpu(1-i,j,lsp)/rhomean
         enddo
         w_gpu(1-i,j,k,I_U) = rhou
         w_gpu(1-i,j,k,I_V) = rhov
         w_gpu(1-i,j,k,I_W) = rhow
         tt                 = p0/rho/rmixt0
         ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                    nsetcv,trange_gpu,1-i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
         w_gpu(1-i,j,k,I_E) = rho*ee + 0.5_rkind*(rhou**2+rhov**2+rhow**2)/rho
        enddo
       enddo
      enddo
      !@cuf iercuda=cudaDeviceSynchronize()
    end subroutine bcrecyc_cuf_3

    subroutine bcrecyc_doublebl_cuf_3(jmin,jmax,nx,ny,nz,nv,nv_aux,ng,p0,u0,rmixt0,w_gpu,w_aux_gpu,wmean_gpu,wrecyc_gpu, &
                             weta_inflow_gpu,map_j_inn_gpu,map_j_out_gpu,map_j_out_blend_gpu, &
                             yplus_inflow_gpu,eta_inflow_gpu,yplus_recyc_gpu,eta_recyc_gpu, &
                             eta_recyc_blend_gpu,betarecyc,glund1,inflow_random_plane_gpu, &
                             indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,rand_type)

     integer, intent(in) :: jmin, jmax, nx, ny, nz, nv, nv_aux, ng, indx_cp_l, indx_cp_r, nsetcv, rand_type
     real(rkind) :: p0, rmixt0, betarecyc, glund1, u0
     real(rkind), dimension(indx_cp_l:indx_cp_r+1), intent(in), device :: cv_coeff_gpu
     real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
     real(rkind), dimension(1-ng:,:,:), intent(in), device :: wmean_gpu
     real(rkind), dimension(:,:,:,:), intent(in), device :: wrecyc_gpu
     real(rkind), dimension(:,:,:), intent(in), device :: inflow_random_plane_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(inout), device :: w_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
     real(rkind), dimension(:), intent(in), device :: weta_inflow_gpu
     real(rkind), dimension(1-ng:), intent(in), device :: yplus_inflow_gpu, eta_inflow_gpu, yplus_recyc_gpu, eta_recyc_gpu
     real(rkind), dimension(1-ng:), intent(in), device :: eta_recyc_blend_gpu
     integer, dimension(:), intent(in), device :: map_j_inn_gpu, map_j_out_gpu, map_j_out_blend_gpu
     integer :: i,j,k,iercuda,lsp
     real(rkind) :: eta, weta, weta1, bdamp, disty_inn, disty_out, rhofluc, ufluc, vfluc, wfluc, rhof_inn, rhof_out
     real(rkind) :: uf_inn, uf_out, vf_inn, vf_out, wf_inn, wf_out
     real(rkind) :: rhomean, uumean , vvmean , wwmean , tmean  , rho , uu , vv , ww , rhou , rhov , rhow, tt, ee
     real(rkind) :: u0_05
     integer :: j_inn, j_out, jj

     u0_05 = 0.05_rkind*u0
     if (rand_type==0) u0_05 = 0._rkind

     !$cuf kernel do(2) <<<*,*>>>
     do k=1,nz
      do j=jmin,jmax
       do i=1,ng
        rho = get_rho_from_w_dev(1-i,j,k,nv,nx,ny,nz,ng,w_gpu)
        do lsp=1,N_S
         w_aux_gpu(1-i,j,k,lsp) = w_gpu(1-i,j,k,lsp)/rho
        enddo
       enddo
      enddo
     enddo
     !@cuf iercuda=cudaDeviceSynchronize()

     !$cuf kernel do(2) <<<*,*>>>
      do k=1,nz
       do j=jmin,jmax
        jj = min(j,ny-j+1)
        eta   = eta_inflow_gpu(jj)
        weta  = weta_inflow_gpu(jj)
        weta1 = 1._rkind-weta
        bdamp = 0.5_rkind*(1._rkind-tanh(4._rkind*(eta_inflow_gpu(jj)-2._rkind)))
        j_inn = map_j_inn_gpu(jj)
        j_out = map_j_out_gpu(jj)
        disty_inn = (yplus_inflow_gpu(jj)-yplus_recyc_gpu(j_inn))/(yplus_recyc_gpu(j_inn+1)-yplus_recyc_gpu(j_inn))
        disty_out = (eta_inflow_gpu(jj)-eta_recyc_gpu(j_out))/(eta_recyc_gpu(j_out+1)-eta_recyc_gpu(j_out))

        do i=1,ng

         if (jj==1.or.j_inn>=ny.or.j_out>=ny) then
          rhofluc = 0._rkind
          ufluc   = 0._rkind
          vfluc   = 0._rkind
          wfluc   = 0._rkind
         else
          rhof_inn = wrecyc_gpu(i,j_inn,k,1)*(1._rkind-disty_inn)+wrecyc_gpu(i,j_inn+1,k,1)*disty_inn
          rhof_out = wrecyc_gpu(i,j_out,k,1)*(1._rkind-disty_out)+wrecyc_gpu(i,j_out+1,k,1)*disty_out
          uf_inn   = wrecyc_gpu(i,j_inn,k,2)*(1._rkind-disty_inn)+wrecyc_gpu(i,j_inn+1,k,2)*disty_inn
          uf_out   = wrecyc_gpu(i,j_out,k,2)*(1._rkind-disty_out)+wrecyc_gpu(i,j_out+1,k,2)*disty_out
          vf_inn   = wrecyc_gpu(i,j_inn,k,3)*(1._rkind-disty_inn)+wrecyc_gpu(i,j_inn+1,k,3)*disty_inn
          vf_out   = wrecyc_gpu(i,j_out,k,3)*(1._rkind-disty_out)+wrecyc_gpu(i,j_out+1,k,3)*disty_out
          wf_inn   = wrecyc_gpu(i,j_inn,k,4)*(1._rkind-disty_inn)+wrecyc_gpu(i,j_inn+1,k,4)*disty_inn
          wf_out   = wrecyc_gpu(i,j_out,k,4)*(1._rkind-disty_out)+wrecyc_gpu(i,j_out+1,k,4)*disty_out
     !
          rhofluc = rhof_inn*weta1+rhof_out*weta
          ufluc   =   uf_inn*weta1+  uf_out*weta
          vfluc   =   vf_inn*weta1+  vf_out*weta
          wfluc   =   wf_inn*weta1+  wf_out*weta
          rhofluc = rhofluc*bdamp
          ufluc   = ufluc  *bdamp*betarecyc
          vfluc   = vfluc  *bdamp*betarecyc
          wfluc   = wfluc  *bdamp*betarecyc
          ufluc   = ufluc+u0_05*(inflow_random_plane_gpu(jj,k,1)-0.5_rkind)*eta
          vfluc   = vfluc+u0_05*(inflow_random_plane_gpu(jj,k,2)-0.5_rkind)*eta
          wfluc   = wfluc+u0_05*(inflow_random_plane_gpu(jj,k,3)-0.5_rkind)*eta
         endif

         rhomean = wmean_gpu(1-i,j,1)
         do lsp=2,N_S
          rhomean = rhomean + wmean_gpu(1-i,j,lsp)
         enddo
         uumean  = wmean_gpu(1-i,j,I_U)/rhomean
         vvmean  = wmean_gpu(1-i,j,I_V)/rhomean
         wwmean  = wmean_gpu(1-i,j,I_W)/rhomean
         tmean   = p0/rhomean/rmixt0
         rho     = rhomean + rhofluc
         uu      = uumean  + ufluc
         vv      = vvmean  + vfluc
         ww      = wwmean  + wfluc
         rhou    = rho*uu
         rhov    = rho*vv
         rhow    = rho*ww
         
         do lsp=1,N_S
          w_gpu(1-i,j,k,lsp) = rho*wmean_gpu(1-i,j,lsp)/rhomean
         enddo
         w_gpu(1-i,j,k,I_U) = rhou
         w_gpu(1-i,j,k,I_V) = rhov
         w_gpu(1-i,j,k,I_W) = rhow
         tt                 = p0/rho/rmixt0
         ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                    nsetcv,trange_gpu,1-i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
         w_gpu(1-i,j,k,I_E) = rho*ee + 0.5_rkind*(rhou**2+rhov**2+rhow**2)/rho
        enddo
       enddo
      enddo
      !@cuf iercuda=cudaDeviceSynchronize()
    end subroutine bcrecyc_doublebl_cuf_3

    subroutine bcfree_cuf(ilat, nx, ny, nz, ng, nv, winf_gpu, w_gpu)
       integer, intent(in) :: ilat,nx,ny,nz,ng,nv
       real(rkind), dimension(1:), intent(in), device :: winf_gpu
       real(rkind), dimension(1-ng:, 1-ng:, 1-ng:, 1:), intent(inout), device :: w_gpu
       real(rkind) :: app
       integer :: j,k,l,m,iercuda

       if (ilat==1) then     ! left side
       !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           do m=1,nv
            w_gpu(1-l,j,k,m) = winf_gpu(m)
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==2) then ! right side
       elseif (ilat==3) then ! lower side
       elseif (ilat==4) then  ! upper side
       elseif (ilat==5) then  ! back side
       elseif (ilat==6) then  ! fore side
       endif
    endsubroutine bcfree_cuf

    subroutine bclam_cuf(ilat, nx, ny, nz, ng, nv, nv_aux, w_gpu, w_aux_gpu, wmean_gpu, p0, rmixt0, &
               indx_cp_l, indx_cp_r, cv_coeff_gpu, nsetcv, trange_gpu)

       integer, intent(in) :: nx, ny, nz, ng, nv, nv_aux, indx_cp_l, indx_cp_r, nsetcv
       real(rkind), intent(in) :: p0, rmixt0
       real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
       real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv), intent(inout), device :: w_gpu
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
       real(rkind), dimension(1-ng:nx+ng+1,1:ny,1:nv), intent(in), device :: wmean_gpu
             
       integer :: ilat,lsp,m
       integer :: j,k,l,iercuda
       real(rkind) :: rho,rhou,rhov,rhow,tt,ee

       if (ilat==1) then     ! left side
       !!$cuf kernel do(2) <<<*,*>>>
       ! do k=1,nz
       !  do j=1,ny
       !   do l=0,ng
       !    rho = get_rho_from_w_dev(1-l,j,k,nv,nx,ny,nz,ng,w_gpu)
       !    do lsp=1,N_S
       !     w_aux_gpu(1-l,j,k,lsp) = w_gpu(1-l,j,k,lsp)/rho
       !    enddo
       !   enddo
       !  enddo
       ! enddo
       !!@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=0,ng
           do m=1,nv
            w_gpu(1-l,j,k,m) = wmean_gpu(1-l,j,m)
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==2) then ! right side
       elseif (ilat==3) then ! lower side
       elseif (ilat==4) then  ! upper side
       elseif (ilat==5) then  ! back side
       elseif (ilat==6) then  ! fore side
       endif
    endsubroutine bclam_cuf

    subroutine bcextr_var_cuf(nx, ny, nz, ng, w_var_gpu)
       integer :: nx,ny,nz,ng
       integer :: ilat
       integer :: i,j,k,l,m,iercuda
       real(rkind), dimension(1-ng:nx+ng, 1-ng:ny+ng, 1-ng:nz+ng, 1:1), intent(inout), device :: w_var_gpu

       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           w_var_gpu(1-l,j,k,1) = w_var_gpu(1,j,k,1)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           w_var_gpu(nx+l,j,k,1) = w_var_gpu(nx,j,k,1)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           w_var_gpu(i,1-l,k,1) = w_var_gpu(i,1,k,1)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           w_var_gpu(i,ny+l,k,1) = w_var_gpu(i,ny,k,1)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          do l=1,ng
           w_var_gpu(i,j,1-l,1) = w_var_gpu(i,j,1,1)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          do l=1,ng
           w_var_gpu(i,j,nz+l,1) = w_var_gpu(i,j,nz,1)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine bcextr_var_cuf

    subroutine bcextr_cuf(ilat, nx, ny, nz, ng, nv, w_gpu)
       integer :: nx,ny,nz,ng,nv
       integer :: ilat
       integer :: i,j,k,l,m,iercuda
       real(rkind), dimension(1-ng:, 1-ng:, 1-ng:, 1:), intent(inout), device :: w_gpu

       if (ilat==1) then     ! left side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           do m=1,nv
            w_gpu(1-l,j,k,m) = w_gpu(1,j,k,m)
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==2) then ! right side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           do m=1,nv
            w_gpu(nx+l,j,k,m) = w_gpu(nx,j,k,m)
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==3) then ! lower side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           do m=1,nv
            w_gpu(i,1-l,k,m) = w_gpu(i,1,k,m)
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==4) then  ! upper side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           do m=1,nv
            w_gpu(i,ny+l,k,m) = w_gpu(i,ny,k,m)
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==5) then  ! back side
       !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          do l=1,ng
           do m=1,nv
            w_gpu(i,j,1-l,m) = w_gpu(i,j,1,m)
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==6) then  ! fore side
       !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          do l=1,ng
           do m=1,nv
            w_gpu(i,j,nz+l,m) = w_gpu(i,j,nz,m)
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       endif
    endsubroutine bcextr_cuf
!
    subroutine bcoutopenfoam_cuf(ilat, nx, ny, nz, ng, nv, w_gpu)
       integer :: nx,ny,nz,ng,nv
       integer :: ilat
       integer :: i,j,k,l,m,iercuda
       real(rkind), dimension(1-ng:, 1-ng:, 1-ng:, 1:), intent(inout), device :: w_gpu

       if (ilat==1) then     ! left side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           do m=1,nv
            if (m .eq. I_U .and. w_gpu(1,j,k,I_U) .gt. 0.0_rkind) then
             w_gpu(1-l,j,k,I_U) = 0._rkind
            else
             w_gpu(1-l,j,k,m) = w_gpu(1,j,k,m)
            endif
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==2) then ! right side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           do m=1,nv
            if (m .eq. I_U .and. w_gpu(nx,j,k,I_U) .lt. 0.0_rkind) then
             w_gpu(nx+l,j,k,I_U) = 0._rkind
            else
             w_gpu(nx+l,j,k,m) = w_gpu(nx,j,k,m)
            endif
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==3) then ! lower side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           do m=1,nv
            if (m .eq. I_V .and. w_gpu(i,1,k,I_V) .gt. 0.0_rkind) then
             w_gpu(i,1-l,k,I_V) = 0._rkind
            else
             w_gpu(i,1-l,k,m) = w_gpu(i,1,k,m)
            endif
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==4) then  ! upper side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           do m=1,nv
            if (m .eq. I_V .and. w_gpu(i,ny,k,I_V) .lt. 0.0_rkind) then
             w_gpu(i,ny+l,k,I_V) = 0._rkind
            else
             w_gpu(i,ny+l,k,m) = w_gpu(i,ny,k,m)
            endif
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==5) then  ! back side
       !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          do l=1,ng
           do m=1,nv
            if (m .eq. I_W .and. w_gpu(i,j,1,I_U) .gt. 0.0_rkind) then
             w_gpu(i,j,1-l,I_W) = 0._rkind
            else
             w_gpu(i,j,1-l,m) = w_gpu(i,j,1,m)
            endif
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==6) then  ! fore side
       !$cuf kernel do(2) <<<*,*>>>
        do j=1,ny
         do i=1,nx
          do l=1,ng
           do m=1,nv
            if (m .eq. I_W .and. w_gpu(i,j,nz,I_W) .lt. 0.0_rkind) then
             w_gpu(i,j,nz+l,I_W) = 0._rkind
            else
             w_gpu(i,j,nz+l,m) = w_gpu(i,j,nz,m)
            endif
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       endif
    endsubroutine bcoutopenfoam_cuf
!
    subroutine bcsym_cuf(ilat, nx, ny, nz, ng, w_gpu)
       integer :: nx,ny,nz,ng,ilat
       integer :: i,j,k,l,lsp, iercuda
       real(rkind), dimension(1-ng:, 1-ng:, 1-ng:, 1:), intent(inout), device :: w_gpu
       
       if (ilat==3) then
       !$cuf kernel do(2) <<<*,*>>>
       do k=1,nz
        do i=1,nx
         w_gpu(i,1,k,I_V) = 0._rkind
         do l=1,ng
          do lsp=1,N_S
           w_gpu(i,1-l,k,lsp) = w_gpu(i,l,k,lsp)
          enddo
          w_gpu(i,1-l,k,I_U) =  w_gpu(i,l,k,I_U)
          w_gpu(i,1-l,k,I_V) = -w_gpu(i,l,k,I_V)
          w_gpu(i,1-l,k,I_W) =  w_gpu(i,l,k,I_W)
          w_gpu(i,1-l,k,I_E) =  w_gpu(i,l,k,I_E)
         enddo
        enddo
       enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==4) then 
       !$cuf kernel do(2) <<<*,*>>>
       do k=1,nz
        do i=1,nx
         w_gpu(i,ny,k,I_V) = 0._rkind
         do l=1,ng
          do lsp=1,N_S
           w_gpu(i,ny+l,k,lsp) = w_gpu(i,ny-l,k,lsp)
          enddo
          w_gpu(i,ny+l,k,I_U) =  w_gpu(i,ny-l,k,I_U)
          w_gpu(i,ny+l,k,I_V) = -w_gpu(i,ny-l,k,I_V)
          w_gpu(i,ny+l,k,I_W) =  w_gpu(i,ny-l,k,I_W)
          w_gpu(i,ny+l,k,I_E) =  w_gpu(i,ny-l,k,I_E)
         enddo
        enddo
       enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       endif 
    endsubroutine bcsym_cuf 

    subroutine bcwall_cuf(ilat,nx,ny,nz,ng,nv,nv_aux,w_gpu,w_aux_gpu,rgas_gpu,&
                          indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,&
                          tol_iter_nr,twall)

       integer, intent(in) :: nx,ny,nz,ng,nv,nv_aux,ilat,indx_cp_l,indx_cp_r,nsetcv
       real(rkind), intent(in) :: twall,tol_iter_nr
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv), intent(inout), device :: w_gpu
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
       real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
       real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
       real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
       integer :: i,j,k,l,lsp, iercuda,m
       real(rkind) :: rho,rhoe,uu,vv,ww,qq,tt,ee,pp,rmixtloc
!
       if (ilat==1) then     ! left side
       elseif (ilat==2) then ! right side
       elseif (ilat==3) then ! lower side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           rho = get_rho_from_w_dev(i,l,k,nv,nx,ny,nz,ng,w_gpu)
           do lsp=1,N_S
            w_aux_gpu(i,l,k,lsp) = w_gpu(i,l,k,lsp)/rho
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           rho  = get_rho_from_w_dev(i,l,k,nv,nx,ny,nz,ng,w_gpu)
           uu   = w_gpu(i,l,k,I_U)/rho
           vv   = w_gpu(i,l,k,I_V)/rho
           ww   = w_gpu(i,l,k,I_W)/rho
           rhoe = w_gpu(i,l,k,I_E)
           qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
           ee   = rhoe/rho-qq
           tt   = get_mixture_temperature_from_e_dev(ee, w_aux_gpu(i,l,k,J_T), cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, &
                                              tol_iter_nr,i,l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           rmixtloc = get_rmixture(i,l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           pp   = rho*tt*rmixtloc
           tt   = 2._rkind*twall-tt ! bc
           if (tt<200._rkind) tt = 200._rkind
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                    nsetcv,trange_gpu,i,l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           rho  = pp/tt/rmixtloc
           do lsp=1,N_S
            w_gpu(i,1-l,k,lsp) =  rho*w_aux_gpu(i,l,k,lsp)
           enddo
           w_gpu(i,1-l,k,I_U) = -rho*uu
           w_gpu(i,1-l,k,I_V) = -rho*vv
           w_gpu(i,1-l,k,I_W) = -rho*ww
           w_gpu(i,1-l,k,I_E) =  rho*(ee+qq)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==4) then  ! upper side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           rho = get_rho_from_w_dev(i,ny+1-l,k,nv,nx,ny,nz,ng,w_gpu)
           do lsp=1,N_S
            w_aux_gpu(i,ny+1-l,k,lsp) = w_gpu(i,ny+1-l,k,lsp)/rho
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           rho  = get_rho_from_w_dev(i,ny+1-l,k,nv,nx,ny,nz,ng,w_gpu)
           uu   = w_gpu(i,ny+1-l,k,I_U)/rho
           vv   = w_gpu(i,ny+1-l,k,I_V)/rho
           ww   = w_gpu(i,ny+1-l,k,I_W)/rho
           rhoe = w_gpu(i,ny+1-l,k,I_E)
           qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
           ee   = rhoe/rho-qq
           tt   = get_mixture_temperature_from_e_dev(ee, w_aux_gpu(i,ny+1-l,k,J_T), cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, &
                                              tol_iter_nr,i,ny+1-l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           rmixtloc = get_rmixture(i,ny+1-l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           pp   = rho*tt*rmixtloc
           tt   = 2._rkind*twall-tt ! bc
           if (tt<200._rkind) tt = 200._rkind
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                    nsetcv,trange_gpu,i,ny+1-l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           rho  = pp/tt/rmixtloc
           do lsp=1,N_S
            w_gpu(i,ny+l,k,lsp) =  rho*w_aux_gpu(i,ny+1-l,k,lsp)
           enddo
           w_gpu(i,ny+l,k,I_U) = -rho*uu
           w_gpu(i,ny+l,k,I_V) = -rho*vv
           w_gpu(i,ny+l,k,I_W) = -rho*ww
           w_gpu(i,ny+l,k,I_E) =  rho*(ee+qq)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==5) then  ! back side
       elseif (ilat==6) then  ! fore side
       endif

    endsubroutine bcwall_cuf

    subroutine bcwall_LE_cuf(ilat,nx,ny,nz,ng,nv,nv_aux,w_gpu,w_aux_gpu,rgas_gpu,&
                          indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,&
                          tol_iter_nr,twall,init_mf_gpu)

       integer, intent(in) :: nx,ny,nz,ng,nv,nv_aux,ilat,indx_cp_l,indx_cp_r,nsetcv
       real(rkind), intent(in) :: twall,tol_iter_nr
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv), intent(inout), device :: w_gpu
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
       real(rkind), dimension(N_S), intent(in), device :: rgas_gpu, init_mf_gpu
       real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
       real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
       integer :: i,j,k,l,lsp, iercuda,m
       real(rkind) :: rho,rhoe,uu,vv,ww,qq,tt,ee,pp,rmixtloc
!
       if (ilat==1) then     ! left side
       elseif (ilat==2) then ! right side
       elseif (ilat==3) then ! lower side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           rho = get_rho_from_w_dev(i,l,k,nv,nx,ny,nz,ng,w_gpu)
           do lsp=1,N_S
            w_aux_gpu(i,l,k,lsp) = w_gpu(i,l,k,lsp)/rho
           enddo
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          do l=1,ng
           rho  = get_rho_from_w_dev(i,l,k,nv,nx,ny,nz,ng,w_gpu)
           uu   = w_gpu(i,l,k,I_U)/rho
           vv   = w_gpu(i,l,k,I_V)/rho
           ww   = w_gpu(i,l,k,I_W)/rho
           rhoe = w_gpu(i,l,k,I_E)
           qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
           ee   = rhoe/rho-qq
           tt   = get_mixture_temperature_from_e_dev(ee, w_aux_gpu(i,l,k,J_T), cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, &
                                              tol_iter_nr,i,l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           rmixtloc = get_rmixture(i,l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           pp   = rho*tt*rmixtloc
           tt   = 2._rkind*twall-tt ! bc
           if (tt<200._rkind) tt = 200._rkind
           do lsp=1,N_S
            w_aux_gpu(i,1-l,k,lsp) = 2._rkind*init_mf_gpu(lsp) - w_aux_gpu(i,l,k,lsp)
            if (w_aux_gpu(i,1-l,k,lsp)<0._rkind) w_aux_gpu(i,1-l,k,lsp) = 0._rkind
           enddo
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                    nsetcv,trange_gpu,i,1-l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           rmixtloc = get_rmixture(i,1-l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           rho  = pp/tt/rmixtloc
           do lsp=1,N_S
            w_gpu(i,1-l,k,lsp) =  rho*w_aux_gpu(i,1-l,k,lsp)
           enddo
           w_gpu(i,1-l,k,I_U) = -rho*uu
           w_gpu(i,1-l,k,I_V) = -rho*vv
           w_gpu(i,1-l,k,I_W) = -rho*ww
           w_gpu(i,1-l,k,I_E) =  rho*(ee+qq)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==4) then  ! upper side
       elseif (ilat==5) then  ! back side
       elseif (ilat==6) then  ! fore side
       endif

    endsubroutine bcwall_LE_cuf

    subroutine bcwall_FC_cuf(ilat,nx,ny,nz,ng,nv,nv_aux,w_gpu,w_aux_gpu,rgas_gpu,mw_gpu,&
                          indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,&
                          tol_iter_nr,twall,y_gpu,idx_O,idx_O2,idx_N,idx_N2,idx_NO)

       integer, intent(in) :: nx,ny,nz,ng,nv,nv_aux,ilat,indx_cp_l,indx_cp_r,nsetcv,idx_O,idx_O2,idx_N,idx_N2,idx_NO
       real(rkind), intent(in) :: twall,tol_iter_nr
       real(rkind), dimension(1-ng:), intent(in), device :: y_gpu
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv), intent(inout), device :: w_gpu
       real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
       real(rkind), dimension(N_S), intent(in), device :: rgas_gpu,mw_gpu
       real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
       real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
       integer :: i,j,k,l,lsp,msp,iercuda,m
       real(rkind) :: rho,rhoe,uu,vv,ww,qq,tt,ee,pp,rmixtloc
       real(rkind) :: aad,bbd,ccd,aai,bbi,cci,dyw,pi
       real(rkind) :: summ,rmixtlocg,rhodifflsp_w,rhodiffmsp_w,xlspg,yylsp_w,rholsp_w,wdotlsp_w,dxmsp_w,mw
       real(rkind) :: grad_O, grad_N
!
       if (ilat==1) then     ! left side
       elseif (ilat==2) then ! right side
       elseif (ilat==3) then ! lower side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do i=1,nx
          pi = acos(-1._rkind)
          aad =  9._rkind/8._rkind
          bbd = -1._rkind/8._rkind
          ccd =  0._rkind/8._rkind
          aai = 150._rkind/128._rkind
          bbi = -25._rkind/128._rkind
          cci =   3._rkind/128._rkind
          dyw = 2._rkind*aad*y_gpu(1)+2._rkind/3._rkind*bbd*y_gpu(2)+2._rkind/5._rkind*ccd*y_gpu(3)
          do l=1,ng
           rho  = w_aux_gpu(i,l,k,J_R) 
           uu   = w_aux_gpu(i,l,k,J_U)
           vv   = w_aux_gpu(i,l,k,J_V)
           ww   = w_aux_gpu(i,l,k,J_W)
           tt   = w_aux_gpu(i,l,k,J_T) 
           pp   = w_aux_gpu(i,l,k,J_P)
           qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
           rmixtloc = pp/(rho*tt) 

           tt   = 2._rkind*twall-tt ! bc
           if (tt<200._rkind) tt = 200._rkind

!          species BC see Barbato et al. 1996
           rmixtlocg = get_rmixture(i,1,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           do lsp=1,N_S
            rhodifflsp_w  = aai*w_aux_gpu(i,1,k,J_D_START+lsp)+bbi*w_aux_gpu(i,2,k,J_D_START+lsp)+cci*w_aux_gpu(i,3,k,J_D_START+lsp)
            xlspg = w_aux_gpu(i,1,k,lsp)*rgas_gpu(lsp)/rmixtlocg
            if (lsp==idx_O .or. lsp==idx_N) then
             summ = 0._rkind 
             yylsp_w = aai*w_aux_gpu(i,1,k,lsp)+bbi*w_aux_gpu(i,2,k,lsp)+cci*w_aux_gpu(i,3,k,lsp)
             do msp=1,N_S
              rhodiffmsp_w = aai*w_aux_gpu(i,1,k,J_D_START+msp)+bbi*w_aux_gpu(i,2,k,J_D_START+msp)+cci*w_aux_gpu(i,3,k,J_D_START+msp)
              dxmsp_w = 2._rkind*aad*w_aux_gpu(i,1,k,J_D_START+msp)+2._rkind/3._rkind*bbd*w_aux_gpu(i,2,k,J_D_START+msp)+&
                       2._rkind/5._rkind*ccd*w_aux_gpu(i,3,k,J_D_START+msp)
              dxmsp_w = dxmsp_w/dyw
              summ = summ + rhodiffmsp_w*dxmsp_w
             enddo
             summ      = summ * yylsp_w
             rholsp_w   = aai*w_gpu(i,1,k,lsp)+bbi*w_gpu(i,2,k,lsp)+cci*w_gpu(i,3,k,lsp)
             wdotlsp_w = 1._rkind*sqrt(rgas_gpu(lsp)*twall/(2._rkind*pi))*rholsp_w
             summ      = summ + wdotlsp_w
             yylsp_w   = xlspg - y_gpu(1)/(rhodifflsp_w)*summ !! xlsp_w
             if (lsp==idx_O) grad_O = (xlspg-yylsp_w)/y_gpu(1) 
             if (lsp==idx_N) grad_N = (xlspg-yylsp_w)/y_gpu(1) 
            elseif (lsp==idx_NO) then ! NO non catalytic
             yylsp_w   = xlspg
            else
             yylsp_w = 0._rkind
            endif
            w_aux_gpu(i,1-l,k,lsp) = yylsp_w ! save it to w_aux_gpu
           enddo 
           ! up until now, N and O have been computed, NO zero gradient, N2 O2 put to zero
           ! compute X_O2
           w_aux_gpu(i,1-l,k,idx_O2) = w_aux_gpu(i,1,k,idx_O2)*rgas_gpu(idx_O2)/rmixtlocg !xO2g
           w_aux_gpu(i,1-l,k,idx_O2) = w_aux_gpu(i,1-l,k,idx_O2) + grad_O*y_gpu(1) ! opposed gradient wrt O, save it to w_aux_gpu

           w_aux_gpu(i,1-l,k,idx_N2) = w_aux_gpu(i,1,k,idx_N2)*rgas_gpu(idx_N2)/rmixtlocg !xN2g
           w_aux_gpu(i,1-l,k,idx_N2) = w_aux_gpu(i,1-l,k,idx_N2) + grad_N*y_gpu(1) ! opposed gradient wrt N, save it to w_aux_gpu
 
           ! normalizing and limiting xi > 0 
           summ = 0._rkind
           do lsp=1,N_S
            if (w_aux_gpu(i,1-l,k,lsp)<0._rkind) w_aux_gpu(i,1-l,k,lsp) = 0._rkind
            summ = summ + w_aux_gpu(i,1-l,k,lsp)
           enddo
           do lsp=1,N_S
            w_aux_gpu(i,1-l,k,lsp) = w_aux_gpu(i,1-l,k,lsp)/summ
           enddo

           ! compute MW
           mw = 0._rkind
           do lsp=1,N_S
            mw = mw+w_aux_gpu(i,1-l,k,lsp)*mw_gpu(lsp) 
           enddo
           summ = 0._rkind
           do lsp=1,N_S
            yylsp_w   = w_aux_gpu(i,1-l,k,lsp)*mw_gpu(lsp)/mw ! from X to Y 
            w_aux_gpu(i,1-l,k,lsp) = 2._rkind*yylsp_w - w_aux_gpu(i,l,k,lsp) 
            if (w_aux_gpu(i,1-l,k,lsp)<0._rkind) w_aux_gpu(i,1-l,k,lsp) = 0._rkind
            summ = summ + w_aux_gpu(i,1-l,k,lsp)
           enddo
           do lsp=1,N_S
            w_aux_gpu(i,1-l,k,lsp) = w_aux_gpu(i,1-l,k,lsp)/summ
           enddo
!           
           ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                    nsetcv,trange_gpu,i,1-l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
           rmixtloc = get_rmixture(i,1-l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
           rho  = pp/tt/rmixtloc
           do lsp=1,N_S
            w_gpu(i,1-l,k,lsp) =  rho*w_aux_gpu(i,1-l,k,lsp)
           enddo
           w_gpu(i,1-l,k,I_U) = -rho*uu
           w_gpu(i,1-l,k,I_V) = -rho*vv
           w_gpu(i,1-l,k,I_W) = -rho*ww
           w_gpu(i,1-l,k,I_E) =  rho*(ee+qq)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==4) then  ! upper side
       elseif (ilat==5) then  ! back side
       elseif (ilat==6) then  ! fore side
       endif

    endsubroutine bcwall_FC_cuf

    subroutine bcwall_ad_cuf(ilat, nx, ny, nz, ng, w_gpu)
       integer :: nx,ny,nz,ng,ilat
       integer :: i,j,k,l,lsp, iercuda
       real(rkind), dimension(1-ng:, 1-ng:, 1-ng:, 1:), intent(inout), device :: w_gpu

       if (ilat==1) then     ! left side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           do lsp=1,N_S
            w_gpu(1-l,j,k,lsp) =  w_gpu(l,j,k,lsp)
           enddo
           w_gpu(1-l,j,k,I_U) = -w_gpu(l,j,k,I_U)
           w_gpu(1-l,j,k,I_V) = -w_gpu(l,j,k,I_V)
           w_gpu(1-l,j,k,I_W) = -w_gpu(l,j,k,I_W)
           w_gpu(1-l,j,k,I_E) =  w_gpu(l,j,k,I_E)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==2) then ! right side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do l=1,ng
           do lsp=1,N_S
            w_gpu(nx+l,j,k,lsp) =  w_gpu(nx+1-l,j,k,lsp)
           enddo
           w_gpu(nx+l,j,k,I_U) = -w_gpu(nx+1-l,j,k,I_U)
           w_gpu(nx+l,j,k,I_V) = -w_gpu(nx+1-l,j,k,I_V)
           w_gpu(nx+l,j,k,I_W) = -w_gpu(nx+1-l,j,k,I_W)
           w_gpu(nx+l,j,k,I_E) =  w_gpu(nx+1-l,j,k,I_E)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==3) then ! lower side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do l=1,ng
          do i=1,nx
           do lsp=1,N_S
            w_gpu(i,1-l,k,lsp) =  w_gpu(i,l,k,lsp)
           enddo
           w_gpu(i,1-l,k,I_U) = -w_gpu(i,l,k,I_U)
           w_gpu(i,1-l,k,I_V) = -w_gpu(i,l,k,I_V)
           w_gpu(i,1-l,k,I_W) = -w_gpu(i,l,k,I_W)
           w_gpu(i,1-l,k,I_E) =  w_gpu(i,l,k,I_E)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==4) then  ! upper side
       !$cuf kernel do(2) <<<*,*>>>
        do k=1,nz
         do l=1,ng
          do i=1,nx
           do lsp=1,N_S
            w_gpu(i,ny+l,k,lsp) =  w_gpu(i,ny+1-l,k,lsp)
           enddo
           w_gpu(i,ny+l,k,I_U) = -w_gpu(i,ny+1-l,k,I_U)
           w_gpu(i,ny+l,k,I_V) = -w_gpu(i,ny+1-l,k,I_V)
           w_gpu(i,ny+l,k,I_W) = -w_gpu(i,ny+1-l,k,I_W)
           w_gpu(i,ny+l,k,I_E) =  w_gpu(i,ny+1-l,k,I_E)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==5) then  ! back side
       !$cuf kernel do(2) <<<*,*>>>
        do l=1,ng
         do j=1,ny
          do i=1,nx
           do lsp=1,N_S
            w_gpu(i,j,1-l,lsp) =  w_gpu(i,j,l,lsp)
           enddo
           w_gpu(i,j,1-l,I_U) = -w_gpu(i,j,l,I_U)
           w_gpu(i,j,1-l,I_V) = -w_gpu(i,j,l,I_V)
           w_gpu(i,j,1-l,I_W) = -w_gpu(i,j,l,I_W)
           w_gpu(i,j,1-l,I_E) =  w_gpu(i,j,l,I_E)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       elseif (ilat==6) then  ! fore side
       !$cuf kernel do(2) <<<*,*>>>
        do l=1,ng
         do j=1,ny
          do i=1,nx
           do lsp=1,N_S
            w_gpu(i,j,nz+l,lsp) =  w_gpu(i,j,nz+1-l,lsp)
           enddo
           w_gpu(i,j,nz+l,I_U) = -w_gpu(i,j,nz+1-l,I_U)
           w_gpu(i,j,nz+l,I_V) = -w_gpu(i,j,nz+1-l,I_V)
           w_gpu(i,j,nz+l,I_W) = -w_gpu(i,j,nz+1-l,I_W)
           w_gpu(i,j,nz+l,I_E) =  w_gpu(i,j,nz+1-l,I_E)
          enddo
         enddo
        enddo
       !@cuf iercuda=cudaDeviceSynchronize()
       endif
!
    endsubroutine bcwall_ad_cuf            

    subroutine bcwall_jcf_cuf(ilat,nx,ny,nz,ng,nv,nv_aux,w_gpu,w_aux_gpu,cv_coeff_gpu, &
               nsetcv,trange_gpu,rgas_gpu,indx_cp_l,indx_cp_r,twall,tol_iter_nr,x_gpu,z_gpu,jcf_jet_num,jcf_parbc_gpu,&
               jcf_coords_gpu,jcf_jet_rad,jcf_relax_factor)

    integer :: nx,ny,nz,ng,ilat,nv,nv_aux
    integer :: i,j,k,l,lsp, iercuda,jcf_jet_num,nsetcv,indx_cp_l,indx_cp_r
    real(rkind) :: jcf_jet_rad,twall,tol_iter_nr,jcf_relax_factor
    real(rkind), dimension(1-ng:), intent(in), device :: x_gpu,z_gpu
    real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv), intent(inout), device :: w_gpu
    real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv_aux), intent(inout), device :: w_aux_gpu
    real(rkind), dimension(1:,1:), intent(in), device :: jcf_parbc_gpu
    real(rkind), dimension(1:,1:), intent(in), device :: jcf_coords_gpu
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
    real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
    real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
    real(rkind) :: rho, rhov, ee, tprod, cv_m, tt, pp, rmixtloc, dist, uu, vv, ww, rhoe, qq
    integer :: ll

    if (ilat==1) then     ! left side
    elseif (ilat==2) then ! right side
    elseif (ilat==3) then ! lower side
    !$cuf kernel do(2) <<<*,*>>>
     do k=1,nz
      do i=1,nx
       do l=1,ng
        rho = get_rho_from_w_dev(i,l,k,nv,nx,ny,nz,ng,w_gpu)
        do lsp=1,N_S
         w_aux_gpu(i,l,k,lsp) = w_gpu(i,l,k,lsp)/rho
        enddo
       enddo
      enddo
     enddo
    !@cuf iercuda=cudaDeviceSynchronize()
    !$cuf kernel do(2) <<<*,*>>>
     do k=1,nz
      do i=1,nx
       !do ll=1,jcf_jet_num
       ll = 1
       dist = ((x_gpu(i)-jcf_coords_gpu(ll,1))**2+(z_gpu(k)-jcf_coords_gpu(ll,3))**2)**0.5_rkind
       if (dist <= jcf_jet_rad) then
        do l=1,ng
         do lsp=1,N_S
          w_gpu(i,1-l,k,lsp) = w_gpu(i,l,k,lsp) + jcf_relax_factor*(jcf_parbc_gpu(ll,lsp)-w_gpu(i,l,k,lsp))
         enddo
         w_gpu(i,1-l,k,I_U) = -w_gpu(i,l,k,I_U) + jcf_relax_factor*(jcf_parbc_gpu(ll,I_U)+w_gpu(i,l,k,I_U))
         w_gpu(i,1-l,k,I_V) = -w_gpu(i,l,k,I_V) + jcf_relax_factor*(jcf_parbc_gpu(ll,I_V)+w_gpu(i,l,k,I_V))
         w_gpu(i,1-l,k,I_W) = -w_gpu(i,l,k,I_W) + jcf_relax_factor*(jcf_parbc_gpu(ll,I_W)+w_gpu(i,l,k,I_W))
         w_gpu(i,1-l,k,I_E) =  w_gpu(i,l,k,I_E) + jcf_relax_factor*(jcf_parbc_gpu(ll,I_E)-w_gpu(i,l,k,I_E))
        enddo
       else
        do l=1,ng
         rho  = get_rho_from_w_dev(i,l,k,nv,nx,ny,nz,ng,w_gpu)
         uu   = w_gpu(i,l,k,I_U)/rho
         vv   = w_gpu(i,l,k,I_V)/rho
         ww   = w_gpu(i,l,k,I_W)/rho
         rhoe = w_gpu(i,l,k,I_E)
         qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
         ee   = rhoe/rho-qq
         tt   = get_mixture_temperature_from_e_dev(ee, w_aux_gpu(i,l,k,J_T), cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, &
                                            tol_iter_nr,i,l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
         rmixtloc = get_rmixture(i,l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
         pp   = rho*tt*rmixtloc
         tt   = 2._rkind*twall-tt ! bc
         ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                  nsetcv,trange_gpu,i,l,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
         rho  = pp/tt/rmixtloc
         do lsp=1,N_S
          w_gpu(i,1-l,k,lsp) =  rho*w_aux_gpu(i,l,k,lsp)
         enddo
         w_gpu(i,1-l,k,I_U) = -rho*uu
         w_gpu(i,1-l,k,I_V) = -rho*vv
         w_gpu(i,1-l,k,I_W) = -rho*ww
         w_gpu(i,1-l,k,I_E) =  rho*(ee+qq)
        enddo
       endif
      enddo
     enddo
    !@cuf iercuda=cudaDeviceSynchronize()
    elseif (ilat==4) then  ! upper side
    elseif (ilat==5) then  ! back side
    elseif (ilat==6) then  ! fore side
    endif

    endsubroutine bcwall_jcf_cuf

    subroutine bcwall_ad_jcf_cuf(ilat,nx,ny,nz,ng,w_gpu,x_gpu,z_gpu,&
                                 jcf_jet_num,jcf_parbc_gpu,jcf_coords_gpu,jcf_jet_rad,jcf_relax_factor)
    integer :: nx,ny,nz,ng,ilat
    integer :: i,j,k,l,lsp,iercuda,jcf_jet_num
    real(rkind) :: jcf_jet_rad,twall,jcf_relax_factor
    real(rkind), dimension(1-ng:), intent(in), device :: x_gpu,z_gpu
    real(rkind), dimension(1-ng:, 1-ng:, 1-ng:, 1:), intent(inout), device :: w_gpu
    real(rkind), dimension(1:,1:), intent(in), device :: jcf_parbc_gpu
    real(rkind), dimension(1:,1:), intent(in), device :: jcf_coords_gpu
    real(rkind) :: dist
    integer :: ll

    if (ilat==1) then     ! left side
    elseif (ilat==2) then ! right side
    elseif (ilat==3) then ! lower side
    !$cuf kernel do(2) <<<*,*>>>
     do k=1,nz
      do i=1,nx
       !do ll=1,jcf_jet_num
       ll = 1
       dist = ((x_gpu(i)-jcf_coords_gpu(ll,1))**2+(z_gpu(k)-jcf_coords_gpu(ll,3))**2)**0.5_rkind
       if (dist <= jcf_jet_rad) then
        do l=1,ng
         do lsp=1,N_S
          w_gpu(i,1-l,k,lsp) = w_gpu(i,l,k,lsp) + jcf_relax_factor*(jcf_parbc_gpu(ll,lsp)-w_gpu(i,l,k,lsp))
         enddo
         w_gpu(i,1-l,k,I_U) = -w_gpu(i,l,k,I_U) + jcf_relax_factor*(jcf_parbc_gpu(ll,I_U)+w_gpu(i,l,k,I_U))
         w_gpu(i,1-l,k,I_V) = -w_gpu(i,l,k,I_V) + jcf_relax_factor*(jcf_parbc_gpu(ll,I_V)+w_gpu(i,l,k,I_V))
         w_gpu(i,1-l,k,I_W) = -w_gpu(i,l,k,I_W) + jcf_relax_factor*(jcf_parbc_gpu(ll,I_W)+w_gpu(i,l,k,I_W))
         w_gpu(i,1-l,k,I_E) =  w_gpu(i,l,k,I_E) + jcf_relax_factor*(jcf_parbc_gpu(ll,I_E)-w_gpu(i,l,k,I_E))
        enddo
       else
        do l=1,ng
         do lsp=1,N_S
          w_gpu(i,1-l,k,lsp) =  w_gpu(i,l,k,lsp)
         enddo
         w_gpu(i,1-l,k,I_U) = -w_gpu(i,l,k,I_U)
         w_gpu(i,1-l,k,I_V) = -w_gpu(i,l,k,I_V)
         w_gpu(i,1-l,k,I_W) = -w_gpu(i,l,k,I_W)
         w_gpu(i,1-l,k,I_E) =  w_gpu(i,l,k,I_E)
        enddo
       endif
      enddo
     enddo
    !@cuf iercuda=cudaDeviceSynchronize()
    elseif (ilat==4) then  ! upper side
    elseif (ilat==5) then  ! back side
    elseif (ilat==6) then  ! fore side
    endif

    endsubroutine bcwall_ad_jcf_cuf

    subroutine compute_residual_cuf(nx, ny, nz, ng, nv, fln_gpu, dt, residual_rhou, fluid_mask_gpu)
        integer :: nx, ny, nz, ng, nv
        real(rkind), intent(out) :: residual_rhou
        real(rkind), intent(in) :: dt
        real(rkind), dimension(1:nx, 1:ny, 1:nz, nv), intent(in), device :: fln_gpu
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: fluid_mask_gpu
        integer :: i,j,k,iercuda
        ! Note: should be modified to include metrics

        residual_rhou = 0._rkind
        !$cuf kernel do(3) <<<*,*>>> reduce(+:residual_rhou)
        do k=1,nz
         do j=1,ny
          do i=1,nx
           if (fluid_mask_gpu(i,j,k)==0) then
            residual_rhou = residual_rhou + (fln_gpu(i,j,k,I_U)/dt)**2
           endif
          enddo
         enddo
       enddo
       !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine compute_residual_cuf

    subroutine compute_vmax_cuf(nx, ny, nz, ng, w_aux_gpu, vmax, fluid_mask_gpu)
        integer :: nx, ny, nz, ng
        real(rkind), intent(out) :: vmax
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(in), device :: w_aux_gpu  
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: fluid_mask_gpu
        integer :: i,j,k,iercuda
        ! Note: should be modified to include metrics

        vmax = 0._rkind
!       vmax = 1000._rkind
        !$cuf kernel do(2) <<<*,*>>> reduce(max:vmax)
!       !$cuf kernel do(2) <<<*,*>>> reduce(min:vmax)
        do k=1,nz
         do j=1,ny
          do i=1,nx
           if (fluid_mask_gpu(i,j,k)==0) then
            vmax = max(vmax,w_aux_gpu(i,j,k,J_V))
           endif
          enddo
         enddo
       enddo
       !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine compute_vmax_cuf

    subroutine compute_rho_t_p_minmax_cuf(nx, ny, nz, ng, w_aux_gpu, &
        rhomin, rhomax, tmin, tmax, pmin, pmax, fluid_mask_gpu)
        integer, intent(in) :: nx, ny, nz, ng
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(in), device :: w_aux_gpu
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: fluid_mask_gpu
        real(rkind), intent(out) :: rhomin, rhomax, tmin, tmax, pmin, pmax
        integer :: i,j,k,iercuda
        real(rkind) :: rho,tt,pp

        rhomin =  huge(1._rkind)
        rhomax = -100._rkind
        tmin   =  huge(1._rkind)
        tmax   = -100._rkind
        pmin   =  huge(1._rkind)
        pmax   = -100._rkind

        !$cuf kernel do(3) <<<*,*>>> reduce(max:rhomax, tmax, pmax) reduce(min:rhomin, tmin, pmin)
        do k=1,nz
         do j=1,ny
          do i=1,nx
           if (fluid_mask_gpu(i,j,k)==0) then
            rho    = w_aux_gpu(i,j,k,J_R)
            tt     = w_aux_gpu(i,j,k,J_T)
            pp     = w_aux_gpu(i,j,k,J_P)
            rhomin = min(rhomin,rho)
            rhomax = max(rhomax,rho)
            tmin   = min(tmin  ,tt )
            tmax   = max(tmax  ,tt )  
            pmin   = min(pmin  ,pp )  
            pmax   = max(pmax  ,pp )  
           endif
          enddo
         enddo
       enddo
       !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine compute_rho_t_p_minmax_cuf

    subroutine compute_dt_cuf(nx, ny, nz, ng, nv, nv_aux, &
                              dcsidx_gpu, detady_gpu, dzitdz_gpu, dcsidxs_gpu, detadys_gpu, dzitdzs_gpu, w_gpu, w_aux_gpu, &
                              dtxi_max, dtyi_max, dtzi_max, dtxv_max, dtyv_max, dtzv_max, dtxk_max, dtyk_max, dtzk_max,    &
                              indx_cp_l, indx_cp_r, cp_coeff_gpu, nsetcv, trange_gpu, fluid_mask_gpu)
     integer :: nx, ny, nz, ng, nv, indx_cp_l, indx_cp_r, nsetcv, nv_aux
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv) , intent(in), device :: w_gpu 
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(in), device :: w_aux_gpu  
     integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: fluid_mask_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cp_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
     real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu, detady_gpu, dzitdz_gpu 
     real(rkind), dimension(1:), intent(in), device :: dcsidxs_gpu, detadys_gpu, dzitdzs_gpu
     real(rkind) :: dtxi, dtyi, dtzi, dtxv, dtyv, dtzv, dtxk, dtyk, dtzk
     integer     :: i,j,k,ll,lsp,iercuda
     real(rkind) :: rho, ri, uu, vv, ww, tt, mu, nu, k_over_rhocp, c
     real(rkind) :: dtxi_max, dtyi_max, dtzi_max, dtxv_max, dtyv_max, dtzv_max, dtxk_max, dtyk_max, dtzk_max
     real(rkind) :: cploc, k_cond

     dtxi_max = 0._rkind
     dtyi_max = 0._rkind
     dtzi_max = 0._rkind
     dtxv_max = 0._rkind
     dtyv_max = 0._rkind
     dtzv_max = 0._rkind
     dtxk_max = 0._rkind
     dtyk_max = 0._rkind
     dtzk_max = 0._rkind

     !$cuf kernel do(3) <<<*,*>>> reduce(max:dtxi_max,dtyi_max,dtzi_max,dtxv_max,dtyv_max,dtzv_max,dtxk_max,dtyk_max,dtzk_max)
     do k=1,nz
      do j=1,ny
       do i=1,nx
!
        if (fluid_mask_gpu(i,j,k)==0) then
!
!        rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
         rho  = w_aux_gpu(i,j,k,J_R)
         ri   = 1._rkind/rho
         uu   = w_aux_gpu(i,j,k,J_U)
         vv   = w_aux_gpu(i,j,k,J_V)
         ww   = w_aux_gpu(i,j,k,J_W)
         tt   = w_aux_gpu(i,j,k,J_T)
         mu   = w_aux_gpu(i,j,k,J_MU)
         c    = w_aux_gpu(i,j,k,J_C)
         k_cond = w_aux_gpu(i,j,k,J_K_COND)
         cploc  = get_cp_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,&
                         i,j,k,nx,ny,nz,ng,nv_aux,w_aux_gpu)
!
         nu  = ri*mu
         k_over_rhocp = k_cond*ri/cploc
         dtxi   = (abs(uu)+c)*dcsidx_gpu(i) 
         dtyi   = (abs(vv)+c)*detady_gpu(j) 
         dtzi   = (abs(ww)+c)*dzitdz_gpu(k) 
         dtxv  = nu*dcsidxs_gpu(i)
         dtyv  = nu*detadys_gpu(j)
         dtzv  = nu*dzitdzs_gpu(k)
         dtxk  = k_over_rhocp*dcsidxs_gpu(i)
         dtyk  = k_over_rhocp*detadys_gpu(j)
         dtzk  = k_over_rhocp*dzitdzs_gpu(k)

         dtxi_max = max(dtxi_max, dtxi)
         dtyi_max = max(dtyi_max, dtyi)
         dtzi_max = max(dtzi_max, dtzi)
         dtxv_max = max(dtxv_max, dtxv)
         dtyv_max = max(dtyv_max, dtyv)
         dtzv_max = max(dtzv_max, dtzv)
         dtxk_max = max(dtxk_max, dtxk)
         dtyk_max = max(dtyk_max, dtyk)
         dtzk_max = max(dtzk_max, dtzk)
!
        endif
!
       enddo
      enddo
     enddo
     !@cuf iercuda=cudaDeviceSynchronize()

    endsubroutine compute_dt_cuf

    subroutine eval_aux_cuf(nx, ny, nz, ng, nv, nv_aux, istart, iend, jstart, jend, kstart, kend, w_gpu, w_aux_gpu, &
            p0, t_min_tab,dt_tab, R_univ, &
            rgas_gpu, &
            cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, tol_iter_nr, stream_id, &
            mw_gpu,mwinv_gpu,visc_species_gpu,lambda_species_gpu,diffbin_species_gpu,num_t_tab,&
            N_EoI_gpu,aw_EoI_gpu,NainSp_gpu,Beta0_gpu,coeff_EoI_gpu)
        integer, intent(in) :: nx, ny, nz, ng, nv, nv_aux, nsetcv, num_t_tab
        integer, intent(in) :: istart, iend, jstart, jend, kstart, kend
        integer, intent(in) :: indx_cp_l, indx_cp_r
        integer, intent(in) :: N_EoI_gpu
        real(rkind), intent(in) :: tol_iter_nr, R_univ, p0
        real(rkind), intent(in) :: t_min_tab,dt_tab
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cp_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu,mw_gpu,mwinv_gpu
        real(rkind), dimension(num_t_tab+1,N_S), intent(in), device :: visc_species_gpu,lambda_species_gpu
        real(rkind), dimension(num_t_tab+1,N_S,N_S), intent(in), device :: diffbin_species_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv), intent(inout), device :: w_gpu  
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv_aux), intent(inout), device :: w_aux_gpu
        integer(kind=cuda_stream_kind), intent(in) :: stream_id
        integer     :: i,j,k,lsp,msp,itt
        real(rkind) :: rho, rhou, rhov, rhow, rhoe, ri, uu, vv, ww, qq, pp, tt, mu, ee, c
        real(rkind) :: cploc,gamloc,rmixtloc
        real(rkind) :: dmu, mulsp, dtt, mu_den, mumsp, phi_lm
        real(rkind) :: mwmixt, k_cond, k_cond2, dlam, xlsp, klsp,yatm
        integer     :: iercuda,le
        real(rkind) :: xmsp, diff_den, ddiff, diff_ij, tloc,Beta
        real(rkind), dimension(N_EoI_gpu), intent(in), device :: aw_EoI_gpu,coeff_EoI_gpu
        integer, dimension(N_S,N_EoI_gpu), intent(in), device :: NainSp_gpu
        real(rkind), dimension(2), intent(in), device :: Beta0_gpu
!
        !$cuf kernel do(3) <<<*,*,stream=stream_id>>>
        do k=kstart,kend
         do j=jstart,jend
          do i=istart,iend
              do lsp=1,N_S
               w_gpu(i,j,k,lsp) = max(w_gpu(i,j,k,lsp),0._rkind)
              enddo
              rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
              ri   = 1._rkind/rho
              do lsp=1,N_S
               w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)*ri
!              w_aux_gpu(i,j,k,lsp) = max(w_aux_gpu(i,j,k,lsp),0._rkind)
!              w_aux_gpu(i,j,k,lsp) = min(w_aux_gpu(i,j,k,lsp),1._rkind)
              enddo
              rhou = w_gpu(i,j,k,I_U)
              rhov = w_gpu(i,j,k,I_V)
              rhow = w_gpu(i,j,k,I_W)
              rhoe = w_gpu(i,j,k,I_E)
              uu   = rhou*ri
              vv   = rhov*ri
              ww   = rhow*ri
              qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
              ee = rhoe*ri-qq
              tt = get_mixture_temperature_from_e_dev(ee, w_aux_gpu(i,j,k,J_T), cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, &
                                              tol_iter_nr,i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
!
              rmixtloc = get_rmixture(i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
              cploc    = get_cp_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,&
                         i,j,k,nx,ny,nz,ng,nv_aux,w_aux_gpu)
              gamloc   = cploc/(cploc-rmixtloc)
              pp       = rho*tt*rmixtloc ! EoS for a perfect gas
              c        = sqrt(gamloc*pp*ri)
!
              w_aux_gpu(i,j,k,J_R)  = rho
              w_aux_gpu(i,j,k,J_U)  = uu
              w_aux_gpu(i,j,k,J_V)  = vv
              w_aux_gpu(i,j,k,J_W)  = ww
              w_aux_gpu(i,j,k,J_H)  = (rhoe+pp)*ri
              w_aux_gpu(i,j,k,J_T)  = tt
              w_aux_gpu(i,j,k,J_P)  = pp
              w_aux_gpu(i,j,k,J_C)  = c
!
              !Mixture Fraction
              Beta = 0._rkind
              if (N_EoI_gpu .ne. 1) then
               do le = 1,N_EoI_gpu
                yatm = 0._rkind
                do lsp = 1,N_S
                 yatm = yatm + NainSp_gpu(lsp,le)*aw_EoI_gpu(le)*w_aux_gpu(i,j,k,lsp)*mwinv_gpu(lsp)
                enddo
                Beta = Beta + coeff_EoI_gpu(le)*yatm/aw_EoI_gpu(le)
               enddo
               w_aux_gpu(i,j,k,J_Z) = (Beta-Beta0_gpu(2))/(Beta0_gpu(1)-Beta0_gpu(2))
              else
               w_aux_gpu(i,j,k,J_Z) = 0.5_rkind
              endif
!
              itt = int((tt-t_min_tab)/dt_tab)+1
              itt = max(itt,1)
              itt = min(itt,num_t_tab)
              tloc = t_min_tab+(itt-1)*dt_tab
              dtt = (tt-tloc)/dt_tab
              ! interpola viscosity and conductivity

!             Mathur et al (1967) mixture thermal conductivity
              mwmixt   = R_univ/rmixtloc 
!             w_aux_gpu(i,j,k,J_MW) = mwmixt
              k_cond   = 0._rkind
              k_cond2  = 0._rkind
              do lsp = 1,N_S
               dlam    = lambda_species_gpu(itt+1,lsp)-lambda_species_gpu(itt,lsp)
               klsp    = lambda_species_gpu(itt,lsp)+dlam*dtt
               xlsp    = w_aux_gpu(i,j,k,lsp)*mwmixt*mwinv_gpu(lsp)
               k_cond  = k_cond  + xlsp*klsp
               k_cond2 = k_cond2 + xlsp/klsp
              enddo
              k_cond = 0.5_rkind*(k_cond + 1._rkind/k_cond2)
!
!             Wilke (1950) mixture dynamic viscosity            
              mu   = 0._rkind
              do lsp=1,N_S
               dmu   = visc_species_gpu(itt+1,lsp)-visc_species_gpu(itt,lsp)
               mulsp = visc_species_gpu(itt,lsp)+dmu*dtt
               mu_den = 0._rkind
               do msp=1,N_S 
                dmu    = visc_species_gpu(itt+1,msp)-visc_species_gpu(itt,msp)
                mumsp  = visc_species_gpu(itt,msp)+dmu*dtt
                phi_lm = 1._rkind/sqrt(1._rkind+mw_gpu(lsp)*mwinv_gpu(msp))*&
                        (1._rkind + sqrt(mumsp/mulsp)*(mw_gpu(msp)*mwinv_gpu(lsp))**0.25_rkind)**2
                mu_den = mu_den + w_aux_gpu(i,j,k,msp)*phi_lm*mwinv_gpu(msp)
               enddo    
               mu = mu + mulsp*w_aux_gpu(i,j,k,lsp)*mwinv_gpu(lsp)/mu_den
              enddo    
              mu = sqrt(8._rkind)*mu
!
              w_aux_gpu(i,j,k,J_MU) = mu
              w_aux_gpu(i,j,k,J_K_COND) = k_cond

!             Bird (1960) species' diffusion into mixture    
              do lsp = 1,N_S
               diff_den = 0._rkind
               do msp = 1,N_S
                if (msp /= lsp) then
                 ddiff = diffbin_species_gpu(itt+1,lsp,msp)-diffbin_species_gpu(itt,lsp,msp)
                 diff_ij = diffbin_species_gpu(itt,lsp,msp)+ddiff*dtt
                 xmsp = w_aux_gpu(i,j,k,msp)*mwmixt*mwinv_gpu(msp)
                 diff_den = diff_den + xmsp/diff_ij
                endif
               enddo
               if (diff_den > 1.0D-015) then
                w_aux_gpu(i,j,k,J_D_START+lsp) = (1._rkind-w_aux_gpu(i,j,k,lsp))/diff_den*p0/pp*rho ! rho * diffusion
               else 
                w_aux_gpu(i,j,k,J_D_START+lsp) = 0._rkind
               endif
              enddo


          enddo
         enddo
        enddo
    endsubroutine eval_aux_cuf

    subroutine eval_aux2_cuf(nx, ny, nz, ng, visc_order, w_aux_gpu, &
                coeff_deriv1_gpu, dcsidx_gpu, detady_gpu, dzitdz_gpu, eps_sensor,sensor_type)

        integer, intent(in) :: nx, ny, nz, ng, visc_order, sensor_type
        real(rkind), intent(in) :: eps_sensor
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: coeff_deriv1_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu, detady_gpu, dzitdz_gpu
        integer     :: i,j,k,l,ll,iercuda
        real(rkind) :: ccl
        real(rkind) :: ux,uy,uz
        real(rkind) :: vx,vy,vz
        real(rkind) :: wx,wy,wz
        real(rkind) :: div,div3l,omegax,omegay,omegaz,omod2
        integer     :: lmax

        lmax = visc_order/2

        !$cuf kernel do(3) <<<*,*>>>
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
            ccl = coeff_deriv1_gpu(l,lmax)
            ux = ux+ccl*(w_aux_gpu(i+l,j,k,J_U)-w_aux_gpu(i-l,j,k,J_U))
            vx = vx+ccl*(w_aux_gpu(i+l,j,k,J_V)-w_aux_gpu(i-l,j,k,J_V))
            wx = wx+ccl*(w_aux_gpu(i+l,j,k,J_W)-w_aux_gpu(i-l,j,k,J_W))
            uy = uy+ccl*(w_aux_gpu(i,j+l,k,J_U)-w_aux_gpu(i,j-l,k,J_U))
            vy = vy+ccl*(w_aux_gpu(i,j+l,k,J_V)-w_aux_gpu(i,j-l,k,J_V))
            wy = wy+ccl*(w_aux_gpu(i,j+l,k,J_W)-w_aux_gpu(i,j-l,k,J_W))
            uz = uz+ccl*(w_aux_gpu(i,j,k+l,J_U)-w_aux_gpu(i,j,k-l,J_U))
            vz = vz+ccl*(w_aux_gpu(i,j,k+l,J_V)-w_aux_gpu(i,j,k-l,J_V))
            wz = wz+ccl*(w_aux_gpu(i,j,k+l,J_W)-w_aux_gpu(i,j,k-l,J_W))
           enddo
           ux = ux*dcsidx_gpu(i)
           vx = vx*dcsidx_gpu(i)
           wx = wx*dcsidx_gpu(i)
           uy = uy*detady_gpu(j)
           vy = vy*detady_gpu(j)
           wy = wy*detady_gpu(j)
           uz = uz*dzitdz_gpu(k)
           vz = vz*dzitdz_gpu(k)
           wz = wz*dzitdz_gpu(k)
!
           div     = ux+vy+wz
           div3l   = div/3._rkind
           omegax = wy-vz
           omegay = uz-wx
           omegaz = vx-uy
           omod2 = omegax*omegax+omegay*omegay+omegaz*omegaz
!          w_aux_gpu(i,j,k,J_DUC) = (max(-div/sqrt(omod2+div**2+(u0/l0)**2),0._rkind))**2
           if (sensor_type == 0) then
            w_aux_gpu(i,j,k,J_DUC) = (-div/sqrt(omod2+div**2+eps_sensor))**2
           else
            w_aux_gpu(i,j,k,J_DUC) = (max(-div/sqrt(omod2+div**2+eps_sensor),0._rkind))**2
           endif
           w_aux_gpu(i,j,k,J_DIV) = div3l
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    end subroutine eval_aux2_cuf

    attributes(device) function get_cp_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu, &
                       nsetcv,trange_gpu,i,j,k,nx,ny,nz,ng,nv_aux,w_aux_gpu)
    real(rkind) :: get_cp_dev
    integer, value :: i,j,k,nv_aux,nx,ny,nz,ng,nsetcv
    integer, value :: indx_cp_l,indx_cp_r
    real(rkind), value :: tt
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
    real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
    real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
    real(rkind) :: cploc,cp_l,tpow
    integer, dimension(N_S) :: nrange 
    integer :: nrangeloc,nmax,l,lsp,jl,jm,ju
!
    nmax = 100000
    do lsp=1,N_S
     nrange(lsp) = 1
    enddo
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do l=1,nmax
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (tt>= trange_gpu(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
      nrange(lsp) = max(nrange(lsp),1)
      nrange(lsp) = min(nrange(lsp),nsetcv)
     enddo
    endif
!    
    cploc = 0._rkind
    do l=indx_cp_l,indx_cp_r
     tpow = tt**l
     cp_l = 0._rkind
     do lsp=1,N_S
      nrangeloc = nrange(lsp)
      cp_l = cp_l+cp_coeff_gpu(l,lsp,nrangeloc)*w_aux_gpu(i,j,k,lsp)
     enddo
     cploc = cploc+cp_l*tpow
    enddo
    get_cp_dev = cploc
!
    endfunction get_cp_dev

    attributes(device) function get_cp_roe_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu, &
                       nsetcv,trange_gpu,yy)
    real(rkind) :: get_cp_roe_dev
    integer, value :: nsetcv,indx_cp_l,indx_cp_r
    real(rkind), value :: tt
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
    real(rkind), dimension(N_S) :: yy
    real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
    real(rkind) :: cploc,cp_l,tpow
    integer, dimension(N_S) :: nrange 
    integer :: nrangeloc,nmax,l,lsp,jl,jm,ju
!        
    nmax = 100000
    do lsp=1,N_S
     nrange(lsp) = 1
    enddo
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do l=1,nmax
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (tt>= trange_gpu(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
      nrange(lsp) = max(nrange(lsp),1)
      nrange(lsp) = min(nrange(lsp),nsetcv)
     enddo
    endif
!    
    cploc = 0._rkind
    do l=indx_cp_l,indx_cp_r
     tpow = tt**l
     cp_l = 0._rkind
     do lsp=1,N_S
      nrangeloc = nrange(lsp)
      cp_l = cp_l+cp_coeff_gpu(l,lsp,nrangeloc)*yy(lsp)
     enddo
     cploc = cploc+cp_l*tpow
    enddo
    get_cp_roe_dev = cploc
!
    endfunction get_cp_roe_dev
!    
    attributes(device) function get_cp_ibm_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu, &
                       nsetcv,trange_gpu,ibm_w_refl_gpu,ll,ibm_num_interface,nv)
    real(rkind) :: get_cp_ibm_dev
    integer, value :: nsetcv,indx_cp_l,indx_cp_r,ll,ibm_num_interface,nv
    real(rkind), value :: tt
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
    real(rkind), dimension(ibm_num_interface,nv) :: ibm_w_refl_gpu
    real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
    real(rkind) :: cploc,cp_l,tpow
    integer, dimension(N_S) :: nrange 
    integer :: nrangeloc,nmax,l,lsp,jl,jm,ju
!
    nmax = 100000
    do lsp=1,N_S
     nrange(lsp) = 1
    enddo
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do l=1,nmax
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (tt>= trange_gpu(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
      nrange(lsp) = max(nrange(lsp),1)
      nrange(lsp) = min(nrange(lsp),nsetcv)
     enddo
    endif
!    
    cploc = 0._rkind
    do l=indx_cp_l,indx_cp_r
     tpow = tt**l
     cp_l = 0._rkind
     do lsp=1,N_S
      nrangeloc = nrange(lsp)
      cp_l = cp_l+cp_coeff_gpu(l,lsp,nrangeloc)*ibm_w_refl_gpu(ll,lsp)
     enddo
     cploc = cploc+cp_l*tpow
    enddo
    get_cp_ibm_dev = cploc
!
    endfunction get_cp_ibm_dev
!
    attributes(device) function get_mixture_temperature_from_e_dev(ee,T_start,cv_coeff_gpu,nsetcv,trange_gpu, &
                                 indx_cp_l, indx_cp_r, tol_iter_nr,i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
    real(rkind) :: get_mixture_temperature_from_e_dev
    real(rkind), value :: ee, T_start, tol_iter_nr
    integer, value :: i,j,k,nv_aux,nx,ny,nz,ng,nsetcv,indx_cp_l,indx_cp_r
    real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), device :: w_aux_gpu
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), device :: cv_coeff_gpu
    real(rkind), dimension(N_S,nsetcv+1), device :: trange_gpu
    real(rkind) :: tt,T_old,ebar,den,num,T_pow,T_powp,tden,tnum,sumb,sum0,tprod,cv_l
    integer :: nrangeloc,nmax,l,iter,max_iter,lsp,jl,jm,ju
    integer, dimension(N_S) :: nrange 
!
    max_iter = 10
!
    nmax = 100000
    do lsp=1,N_S
     nrange(lsp) = 1
    enddo
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do l=1,nmax
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (T_start >= trange_gpu(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
      nrange(lsp) = max(nrange(lsp),1)
      nrange(lsp) = min(nrange(lsp),nsetcv)
     enddo
    endif
!
    T_old = T_start
    do iter=1,max_iter
!
     if (nsetcv>1) then
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
!      if (T_old>trange_gpu(lsp,nrangeloc).and.T_old<trange_gpu(lsp,nrangeloc+1)) then
!      else
!      endif
!      Assume maximum range jump in the iterative process equal to 1
       if (T_old<trange_gpu(lsp,nrangeloc)) then
        nrange(lsp) = nrange(lsp)-1
       elseif(T_old>trange_gpu(lsp,nrangeloc+1)) then 
        nrange(lsp) = nrange(lsp)+1
       endif
       nrange(lsp) = max(nrange(lsp),1)
       nrange(lsp) = min(nrange(lsp),nsetcv)
      enddo
     endif
!
     sumb = 0._rkind
     do lsp=1,N_S
      nrangeloc = nrange(lsp)
      sumb = sumb+cv_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)*w_aux_gpu(i,j,k,lsp)
     enddo
     ebar = ee-sumb
!
     den = 0._rkind
     num = 0._rkind
     do l=indx_cp_l,indx_cp_r
      T_pow  = T_old**l
      tden   = T_pow
      if (l==-1) then
       tnum   = log(T_old)
      else
       T_powp = T_old*T_pow
       tnum   = T_powp/(l+1._rkind)
      endif
      cv_l = 0._rkind
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
       cv_l = cv_l+cv_coeff_gpu(l,lsp,nrangeloc)*w_aux_gpu(i,j,k,lsp)
      enddo
      num = num+cv_l*tnum
      den = den+cv_l*tden
     enddo
     tt = T_old+(ebar-num)/den
     if (abs(tt-T_old) < tol_iter_nr) exit
     T_old = tt
!
    enddo
!
    get_mixture_temperature_from_e_dev = tt
    endfunction get_mixture_temperature_from_e_dev

    attributes(device) function get_mixture_temperature_from_e_ibm_dev(ee,T_start,cv_coeff_gpu,nsetcv,trange_gpu, &
                                indx_cp_l,indx_cp_r,tol_iter_nr,ibm_num_interface,nv,ll,ibm_w_refl_gpu)

    real(rkind) :: get_mixture_temperature_from_e_ibm_dev
    real(rkind), value :: ee, T_start, tol_iter_nr
    integer, value :: ibm_num_interface,nv,ll,nsetcv,indx_cp_l,indx_cp_r
    real(rkind), dimension(ibm_num_interface,nv), device :: ibm_w_refl_gpu
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), device :: cv_coeff_gpu
    real(rkind), dimension(N_S,nsetcv+1), device :: trange_gpu
    real(rkind) :: tt,T_old,ebar,den,num,T_pow,T_powp,tden,tnum,sumb,sum0,tprod,cv_l
    integer :: nrangeloc,nmax,l,iter,max_iter,lsp,jl,jm,ju
    integer, dimension(N_S) :: nrange 
!
    max_iter = 10
!
    nmax = 100000
    do lsp=1,N_S
     nrange(lsp) = 1
    enddo
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do l=1,nmax
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (T_start >= trange_gpu(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
      nrange(lsp) = max(nrange(lsp),1)
      nrange(lsp) = min(nrange(lsp),nsetcv)
     enddo
    endif
!
    T_old = T_start
    do iter=1,max_iter
!
     if (nsetcv>1) then
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
!      if (T_old>trange_gpu(lsp,nrangeloc).and.T_old<trange_gpu(lsp,nrangeloc+1)) then
!      else
!      endif
!      Assume maximum range jump in the iterative process equal to 1
       if (T_old<trange_gpu(lsp,nrangeloc)) then
        nrange(lsp) = nrange(lsp)-1
       elseif(T_old>trange_gpu(lsp,nrangeloc+1)) then 
        nrange(lsp) = nrange(lsp)+1
       endif
       nrange(lsp) = max(nrange(lsp),1)
       nrange(lsp) = min(nrange(lsp),nsetcv)
      enddo
     endif
!
     sumb = 0._rkind
     do lsp=1,N_S
      nrangeloc = nrange(lsp)
      sumb = sumb+cv_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)*ibm_w_refl_gpu(ll,lsp)
     enddo
     ebar = ee-sumb
!
     den = 0._rkind
     num = 0._rkind
     do l=indx_cp_l,indx_cp_r
      T_pow  = T_old**l
      tden   = T_pow
      if (l==-1) then
       tnum   = log(T_old)
      else
       T_powp = T_old*T_pow
       tnum   = T_powp/(l+1._rkind)
      endif
      cv_l = 0._rkind
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
       cv_l = cv_l+cv_coeff_gpu(l,lsp,nrangeloc)*ibm_w_refl_gpu(ll,lsp)
      enddo
      num = num+cv_l*tnum
      den = den+cv_l*tden
     enddo
     tt = T_old+(ebar-num)/den
     if (abs(tt-T_old) < tol_iter_nr) exit
     T_old = tt
!
    enddo
!
    get_mixture_temperature_from_e_ibm_dev = tt
    endfunction get_mixture_temperature_from_e_ibm_dev


    attributes(device) function get_mixture_temperature_from_e_roe_dev(ee,T_start,cv_coeff_gpu,nsetcv,trange_gpu, &
                                 indx_cp_l,indx_cp_r,tol_iter_nr,yyroe)
    real(rkind) :: get_mixture_temperature_from_e_roe_dev
    integer, value :: nsetcv,indx_cp_l,indx_cp_r
    real(rkind), value :: ee, T_start, tol_iter_nr
    real(rkind), dimension(N_S) :: yyroe
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
    real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
    real(rkind) :: tt,T_old,ebar,den,num,T_pow,T_powp,tden,tnum,sumb,sum0,tprod,cv_l
    integer :: nrangeloc,nmax,l,iter,max_iter,lsp,jl,jm,ju
    integer, dimension(N_S) :: nrange 
!
    max_iter = 10
!
    nmax = 100000
    do lsp=1,N_S
     nrange(lsp) = 1
    enddo
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do l=1,nmax
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (T_start >= trange_gpu(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
      nrange(lsp) = max(nrange(lsp),1)
      nrange(lsp) = min(nrange(lsp),nsetcv)
     enddo
    endif
!
    T_old = T_start
    do iter=1,max_iter
!
     if (nsetcv>1) then
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
!      if (T_old>trange_gpu(lsp,nrangeloc).and.T_old<trange_gpu(lsp,nrangeloc+1)) then
!      else
!      endif
!      Assume maximum range jump in the iterative process equal to 1
       if (T_old<trange_gpu(lsp,nrangeloc)) then
        nrange(lsp) = nrange(lsp)-1
       elseif(T_old>trange_gpu(lsp,nrangeloc+1)) then 
        nrange(lsp) = nrange(lsp)+1
       endif
       nrange(lsp) = max(nrange(lsp),1)
       nrange(lsp) = min(nrange(lsp),nsetcv)
      enddo
     endif
!
     sumb = 0._rkind
     do lsp=1,N_S
      nrangeloc = nrange(lsp)
      sumb = sumb+cv_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)*yyroe(lsp)
     enddo
     ebar = ee-sumb
!
     den = 0._rkind
     num = 0._rkind
     do l=indx_cp_l,indx_cp_r
      T_pow  = T_old**l
      tden   = T_pow
      if (l==-1) then
       tnum   = log(T_old)
      else
       T_powp = T_old*T_pow
       tnum   = T_powp/(l+1._rkind)
      endif
      cv_l = 0._rkind
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
       cv_l = cv_l+cv_coeff_gpu(l,lsp,nrangeloc)*yyroe(lsp)
      enddo
      num = num+cv_l*tnum
      den = den+cv_l*tden
     enddo
     tt = T_old+(ebar-num)/den
     if (abs(tt-T_old) < tol_iter_nr) exit
     T_old = tt
!
    enddo
!
    get_mixture_temperature_from_e_roe_dev = tt
    endfunction get_mixture_temperature_from_e_roe_dev

    attributes(device) function get_mixture_temperature_from_h_roe_dev(hh,T_start,cp_coeff_gpu,nsetcv,trange_gpu, &
                                 indx_cp_l,indx_cp_r,tol_iter_nr,yyroe)
    real(rkind) :: get_mixture_temperature_from_h_roe_dev
    integer, value :: nsetcv,indx_cp_l,indx_cp_r
    real(rkind), value :: hh, T_start, tol_iter_nr
    real(rkind), dimension(N_S) :: yyroe
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
    real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
    real(rkind) :: tt,T_old,hbar,den,num,T_pow,T_powp,tden,tnum,sumb,sum0,tprod,cp_l
    integer :: nrangeloc,nmax,l,iter,max_iter,lsp,jl,jm,ju
    integer, dimension(N_S) :: nrange 
!
    max_iter = 10
!
    nmax = 100000
    do lsp=1,N_S
     nrange(lsp) = 1
    enddo
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do l=1,nmax
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (T_start >= trange_gpu(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
      nrange(lsp) = max(nrange(lsp),1)
      nrange(lsp) = min(nrange(lsp),nsetcv)
     enddo
    endif
!
    T_old = T_start
    do iter=1,max_iter
!
     if (nsetcv>1) then
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
!      if (T_old>trange_gpu(lsp,nrangeloc).and.T_old<trange_gpu(lsp,nrangeloc+1)) then
!      else
!      endif
!      Assume maximum range jump in the iterative process equal to 1
       if (T_old<trange_gpu(lsp,nrangeloc)) then
        nrange(lsp) = nrange(lsp)-1
       elseif(T_old>trange_gpu(lsp,nrangeloc+1)) then 
        nrange(lsp) = nrange(lsp)+1
       endif
       nrange(lsp) = max(nrange(lsp),1)
       nrange(lsp) = min(nrange(lsp),nsetcv)
      enddo
     endif
!
     sumb = 0._rkind
     do lsp=1,N_S
      nrangeloc = nrange(lsp)
      sumb = sumb+cp_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)*yyroe(lsp)
     enddo
     hbar = hh-sumb
!
     den = 0._rkind
     num = 0._rkind
     do l=indx_cp_l,indx_cp_r
      T_pow  = T_old**l
      tden   = T_pow
      if (l==-1) then
       tnum   = log(T_old)
      else
       T_powp = T_old*T_pow
       tnum   = T_powp/(l+1._rkind)
      endif
      cp_l = 0._rkind
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
       cp_l = cp_l+cp_coeff_gpu(l,lsp,nrangeloc)*yyroe(lsp)
      enddo
      num = num+cp_l*tnum
      den = den+cp_l*tden
     enddo
     tt = T_old+(hbar-num)/den
     if (abs(tt-T_old) < tol_iter_nr) exit
     T_old = tt
!
    enddo
!
    get_mixture_temperature_from_h_roe_dev = tt
    endfunction get_mixture_temperature_from_h_roe_dev

    attributes(device) function get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
        real(rkind) :: get_rho_from_w_dev
        integer, value :: i,j,k,nv,nx,ny,nz,ng
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv) :: w_gpu
        real(rkind) :: rho
        integer :: lsp
!
        rho = w_gpu(i,j,k,1)
        do lsp=2,N_S
         rho = rho+w_gpu(i,j,k,lsp)
        enddo
!
        get_rho_from_w_dev = rho
    endfunction get_rho_from_w_dev

    attributes(device) function get_rmixture(i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
        real(rkind) :: get_rmixture
        integer, value :: i,j,k,nv_aux,nx,ny,nz,ng
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
        real(rkind), dimension(N_S) :: rgas_gpu
        real(rkind) :: rm
        integer :: lsp
!
        rm = rgas_gpu(1)*w_aux_gpu(i,j,k,1)
        do lsp=2,N_S
         rm = rm+rgas_gpu(lsp)*w_aux_gpu(i,j,k,lsp)
        enddo
!
        get_rmixture = rm
    endfunction get_rmixture

    attributes(device) function get_mixture_e_from_temperature_roe_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                nsetcv,trange_gpu,yy)
        real(rkind) :: get_mixture_e_from_temperature_roe_dev
        real(rkind), value :: tt
        real(rkind), dimension(N_S) :: yy
        integer,value :: indx_cp_l, indx_cp_r, nsetcv
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind) :: ee,tprod,cv_l
        integer, dimension(N_S) :: nrange 
        integer :: l,lsp,jl,ju,jm,nmax,nrangeloc
!
        nmax = 100000
        do lsp=1,N_S
         nrange(lsp) = 1
        enddo
        if (nsetcv>1) then ! Replicate locate function of numerical recipes
         do lsp=1,N_S
          jl = 0
          ju = nsetcv+1+1
          do l=1,nmax
           if (ju-jl <= 1) exit
           jm = (ju+jl)/2
           if (tt >= trange_gpu(lsp,jm)) then
            jl=jm
           else
            ju=jm
           endif
          enddo
          nrange(lsp) = jl
          nrange(lsp) = max(nrange(lsp),1)
          nrange(lsp) = min(nrange(lsp),nsetcv)
         enddo
        endif
!
        ee = 0._rkind
        do lsp=1,N_S
         nrangeloc = nrange(lsp)
         ee = ee+cv_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)*yy(lsp)
        enddo
        do l=indx_cp_l,indx_cp_r
         if (l==-1) then
          tprod = log(tt)
         else
          tprod = tt**(l+1)/(l+1._rkind)
         endif
         cv_l = 0._rkind
         do lsp=1,N_S
          nrangeloc = nrange(lsp)
          cv_l = cv_l+cv_coeff_gpu(l,lsp,nrangeloc)*yy(lsp)
         enddo
         ee = ee+cv_l*tprod
        enddo
        get_mixture_e_from_temperature_roe_dev = ee

    endfunction get_mixture_e_from_temperature_roe_dev

    attributes(device) function get_mixture_e_from_temperature_ibm_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                nsetcv,trange_gpu,ibm_w_refl_gpu,ll,ibm_num_interface,nv)
        real(rkind) :: get_mixture_e_from_temperature_ibm_dev
        real(rkind), value :: tt
        integer,value :: indx_cp_l, indx_cp_r, nsetcv, ll, ibm_num_interface, nv
        real(rkind), dimension(ibm_num_interface,nv) :: ibm_w_refl_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind) :: ee,tprod,cv_l
        integer, dimension(N_S) :: nrange 
        integer :: nrangeloc,nmax,l,lsp,jl,ju,jm
!
        nmax = 100000
        do lsp=1,N_S
         nrange(lsp) = 1
        enddo
        if (nsetcv>1) then ! Replicate locate function of numerical recipes
         do lsp=1,N_S
          jl = 0
          ju = nsetcv+1+1
          do l=1,nmax
           if (ju-jl <= 1) exit
           jm = (ju+jl)/2
           if (tt >= trange_gpu(lsp,jm)) then
            jl=jm
           else
            ju=jm
           endif
          enddo
          nrange(lsp) = jl
          nrange(lsp) = max(nrange(lsp),1)
          nrange(lsp) = min(nrange(lsp),nsetcv)
         enddo
        endif
!
        ee = 0._rkind
        do lsp=1,N_S
         nrangeloc = nrange(lsp)
         ee = ee+cv_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)*ibm_w_refl_gpu(ll,lsp)
        enddo
        do l=indx_cp_l,indx_cp_r
         if (l==-1) then
          tprod = log(tt)
         else
          tprod = tt**(l+1)/(l+1._rkind)
         endif
         cv_l = 0._rkind
         do lsp=1,N_S
          nrangeloc = nrange(lsp)
          cv_l = cv_l+cv_coeff_gpu(l,lsp,nrangeloc)*ibm_w_refl_gpu(ll,lsp)
         enddo
         ee = ee+cv_l*tprod
        enddo
        get_mixture_e_from_temperature_ibm_dev = ee

    endfunction get_mixture_e_from_temperature_ibm_dev

    attributes(device) function get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                nsetcv,trange_gpu,i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
        real(rkind) :: get_mixture_e_from_temperature_dev
        real(rkind), value :: tt
        integer, value :: i,j,k,nv_aux,nx,ny,nz,ng,nsetcv,indx_cp_l, indx_cp_r
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind) :: ee,tprod,cv_l
        integer, dimension(N_S) :: nrange 
        integer :: nrangeloc,nmax,l,lsp,jl,ju,jm
!
        nmax = 100000
        do lsp=1,N_S
         nrange(lsp) = 1
        enddo
        if (nsetcv>1) then ! Replicate locate function of numerical recipes
         do lsp=1,N_S
          jl = 0
          ju = nsetcv+1+1
          do l=1,nmax
           if (ju-jl <= 1) exit
           jm = (ju+jl)/2
           if (tt >= trange_gpu(lsp,jm)) then
            jl=jm
           else
            ju=jm
           endif
          enddo
          nrange(lsp) = jl
          nrange(lsp) = max(nrange(lsp),1)
          nrange(lsp) = min(nrange(lsp),nsetcv)
         enddo
        endif
!
        ee = 0._rkind
        do lsp=1,N_S
         nrangeloc = nrange(lsp)
         ee = ee+cv_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)*w_aux_gpu(i,j,k,lsp)
        enddo
        do l=indx_cp_l,indx_cp_r
         if (l==-1) then
          tprod = log(tt)
         else
          tprod = tt**(l+1)/(l+1._rkind)
         endif
         cv_l = 0._rkind
         do lsp=1,N_S
          nrangeloc = nrange(lsp)
          cv_l = cv_l+cv_coeff_gpu(l,lsp,nrangeloc)*w_aux_gpu(i,j,k,lsp)
         enddo
         ee = ee+cv_l*tprod
        enddo
        get_mixture_e_from_temperature_dev = ee
!
    endfunction get_mixture_e_from_temperature_dev

    attributes(device) function get_species_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                nsetcv,trange_gpu,isp)
        real(rkind) :: get_species_e_from_temperature_dev
        real(rkind), value :: tt
        integer,value :: indx_cp_l,indx_cp_r,isp,nsetcv
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind) :: ee
        integer :: nmax,l,nrange,jl,jm,ju
!
        nmax = 100000
        nrange = 1
        if (nsetcv>1) then
         jl = 0
         ju = nsetcv+1+1
         do l=1,nmax
          if (ju-jl <= 1) exit
          jm = (ju+jl)/2
          if (tt >= trange_gpu(isp,jm)) then
           jl=jm
          else
           ju=jm
          endif
         enddo
         nrange = jl
         nrange = max(nrange,1)
         nrange = min(nrange,nsetcv)
        endif
        ee = cv_coeff_gpu(indx_cp_r+1,isp,nrange)
        do l=indx_cp_l,indx_cp_r
         if (l==-1) then
          ee = ee+cv_coeff_gpu(l,isp,nrange)*log(tt)
         else
          ee = ee+cv_coeff_gpu(l,isp,nrange)/(l+1._rkind)*(tt)**(l+1)
         endif
        enddo
        get_species_e_from_temperature_dev = ee
!
    endfunction get_species_e_from_temperature_dev

    attributes(device) function get_species_h_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu,&
                                nsetcv,trange_gpu,isp)
        real(rkind) :: get_species_h_from_temperature_dev
        real(rkind), value :: tt
        integer,value :: indx_cp_l,indx_cp_r,isp,nsetcv
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
        real(rkind) :: hh
        integer :: nmax,l,nrange,jl,jm,ju
!
        nmax = 100000
        nrange = 1
        if (nsetcv>1) then
         jl = 0
         ju = nsetcv+1+1
         do l=1,nmax
          if (ju-jl <= 1) exit
          jm = (ju+jl)/2
          if (tt >= trange_gpu(isp,jm)) then
           jl=jm
          else
           ju=jm
          endif
         enddo
         nrange = jl
         nrange = max(nrange,1)
         nrange = min(nrange,nsetcv)
        endif
        hh = cp_coeff_gpu(indx_cp_r+1,isp,nrange)
        do l=indx_cp_l,indx_cp_r
         if (l==-1) then
          hh = hh+cp_coeff_gpu(l,isp,nrange)*log(tt)
         else
          hh = hh+cp_coeff_gpu(l,isp,nrange)/(l+1._rkind)*(tt)**(l+1)
         endif
        enddo
        get_species_h_from_temperature_dev = hh
!
    endfunction get_species_h_from_temperature_dev

!    subroutine ibm_interpolation_cuf(ibm_num_interface,nx,ny,nz,ng,nv,nv_aux,indx_cp_l,indx_cp_r,ibm_ijk_refl_gpu, &
!               ibm_refl_type_gpu, w_gpu,w_aux_gpu,ibm_is_interface_node_gpu,ibm_coeff_d_gpu,ibm_coeff_n_gpu, &
!               ibm_bc_gpu,cv_coeff_gpu,nsetcv,trange_gpu,ibm_w_refl_gpu,ibm_nxyz_interface_gpu, &
!               rgas_gpu,ibm_parbc_gpu)
!!
!        integer, intent(in) :: ibm_num_interface,nx,ny,nz,ng,indx_cp_l,indx_cp_r,nv,nv_aux,nsetcv
!        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(in), device :: w_gpu
!        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(in), device :: w_aux_gpu
!        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: ibm_coeff_d_gpu
!        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: ibm_coeff_n_gpu
!        real(rkind), dimension(1:,1:), intent(in), device :: ibm_nxyz_interface_gpu
!        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: ibm_is_interface_node_gpu
!        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_refl_gpu
!        integer, dimension(1:), intent(in), device :: ibm_refl_type_gpu
!        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
!        real(rkind), dimension(1:,1:), intent(in), device :: ibm_parbc_gpu
!        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
!        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
!        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
!        real(rkind), dimension(1:,1:), intent(inout), device :: ibm_w_refl_gpu
!!
!        integer :: indx_patch
!        integer :: l,i,j,k,ii,jj,kk,lsp,iercuda
!        real(rkind) :: wi1,wi2,wi3,wi4,wi5,wi6,wsp
!        real(rkind) :: rho,uu,vv,ww,pp,tt
!        real(rkind) :: un,ut1,ut2,ut3
!        real(rkind) :: rmixtloc
!        real(rkind) :: twall_ibm
!!
!        if (ibm_num_interface>0) then
!         !$cuf kernel do(1) <<<*,*>>>
!         do l=1,ibm_num_interface
!!
!          i = ibm_ijk_refl_gpu(1,l)
!          j = ibm_ijk_refl_gpu(2,l)
!          k = ibm_ijk_refl_gpu(3,l)
!!
!          wi1 = 0._rkind
!          wi2 = 0._rkind
!          wi3 = 0._rkind
!          wi4 = 0._rkind
!          wi5 = 0._rkind
!          wi6 = 0._rkind
!!
!          type_refl: if (ibm_refl_type_gpu(l)==0) then
!!
!           do kk=1,2
!            do jj=1,2
!             do ii=1,2
!              wi1 = wi1 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
!              wi2 = wi2 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
!              wi3 = wi3 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
!              wi4 = wi4 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
!              wi5 = wi5 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
!             enddo
!            enddo
!           enddo
!           do lsp=1,N_S
!            wsp = 0._rkind
!            do kk=1,2
!             do jj=1,2
!              do ii=1,2
!               wsp = wsp + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,lsp)
!              enddo
!             enddo
!            enddo
!            ibm_w_refl_gpu(l,lsp) = wsp
!           enddo
!!
!          else
!!
!           rmixtloc = 0._rkind
!           do lsp=1,N_S ! Apply Neumann bc for species
!            wsp = 0._rkind
!            do kk=1,2
!             do jj=1,2
!              do ii=1,2
!!              wsp = wsp + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,lsp)
!               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
!               else
!                wsp = wsp + ibm_coeff_n_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,lsp)
!               endif
!              enddo
!             enddo
!            enddo
!            ibm_w_refl_gpu(l,lsp) = wsp
!            rmixtloc = rmixtloc+rgas_gpu(lsp)*wsp
!           enddo
!!
!           type_bc: select case(ibm_bc_gpu(1,l))
!!
!           case(5) ! Inviscid wall (un=> D, p,T,ut => N)
!!
!            do kk=1,2
!             do jj=1,2
!              do ii=1,2
!!
!               rho = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
!               uu  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
!               vv  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
!               ww  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
!               tt  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
!               pp  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_P)
!               un  = uu*ibm_nxyz_interface_gpu(1,l)+vv*ibm_nxyz_interface_gpu(2,l)+ww*ibm_nxyz_interface_gpu(3,l)
!               ut1 = uu-un*ibm_nxyz_interface_gpu(1,l)
!               ut2 = vv-un*ibm_nxyz_interface_gpu(2,l)
!               ut3 = ww-un*ibm_nxyz_interface_gpu(3,l)
!!
!               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
!                wi1 = wi1!+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
!               else
!                wi1 = wi1 + ibm_coeff_d_gpu(ii,jj,kk,l)*un
!                wi2 = wi2 + ibm_coeff_n_gpu(ii,jj,kk,l)*ut1
!                wi3 = wi3 + ibm_coeff_n_gpu(ii,jj,kk,l)*ut2
!                wi4 = wi4 + ibm_coeff_n_gpu(ii,jj,kk,l)*ut3
!                wi5 = wi5 + ibm_coeff_n_gpu(ii,jj,kk,l)*tt
!                wi6 = wi6 + ibm_coeff_n_gpu(ii,jj,kk,l)*pp
!               endif
!!
!              enddo
!             enddo
!            enddo
!!
!            wi2 = wi2 + wi1*ibm_nxyz_interface_gpu(1,l)
!            wi3 = wi3 + wi1*ibm_nxyz_interface_gpu(2,l)
!            wi4 = wi4 + wi1*ibm_nxyz_interface_gpu(3,l)
!            wi1 = wi6/wi5/rmixtloc ! rho = p/tt/R
!!
!           case(6,16) ! Viscous isothermal wall (u,v,w,T => D, p => N)
!          
!           indx_patch = ibm_bc_gpu(2,l)
!           twall_ibm  = ibm_parbc_gpu(indx_patch,1)
!
!            do kk=1,2
!             do jj=1,2
!              do ii=1,2
!!
!               rho = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
!               uu  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
!               vv  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
!               ww  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
!               tt  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
!               pp  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_P)
!!
!               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
!                wi2 = wi2 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
!                wi3 = wi3 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
!                wi4 = wi4 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
!                wi5 = wi5  + ibm_coeff_d_gpu(ii,jj,kk,l)*twall_ibm
!               else
!                wi2 = wi2 + ibm_coeff_d_gpu(ii,jj,kk,l)*uu
!                wi3 = wi3 + ibm_coeff_d_gpu(ii,jj,kk,l)*vv
!                wi4 = wi4 + ibm_coeff_d_gpu(ii,jj,kk,l)*ww
!                wi5 = wi5 + ibm_coeff_d_gpu(ii,jj,kk,l)*tt
!                wi6 = wi6 + ibm_coeff_n_gpu(ii,jj,kk,l)*pp
!               endif
!!
!              enddo
!             enddo
!            enddo
!!
!            wi1 = wi6/wi5/rmixtloc ! rho = p/tt/R
!!
!           case(2,8,18) ! Viscous adiabatic wall (u,v,w => D, p,T => N)
!          
!            do kk=1,2
!             do jj=1,2
!              do ii=1,2
!!
!!              rho = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
!               uu  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
!               vv  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
!               ww  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
!               tt  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
!               pp  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_P)
!!
!               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
!                wi2 = wi2 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
!                wi3 = wi3 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
!                wi4 = wi4 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
!               else
!                wi2 = wi2 + ibm_coeff_d_gpu(ii,jj,kk,l)*uu
!                wi3 = wi3 + ibm_coeff_d_gpu(ii,jj,kk,l)*vv
!                wi4 = wi4 + ibm_coeff_d_gpu(ii,jj,kk,l)*ww
!                wi5 = wi5 + ibm_coeff_n_gpu(ii,jj,kk,l)*tt
!                wi6 = wi6 + ibm_coeff_n_gpu(ii,jj,kk,l)*pp
!               endif
!!
!              enddo
!             enddo
!            enddo
!!
!            wi1 = wi6/wi5/rmixtloc ! rho = p/tt/R
!!
!           case(1,9) ! Supersonic inflow (no interpolation needed because ibm_w_refl_gpu not used in forcing)
!!
!           endselect type_bc
!!
!          endif type_refl
!!
!          do lsp=1,N_S
!           ibm_w_refl_gpu(l,lsp) = ibm_w_refl_gpu(l,lsp)*wi1
!          enddo
!          ibm_w_refl_gpu(l,I_U) = wi2
!          ibm_w_refl_gpu(l,I_V) = wi3
!          ibm_w_refl_gpu(l,I_W) = wi4
!          ibm_w_refl_gpu(l,I_E) = wi5
!!
!         enddo
!         !@cuf iercuda=cudaDeviceSynchronize()
!        endif  
!!
!    end subroutine ibm_interpolation_cuf

    subroutine ibm_interpolation_cons_cuf(ibm_num_interface,nx,ny,nz,ng,ibm_ijk_refl_gpu,ibm_refl_type_gpu,&
               w_gpu,ibm_is_interface_node_gpu,ibm_coeff_d_gpu,ibm_coeff_n_gpu,ibm_w_refl_gpu,ibm_bc_gpu)
!
        integer, intent(in) :: ibm_num_interface,nx,ny,nz,ng
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(in), device :: w_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: ibm_coeff_d_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: ibm_coeff_n_gpu
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: ibm_is_interface_node_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_refl_gpu
        integer, dimension(1:), intent(in), device :: ibm_refl_type_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
        real(rkind), dimension(1:,1:), intent(inout), device :: ibm_w_refl_gpu
!
        integer :: l,i,j,k,ii,jj,kk,lsp,iercuda
        real(rkind) :: wi2,wi3,wi4,wi5,wsp
!
        if (ibm_num_interface>0) then
         !$cuf kernel do(1) <<<*,*>>>
         do l=1,ibm_num_interface
!
          i = ibm_ijk_refl_gpu(1,l)
          j = ibm_ijk_refl_gpu(2,l)
          k = ibm_ijk_refl_gpu(3,l)
!
          wi2 = 0._rkind
          wi3 = 0._rkind
          wi4 = 0._rkind
          wi5 = 0._rkind
!
          type_refl: if (ibm_refl_type_gpu(l)==0) then
!
           do kk=1,2
            do jj=1,2
             do ii=1,2
              wi2 = wi2 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_U)
              wi3 = wi3 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_V)
              wi4 = wi4 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_W)
              wi5 = wi5 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_E)
             enddo
            enddo
           enddo
           do lsp=1,N_S
            wsp = 0._rkind
            do kk=1,2
             do jj=1,2
              do ii=1,2
               wsp = wsp + ibm_coeff_d_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,lsp)
              enddo
             enddo
            enddo
            ibm_w_refl_gpu(l,lsp) = wsp
           enddo
!
          else
!
           if (ibm_bc_gpu(1,l) == 5 ) then
           !case(5) ! Inviscid wall (un=> D, p,T,ut => N)
            do kk=1,2
             do jj=1,2
              do ii=1,2
               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
               else
                wi2 = wi2 + ibm_coeff_n_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_U) 
                wi3 = wi3 + ibm_coeff_n_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_V)
                wi4 = wi4 + ibm_coeff_n_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_W)
                wi5 = wi5 + ibm_coeff_n_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_E)
               endif
              enddo
             enddo
            enddo
            do lsp=1,N_S
             wsp = 0._rkind
             do kk=1,2
              do jj=1,2
               do ii=1,2
                if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
                else
                 wsp = wsp + ibm_coeff_n_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,lsp)
                endif
               enddo
              enddo
             enddo
             ibm_w_refl_gpu(l,lsp) = wsp
            enddo
!
           !case default  
           else
            do kk=1,2
             do jj=1,2
              do ii=1,2
               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
                wi2 = wi2 
                wi3 = wi3 
                wi4 = wi4 
               else
                wi2 = wi2 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_U) 
                wi3 = wi3 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_V)
                wi4 = wi4 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_W)
                wi5 = wi5 + ibm_coeff_n_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,I_E)
               endif
              enddo
             enddo
            enddo
            do lsp=1,N_S
             wsp = 0._rkind
             do kk=1,2
              do jj=1,2
               do ii=1,2
                if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
                else
                 wsp = wsp + ibm_coeff_n_gpu(ii,jj,kk,l)*w_gpu(i+ii-1,j+jj-1,k+kk-1,lsp)
                endif
               enddo
              enddo
             enddo
             ibm_w_refl_gpu(l,lsp) = wsp
            enddo
!
           endif
           !endselect
!
          endif type_refl
!
          ibm_w_refl_gpu(l,I_U) = wi2
          ibm_w_refl_gpu(l,I_V) = wi3
          ibm_w_refl_gpu(l,I_W) = wi4
          ibm_w_refl_gpu(l,I_E) = wi5
!
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()
        endif  
!
    end subroutine ibm_interpolation_cons_cuf
!
    subroutine ibm_forcing_cuf(ibm_num_interface,nx,ny,nz,ng,nv,indx_cp_l,indx_cp_r,ibm_ijk_interface_gpu,w_gpu,w_aux_gpu,ibm_bc_gpu, &
                               cv_coeff_gpu,nsetcv,trange_gpu,ibm_w_refl_gpu, ibm_nxyz_interface_gpu,  &
                               ibm_bc_relax_factor,rgas_gpu,ibm_parbc_gpu,ibm_num_bc,tol_iter_nr, randvar_a_gpu,randvar_p_gpu,&
                               time,x_gpu,y_gpu,z_gpu)
!
        integer, intent(in) :: ibm_num_interface,nx,ny,nz,ng,indx_cp_l,indx_cp_r,nsetcv,nv,ibm_num_bc
        real(rkind), intent(in) :: ibm_bc_relax_factor,tol_iter_nr,time
        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_interface_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_nxyz_interface_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(ibm_num_interface,nv), intent(inout), device :: ibm_w_refl_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1-ng:), intent(in), device :: x_gpu,y_gpu,z_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_parbc_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
        real(rkind), dimension(8), intent(in),device :: randvar_a_gpu,randvar_p_gpu
!
        integer :: l,i,j,k,lsp,iercuda
        integer :: indx_patch,modes
        real(rkind) :: rho,uu,vv,ww,pp,tt,yy
        real(rkind) :: rhoi,uui,vvi,wwi,ppi,tti,eei,rhoui,rhovi,rhowi
        real(rkind) :: rad,unorm2,rmixtloc
        real(rkind) :: twall_ibm,pp_patch,tt_patch,vel_patch
        real(rkind) :: rhou,rhov,rhow
        real(rkind) :: cvloc,gamloc
        real(rkind) :: ptotal,ttotal,del,rfac,rmfac,rml,vel_mod
        real(rkind) :: qq,ee,rhoe,ri
        real(rkind) :: avar_gpu,inlet_phi,utmp,upert,yc_patch,zc_patch
!
        if (ibm_num_interface>0) then
         !$cuf kernel do(1) <<<*,*>>>
         do l=1,ibm_num_interface
!
          i = ibm_ijk_interface_gpu(1,l)
          j = ibm_ijk_interface_gpu(2,l)
          k = ibm_ijk_interface_gpu(3,l)
!
!         Storing momentum for force computation
!         w2   = w_gpu(i,j,k,I_U)
!         w3   = w_gpu(i,j,k,I_V)
!         w4   = w_gpu(i,j,k,I_W)
!
!         Quantities at reflected node


          rho = ibm_w_refl_gpu(l,1)
          do lsp=2,N_S
           rho  = rho+ibm_w_refl_gpu(l,lsp)
          enddo
          ri = 1._rkind/rho
          uu   = ibm_w_refl_gpu(l,I_U)*ri
          vv   = ibm_w_refl_gpu(l,I_V)*ri
          ww   = ibm_w_refl_gpu(l,I_W)*ri
          rhoe = ibm_w_refl_gpu(l,I_E)
          qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
          ee   = rhoe*ri-qq
          rmixtloc = 0._rkind
          do lsp=1,N_S
           yy = ibm_w_refl_gpu(l,lsp)*ri
           ibm_w_refl_gpu(l,lsp) = yy !storing yi in ibm_w_refl_gpu(1:N_S)
           rmixtloc = rmixtloc+rgas_gpu(lsp)*yy
          enddo
          tt   = get_mixture_temperature_from_e_ibm_dev(ee,w_aux_gpu(i,j,k,J_T),cv_coeff_gpu,nsetcv,trange_gpu, &
                 indx_cp_l,indx_cp_r,tol_iter_nr,ibm_num_interface,nv,l,ibm_w_refl_gpu)

          pp   = rho*tt*rmixtloc
!
          !select case(ibm_bc_gpu(1,l))
          if (ibm_bc_gpu(1,l) == 1) then 
          !case(1) ! supersonic inflow (input pp_patch, tt_patch, vel_patch, composizione)
!
           indx_patch = ibm_bc_gpu(2,l) 
           rmixtloc = 0._rkind
           do lsp=1,N_S
!           yy = w_aux_gpu(i,j,k,lsp)+ibm_bc_relax_factor*(ibm_parbc_gpu(indx_patch,3+lsp)-w_aux_gpu(i,j,k,lsp))
            yy = ibm_parbc_gpu(indx_patch,3+lsp)
            ibm_w_refl_gpu(l,lsp) = yy
            rmixtloc = rmixtloc+rgas_gpu(lsp)*yy
           enddo
           pp_patch  = ibm_parbc_gpu(indx_patch,1)
           tt_patch  = ibm_parbc_gpu(indx_patch,2)
           vel_patch = ibm_parbc_gpu(indx_patch,3)
           ppi = pp_patch
           tti = tt_patch
           uui = vel_patch*ibm_nxyz_interface_gpu(1,l)
           vvi = vel_patch*ibm_nxyz_interface_gpu(2,l)
           wwi = vel_patch*ibm_nxyz_interface_gpu(3,l)
!
          !case(2) ! subsonic inflow (p0, T0, composizione)
           elseif (ibm_bc_gpu(1,l) == 2) then
!
           cvloc  = get_cp_ibm_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,ibm_w_refl_gpu,l,ibm_num_interface,nv)
           gamloc = (cvloc+rmixtloc)/cvloc
           indx_patch = ibm_bc_gpu(2,l)
           ptotal = ibm_parbc_gpu(indx_patch,1)
           ttotal = ibm_parbc_gpu(indx_patch,2)
           rmixtloc = 0._rkind
           do lsp=1,N_S
            yy = ibm_parbc_gpu(indx_patch,2+lsp)
            ibm_w_refl_gpu(l,lsp) = yy
            rmixtloc = rmixtloc+rgas_gpu(lsp)*yy
           enddo
           del = 0.5_rkind*(gamloc-1._rkind)
           rmfac  = (pp/ptotal)**(-(gamloc-1._rkind)/gamloc)
           rml = sqrt((rmfac-1._rkind)/del)
           ppi = pp
           tti = ttotal/rmfac
           vel_mod = rml*sqrt(gamloc*rmixtloc*tti)
           uui = vel_mod*ibm_nxyz_interface_gpu(1,l)
           vvi = vel_mod*ibm_nxyz_interface_gpu(2,l)
           wwi = vel_mod*ibm_nxyz_interface_gpu(3,l)
!
          !case (3) ! Supersonic Turbulent Inflow (yp,zp,pp_patch,tt_patch,vel_patch, composizione))
          elseif (ibm_bc_gpu(1,l) == 3) then
           indx_patch = ibm_bc_gpu(2,l)
           !rintinv = 436.681222707_rkind !1/rint
           !nexp = 0.15151515_rkind !1/6.6
           rmixtloc = 0._rkind
           do lsp=1,N_S
!           yy = w_aux_gpu(i,j,k,lsp)+ibm_bc_relax_factor*(ibm_parbc_gpu(indx_patch,3+lsp)-w_aux_gpu(i,j,k,lsp))
            yy = ibm_parbc_gpu(indx_patch,5+lsp)
            ibm_w_refl_gpu(l,lsp) = yy
            rmixtloc = rmixtloc+rgas_gpu(lsp)*yy
           enddo
           yc_patch  = ibm_parbc_gpu(indx_patch,1)
           zc_patch  = ibm_parbc_gpu(indx_patch,2)
           pp_patch  = ibm_parbc_gpu(indx_patch,3)
           tt_patch  = ibm_parbc_gpu(indx_patch,4)
           vel_patch = ibm_parbc_gpu(indx_patch,5)
           ppi = pp_patch
           tti = tt_patch

           rad = (((y_gpu(j) - yc_patch)**2_rkind + (z_gpu(k) - zc_patch)**2_rkind)**0.5_rkind)*436.681222707_rkind
           inlet_phi = datan2(z_gpu(k)-zc_patch,y_gpu(j)-yc_patch)
           utmp = vel_patch*((1-rad)**0.151515151515_rkind)*ibm_nxyz_interface_gpu(1,l)
           avar_gpu = 0.5_rkind*0.05_rkind*(utmp-0.7_rkind)
           upert = 0._rkind
           do modes=1,8
            if (modes .le. 6) then
             upert = upert + avar_gpu*sin(randvar_a_gpu(modes) + modes*inlet_phi+randvar_p_gpu(modes)*time)
            else
             upert = upert + avar_gpu*sin(randvar_p_gpu(modes)*time)
            endif
           enddo

           uui = utmp + upert
           vvi = vel_patch*ibm_nxyz_interface_gpu(2,l)
           wwi = vel_patch*ibm_nxyz_interface_gpu(3,l)
!
          !case (4) ! Subsonic Turbulent Inflow (yp,zp,p0,T0,composizione)
          elseif (ibm_bc_gpu(1,l) == 4) then
           cvloc  = get_cp_ibm_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,ibm_w_refl_gpu,l,ibm_num_interface,nv)
           gamloc = (cvloc+rmixtloc)/cvloc
           indx_patch = ibm_bc_gpu(2,l)
           yc_patch  = ibm_parbc_gpu(indx_patch,1)
           zc_patch  = ibm_parbc_gpu(indx_patch,2)
           ptotal = ibm_parbc_gpu(indx_patch,3)
           ttotal = ibm_parbc_gpu(indx_patch,4)
           rmixtloc = 0._rkind
           do lsp=1,N_S
            yy = ibm_parbc_gpu(indx_patch,4+lsp)
            ibm_w_refl_gpu(l,lsp) = yy
            rmixtloc = rmixtloc+rgas_gpu(lsp)*yy
           enddo
           del = 0.5_rkind*(gamloc-1._rkind)
           rmfac  = (pp/ptotal)**(-(gamloc-1._rkind)/gamloc)
           rml = sqrt((rmfac-1._rkind)/del)
           ppi = pp
           tti = ttotal/rmfac
           vel_mod = rml*sqrt(gamloc*rmixtloc*tti)

           rad = (((y_gpu(j) - yc_patch)**2_rkind + (z_gpu(k) - zc_patch)**2_rkind)**0.5_rkind)*436.681222707_rkind
           inlet_phi = datan2(z_gpu(k)-zc_patch,y_gpu(j)-yc_patch)
           utmp = vel_mod*((1-rad)**0.151515151515_rkind)*ibm_nxyz_interface_gpu(1,l)
           avar_gpu = 0.5_rkind*0.05_rkind*(utmp-0.7_rkind)
           upert = 0._rkind
           do modes=1,8
            if (modes .le. 6) then
             upert = upert + avar_gpu*sin(randvar_a_gpu(modes) + modes*inlet_phi+randvar_p_gpu(modes)*time)
            else
             upert = upert + avar_gpu*sin(randvar_p_gpu(modes)*time)
            endif
           enddo

           uui = utmp + upert
           vvi = vel_mod*ibm_nxyz_interface_gpu(2,l)
           wwi = vel_mod*ibm_nxyz_interface_gpu(3,l)
!
          !case (5) ! Inviscid adiabatic wall
          elseif (ibm_bc_gpu(1,l) == 5) then
!
           unorm2 = 2._rkind*(uu*ibm_nxyz_interface_gpu(1,l)+vv*ibm_nxyz_interface_gpu(2,l)+ww*ibm_nxyz_interface_gpu(3,l))
           uui  = uu-unorm2*ibm_nxyz_interface_gpu(1,l)
           vvi  = vv-unorm2*ibm_nxyz_interface_gpu(2,l)
           wwi  = ww-unorm2*ibm_nxyz_interface_gpu(3,l)
           ppi  = pp ! extrapolate from interior
           tti  = tt ! extrapolate from interior
!
          !case (6,16) ! Viscous isothermal wall
          elseif (ibm_bc_gpu(1,l) == 6 .or. ibm_bc_gpu(1,l) == 16) then
!
           indx_patch = ibm_bc_gpu(2,l) 
           twall_ibm  = ibm_parbc_gpu(indx_patch,1)
           ppi  = pp ! extrapolate from interior
           tti  = 2._rkind*twall_ibm-tt
           if (tti < 200._rkind) tti=200._rkind
           uui  = -uu
           vvi  = -vv
           wwi  = -ww
!
          !case (8,18) ! Adiabatic wall
          elseif (ibm_bc_gpu(1,l) == 8 .or. ibm_bc_gpu(1,l) == 18) then
!
           ppi  = pp ! extrapolate from interior
           tti  = tt ! extrapolate from interior
           uui  = -uu
           vvi  = -vv
           wwi  = -ww
!
          !end select 
          endif
!
          ppi = pp+ibm_bc_relax_factor*(ppi-pp)
          uui = uu+ibm_bc_relax_factor*(uui-uu)
          vvi = vv+ibm_bc_relax_factor*(vvi-vv)
          wwi = ww+ibm_bc_relax_factor*(wwi-ww)
          tti = tt+ibm_bc_relax_factor*(tti-tt)
!
          rhoi  = ppi/tti/rmixtloc
          rhoui = rhoi*uui
          rhovi = rhoi*vvi
          rhowi = rhoi*wwi
!         Compute energy mixture
          eei = get_mixture_e_from_temperature_ibm_dev(tti,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                nsetcv,trange_gpu,ibm_w_refl_gpu,l,ibm_num_interface,nv)
!
          do lsp=1,N_S
           w_gpu(i,j,k,lsp) = rhoi*ibm_w_refl_gpu(l,lsp)
          enddo
          w_gpu(i,j,k,I_U) = rhoui
          w_gpu(i,j,k,I_V) = rhovi
          w_gpu(i,j,k,I_W) = rhowi
          w_gpu(i,j,k,I_E) = rhoi*eei+0.5_rkind*(rhoui*rhoui+rhovi*rhovi+rhowi*rhowi)/rhoi
!
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()
!
        endif
!
    end subroutine ibm_forcing_cuf
!
    subroutine ibm_eikonal_cons_cuf(nx,ny,nz,ng,w_gpu,x_gpu,y_gpu,z_gpu,ibm_dw_aux_eikonal_gpu,         &
                               ibm_body_dist_gpu,ibm_is_interface_node_gpu,                                     &
                               ibm_reflection_coeff_gpu,ibm_num_interface,ibm_w_refl_gpu,ibm_ijk_interface_gpu, &
                               ibm_bc_gpu,ibm_eikonal_cfl,ibm_indx_eikonal)
        integer :: nx,ny,nz,ng,ibm_num_interface,ibm_indx_eikonal
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: ibm_is_interface_node_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_interface_gpu 
        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
        real(rkind), intent(in) :: ibm_eikonal_cfl
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_gpu
        real(rkind), dimension(1-ng:), intent(in), device :: x_gpu,y_gpu,z_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: ibm_dw_aux_eikonal_gpu
        real(rkind), dimension(0-ng:,0-ng:,0-ng:), intent(inout), device :: ibm_body_dist_gpu
        real(rkind), dimension(1:,1:,1:), intent(in), device :: ibm_reflection_coeff_gpu
        real(rkind), dimension(1:,1:), intent(inout), device :: ibm_w_refl_gpu
        real(rkind) :: rho,rhou,rhov,rhow,rhoe,uu,vv,ww,qq,pp,tt,ee
        real(rkind) :: dtau,dxloc,dyloc,dzloc,normxdxi,normydyi,normzdzi
        real(rkind) :: normx,normy,normz,nmod,dxi,dyi,dzi,dtime
        real(rkind) :: de,drho,du,dv,dw,refl_coeff,drhoi
        integer :: i,j,k,l,lsp,iercuda,ii,jj,kk
        integer :: indx_eikonal
!
!       Steady-state solution of convection along normal (towards inside of the body)
!       (Di Mascio and Zaghi, 2021)
!
        do indx_eikonal=1,ibm_indx_eikonal ! loop on eikonal iterations
!        
        !$cuf kernel do(3) <<<*,*>>>
         do k=1,nz
          do j=1,ny
           do i=1,nx
!           if (ibm_body_dist_gpu(i,j,k)>0._rkind) then
            if (ibm_is_interface_node_gpu(i,j,k)==1) then
             normx = ibm_body_dist_gpu(i+1,j,k)-ibm_body_dist_gpu(i-1,j,k)
             normy = ibm_body_dist_gpu(i,j+1,k)-ibm_body_dist_gpu(i,j-1,k)
             normz = ibm_body_dist_gpu(i,j,k+1)-ibm_body_dist_gpu(i,j,k-1)
             dxloc = 0.5_rkind*(x_gpu(i+1)-x_gpu(i-1))
             dyloc = 0.5_rkind*(y_gpu(j+1)-y_gpu(j-1))
             dzloc = 0.5_rkind*(z_gpu(k+1)-z_gpu(k-1))
             normx = normx/(x_gpu(i+1)-x_gpu(i-1))
             normy = normy/(y_gpu(j+1)-y_gpu(j-1))
             normz = normz/(z_gpu(k+1)-z_gpu(k-1))
!            nmod = abs(normx) + abs(normy) + abs(normz)
!            nmod = 0.9_rkind/nmod ! cfl = 0.9
!            normx = normx*nmod
!            normy = normy*nmod
!            normz = normz*nmod
             dtau  = min(dxloc,dyloc)
             dtau  = min(dtau,dzloc)
!            dtau  = dtau*0.9_rkind
             dtau  = dtau*ibm_eikonal_cfl
!             drho  = 0._rkind
             du    = 0._rkind
             dv    = 0._rkind
             dw    = 0._rkind
             de    = 0._rkind
             if (normx > 0._rkind) then
                dxi  = dtau/(x_gpu(i)-x_gpu(i-1))
                normxdxi = normx*dxi
!                drho = drho + normxdxi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i-1,j,k,J_R))
                du   = du   + normxdxi*(w_gpu(i,j,k,I_U)-w_gpu(i-1,j,k,I_U))
                dv   = dv   + normxdxi*(w_gpu(i,j,k,I_V)-w_gpu(i-1,j,k,I_V))
                dw   = dw   + normxdxi*(w_gpu(i,j,k,I_W)-w_gpu(i-1,j,k,I_W))
                de   = de   + normxdxi*(w_gpu(i,j,k,I_E)-w_gpu(i-1,j,k,I_E))
             else
                dxi  = dtau/(x_gpu(i+1)-x_gpu(i))
                normxdxi = normx*dxi
!                drho = drho + normxdxi*(w_aux_gpu(i+1,j,k,J_R)-w_aux_gpu(i,j,k,J_R))
                du   = du   + normxdxi*(w_gpu(i+1,j,k,I_U)-w_gpu(i,j,k,I_U))
                dv   = dv   + normxdxi*(w_gpu(i+1,j,k,I_V)-w_gpu(i,j,k,I_V))
                dw   = dw   + normxdxi*(w_gpu(i+1,j,k,I_W)-w_gpu(i,j,k,I_W))
                de   = de   + normxdxi*(w_gpu(i+1,j,k,I_E)-w_gpu(i,j,k,I_E))
             endif
             if (normy > 0._rkind) then
                dyi = dtau/(y_gpu(j)-y_gpu(j-1))
                normydyi = normy*dyi
!                drho = drho + normydyi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i,j-1,k,J_R))
                du   = du   + normydyi*(w_gpu(i,j,k,I_U)-w_gpu(i,j-1,k,I_U))
                dv   = dv   + normydyi*(w_gpu(i,j,k,I_V)-w_gpu(i,j-1,k,I_V))
                dw   = dw   + normydyi*(w_gpu(i,j,k,I_W)-w_gpu(i,j-1,k,I_W))
                de   = de   + normydyi*(w_gpu(i,j,k,I_E)-w_gpu(i,j-1,k,I_E))
             else
                dyi  = dtau/(y_gpu(j+1)-y_gpu(j))
                normydyi = normy*dyi
!                drho = drho + normydyi*(w_aux_gpu(i,j+1,k,J_R)-w_aux_gpu(i,j,k,J_R))
                du   = du   + normydyi*(w_gpu(i,j+1,k,I_U)-w_gpu(i,j,k,I_U))
                dv   = dv   + normydyi*(w_gpu(i,j+1,k,I_V)-w_gpu(i,j,k,I_V))
                dw   = dw   + normydyi*(w_gpu(i,j+1,k,I_W)-w_gpu(i,j,k,I_W))
                de   = de   + normydyi*(w_gpu(i,j+1,k,I_E)-w_gpu(i,j,k,I_E))
             endif
             if (normz > 0._rkind) then
                dzi  = dtau/(z_gpu(k)-z_gpu(k-1))
                normzdzi = normz*dzi
!                drho = drho + normzdzi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i,j,k-1,J_R))
                du   = du   + normzdzi*(w_gpu(i,j,k,I_U)-w_gpu(i,j,k-1,I_U))
                dv   = dv   + normzdzi*(w_gpu(i,j,k,I_V)-w_gpu(i,j,k-1,I_V))
                dw   = dw   + normzdzi*(w_gpu(i,j,k,I_W)-w_gpu(i,j,k-1,I_W))
                de   = de   + normzdzi*(w_gpu(i,j,k,I_E)-w_gpu(i,j,k-1,I_E))
             else
                dzi = dtau/(z_gpu(k+1)-z_gpu(k))
                normzdzi = normz*dzi
!                drho = drho + normzdzi*(w_aux_gpu(i,j,k+1,J_R)-w_aux_gpu(i,j,k,J_R))
                du   = du   + normzdzi*(w_gpu(i,j,k+1,I_U)-w_gpu(i,j,k,I_U))
                dv   = dv   + normzdzi*(w_gpu(i,j,k+1,I_V)-w_gpu(i,j,k,I_V))
                dw   = dw   + normzdzi*(w_gpu(i,j,k+1,I_W)-w_gpu(i,j,k,I_W))
                de   = de   + normzdzi*(w_gpu(i,j,k+1,I_E)-w_gpu(i,j,k,I_E))
             endif
!             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+1) = drho
             ibm_dw_aux_eikonal_gpu(i,j,k,I_U) = du
             ibm_dw_aux_eikonal_gpu(i,j,k,I_V) = dv
             ibm_dw_aux_eikonal_gpu(i,j,k,I_W) = dw
             ibm_dw_aux_eikonal_gpu(i,j,k,I_E) = de
             do lsp=1,N_S
              drhoi = 0._rkind
              if (normx > 0._rkind) then
                 dxi  = dtau/(x_gpu(i)-x_gpu(i-1))
                 normxdxi = normx*dxi
                 drhoi = drhoi + normxdxi*(w_gpu(i,j,k,lsp)-w_gpu(i-1,j,k,lsp))
              else
                 dxi  = dtau/(x_gpu(i+1)-x_gpu(i))
                 normxdxi = normx*dxi
                 drhoi = drhoi + normxdxi*(w_gpu(i+1,j,k,lsp)-w_gpu(i,j,k,lsp))
              endif
              if (normy > 0._rkind) then
                 dyi = dtau/(y_gpu(j)-y_gpu(j-1))
                 normydyi = normy*dyi
                 drhoi = drhoi + normydyi*(w_gpu(i,j,k,lsp)-w_gpu(i,j-1,k,lsp))
              else
                 dyi = dtau/(y_gpu(j+1)-y_gpu(j))
                 normydyi = normy*dyi
                 drhoi = drhoi + normydyi*(w_gpu(i,j+1,k,lsp)-w_gpu(i,j,k,lsp))
              endif
              if (normz > 0._rkind) then
                 dzi  = dtau/(z_gpu(k)-z_gpu(k-1))
                 normzdzi = normz*dzi
                 drhoi = drhoi + normzdzi*(w_gpu(i,j,k,lsp)-w_gpu(i,j,k-1,lsp))
              else
                 dzi  = dtau/(z_gpu(k+1)-z_gpu(k))
                 normzdzi = normz*dzi
                 drhoi = drhoi + normzdzi*(w_gpu(i,j,k+1,lsp)-w_gpu(i,j,k,lsp))
              endif
              ibm_dw_aux_eikonal_gpu(i,j,k,lsp) = drhoi
             enddo
            endif
           enddo
          enddo
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()

         !$cuf kernel do(3) <<<*,*>>>
          do k=1,nz
           do j=1,ny
            do i=1,nx
             if (ibm_is_interface_node_gpu(i,j,k)==1) then
!              w_aux_gpu(i,j,k,J_R) = w_aux_gpu(i,j,k,J_R)-ibm_dw_aux_eikonal_gpu(i,j,k,N_S+1)
              w_gpu(i,j,k,I_U) = w_gpu(i,j,k,I_U)-ibm_dw_aux_eikonal_gpu(i,j,k,I_U) 
              w_gpu(i,j,k,I_V) = w_gpu(i,j,k,I_V)-ibm_dw_aux_eikonal_gpu(i,j,k,I_V)
              w_gpu(i,j,k,I_W) = w_gpu(i,j,k,I_W)-ibm_dw_aux_eikonal_gpu(i,j,k,I_W)
              w_gpu(i,j,k,I_E) = w_gpu(i,j,k,I_E)-ibm_dw_aux_eikonal_gpu(i,j,k,I_E)
              do lsp=1,N_S
               w_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)-ibm_dw_aux_eikonal_gpu(i,j,k,lsp)
              enddo
             endif
            enddo
           enddo
          enddo
         !@cuf iercuda=cudaDeviceSynchronize()
!         
        enddo
!
        if (ibm_num_interface>0) then
         !$cuf kernel do(1) <<<*,*>>>
         do l=1,ibm_num_interface
          i = ibm_ijk_interface_gpu(1,l)
          j = ibm_ijk_interface_gpu(2,l)
          k = ibm_ijk_interface_gpu(3,l)
          do lsp=1,N_S
           ibm_w_refl_gpu(l,lsp) = w_gpu(i,j,k,lsp)
          enddo
          ibm_w_refl_gpu(l,I_U) = w_gpu(i,j,k,I_U)
          ibm_w_refl_gpu(l,I_V) = w_gpu(i,j,k,I_V)
          ibm_w_refl_gpu(l,I_W) = w_gpu(i,j,k,I_W)
          ibm_w_refl_gpu(l,I_E) = w_gpu(i,j,k,I_E)
         enddo
!        !@cuf iercuda=cudaDeviceSynchronize()
        endif
!
    end subroutine ibm_eikonal_cons_cuf
!
!    subroutine ibm_eikonal_cuf(nx,ny,nz,ng,w_gpu,x_gpu,y_gpu,z_gpu,w_aux_gpu,ibm_dw_aux_eikonal_gpu,         &
!                               ibm_body_dist_gpu,ibm_is_interface_node_gpu,                                     &
!                               ibm_reflection_coeff_gpu,ibm_num_interface,ibm_w_refl_gpu,ibm_ijk_interface_gpu, &
!                               ibm_bc_gpu,ibm_eikonal_cfl)
!        integer :: nx,ny,nz,ng,ibm_num_interface
!        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: ibm_is_interface_node_gpu
!        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_interface_gpu
!        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
!        real(rkind), intent(in) :: ibm_eikonal_cfl
!        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_gpu
!        real(rkind), dimension(1-ng:), intent(in), device :: x_gpu,y_gpu,z_gpu
!        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_aux_gpu
!        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: ibm_dw_aux_eikonal_gpu
!        real(rkind), dimension(0-ng:,0-ng:,0-ng:), intent(inout), device :: ibm_body_dist_gpu
!        real(rkind), dimension(1:,1:,1:), intent(in), device :: ibm_reflection_coeff_gpu
!        real(rkind), dimension(1:,1:), intent(inout), device :: ibm_w_refl_gpu
!        real(rkind) :: rho,rhou,rhov,rhow,rhoe,uu,vv,ww,qq,pp,tt,ee
!        real(rkind) :: dtau,dxloc,dyloc,dzloc,normxdxi,normydyi,normzdzi
!        real(rkind) :: normx,normy,normz,nmod,dxi,dyi,dzi,dtime
!        real(rkind) :: dt,drho,du,dv,dw,refl_coeff,dysp
!        integer :: i,j,k,l,lsp,iercuda,ii,jj,kk
!        integer :: indx_eikonal,eikonal_num_steps
!!
!!       Steady-state solution of convection along normal (towards inside of the body)
!!       (Di Mascio and Zaghi, 2021)
!!
!!       making sure enough steps are done also with a low eikonal cfl
!        eikonal_num_steps = ceiling(0.9_rkind*3._rkind/ibm_eikonal_cfl)
!!
!        do indx_eikonal=1,eikonal_num_steps ! loop on eikonal iterations
!!
!        !$cuf kernel do(3) <<<*,*>>>
!         do k=1,nz
!          do j=1,ny
!           do i=1,nx
!!           if (ibm_body_dist_gpu(i,j,k)>0._rkind) then
!            if (ibm_is_interface_node_gpu(i,j,k)==1) then
!             normx = ibm_body_dist_gpu(i+1,j,k)-ibm_body_dist_gpu(i-1,j,k)
!             normy = ibm_body_dist_gpu(i,j+1,k)-ibm_body_dist_gpu(i,j-1,k)
!             normz = ibm_body_dist_gpu(i,j,k+1)-ibm_body_dist_gpu(i,j,k-1)
!             dxloc = 0.5_rkind*(x_gpu(i+1)-x_gpu(i-1))
!             dyloc = 0.5_rkind*(y_gpu(j+1)-y_gpu(j-1))
!             dzloc = 0.5_rkind*(z_gpu(k+1)-z_gpu(k-1))
!             normx = normx/(x_gpu(i+1)-x_gpu(i-1))
!             normy = normy/(y_gpu(j+1)-y_gpu(j-1))
!             normz = normz/(z_gpu(k+1)-z_gpu(k-1))
!!            nmod = abs(normx) + abs(normy) + abs(normz)
!!            nmod = 0.9_rkind/nmod ! cfl = 0.9
!!            normx = normx*nmod
!!            normy = normy*nmod
!!            normz = normz*nmod
!             dtau  = min(dxloc,dyloc)
!             dtau  = min(dtau,dzloc)
!!            dtau  = dtau*0.9_rkind
!             dtau  = dtau*ibm_eikonal_cfl
!             drho  = 0._rkind
!             du    = 0._rkind
!             dv    = 0._rkind
!             dw    = 0._rkind
!             dt    = 0._rkind
!             if (normx > 0._rkind) then
!                dxi  = dtau/(x_gpu(i)-x_gpu(i-1))
!                normxdxi = normx*dxi
!                drho = drho + normxdxi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i-1,j,k,J_R))
!                du   = du   + normxdxi*(w_aux_gpu(i,j,k,J_U)-w_aux_gpu(i-1,j,k,J_U))
!                dv   = dv   + normxdxi*(w_aux_gpu(i,j,k,J_V)-w_aux_gpu(i-1,j,k,J_V))
!                dw   = dw   + normxdxi*(w_aux_gpu(i,j,k,J_W)-w_aux_gpu(i-1,j,k,J_W))
!                dt   = dt   + normxdxi*(w_aux_gpu(i,j,k,J_T)-w_aux_gpu(i-1,j,k,J_T))
!             else
!                dxi  = dtau/(x_gpu(i+1)-x_gpu(i))
!                normxdxi = normx*dxi
!                drho = drho + normxdxi*(w_aux_gpu(i+1,j,k,J_R)-w_aux_gpu(i,j,k,J_R))
!                du   = du   + normxdxi*(w_aux_gpu(i+1,j,k,J_U)-w_aux_gpu(i,j,k,J_U))
!                dv   = dv   + normxdxi*(w_aux_gpu(i+1,j,k,J_V)-w_aux_gpu(i,j,k,J_V))
!                dw   = dw   + normxdxi*(w_aux_gpu(i+1,j,k,J_W)-w_aux_gpu(i,j,k,J_W))
!                dt   = dt   + normxdxi*(w_aux_gpu(i+1,j,k,J_T)-w_aux_gpu(i,j,k,J_T))
!             endif
!             if (normy > 0._rkind) then
!                dyi = dtau/(y_gpu(j)-y_gpu(j-1))
!                normydyi = normy*dyi
!                drho = drho + normydyi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i,j-1,k,J_R))
!                du   = du   + normydyi*(w_aux_gpu(i,j,k,J_U)-w_aux_gpu(i,j-1,k,J_U))
!                dv   = dv   + normydyi*(w_aux_gpu(i,j,k,J_V)-w_aux_gpu(i,j-1,k,J_V))
!                dw   = dw   + normydyi*(w_aux_gpu(i,j,k,J_W)-w_aux_gpu(i,j-1,k,J_W))
!                dt   = dt   + normydyi*(w_aux_gpu(i,j,k,J_T)-w_aux_gpu(i,j-1,k,J_T))
!             else
!                dyi  = dtau/(y_gpu(j+1)-y_gpu(j))
!                normydyi = normy*dyi
!                drho = drho + normydyi*(w_aux_gpu(i,j+1,k,J_R)-w_aux_gpu(i,j,k,J_R))
!                du   = du   + normydyi*(w_aux_gpu(i,j+1,k,J_U)-w_aux_gpu(i,j,k,J_U))
!                dv   = dv   + normydyi*(w_aux_gpu(i,j+1,k,J_V)-w_aux_gpu(i,j,k,J_V))
!                dw   = dw   + normydyi*(w_aux_gpu(i,j+1,k,J_W)-w_aux_gpu(i,j,k,J_W))
!                dt   = dt   + normydyi*(w_aux_gpu(i,j+1,k,J_T)-w_aux_gpu(i,j,k,J_T))
!             endif
!             if (normz > 0._rkind) then
!                dzi  = dtau/(z_gpu(k)-z_gpu(k-1))
!                normzdzi = normz*dzi
!                drho = drho + normzdzi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i,j,k-1,J_R))
!                du   = du   + normzdzi*(w_aux_gpu(i,j,k,J_U)-w_aux_gpu(i,j,k-1,J_U))
!                dv   = dv   + normzdzi*(w_aux_gpu(i,j,k,J_V)-w_aux_gpu(i,j,k-1,J_V))
!                dw   = dw   + normzdzi*(w_aux_gpu(i,j,k,J_W)-w_aux_gpu(i,j,k-1,J_W))
!                dt   = dt   + normzdzi*(w_aux_gpu(i,j,k,J_T)-w_aux_gpu(i,j,k-1,J_T))
!             else
!                dzi = dtau/(z_gpu(k+1)-z_gpu(k))
!                normzdzi = normz*dzi
!                drho = drho + normzdzi*(w_aux_gpu(i,j,k+1,J_R)-w_aux_gpu(i,j,k,J_R))
!                du   = du   + normzdzi*(w_aux_gpu(i,j,k+1,J_U)-w_aux_gpu(i,j,k,J_U))
!                dv   = dv   + normzdzi*(w_aux_gpu(i,j,k+1,J_V)-w_aux_gpu(i,j,k,J_V))
!                dw   = dw   + normzdzi*(w_aux_gpu(i,j,k+1,J_W)-w_aux_gpu(i,j,k,J_W))
!                dt   = dt   + normzdzi*(w_aux_gpu(i,j,k+1,J_T)-w_aux_gpu(i,j,k,J_T))
!             endif
!             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+1) = drho
!             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+2) = du
!             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+3) = dv
!             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+4) = dw
!             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+5) = dt
!             do lsp=1,N_S
!              dysp = 0._rkind
!              if (normx > 0._rkind) then
!                 dxi  = dtau/(x_gpu(i)-x_gpu(i-1))
!                 normxdxi = normx*dxi
!                 dysp = dysp + normxdxi*(w_aux_gpu(i,j,k,lsp)-w_aux_gpu(i-1,j,k,lsp))
!              else
!                 dxi  = dtau/(x_gpu(i+1)-x_gpu(i))
!                 normxdxi = normx*dxi
!                 dysp = dysp + normxdxi*(w_aux_gpu(i+1,j,k,lsp)-w_aux_gpu(i,j,k,lsp))
!              endif
!              if (normy > 0._rkind) then
!                 dyi = dtau/(y_gpu(j)-y_gpu(j-1))
!                 normydyi = normy*dyi
!                 dysp = dysp + normydyi*(w_aux_gpu(i,j,k,lsp)-w_aux_gpu(i,j-1,k,lsp))
!              else
!                 dyi = dtau/(y_gpu(j+1)-y_gpu(j))
!                 normydyi = normy*dyi
!                 dysp = dysp + normydyi*(w_aux_gpu(i,j+1,k,lsp)-w_aux_gpu(i,j,k,lsp))
!              endif
!              if (normz > 0._rkind) then
!                 dzi  = dtau/(z_gpu(k)-z_gpu(k-1))
!                 normzdzi = normz*dzi
!                 dysp = dysp + normzdzi*(w_aux_gpu(i,j,k,lsp)-w_aux_gpu(i,j,k-1,lsp))
!              else
!                 dzi  = dtau/(z_gpu(k+1)-z_gpu(k))
!                 normzdzi = normz*dzi
!                 dysp = dysp + normzdzi*(w_aux_gpu(i,j,k+1,lsp)-w_aux_gpu(i,j,k,lsp))
!              endif
!              ibm_dw_aux_eikonal_gpu(i,j,k,lsp) = dysp
!             enddo
!            endif
!           enddo
!          enddo
!         enddo
!         !@cuf iercuda=cudaDeviceSynchronize()
!
!         !$cuf kernel do(3) <<<*,*>>>
!          do k=1,nz
!           do j=1,ny
!            do i=1,nx
!             if (ibm_is_interface_node_gpu(i,j,k)==1) then
!              w_aux_gpu(i,j,k,J_R) = w_aux_gpu(i,j,k,J_R)-ibm_dw_aux_eikonal_gpu(i,j,k,N_S+1)
!              w_aux_gpu(i,j,k,J_U) = w_aux_gpu(i,j,k,J_U)-ibm_dw_aux_eikonal_gpu(i,j,k,N_S+2)
!              w_aux_gpu(i,j,k,J_V) = w_aux_gpu(i,j,k,J_V)-ibm_dw_aux_eikonal_gpu(i,j,k,N_S+3)
!              w_aux_gpu(i,j,k,J_W) = w_aux_gpu(i,j,k,J_W)-ibm_dw_aux_eikonal_gpu(i,j,k,N_S+4)
!              w_aux_gpu(i,j,k,J_T) = w_aux_gpu(i,j,k,J_T)-ibm_dw_aux_eikonal_gpu(i,j,k,N_S+5)
!              do lsp=1,N_S
!               w_aux_gpu(i,j,k,lsp) = w_aux_gpu(i,j,k,lsp)-ibm_dw_aux_eikonal_gpu(i,j,k,lsp)
!              enddo
!             endif
!            enddo
!           enddo
!          enddo
!         !@cuf iercuda=cudaDeviceSynchronize()
!!
!        enddo
!!
!        if (ibm_num_interface>0) then
!         !$cuf kernel do(1) <<<*,*>>>
!         do l=1,ibm_num_interface
!          i = ibm_ijk_interface_gpu(1,l)
!          j = ibm_ijk_interface_gpu(2,l)
!          k = ibm_ijk_interface_gpu(3,l)
!          do lsp=1,N_S
!           ibm_w_refl_gpu(l,lsp) = w_aux_gpu(i,j,k,lsp)*w_aux_gpu(i,j,k,J_R)
!          enddo
!          ibm_w_refl_gpu(l,I_U) = w_aux_gpu(i,j,k,J_U)
!          ibm_w_refl_gpu(l,I_V) = w_aux_gpu(i,j,k,J_V)
!          ibm_w_refl_gpu(l,I_W) = w_aux_gpu(i,j,k,J_W)
!          ibm_w_refl_gpu(l,I_E) = w_aux_gpu(i,j,k,J_T)
!         enddo
!!        !@cuf iercuda=cudaDeviceSynchronize()
!        endif
!!
!    end subroutine ibm_eikonal_cuf
!
    subroutine ibm_interpolate_hwm_cuf(nx,ny,nz,ng,w_aux_gpu,ibm_num_interface,ibm_ijk_hwm_gpu,ibm_coeff_hwm_gpu,ibm_w_hwm_gpu, &      
                                   ibm_nxyz_interface_gpu,ibm_bc_gpu)
    
        integer, intent(in) :: nx,ny,nz,ng,ibm_num_interface
        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_hwm_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_nxyz_interface_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: ibm_coeff_hwm_gpu
        real(rkind), dimension(1:,1:), intent(inout), device :: ibm_w_hwm_gpu
        real(rkind) :: wi1,wi2,wi3,wi4,wi5,un,ut1,ut2,ut3,upar
        integer :: i,j,k,l,iercuda,ii,jj,kk

        !$cuf kernel do(1) <<<*,*>>>
        do l=1,ibm_num_interface
         if (ibm_bc_gpu(1,l) == 16 .or. ibm_bc_gpu(1,l) == 18) then
          i = ibm_ijk_hwm_gpu(1,l)
          j = ibm_ijk_hwm_gpu(2,l)
          k = ibm_ijk_hwm_gpu(3,l)
          wi1 = 0._rkind
          wi2 = 0._rkind
          wi3 = 0._rkind
          wi4 = 0._rkind
          wi5 = 0._rkind
          do kk=1,2
           do jj=1,2
            do ii=1,2
             wi1 = wi1 + ibm_coeff_hwm_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
             wi2 = wi2 + ibm_coeff_hwm_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
             wi3 = wi3 + ibm_coeff_hwm_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
             wi4 = wi4 + ibm_coeff_hwm_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
             wi5 = wi5 + ibm_coeff_hwm_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
            enddo
           enddo
          enddo
          un  = wi2*ibm_nxyz_interface_gpu(1,l)+wi3*ibm_nxyz_interface_gpu(2,l)+wi4*ibm_nxyz_interface_gpu(3,l)
          ut1 = wi2-un*ibm_nxyz_interface_gpu(1,l)
          ut2 = wi3-un*ibm_nxyz_interface_gpu(2,l)
          ut3 = wi4-un*ibm_nxyz_interface_gpu(3,l)
          upar = sqrt(ut1*ut1+ut2*ut2+ut3*ut3)
!
          ibm_w_hwm_gpu(l,1) = wi1 ! rho
          ibm_w_hwm_gpu(l,2) = upar
          ibm_w_hwm_gpu(l,3) = wi5 ! T
         endif
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
!
    endsubroutine ibm_interpolate_hwm_cuf                           
!
    subroutine ibm_solve_wm_cuf(nx,ny,nz,ng,nv_aux,w_aux_gpu,ibm_num_interface,ibm_ijk_interface_gpu,ibm_w_hwm_gpu, &
                                ibm_dist_hwm_gpu,ibm_bc_gpu,ibm_wm_correction_gpu,ibm_wm_wallprop_gpu,&
                                indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,ibm_parbc_gpu,u0)

        integer, intent(in) :: nx,ny,nz,ng,ibm_num_interface,indx_cp_l,indx_cp_r,nsetcv,nv_aux
        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_interface_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
        real(rkind) :: u0
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cp_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(in), device :: w_aux_gpu
        real(rkind), dimension(1:,1:), intent(inout), device :: ibm_w_hwm_gpu
        real(rkind), dimension(1:), intent(in), device :: ibm_dist_hwm_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_parbc_gpu
        real(rkind), dimension(1:,1:), intent(out), device :: ibm_wm_correction_gpu,ibm_wm_wallprop_gpu
        real(rkind) :: rho,rhow,upar,tt,dist,muw,nuw,up,utauold,utau,tauw,yp,fyp,lamw
        real(rkind) :: mu_wm,cploc,ttw,uthr,prandtl,typ,ttau,twall_ibm
        integer :: max_iter,i,j,k,ll,l,iercuda,ii,jj,kk,indx_patch
!
        max_iter = 10
        uthr = 0.00001_rkind*u0

        !$cuf kernel do(1) <<<*,*>>>
         do l=1,ibm_num_interface
          if (ibm_bc_gpu(1,l)==16.or.ibm_bc_gpu(1,l)==18) then
!
           i = ibm_ijk_interface_gpu(1,l)
           j = ibm_ijk_interface_gpu(2,l)
           k = ibm_ijk_interface_gpu(3,l)
!             
           rho  = ibm_w_hwm_gpu(l,1)
           upar = ibm_w_hwm_gpu(l,2)
           tt   = ibm_w_hwm_gpu(l,3)
!
           rhow = w_aux_gpu(i,j,k,J_R)
           muw  = w_aux_gpu(i,j,k,J_MU)
           nuw  = muw/rhow
           dist = ibm_dist_hwm_gpu(l)
           upar = max(upar,uthr)
           utau = sqrt(nuw*upar/dist)
!
           do ll=1,max_iter
            utauold = utau
            yp      = dist*utau/nuw
            fyp     = vel_law_of_the_wall(yp)
            utau    = upar/fyp
            if (abs(utau-utauold)<uthr) exit
           enddo
           tauw = rhow*utau**2
           ibm_wm_correction_gpu(1,l) = max(yp/fyp,1._rkind)
           ibm_wm_wallprop_gpu(1,l) = rhow*utau**2
           if (ibm_bc_gpu(1,l)==18) then
            ibm_wm_correction_gpu(2,l) = 1._rkind
            ibm_wm_wallprop_gpu(2,l) = 0._rkind
           else
            indx_patch = ibm_bc_gpu(2,l)
            twall_ibm  = ibm_parbc_gpu(indx_patch,1)
            lamw    = w_aux_gpu(i,j,k,J_K_COND)
            ttw     = w_aux_gpu(i,j,k,J_T)
            cploc   = get_cp_dev(ttw,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,&
                               i,j,k,nx,ny,nz,ng,nv_aux,w_aux_gpu)
            prandtl = muw*cploc/lamw
            typ     = tem_law_of_the_wall(yp,prandtl)
            ibm_wm_correction_gpu(2,l) = yp/typ
            ttau    = (tt-twall_ibm)/typ
            ibm_wm_wallprop_gpu(2,l) = cploc*rhow*utau*ttau
           endif
          endif
!
         enddo
!        !@cuf iercuda=cudaDeviceSynchronize()           
!        
    endsubroutine ibm_solve_wm_cuf                    
!
    attributes(device) function vel_law_of_the_wall(yp)
         real(rkind) :: vel_law_of_the_wall
         real(rkind), value :: yp
         real(rkind) :: vkc, vkci, b, blogk, up
!
         vkc = 0.41_rkind
         vkci = 1._rkind/vkc
         b = 5.25_rkind
         blogk = b-vkci*log(vkc)
         up = vkci*log(1._rkind+vkc*yp)
         up = up+blogk*(1._rkind-exp(-yp/11._rkind)-yp/11._rkind*exp(-yp/3._rkind))
         vel_law_of_the_wall = max(11._rkind,up)
!
    endfunction vel_law_of_the_wall
!
    attributes(device) function tem_law_of_the_wall(yp,pr)
         real(rkind) :: tem_law_of_the_wall
         real(rkind), value :: yp,pr
         real(rkind) :: vkt,vkti,big_gam,beta,onethird,tp
!
         vkt  = 1._rkind/2.12_rkind
         vkti = 2.12_rkind
         big_gam = 0.01_rkind*(yp*pr)**4
         big_gam = big_gam/(1._rkind+5._rkind*pr**3*yp)
         onethird = 1._rkind/3._rkind
         beta = vkti*log(pr)+(3.85_rkind*pr**onethird-1.3_rkind)**2

         tp = pr*yp*exp(-big_gam)+exp(-1._rkind/big_gam)*(beta+vkti*log(1+yp))
         tem_law_of_the_wall = tp
!
    endfunction tem_law_of_the_wall
!
    subroutine ibm_apply_wm_cuf(nx,ny,nz,ng,w_aux_gpu,ibm_num_interface,ibm_ijk_interface_gpu,ibm_bc_gpu,ibm_wm_correction_gpu)
        integer, intent(in) :: nx,ny,nz,ng,ibm_num_interface
        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_interface_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_wm_correction_gpu
        real(rkind) :: mu,mu_correction,lam,lam_correction 
        integer :: i,j,k,l,iercuda
!
         !$cuf kernel do(1) <<<*,*>>>
         do l=1,ibm_num_interface
          if (ibm_bc_gpu(1,l)==16.or.ibm_bc_gpu(1,l)==18) then
           i = ibm_ijk_interface_gpu(1,l)
           j = ibm_ijk_interface_gpu(2,l)
           k = ibm_ijk_interface_gpu(3,l)
           mu  = w_aux_gpu(i,j,k,J_MU)
           lam = w_aux_gpu(i,j,k,J_K_COND)
           mu_correction  = mu *max(1._rkind,ibm_wm_correction_gpu(1,l))
           lam_correction = lam*max(1._rkind,ibm_wm_correction_gpu(2,l))
           w_aux_gpu(i,j,k,J_MU)     = 2._rkind*mu_correction  - mu
           w_aux_gpu(i,j,k,J_K_COND) = 2._rkind*lam_correction - lam 
          endif
         enddo
!        !@cuf iercuda=cudaDeviceSynchronize()
!
    endsubroutine ibm_apply_wm_cuf
!
    subroutine ibm_compute_force_cuf(ibm_num_interface,nx,ny,nz,ng,ibm_ijk_interface_gpu,fln_gpu,w_gpu,w_aux_gpu,ibm_bc_gpu,&
        dcsidx_gpu,detady_gpu,dzitdz_gpu,ibm_force_x_s,ibm_force_y_s,ibm_force_z_s,fluid_mask_gpu)
!
        integer, intent(in) :: ibm_num_interface,nx,ny,nz,ng
        real(rkind), intent(inout) :: ibm_force_x_s,ibm_force_y_s,ibm_force_z_s
        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_interface_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu,detady_gpu,dzitdz_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: fln_gpu
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: fluid_mask_gpu
!
        integer :: i,j,k,iercuda
!       real(rkind) :: w1,w2,w3,w4
        real(rkind) :: dvol
!
        ibm_force_x_s = 0._rkind
        ibm_force_y_s = 0._rkind
        ibm_force_z_s = 0._rkind
        if (ibm_num_interface>0) then
!
!         !$cuf kernel do(1) <<<*,*>>> !reduce(+:ibm_force_x,ibm_force_y,ibm_force_z)
!         do l=1,ibm_num_interface
!!
!          i = ibm_ijk_interface_gpu(1,l)
!          j = ibm_ijk_interface_gpu(2,l)
!          k = ibm_ijk_interface_gpu(3,l)
!          dvol = 1._rkind/(dcsidx_gpu(i)*detady_gpu(j)*dzitdz_gpu(k))
!          w1   =    w_aux_gpu(i,j,k,1)
!          w2   = w1*w_aux_gpu(i,j,k,2)
!          w3   = w1*w_aux_gpu(i,j,k,3)
!          w4   = w1*w_aux_gpu(i,j,k,J_W)
!          ibm_force_x = ibm_force_x+(w_gpu(i,j,k,I_U)-w2)*dvol
!          ibm_force_y = ibm_force_y+(w_gpu(i,j,k,I_V)-w3)*dvol
!          ibm_force_z = ibm_force_z+(w_gpu(i,j,k,I_W)-w4)*dvol
!!
!         enddo
!         !@cuf iercuda=cudaDeviceSynchronize()
!
         !$cuf kernel do(3) <<<*,*>>> reduce(+:ibm_force_x_s,ibm_force_y_s,ibm_force_z_s)
         do k=1,nz
          do j=1,ny
           do i=1,nx
            if (fluid_mask_gpu(i,j,k)/=0) then
             dvol = 1._rkind/(dcsidx_gpu(i)*detady_gpu(j)*dzitdz_gpu(k))
             ibm_force_x_s = ibm_force_x_s+(-fln_gpu(i,j,k,I_U)*dvol)
             ibm_force_y_s = ibm_force_y_s+(-fln_gpu(i,j,k,I_V)*dvol)
             ibm_force_z_s = ibm_force_z_s+(-fln_gpu(i,j,k,I_W)*dvol)
            endif
           enddo
          enddo
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()
!
        endif
!
    end subroutine ibm_compute_force_cuf

    attributes(global) launch_bounds(256) subroutine insitu_swirling_kernel(nv, nx, ny, nz, visc_order, ng, npsi,mpsi,&
       dcsidx_gpu, detady_gpu, dzitdz_gpu, w_aux_gpu, coeff_deriv1_gpu, &
       psi_gpu, x_gpu )
    !
    ! Evaluation of the swirling strength
    !
     implicit none
    !
     integer, value :: nv, nx, ny, nz, visc_order, ng, npsi,mpsi
     real(rkind), dimension(nx) :: dcsidx_gpu
     real(rkind), dimension(ny) :: detady_gpu
     real(rkind), dimension(nz) :: dzitdz_gpu
     real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:)  :: psi_gpu
     real(rkind), dimension(1-ng:, 1-ng:,1-ng:,1:) :: w_aux_gpu
     real(rkind), dimension(1:,1:), intent(in) :: coeff_deriv1_gpu
     real(rkind), dimension(1-ng:nx+ng) :: x_gpu
     real(rkind) :: uu,vv,ww
     real(rkind) :: ux,uy,uz
     real(rkind) :: vx,vy,vz
     real(rkind) :: wx,wy,wz
     real(rkind), dimension(3) ::  eigr_a, eigi_a
     real(rkind), dimension(3,3) :: astar
     real(rkind) :: ccl, div3l
     real(rkind) :: omx, omy, omz, omod2, div2, div
     integer :: i,j,k,l
    !
     i = blockDim%x * (blockIdx%x - 1) + threadIdx%x 
     k = blockDim%y * (blockIdx%y - 1) + threadIdx%y
     if(i > nx .or. k > nz) return
    !
     do j=1,ny
    !
      uu = w_aux_gpu(i,j,k,J_U)
      vv = w_aux_gpu(i,j,k,J_V)
      ww = w_aux_gpu(i,j,k,J_W)
    !
      ux = 0._rkind
      vx = 0._rkind
      wx = 0._rkind
      uy = 0._rkind
      vy = 0._rkind
      wy = 0._rkind
      uz = 0._rkind
      vz = 0._rkind
      wz = 0._rkind
      do l=1,visc_order/2
       ccl = coeff_deriv1_gpu(l,visc_order/2)
       ux = ux+ccl*(w_aux_gpu(i+l,j,k,J_U)-w_aux_gpu(i-l,j,k,J_U))
       vx = vx+ccl*(w_aux_gpu(i+l,j,k,J_V)-w_aux_gpu(i-l,j,k,J_V))
       wx = wx+ccl*(w_aux_gpu(i+l,j,k,J_W)-w_aux_gpu(i-l,j,k,J_W))
    !
       uy = uy+ccl*(w_aux_gpu(i,j+l,k,J_U)-w_aux_gpu(i,j-l,k,J_U))
       vy = vy+ccl*(w_aux_gpu(i,j+l,k,J_V)-w_aux_gpu(i,j-l,k,J_V))
       wy = wy+ccl*(w_aux_gpu(i,j+l,k,J_W)-w_aux_gpu(i,j-l,k,J_W))
    !
       uz = uz+ccl*(w_aux_gpu(i,j,k+l,J_U)-w_aux_gpu(i,j,k-l,J_U))
       vz = vz+ccl*(w_aux_gpu(i,j,k+l,J_V)-w_aux_gpu(i,j,k-l,J_V))
       wz = wz+ccl*(w_aux_gpu(i,j,k+l,J_W)-w_aux_gpu(i,j,k-l,J_W))
      enddo
      ux = ux*dcsidx_gpu(i)
      vx = vx*dcsidx_gpu(i)
      wx = wx*dcsidx_gpu(i)
      uy = uy*detady_gpu(j)
      vy = vy*detady_gpu(j)
      wy = wy*detady_gpu(j)
      uz = uz*dzitdz_gpu(k)
      vz = vz*dzitdz_gpu(k)
      wz = wz*dzitdz_gpu(k)
    !
      div   = ux+vy+wz
      div3l   = div/3._rkind
    !
      omz = vx-uy
      omx = wy-vz
      omy = uz-wx
      omod2 = omx*omx+omy*omy+omz*omz
      div2 = div*div
    !
      astar(1,1) = ux-div3l
      astar(1,2) = uy
      astar(1,3) = uz
      astar(2,1) = vx
      astar(2,2) = vy-div3l
      astar(2,3) = vz
      astar(3,1) = wx
      astar(3,2) = wy
      astar(3,3) = wz-div3l
    !
      call eigs33(astar,eigr_a,eigi_a)
      psi_gpu(i,j,k,mpsi) = 2._rkind*max(0._rkind,eigi_a(2)) ! swirling strength
    !
     enddo
    !
    end subroutine insitu_swirling_kernel

    attributes(global) launch_bounds(256) subroutine insitu_schlieren_kernel(nv, nx, ny, nz, visc_order, ng, npsi,mpsi,&
       dcsidx_gpu, detady_gpu, dzitdz_gpu, w_aux_gpu, coeff_deriv1_gpu, &
       psi_gpu, x_gpu )
    !
    ! Evaluation of the swirling strength
    !
     implicit none
    !
     integer, value :: nv, nx, ny, nz, visc_order, ng, npsi,mpsi
     real(rkind), dimension(nx) :: dcsidx_gpu
     real(rkind), dimension(ny) :: detady_gpu
     real(rkind), dimension(nz) :: dzitdz_gpu
     real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:)  :: psi_gpu
     real(rkind), dimension(1-ng:, 1-ng:,1-ng:,1:) :: w_aux_gpu
     real(rkind), dimension(1:,1:), intent(in) :: coeff_deriv1_gpu
     real(rkind), dimension(1-ng:nx+ng) :: x_gpu
     real(rkind) :: rhox, rhoy, rhoz
     real(rkind) :: ccl
     integer :: i,j,k,l
    !
     i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
     k = blockDim%y * (blockIdx%y - 1) + threadIdx%y
     if(i > nx .or. k > nz) return
    !
     do j=1,ny
    !
      rhox = 0._rkind
      rhoy = 0._rkind
      rhoz = 0._rkind
      do l=1,visc_order/2
       ccl = coeff_deriv1_gpu(l,visc_order/2)
       rhox = rhox+ccl*(w_aux_gpu(i+l,j,k,J_R)-w_aux_gpu(i-l,j,k,J_R))
       rhoy = rhoy+ccl*(w_aux_gpu(i,j+l,k,J_R)-w_aux_gpu(i,j-l,k,J_R))
       rhoz = rhoz+ccl*(w_aux_gpu(i,j,k+l,J_R)-w_aux_gpu(i,j,k-l,J_R))
      enddo
      rhox = rhox*dcsidx_gpu(i)
      rhoy = rhoy*detady_gpu(j)
      rhoz = rhoz*dzitdz_gpu(k)
    !
      psi_gpu(i,j,k,mpsi) = exp(-sqrt((rhox)**2+(rhoy)**2+(rhoz)**2))
    !
     enddo
    !
    endsubroutine insitu_schlieren_kernel
    
    ! eigs33
    attributes(device) subroutine eigs33(rmat,rex,rimx)
     implicit none
     integer, parameter :: doubtype = REAL64
     real(rkind), dimension(3) :: rex,rimx
     real(rkind), dimension(3) :: rey,rimy
     real(rkind), dimension(3) :: reu,rimu
     real(rkind), dimension(3) :: rev,rimv
     real(rkind), dimension(3,3) :: rmat,S,WW,temp
     real(doubtype) :: ddbl,pdbl,qdbl
     real(rkind) :: otrd, pi, a, temps, tempw, b, somma, c, p, q, sqdel, &
                     sqp, teta
     integer :: ii, jj, k

     otrd = 1./3._rkind
     pi = acos(-1._rkind)
     do ii = 1,3
      do jj = 1,3
       S(ii,jj) = 0.5*(rmat(ii,jj)+rmat(jj,ii))
       WW(ii,jj) = 0.5*(rmat(ii,jj)-rmat(jj,ii))
      enddo
     enddo
     a = -(S(1,1)+S(2,2)+S(3,3))
     temps = 0.
     tempw = 0.
     do ii = 1,3
      do jj = 1,3
       temps = temps + S(ii,jj)*S(ii,jj)
       tempw = tempw + WW(ii,jj)*WW(ii,jj)
      enddo
     enddo
     b = 0.5*(a**2-temps+tempw)
     do ii = 1,3
      do jj = 1,3
       temp(ii,jj)=0.
       do k = 1,3
        temp(ii,jj) = temp(ii,jj)+S(ii,k)*S(k,jj)+3.*WW(ii,k)*WW(k,jj)
       enddo
      enddo
     enddo
     somma=0.
     do ii=1,3
      do jj=1,3
       somma= somma+temp(ii,jj)*S(jj,ii)
      enddo
     enddo
     c = -1./3.*(a**3-3.*a*b+somma)
     pdbl = real(b-a**2/3.,          doubtype)
     qdbl = real(c-a*b/3.+2*a**3/27.,doubtype)
     ddbl = (qdbl**2)/4.D0 + (pdbl**3)/27.D0
     p = real(pdbl,rkind)
     q = real(qdbl,rkind)
     if(ddbl.gt.0.D0) then
      sqdel   = real(sqrt(ddbl), rkind)
      reu(1)  =-0.5*q+sqdel
      rev(1)  =-0.5*q-sqdel
      reu(1)  = sign(1._rkind,reu(1))*(abs(reu(1)))**otrd
      rev(1)  = sign(1._rkind,rev(1))*(abs(rev(1)))**otrd
      reu(2)  = -0.5*reu(1)
      rev(2)  = -0.5*rev(1)
      reu(3)  = reu(2)
      rev(3)  = rev(2)
      rimu(1) = 0.
      rimv(1) = 0.
      rimu(2) = sqrt(3._rkind)/2.*reu(1)
      rimv(2) = sqrt(3._rkind)/2.*rev(1)
      rimu(3) = -rimu(2)
      rimv(3) = -rimv(2)
      rey(1)  = reu(1)+rev(1)
      rimy(1) = rimu(1)+rimv(1)
      rey(2)  = reu(2)+rev(3)
      rimy(2) = rimu(2)+rimv(3)
      rey(3)  = reu(3)+rev(2)
      rimy(3) = rimu(3)+rimv(2)
     else
      if (q.eq.0.) then
       rey(1) = 0.
       rey(2) = sqrt(-p)
       rey(3) = -rey(2)
      else
       sqp    = 2.*sqrt(-p/3.)
       sqdel  = real(sqrt(-ddbl), rkind)
       if (q.lt.0.) then
        teta = atan(-2.*sqdel/q)
       else
        teta = pi+atan(-2.*sqdel/q)
       endif
       rey(1) = sqp*cos(teta/3.)
       rey(2) = sqp*cos((teta+2*pi)/3.)
       rey(3) = sqp*cos((teta+4*pi)/3.)
      endif
      rimy(1) = 0.
      rimy(2) = 0.
      rimy(3) = 0.
     endif
     rex(1)   = rey(1)-(a/3.)
     rimx(1)  = rimy(1)
     rex(2)   = rey(2)-(a/3.)
     rimx(2)  = rimy(2)
     rex(3)   = rey(3)-(a/3.)
     rimx(3)  = rimy(3)
     if (rimy(2).lt.0.) then ! exchanging eigenvalues
      rex(2)   = rey(3)-(a/3.)
      rimx(2)  = rimy(3)
      rex(3)   = rey(2)-(a/3.)
      rimx(3)  = rimy(2)
     endif
    endsubroutine eigs33

    subroutine insitu_div_cuf(nx, ny, nz, ng, visc_order, npsi,mpsi, &
            w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, detady_gpu, dzitdz_gpu, psi_gpu )

        integer, intent(in) :: nx, ny, nz, ng, visc_order, npsi, mpsi
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: w_aux_gpu 
        real(rkind), dimension(1:,1:), intent(in), device :: coeff_deriv1_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu, detady_gpu, dzitdz_gpu 
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: psi_gpu
        integer     :: i,j,k,l,iercuda
        real(rkind) :: ccl
        real(rkind) :: uu,vv,ww
        real(rkind) :: ux
        real(rkind) :: vy
        real(rkind) :: wz
        real(rkind) :: div
        integer     :: lmax

        lmax = visc_order/2
     
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do i=1,nx
      
           uu = w_aux_gpu(i,j,k,J_U)
           vv = w_aux_gpu(i,j,k,J_V)
           ww = w_aux_gpu(i,j,k,J_W)
     
           ux = 0._rkind
           vy = 0._rkind
           wz = 0._rkind

           do l=1,lmax
            ccl = coeff_deriv1_gpu(l,lmax)
            ux = ux+ccl*(w_aux_gpu(i+l,j,k,J_U)-w_aux_gpu(i-l,j,k,J_U))
      
            vy = vy+ccl*(w_aux_gpu(i,j+l,k,J_V)-w_aux_gpu(i,j-l,k,J_V))
     
            wz = wz+ccl*(w_aux_gpu(i,j,k+l,J_W)-w_aux_gpu(i,j,k-l,J_W))
           enddo
           ux = ux*dcsidx_gpu(i)
           vy = vy*detady_gpu(j)
           wz = wz*dzitdz_gpu(k)
       
           div     = ux+vy+wz
           psi_gpu(i,j,k,mpsi) = div
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    end subroutine insitu_div_cuf

    subroutine insitu_omega_cuf(nx, ny, nz, ng, visc_order, npsi,mpsi, &
            w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, detady_gpu, dzitdz_gpu, psi_gpu )

        integer, intent(in) :: nx, ny, nz, ng, visc_order, npsi, mpsi
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: w_aux_gpu 
        real(rkind), dimension(1:,1:), intent(in), device :: coeff_deriv1_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu, detady_gpu, dzitdz_gpu 
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: psi_gpu
        integer     :: i,j,k,l,iercuda
        real(rkind) :: ccl
        real(rkind) :: uu,vv,ww
        real(rkind) :: uy,uz
        real(rkind) :: vx,vz
        real(rkind) :: wx,wy
        real(rkind) :: omegax, omegay, omegaz, omod2
        integer     :: lmax

        lmax = visc_order/2
     
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do i=1,nx
      
           uu = w_aux_gpu(i,j,k,J_U)
           vv = w_aux_gpu(i,j,k,J_V)
           ww = w_aux_gpu(i,j,k,J_W)
     
           vx = 0._rkind
           wx = 0._rkind
           uy = 0._rkind
           wy = 0._rkind
           uz = 0._rkind
           vz = 0._rkind

           do l=1,lmax
            ccl = coeff_deriv1_gpu(l,lmax)
            vx = vx+ccl*(w_aux_gpu(i+l,j,k,J_V)-w_aux_gpu(i-l,j,k,J_V))
            wx = wx+ccl*(w_aux_gpu(i+l,j,k,J_W)-w_aux_gpu(i-l,j,k,J_W))
      
            uy = uy+ccl*(w_aux_gpu(i,j+l,k,J_U)-w_aux_gpu(i,j-l,k,J_U))
            wy = wy+ccl*(w_aux_gpu(i,j+l,k,J_W)-w_aux_gpu(i,j-l,k,J_W))
     
            uz = uz+ccl*(w_aux_gpu(i,j,k+l,J_U)-w_aux_gpu(i,j,k-l,J_U))
            vz = vz+ccl*(w_aux_gpu(i,j,k+l,J_V)-w_aux_gpu(i,j,k-l,J_V))
           enddo
           vx = vx*dcsidx_gpu(i)
           wx = wx*dcsidx_gpu(i)
           uy = uy*detady_gpu(j)
           wy = wy*detady_gpu(j)
           uz = uz*dzitdz_gpu(k)
           vz = vz*dzitdz_gpu(k)
       
           omegax = wy-vz
           omegay = uz-wx
           omegaz = vx-uy
           omod2 = omegax*omegax+omegay*omegay+omegaz*omegaz
           psi_gpu(i,j,k,mpsi) = sqrt(omod2)
     
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine insitu_omega_cuf

    subroutine insitu_ducros_cuf(nx, ny, nz, ng, visc_order, npsi,mpsi, &
            w_aux_gpu, coeff_deriv1_gpu, dcsidx_gpu, detady_gpu, dzitdz_gpu, psi_gpu, eps_sensor)

        integer, intent(in) :: nx, ny, nz, ng, visc_order, npsi, mpsi
        real(rkind), intent(in) :: eps_sensor
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: w_aux_gpu 
        real(rkind), dimension(1:,1:), intent(in), device :: coeff_deriv1_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu, detady_gpu, dzitdz_gpu 
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: psi_gpu
        integer     :: i,j,k,l,iercuda
        real(rkind) :: ccl
        real(rkind) :: uu,vv,ww
        real(rkind) :: ux,uy,uz
        real(rkind) :: vx,vy,vz
        real(rkind) :: wx,wy,wz
        real(rkind) :: div, omegax, omegay, omegaz, omod2
        integer     :: lmax

        lmax = visc_order/2
     
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do i=1,nx
      
           uu = w_aux_gpu(i,j,k,J_U)
           vv = w_aux_gpu(i,j,k,J_V)
           ww = w_aux_gpu(i,j,k,J_W)
     
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
            ccl = coeff_deriv1_gpu(l,lmax)
            ux = ux+ccl*(w_aux_gpu(i+l,j,k,J_U)-w_aux_gpu(i-l,j,k,J_U))
            vx = vx+ccl*(w_aux_gpu(i+l,j,k,J_V)-w_aux_gpu(i-l,j,k,J_V))
            wx = wx+ccl*(w_aux_gpu(i+l,j,k,J_W)-w_aux_gpu(i-l,j,k,J_W))
      
            uy = uy+ccl*(w_aux_gpu(i,j+l,k,J_U)-w_aux_gpu(i,j-l,k,J_U))
            vy = vy+ccl*(w_aux_gpu(i,j+l,k,J_V)-w_aux_gpu(i,j-l,k,J_V))
            wy = wy+ccl*(w_aux_gpu(i,j+l,k,J_W)-w_aux_gpu(i,j-l,k,J_W))
     
            uz = uz+ccl*(w_aux_gpu(i,j,k+l,J_U)-w_aux_gpu(i,j,k-l,J_U))
            vz = vz+ccl*(w_aux_gpu(i,j,k+l,J_V)-w_aux_gpu(i,j,k-l,J_V))
            wz = wz+ccl*(w_aux_gpu(i,j,k+l,J_W)-w_aux_gpu(i,j,k-l,J_W))
           enddo
           ux = ux*dcsidx_gpu(i)
           vx = vx*dcsidx_gpu(i)
           wx = wx*dcsidx_gpu(i)
           uy = uy*detady_gpu(j)
           vy = vy*detady_gpu(j)
           wy = wy*detady_gpu(j)
           uz = uz*dzitdz_gpu(k)
           vz = vz*dzitdz_gpu(k)
           wz = wz*dzitdz_gpu(k)
       
           div     = ux+vy+wz
           omegax = wy-vz
           omegay = uz-wx
           omegaz = vx-uy
           omod2 = omegax*omegax+omegay*omegay+omegaz*omegaz

!          psi_gpu(i,j,k,mpsi) = max(-div/sqrt(omod2+div**2+(u0/l0)**2),0._rkind)
           psi_gpu(i,j,k,mpsi) = max(-div/sqrt(omod2+div**2+eps_sensor),0._rkind)
           ! Original Ducros shock sensor
           ! psi_gpu(i,j,k,mpsi) = div**2/(omod2+div**2+1.D-12)
     
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
    endsubroutine insitu_ducros_cuf

    !NOMANAGEDsubroutine copy_to_psi_pv_managed_cuf(nxsl_ins,nxel_ins,nysl_ins,nyel_ins,nzsl_ins,nzel_ins, &
    !NOMANAGED         ng,nx,ny,nz,npsi, npsi_pv, nv_aux, n_aux_list, &
    !NOMANAGED         psi_gpu,psi_pv_managed,w_aux_gpu,aux_list_gpu)
    !NOMANAGED integer :: nxsl_ins,nxel_ins,nysl_ins,nyel_ins,nzsl_ins,nzel_ins
    !NOMANAGED integer :: ng,nx,ny,nz,npsi, npsi_pv, nv_aux, n_aux_list
    !NOMANAGED integer :: i,j,k,l,ll, iercuda
    !NOMANAGED integer, dimension(1:n_aux_list), device :: aux_list_gpu
    !NOMANAGED real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,npsi), device :: psi_gpu
    !NOMANAGED real(rkind), dimension(nxsl_ins:nxel_ins,nysl_ins:nyel_ins,nzsl_ins:nzel_ins,npsi_pv), managed :: psi_pv_managed
    !NOMANAGED real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), device :: w_aux_gpu

    !NOMANAGED !$cuf kernel do(3) <<<*,*>>> 
    !NOMANAGED do k=nzsl_ins,nzel_ins
    !NOMANAGED  do j=nysl_ins,nyel_ins
    !NOMANAGED   do i=nxsl_ins,nxel_ins
    !NOMANAGED    do l=1,n_aux_list
    !NOMANAGED     ll = aux_list_gpu(l)
    !NOMANAGED     psi_pv_managed(i,j,k,l)  = w_aux_gpu(i,j,k,ll)
    !NOMANAGED    enddo
    !NOMANAGED    do l=1,npsi
    !NOMANAGED     psi_pv_managed(i,j,k,n_aux_list+l) = psi_gpu(i,j,k,l)
    !NOMANAGED    enddo
    !NOMANAGED   enddo
    !NOMANAGED  enddo
    !NOMANAGED enddo
    !NOMANAGED !@cuf iercuda=cudaDeviceSynchronize()
    !NOMANAGEDend subroutine copy_to_psi_pv_managed_cuf

    subroutine probe_interpolation_cuf(num_probe,nx,ny,nz,ng,nv_aux,ijk_probe_gpu,w_aux_probe_gpu,w_aux_gpu,probe_coeff_gpu)
        integer, intent(in) :: num_probe,nx,ny,nz,ng,nv_aux
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(in), device :: w_aux_gpu
        real(rkind), dimension(6,num_probe), intent(inout), device :: w_aux_probe_gpu
        real(rkind), dimension(2,2,2,num_probe), intent(in), device :: probe_coeff_gpu
        integer, dimension(3,num_probe), intent(in), device :: ijk_probe_gpu
        integer :: i,j,k,ii,jj,kk,l,iercuda
        real(rkind) :: w1,w2,w3,w4,w5,w6

        !$cuf kernel do(1) <<<*,*>>>
        do l=1,num_probe
         ii = ijk_probe_gpu(1,l)
         jj = ijk_probe_gpu(2,l)
         kk = ijk_probe_gpu(3,l)
         w1 = 0._rkind
         w2 = 0._rkind
         w3 = 0._rkind
         w4 = 0._rkind
         w5 = 0._rkind
         w6 = 0._rkind
         do k=1,2
          do j=1,2
           do i=1,2
            w1 = w1 + probe_coeff_gpu(i,j,k,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
            w2 = w2 + probe_coeff_gpu(i,j,k,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
            w3 = w3 + probe_coeff_gpu(i,j,k,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
            w4 = w4 + probe_coeff_gpu(i,j,k,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
            w5 = w5 + probe_coeff_gpu(i,j,k,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_H)
            w6 = w6 + probe_coeff_gpu(i,j,k,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
           enddo
          enddo
         enddo
         w_aux_probe_gpu(1,l) = w1
         w_aux_probe_gpu(2,l) = w2
         w_aux_probe_gpu(3,l) = w3
         w_aux_probe_gpu(4,l) = w4
         w_aux_probe_gpu(5,l) = w5
         w_aux_probe_gpu(6,l) = w6
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
!
    endsubroutine probe_interpolation_cuf
!
!
    attributes(global) launch_bounds(512) subroutine rosenbrock_kernel(nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,&
                                                     nreactions,arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,&
                                                     isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,&
                                                     cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l,indx_cp_r,dttry,time_start,time_end,&
                                                     maxsteps,maxtry,tol,simpler_splitting,enable_pasr,les_c_yoshi,les_c_mix,les_c_eps,&
                                                     dcsidx_gpu,detady_gpu,dzitdz_gpu)
     implicit none
     ! passed arguments
     integer, value     :: nv,nv_aux,nx,ny,nz,ng,nreactions,num_t_tab,simpler_splitting,enable_pasr
     integer, value     :: indx_cp_l,indx_cp_r,nsetcv,maxsteps,maxtry
     integer, dimension(nreactions) :: reac_ty_gpu,isRev_gpu
     real(rkind), value :: dttry,t_min_tab,dt_tab,tol_iter_nr,R_univ,time_start,time_end,tol,les_c_yoshi,les_c_mix,les_c_eps
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv    ) :: w_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ) :: fl_gpu
     real(rkind), dimension(N_S)          :: mw_gpu
     real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
     real(rkind), dimension(nreactions,2)           :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
     real(rkind), intent(in), dimension(nreactions,5) :: falloff_coeffs_gpu
     real(rkind), dimension(nreactions,N_S)         :: tb_eff_gpu,r_coeffs_gpu,p_coeffs_gpu
     real(rkind), dimension(num_t_tab+1,nreactions) :: kc_tab_gpu
     real(rkind), dimension(1:) :: dcsidx_gpu, detady_gpu, dzitdz_gpu
     ! local variables
     integer :: i,j,k,iter,jtry,m,finish
     real(rkind) :: time,dt,errmax,dtnext,t0,remaining,itol
     real(rkind), dimension(4+N_S)    :: w_sav,rhs_sav,err,yscal
     real(rkind), dimension(4+N_S,4+N_S) :: jacobian
     real(rkind), parameter :: safety = 0.9_rkind, grow = 1.5_rkind, pgrow = -0.25_rkind
     real(rkind), parameter :: shrnk = 0.5_rkind, pshrnk = -1._rkind/3._rkind, errcon = 0.1296_rkind
     real(rkind) :: ri, delta, nu_sgs, eps_sgs, tau_k, nu_eff

     i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
     j = blockDim%y * (blockIdx%y - 1) + threadIdx%y
     k = blockDim%z * (blockIdx%z - 1) + threadIdx%z
     if (i > nx .or. j > ny .or. k > nz) return
      finish = 0
      time   = time_start
      dt     = dttry
      itol   = 1._rkind/tol
 
      !   preparaing variables for PaSR model
      if (enable_pasr > 0) then
       ri    = 1._rkind/w_aux_gpu(i,j,k,J_R)
       delta = (dcsidx_gpu(i)*detady_gpu(j)*dzitdz_gpu(k))**(-1._rkind/3._rkind)
       nu_sgs  = w_aux_gpu(i,j,k,J_LES1)*ri
       eps_sgs = les_c_eps*les_c_yoshi**(3._rkind/2._rkind)*nu_sgs**3/delta**4
       nu_eff  = w_aux_gpu(i,j,k,J_MU)*ri
       tau_k   = les_c_mix*sqrt(max(nu_eff/(eps_sgs+tiny(1._rkind)),0._rkind))
      else
       tau_k = 0._rkind
      endif

      steploop: do iter=1,maxsteps
       t0 = time

       call rosenbrock_jacobian(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                indx_cp_l,indx_cp_r,simpler_splitting,jacobian,enable_pasr,tau_k)

       call rosenbrock_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                           arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                           kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                           indx_cp_l,indx_cp_r,simpler_splitting,rhs_sav,enable_pasr,tau_k)

       do m=1,nv
        w_sav(m)   = w_gpu(i,j,k,m)
        yscal(m)   = max(1._rkind,abs(w_sav(m)))
       enddo

       tryloop: do jtry=1,maxtry
        ! advancing the solution 
        call rosenbrock_step(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,&
                             nreactions,arr_a_gpu,arr_b_gpu,arr_ea_gpu,falloff_coeffs_gpu,tb_eff_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                             kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                             indx_cp_l,indx_cp_r,simpler_splitting,dt,t0,time,jacobian,rhs_sav,w_sav,err,enable_pasr,tau_k)

        ! check dt size
        errmax = 0._rkind
        do m=1,nv
         errmax = max(errmax, abs(err(m)/yscal(m)))
        enddo
        errmax = errmax*itol

        if (errmax < 1._rkind .or. finish == 1) then
         if (errmax > errcon) then
          dtnext = safety*dt*errmax**pgrow
         else
          dtnext = grow*dt
         endif
         !print *, iter, dt, time, w_aux_gpu(i,j,k,J_T)
         exit tryloop
        else
         dtnext = safety*dt*errmax**pshrnk
         dt = sign(max(abs(dtnext), SHRNK*abs(dt)), dt)
        endif

       enddo tryloop

       ! check integration time 
       remaining = time_end - time
       if (finish == 1 .or. remaining < 1.D-012*time_end) then
        exit steploop
       else
        if (dtnext*1.5_rkind > remaining) then
         dt = remaining
         finish = 1
        else
         dt = dtnext
         finish = 0
        endif
       endif

      enddo steploop
!
    endsubroutine rosenbrock_kernel
!
    attributes(device) subroutine rosenbrock_step(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                  arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                  kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                  indx_cp_l,indx_cp_r,simpler_splitting,dt,t0,time,jacobian,rhs_sav,w_sav,err,enable_pasr,tau_k)
    ! compute one rosenbrock step
     implicit none
     ! passed arguments
     integer, intent(in) :: i,j,k,nv,nv_aux,nx,ny,nz,ng,nreactions,num_t_tab,simpler_splitting,enable_pasr
     integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
     integer, dimension(nreactions), intent(in) :: reac_ty_gpu,isRev_gpu
     real(rkind), intent(in) :: t_min_tab,dt_tab,tol_iter_nr,R_univ,dt,t0,tau_k
     real(rkind), intent(inout) :: time
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv    ), intent(inout) :: w_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout) :: w_aux_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ), intent(in)    :: fl_gpu
     real(rkind), dimension(N_S)         , intent(in) :: mw_gpu
     real(rkind), dimension(N_S,nsetcv+1), intent(in) :: trange_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in) :: cv_coeff_gpu
     real(rkind), dimension(nreactions,2)          , intent(in) :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
     real(rkind), dimension(nreactions,5)          , intent(in) :: falloff_coeffs_gpu 
     real(rkind), dimension(nreactions,N_S)        , intent(in) :: tb_eff_gpu,r_coeffs_gpu,p_coeffs_gpu
     real(rkind), dimension(num_t_tab+1,nreactions), intent(in) :: kc_tab_gpu
     real(rkind), dimension(4+N_S)   , intent(in) :: rhs_sav,w_sav
     real(rkind), dimension(4+N_S,4+N_S), intent(in) :: jacobian
     ! output arguments
     real(rkind), dimension(4+N_S), intent(out) :: err
     ! local variables
     integer :: n,m
     integer, dimension(4+N_S) :: indx
     real(rkind), dimension(4+N_S)    :: rhs,g1,g2,g3,g4
     real(rkind), dimension(4+N_S,4+N_S) :: amat

     real(rkind), parameter :: gam = 1._rkind/2._rkind, a21 = 2._rkind, a31 = 48._rkind/25._rkind, a32 = 6._rkind/25._rkind
     real(rkind), parameter :: c21 = -8._rkind, c31 = 372._rkind/25._rkind, c32 = 12._rkind/5._rkind
     real(rkind), parameter :: c41 = -112._rkind/125._rkind, c42 = -54._rkind/125._rkind, c43 = -2._rkind/5._rkind
     real(rkind), parameter :: b1 = 19._rkind/9._rkind, b2 = 1._rkind/2._rkind, b3 = 25._rkind/108._rkind, b4 = 125._rkind/108._rkind
     real(rkind), parameter :: e1 = 17._rkind/54._rkind, e2 = 7._rkind/36._rkind, e3 = 0._rkind, e4 = 125._rkind/108._rkind
     real(rkind), parameter :: c1x = 1._rkind/2._rkind, c2x = -3._rkind/2._rkind, c3x = 121._rkind/50._rkind, c4x = 29._rkind/250._rkind
     real(rkind), parameter :: a2x = 1._rkind, a3x = 3._rkind/5._rkind

     do n=1,nv
      do m=1,nv
       amat(m,n) = -jacobian(m,n)
      enddo
      amat(n,n) = amat(n,n) + 1._rkind/(gam*dt)
     enddo
     call ludcmp(amat,indx)

     ! substep 1
     do m=1,nv
      g1(m) = rhs_sav(m)
     enddo
     call lubksb(amat,indx,g1)
     do m=1,nv
      w_gpu(i,j,k,m) = w_sav(m) + a21*g1(m)
     enddo
     time = t0+a2x*dt

     ! substep 2
     call rosenbrock_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                         arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                         kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                         indx_cp_l,indx_cp_r,simpler_splitting,rhs,enable_pasr,tau_k)
     do m=1,nv
      g2(m) = rhs(m)+c21*g1(m)/dt
     enddo
     call lubksb(amat,indx,g2)
     do m=1,nv
      w_gpu(i,j,k,m) = w_sav(m) + a31*g1(m) + a32*g2(m)
     enddo
     time = t0+a3x*dt

     ! substep 3
     call rosenbrock_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                         arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                         kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                         indx_cp_l,indx_cp_r,simpler_splitting,rhs,enable_pasr,tau_k)
     do m=1,nv
      g3(m) = rhs(m) + (c31*g1(m) + c32*g2(m))/dt
     enddo
     call lubksb(amat,indx,g3)

     ! substep 4
     do m=1,nv
      g4(m) = rhs(m) + (c41*g1(m) + c42*g2(m) + c43*g3(m))/dt
     enddo
     call lubksb(amat,indx,g4)
     do m=1,nv
      w_gpu(i,j,k,m) = w_sav(m) + B1*g1(m) + B2*g2(m) + B3*g3(m) + B4*g4(m)
      err(m)         =            E1*g1(m) + E2*g2(m) + E3*g3(m) + E4*g4(m)
     enddo
     time = t0 + dt

    endsubroutine rosenbrock_step
!
    attributes(device) subroutine rosenbrock_jacobian(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                  arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                  kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                  indx_cp_l,indx_cp_r,simpler_splitting,jacobian,enable_pasr,tau_k)
    ! compute the rhs's jacobian
     implicit none
     ! passed arguments
     integer, intent(in) :: i,j,k,nv,nv_aux,nx,ny,nz,ng,nreactions,num_t_tab,simpler_splitting,enable_pasr
     integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
     integer, dimension(nreactions), intent(in) :: reac_ty_gpu,isRev_gpu
     real(rkind), intent(in) :: t_min_tab,dt_tab,tol_iter_nr,R_univ,tau_k
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv    ), intent(inout) :: w_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout) :: w_aux_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ), intent(in)    :: fl_gpu
     real(rkind), dimension(N_S)         , intent(in) :: mw_gpu
     real(rkind), dimension(N_S,nsetcv+1), intent(in) :: trange_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in) :: cv_coeff_gpu
     real(rkind), dimension(nreactions,2)          , intent(in) :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
     real(rkind), dimension(nreactions,5)          , intent(in) :: falloff_coeffs_gpu 
     real(rkind), dimension(nreactions,N_S)        , intent(in) :: tb_eff_gpu,r_coeffs_gpu,p_coeffs_gpu
     real(rkind), dimension(num_t_tab+1,nreactions), intent(in) :: kc_tab_gpu
     ! output arguments
     real(rkind), dimension(4+N_S,4+N_S), intent(out) :: jacobian
     ! local variables
     integer :: n,m
     real(rkind) :: eps,delta,wtmp,h
     real(rkind), dimension(4+N_S) :: rhsp,rhsm

     eps   = 1.0D-06
     delta = 1.0D-14

     do n=1,nv
      wtmp = w_gpu(i,j,k,n)
      h    = wtmp*eps + delta
      w_gpu(i,j,k,n) = wtmp + h
      call rosenbrock_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                          arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                          kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                          indx_cp_l,indx_cp_r,simpler_splitting,rhsp,enable_pasr,tau_k)

      w_gpu(i,j,k,n) = wtmp - h
      call rosenbrock_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                          arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                          kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                          indx_cp_l,indx_cp_r,simpler_splitting,rhsm,enable_pasr,tau_k)

      w_gpu(i,j,k,n) = wtmp
      do m=1,nv
       jacobian(m,n) = (rhsp(m) - rhsm(m))/(2._rkind*h)
      enddo
     enddo

    endsubroutine rosenbrock_jacobian
!
    attributes(device) subroutine rosenbrock_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                  arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                  kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                  indx_cp_l,indx_cp_r,simpler_splitting,rhs,enable_pasr,tau_k)
    ! compute the rhs
     implicit none
     ! passed arguments
     integer, intent(in) :: i,j,k,nv,nv_aux,nx,ny,nz,ng,nreactions,num_t_tab,simpler_splitting,enable_pasr
     integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
     integer, dimension(nreactions), intent(in) :: reac_ty_gpu,isRev_gpu
     real(rkind), intent(in) :: t_min_tab,dt_tab,tol_iter_nr,R_univ,tau_k
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv    ), intent(inout) :: w_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout) :: w_aux_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ), intent(in)    :: fl_gpu
     real(rkind), dimension(N_S)         , intent(in) :: mw_gpu
     real(rkind), dimension(N_S,nsetcv+1), intent(in) :: trange_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in) :: cv_coeff_gpu
     real(rkind), dimension(nreactions,2)          , intent(in) :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
     real(rkind), dimension(nreactions,5)          , intent(in) :: falloff_coeffs_gpu 
     real(rkind), dimension(nreactions,N_S)        , intent(in) :: tb_eff_gpu,r_coeffs_gpu,p_coeffs_gpu
     real(rkind), dimension(num_t_tab+1,nreactions), intent(in) :: kc_tab_gpu
     ! output arguments
     real(rkind), dimension(4+N_S), intent(out) :: rhs
     ! local arguments
     integer :: lsp,m,itt,lr
     real(rkind) :: rho,ri,ee,tt,ttleft,dtt,dkc,kc,kf,kb,q1,q2,rr,pp,tb,qlr,wdtkj
     real(rkind) :: rhou,rhov,rhow,rhoe,uu,vv,ww,qq
     real(rkind) :: arr_a,arr_b,arr_ea,Rtti
     real(rkind) :: gam_pasr, tau_c, tau_cs, tau_cf, tau_lsp
     real(rkind) :: k0,kinf,rpres,fcent,coff,noff,doff,foff,xoff
     real(rkind),dimension(N_S) :: conc


     do lsp=1,N_S
      w_gpu(i,j,k,lsp) = max(w_gpu(i,j,k,lsp),0._rkind)
     enddo
     rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
     ri   = 1._rkind/rho
     do lsp=1,N_S
      w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)*ri
      conc(lsp) = rho*w_aux_gpu(i,j,k,lsp)/mw_gpu(lsp)
     enddo
     rhou = w_gpu(i,j,k,I_U)
     rhov = w_gpu(i,j,k,I_V)
     rhow = w_gpu(i,j,k,I_W)
     rhoe = w_gpu(i,j,k,I_E)
     uu   = rhou*ri
     vv   = rhov*ri
     ww   = rhow*ri
     qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
     ee = rhoe*ri-qq
     tt = get_mixture_temperature_from_e_dev(ee, w_aux_gpu(i,j,k,J_T), cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, &
                                             tol_iter_nr,i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
     w_aux_gpu(i,j,k,J_T) = tt
     Rtti=1._rkind/(R_univ*tt)

     itt = int((tt-t_min_tab)/dt_tab)+1
     itt = max(itt,1)
     itt = min(itt,num_t_tab)

     ttleft = (itt-1)*dt_tab + t_min_tab
     dtt = (tt-ttleft)/dt_tab
!     
     do m=1,nv
      rhs(m) = 0._rkind
     enddo
!                                    
     do lr=1,nreactions
      ! compute kf
      select case(reac_ty_gpu(lr))
       case(0,1) !0 => Arrhenius, 1 => Three-body
        kf = arr_a_gpu(lr,1)
        if (arr_b_gpu(lr,1)  .ne. 0._rkind) kf = kf*(tt**arr_b_gpu(lr,1))
        if (arr_ea_gpu(lr,1) .ne. 0._rkind) kf = kf* exp(-arr_ea_gpu(lr,1)*Rtti)
       case(2) !falloff-Lindemann
        k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
        kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
        tb = 0._rkind
        do lsp=1,N_S
         tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
        enddo
        rpres = k0*tb/kinf
        kf = kinf*(rpres/(1._rkind + rpres))
       case(3) !falloff-Troe
        k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
        kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
        tb = 0._rkind
        do lsp=1,N_S
         tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
        enddo
        rpres = k0*tb/kinf
        if (rpres .lt. 1E-30_rkind) then
         kf = K0*tb
        else
         fcent = (1._rkind - falloff_coeffs_gpu(lr,1))*exp(-tt/falloff_coeffs_gpu(lr,2)) + &
                 falloff_coeffs_gpu(lr,1)*exp(-tt/falloff_coeffs_gpu(lr,3))
         if (falloff_coeffs_gpu(lr,4) .ne. -3.14_rkind) then
          fcent = fcent + exp(-falloff_coeffs_gpu(lr,4)/tt)
         end if
         fcent = log10(fcent)
         coff = -0.4_rkind - 0.67_rkind*fcent
         noff = 0.75_rkind - 1.27_rkind*fcent
         doff = 0.14_rkind
         foff = 10_rkind**((((1._rkind + ((log10(rpres) + coff)/(noff -doff*(log10(rpres) + &
         coff)))**2._rkind)**(-1._rkind))*fcent))
         kf = kinf*(rpres/(1._rkind + rpres))*foff
        endif
       case(4) !falloff-SRI
        k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
        kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
        tb = 0._rkind
        do lsp=1,N_S
         tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
        enddo
        rpres = k0*tb/kinf
        xoff = 1._rkind/(1._rkind + log10(rpres)**2._rkind)
        foff = falloff_coeffs_gpu(lr,4)*((falloff_coeffs_gpu(lr,1)*exp(-falloff_coeffs_gpu(lr,2)/tt) + &
               exp(-tt/falloff_coeffs_gpu(lr,3)))**xoff)*(tt**falloff_coeffs_gpu(lr,5))
        kf = kinf*(rpres/(1._rkind + rpres))*foff
      endselect

      if (isRev_gpu(lr) .eq. 1) then
        dkc = kc_tab_gpu(itt+1,lr)-kc_tab_gpu(itt,lr)
        kc  = kc_tab_gpu(itt,lr)+dkc*dtt
        kb = kf/kc
      else
        kb = 0._rkind
      endif

      q1  = 1._rkind
      q2  = 1._rkind

      do lsp=1,N_S
       rr = r_coeffs_gpu(lr,lsp)
       pp = p_coeffs_gpu(lr,lsp)
       if (rr == 1._rkind) then
        q1 = q1*conc(lsp)
       elseif (rr .ne. 0._rkind) then
        q1 = q1*conc(lsp)**rr
       endif
       if (pp == 1._rkind) then
        q2 = q2*conc(lsp)
       elseif (pp .ne. 0._rkind) then
        q2 = q2*conc(lsp)**pp
       endif
      enddo
      tb = 1._rkind
      if (reac_ty_gpu(lr) == 1) then
       tb = 0._rkind
       do lsp=1,N_S
        tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
       enddo
      endif
      qlr = tb*(kf*q1-kb*q2)
      do lsp=1,N_S
       rr = r_coeffs_gpu(lr,lsp)
       pp = p_coeffs_gpu(lr,lsp)
       if (rr .ne. 0._rkind .or. pp .ne. 0._rkind) then
        wdtkj = (pp-rr)*qlr
        rhs(lsp) = rhs(lsp) + wdtkj*mw_gpu(lsp)
       endif
      enddo
     enddo

!    PaSR model
     if (enable_pasr > 0) then
      tau_cf = 0._rkind
      tau_cs = huge(1._rkind)
      do lsp=1,N_S
       if (abs(rhs(lsp)*ri) > 1.D-08) then
        tau_lsp = w_aux_gpu(i,j,k,lsp)/abs(rhs(lsp)*ri)
        tau_cf  = max(tau_cf,tau_lsp)
        if (tau_lsp > 0._rkind) tau_cs = min(tau_cs,tau_lsp)
       endif
      enddo
      tau_c = sqrt(tau_cs*tau_cf)
      if (tau_k > tiny(1._rkind)) then
       gam_pasr = tau_c/(tau_c + tau_k)
      else
       gam_pasr = 1._rkind
      endif
      do lsp=1,N_S
       rhs(lsp) = gam_pasr*rhs(lsp)
      enddo
     endif
!
    endsubroutine rosenbrock_rhs
!
    !*******************************************************
    !*    LU decomposition routines used by test_lu.f90    *
    !*                                                     *
    !*                 F90 version by J-P Moreau, Paris    *
    !* --------------------------------------------------- *
    !* Reference:                                          *
    !*                                                     *
    !* "Numerical Recipes By W.H. Press, B. P. Flannery,   *
    !*  S.A. Teukolsky and W.T. Vetterling, Cambridge      *
    !*  University Press, 1986" [BIBLI 08].                *
    !*                                                     * 
    !*******************************************************
    attributes(device) subroutine ludcmp(a,indx)
       implicit none
       real(rkind), parameter :: tiny = 1.5D-16

       real(rkind), intent(inout), dimension(4+N_S,4+N_S) :: A
       integer, intent(out), dimension(4+N_S) :: INDX
       !f2py depend(N) A, indx

       real(rkind), dimension(4+N_S) :: VV
       real(rkind)  :: AMAX, DUM, SUMM
       integer :: i, j, k, imax

       DO I=1,4+N_S
        AMAX=0._rkind
        DO J=1,4+N_S
         IF (DABS(A(I,J)).GT.AMAX) AMAX=DABS(A(I,J))
        END DO ! j loop
        IF(AMAX.LT.TINY) THEN
         RETURN
        END IF
        VV(I) = 1._rkind / AMAX
       END DO ! i loop

       DO J=1,4+N_S
        DO I=1,J-1
         SUMM = A(I,J)
         DO K=1,I-1
          SUMM = SUMM - A(I,K)*A(K,J)
         END DO ! k loop
         A(I,J) = SUMM
        END DO ! i loop
        AMAX = 0._rkind
        DO I=J,4+N_S
         SUMM = A(I,J)
         DO K=1,J-1
          SUMM = SUMM - A(I,K)*A(K,J)
         END DO ! k loop
         A(I,J) = SUMM
         DUM = VV(I)*DABS(SUMM)
         IF(DUM.GE.AMAX) THEN
          IMAX = I
          AMAX = DUM
         END IF
        END DO ! i loop  

        IF(J.NE.IMAX) THEN
         DO K=1,4+N_S
           DUM = A(IMAX,K)
           A(IMAX,K) = A(J,K)
           A(J,K) = DUM
         END DO ! k loop
         VV(IMAX) = VV(J)
        END IF

        INDX(J) = IMAX
        IF(DABS(A(J,J)) < TINY) A(J,J) = TINY

        IF(J.NE.4+N_S) THEN
         DUM = 1._rkind / A(J,J)
         DO I=J+1,4+N_S
           A(I,J) = A(I,J)*DUM
         END DO ! i loop
        END IF
       END DO ! j loop

     RETURN

    endsubroutine ludcmp
!
    attributes(device) subroutine lubksb(a,indx,b)
       implicit none
       real(rkind), dimension(4+N_S,4+N_S) :: A
       real(rkind), dimension(4+N_S)   :: b
       integer, dimension(4+N_S) :: INDX
       !f2py depend(4+N_S) A, indx

       integer :: i,ii,j,ll
       real(rkind)  :: SUMM

       II = 0

       DO I=1,4+N_S
         LL = INDX(I)
         SUMM = B(LL)
         B(LL) = B(I)
         IF(II.NE.0) THEN
           DO J=II,I-1
             SUMM = SUMM - A(I,J)*B(J)
           END DO ! j loop
         ELSE IF(SUMM.NE.0._rkind) THEN
           II = I
         END IF
         B(I) = SUMM
       END DO ! i loop

       DO I=4+N_S,1,-1
         SUMM = B(I)
         IF(I < 4+N_S) THEN
           DO J=I+1,4+N_S
             SUMM = SUMM - A(I,J)*B(J)
           END DO ! j loop
         END IF
         B(I) = SUMM / A(I,I)
       END DO ! i loop

       RETURN
    endsubroutine lubksb
!
    subroutine compute_chemistry_cuf(nx,ny,nz,nv,nv_aux,ng,nreactions,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,arr_a_gpu,arr_b_gpu,arr_ea_gpu,&
                                     tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,kc_tab_gpu,num_t_tab,&
                                     t_min_tab,dt_tab,R_univ,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,tol_iter_nr)

        integer, intent(in) :: nx,ny,nz,ng,nreactions,num_t_tab,nv,nv_aux
        integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
        integer, dimension(nreactions), intent(in), device :: reac_ty_gpu,isRev_gpu
        real(rkind), intent(in) :: t_min_tab,dt_tab,R_univ,tol_iter_nr
        real(rkind), dimension(nreactions,2), intent(in), device :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
        real(rkind), dimension(nreactions,5), intent(in), device :: falloff_coeffs_gpu
        real(rkind), dimension(num_t_tab+1,nreactions), intent(in), device :: kc_tab_gpu
        real(rkind), dimension(nreactions,N_S), intent(in), device :: tb_eff_gpu, r_coeffs_gpu, p_coeffs_gpu
        real(rkind), dimension(N_S), intent(in), device :: mw_gpu
        real(rkind), dimension(nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), device :: cv_coeff_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv    ), intent(inout), device :: w_gpu 
        real(rkind), dimension(1:nx,1:ny,1:nz,1:nv), intent(inout), device :: fl_gpu
        integer :: i,j,k,l,lsp,lr,iercuda,itt
        real(rkind) :: wdtkj,qlr
        real(rkind) :: rho,tt,dtt,dkc,ttleft
        real(rkind) :: kf,kb,kc,q1,q2,wdt,tb,h0lsp,wdt_t
        real(rkind) :: rr,pp,arr_a,arr_b,arr_ea,uu,vv,ww,rhoe,ee,qq,ri,Rtti
        real(rkind) :: k0,kinf,rpres,fcent,coff,noff,doff,foff,xoff
        real(rkind),dimension(N_S),device :: conc 

    !$cuf kernel do(3) <<<*,*>>>
    do k=1,nz
     do j=1,ny
      do i=1,nx
       do lsp=1,N_S
        w_gpu(i,j,k,lsp) = max(w_gpu(i,j,k,lsp),0._rkind)
       enddo
       rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
       ri   = 1._rkind/rho
       do lsp=1,N_S
        w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)*ri
        conc(lsp) = rho*w_aux_gpu(i,j,k,lsp)/mw_gpu(lsp)
       enddo
       uu   = w_gpu(i,j,k,I_U)*ri 
       vv   = w_gpu(i,j,k,I_V)*ri
       ww   = w_gpu(i,j,k,I_W)*ri
       rhoe = w_gpu(i,j,k,I_E)
       qq   = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
       ee = rhoe*ri-qq
       tt = get_mixture_temperature_from_e_dev(ee, w_aux_gpu(i,j,k,J_T), cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, &
                                               tol_iter_nr,i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
       w_aux_gpu(i,j,k,J_T) = tt
       Rtti = 1._rkind/(R_univ*tt)
  
       itt = int((tt-t_min_tab)/dt_tab)+1
       itt = max(itt,1)
       itt = min(itt,num_t_tab)
       ttleft = (itt-1)*dt_tab + t_min_tab
       dtt = (tt-ttleft)/dt_tab
       ! interpola kc 

       do lr=1,nreactions
        ! compute kf
        select case(reac_ty_gpu(lr))
         case(0,1) !0 => Arrhenius, 1 => Three-body
          kf = arr_a_gpu(lr,1)
          if (arr_b_gpu(lr,1)  .ne. 0._rkind) kf = kf*(tt**arr_b_gpu(lr,1))
          if (arr_ea_gpu(lr,1) .ne. 0._rkind) kf = kf* exp(-arr_ea_gpu(lr,1)*Rtti)
         case(2) !falloff-Lindemann
          k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
          kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
          tb = 0._rkind
          do lsp=1,N_S
           tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
          enddo
          rpres = k0*tb/kinf
          kf = kinf*(rpres/(1._rkind + rpres))
         case(3) !falloff-Troe
          k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
          kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
          tb = 0._rkind
          do lsp=1,N_S
           tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
          enddo
          rpres = k0*tb/kinf
          if (rpres .lt. 1E-30_rkind) then
           kf = K0*tb
          else
           fcent = (1._rkind - falloff_coeffs_gpu(lr,1))*exp(-tt/falloff_coeffs_gpu(lr,2)) + &
                   falloff_coeffs_gpu(lr,1)*exp(-tt/falloff_coeffs_gpu(lr,3))
           if (falloff_coeffs_gpu(lr,4) .ne. -3.14_rkind) then
            fcent = fcent + exp(-falloff_coeffs_gpu(lr,4)/tt)
           end if
           fcent = log10(fcent)
           coff = -0.4_rkind - 0.67_rkind*fcent
           noff = 0.75_rkind - 1.27_rkind*fcent
           doff = 0.14_rkind
           foff = 10_rkind**((((1._rkind + ((log10(rpres) + coff)/(noff -doff*(log10(rpres) + &
           coff)))**2._rkind)**(-1._rkind))*fcent))
           kf = kinf*(rpres/(1._rkind + rpres))*foff
          endif
         case(4) !falloff-SRI
          k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
          kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
          tb = 0._rkind
          do lsp=1,N_S
           tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
          enddo
          rpres = k0*tb/kinf
          xoff = 1._rkind/(1._rkind + log10(rpres)**2._rkind)
          foff = falloff_coeffs_gpu(lr,4)*((falloff_coeffs_gpu(lr,1)*exp(-falloff_coeffs_gpu(lr,2)/tt) + &
                 exp(-tt/falloff_coeffs_gpu(lr,3)))**xoff)*(tt**falloff_coeffs_gpu(lr,5))
          kf = kinf*(rpres/(1._rkind + rpres))*foff
        endselect

        if (isRev_gpu(lr) .eq. 1) then
         dkc = kc_tab_gpu(itt+1,lr)-kc_tab_gpu(itt,lr)
         kc  = kc_tab_gpu(itt,lr)+dkc*dtt
         kb = kf/kc
        else
         kb = 0._rkind
        endif

        q1  = 1._rkind
        q2  = 1._rkind
        do lsp=1,N_S
         rr = r_coeffs_gpu(lr,lsp)
         pp = p_coeffs_gpu(lr,lsp)
         if (rr == 1._rkind) then
          q1 = q1*conc(lsp)
         elseif (rr .ne. 0._rkind) then
          q1 = q1*conc(lsp)**rr
         endif
         if (pp == 1._rkind) then
          q2 = q2*conc(lsp)
         elseif (pp .ne. 0._rkind) then
          q2 = q2*conc(lsp)**pp
         endif
        enddo
        tb = 1._rkind
        if (reac_ty_gpu(lr) == 1) then
         tb = 0._rkind
         do lsp=1,N_S
           tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
         enddo
        endif
        qlr = tb*(kf*q1-kb*q2)

        do lsp=1,N_S
         wdtkj = (p_coeffs_gpu(lr,lsp)-r_coeffs_gpu(lr,lsp))*qlr*mw_gpu(lsp)
         fl_gpu(i,j,k,lsp) = fl_gpu(i,j,k,lsp) - wdtkj
!         
!         w_aux_gpu(i,j,k,J_WDOT_START+lsp) = w_aux_gpu(i,j,k,J_WDOT_START+lsp) + wdtkj
!
        enddo
       enddo
!
!       wdot_t = 0._rkind
!       do lsp=1,N_S
!        wdot = 0._rkind
!        do lr=1,nreactions
!         wdot = wdot + (p_coeffs_gpu(lr,lsp)-r_coeffs_gpu(lr,lsp))*q(lr)
!        enddo
!        h0lsp = get_species_h_from_temperature_dev(298.15_rkind,indx_cp_l,indx_cp_r,cp_coeff_gpu,&
!                                                nsetcv,trange_gpu,lsp)
!        wdot_t = wdot_t-wdot*h0lsp
!
!        fl_gpu(i,j,k,lsp) = fl_gpu(i,j,k,lsp) - wdot*mw_gpu(lsp)
!       enddo
       !fl_gpu(i,j,k,I_E) = fl_gpu(i,j,k,I_E) - wdot_t
      enddo
     enddo
    enddo
    !@cuf iercuda=cudaDeviceSynchronize()

    endsubroutine compute_chemistry_cuf

    subroutine eval_chem_aux_cuf(nx,ny,nz,nv_aux,ng,nreactions,w_aux_gpu,mw_gpu,arr_a_gpu,arr_b_gpu,arr_ea_gpu,&
                                 tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,kc_tab_gpu,num_t_tab,&
                                 t_min_tab,dt_tab,R_univ,dcsidx_gpu,detady_gpu,dzitdz_gpu,h298_gpu,enable_pasr,les_c_yoshi,&
                                 les_c_mix,les_c_eps)

        integer, intent(in) :: nx,ny,nz,ng,nreactions,num_t_tab,nv_aux,enable_pasr
        integer, dimension(nreactions), intent(in), device :: reac_ty_gpu,isRev_gpu
        real(rkind), intent(in) :: t_min_tab,dt_tab,R_univ,les_c_yoshi,les_c_mix,les_c_eps
        real(rkind), dimension(nreactions,2), intent(in), device :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
        real(rkind), dimension(nreactions,5), intent(in), device :: falloff_coeffs_gpu
        real(rkind), dimension(num_t_tab+1,nreactions), intent(in), device :: kc_tab_gpu
        real(rkind), dimension(nreactions,N_S), intent(in), device :: tb_eff_gpu, r_coeffs_gpu, p_coeffs_gpu
        real(rkind), dimension(N_S), intent(in), device :: mw_gpu, h298_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu, detady_gpu, dzitdz_gpu

        integer :: i,j,k,l,lsp,lr,iercuda,itt
        real(rkind) :: wdtkj,qlr
        real(rkind) :: rho,tt,dtt,dkc,ttleft
        real(rkind) :: kf,kb,kc,q1,q2,wdt,tb,h0lsp,wdt_t,Rtti
        real(rkind) :: rr,pp,arr_a,arr_b,arr_ea,uu,vv,ww,rhoe,ee,qq,ri
        real(rkind) :: delta, nu_sgs, eps_sgs, tau_k, nu_eff
        real(rkind) :: tau_lsp,tau_cf,tau_cs,tau_c,gam_pasr
        real(rkind) :: k0,kinf,rpres,fcent,coff,noff,doff,foff,xoff
        real(rkind), dimension(N_S),device :: conc

    !$cuf kernel do(3) <<<*,*>>>
    do k=1,nz
     do j=1,ny
      do i=1,nx
       rho  = w_aux_gpu(i,j,k,J_R) 
       ri   = 1._rkind/rho
       uu   = w_aux_gpu(i,j,k,J_U) 
       vv   = w_aux_gpu(i,j,k,J_V)
       ww   = w_aux_gpu(i,j,k,J_W)
       tt   = w_aux_gpu(i,j,k,J_T) 
       Rtti = 1._rkind/(R_univ*tt)
  
       do lsp=1,N_S
        w_aux_gpu(i,j,k,J_WDOT_START+lsp) = 0._rkind
        conc(lsp) = rho*w_aux_gpu(i,j,k,lsp)/mw_gpu(lsp)
       enddo

       !   preparaing variables for PaSR model
       if (enable_pasr > 0) then
        delta = (dcsidx_gpu(i)*detady_gpu(j)*dzitdz_gpu(k))**(-1._rkind/3._rkind)
        nu_sgs  = w_aux_gpu(i,j,k,J_LES1)*ri
        eps_sgs = les_c_eps*les_c_yoshi**(3._rkind/2._rkind)*nu_sgs**3/delta**4
        nu_eff  = w_aux_gpu(i,j,k,J_MU)*ri
        tau_k   = les_c_mix*sqrt(max(nu_eff/(eps_sgs+tiny(1._rkind)),0._rkind))
       else
        tau_k = 0._rkind
       endif

       itt = int((tt-t_min_tab)/dt_tab)+1
       itt = max(itt,1)
       itt = min(itt,num_t_tab)
       ttleft = (itt-1)*dt_tab + t_min_tab
       dtt = (tt-ttleft)/dt_tab
       ! interpola kc 

       do lr=1,nreactions
        ! compute kf
        select case(reac_ty_gpu(lr))
         case(0,1) !0 => Arrhenius, 1 => Three-body
          kf = arr_a_gpu(lr,1)
          if (arr_b_gpu(lr,1)  .ne. 0._rkind) kf = kf*(tt**arr_b_gpu(lr,1))
          if (arr_ea_gpu(lr,1) .ne. 0._rkind) kf = kf* exp(-arr_ea_gpu(lr,1)*Rtti)
         case(2) !falloff-Lindemann
          k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
          kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
          tb = 0._rkind
          do lsp=1,N_S
           tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
          enddo
          rpres = k0*tb/kinf
          kf = kinf*(rpres/(1._rkind + rpres))
         case(3) !falloff-Troe
          k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
          kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
          tb = 0._rkind
          do lsp=1,N_S
           tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
          enddo
          rpres = k0*tb/kinf
          if (rpres .lt. 1E-30_rkind) then
           kf = K0*tb
          else
           fcent = (1._rkind - falloff_coeffs_gpu(lr,1))*exp(-tt/falloff_coeffs_gpu(lr,2)) + &
                   falloff_coeffs_gpu(lr,1)*exp(-tt/falloff_coeffs_gpu(lr,3))
           if (falloff_coeffs_gpu(lr,4) .ne. -3.14_rkind) then
            fcent = fcent + exp(-falloff_coeffs_gpu(lr,4)/tt)
           end if
           fcent = log10(fcent)
           coff = -0.4_rkind - 0.67_rkind*fcent
           noff = 0.75_rkind - 1.27_rkind*fcent
           doff = 0.14_rkind
           foff = 10_rkind**((((1._rkind + ((log10(rpres) + coff)/(noff -doff*(log10(rpres) + &
           coff)))**2._rkind)**(-1._rkind))*fcent))
           kf = kinf*(rpres/(1._rkind + rpres))*foff
          endif
         case(4) !falloff-SRI
          k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
          kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
          tb = 0._rkind
          do lsp=1,N_S
           tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
          enddo
          rpres = k0*tb/kinf
          xoff = 1._rkind/(1._rkind + log10(rpres)**2._rkind)
          foff = falloff_coeffs_gpu(lr,4)*((falloff_coeffs_gpu(lr,1)*exp(-falloff_coeffs_gpu(lr,2)/tt) + &
                 exp(-tt/falloff_coeffs_gpu(lr,3)))**xoff)*(tt**falloff_coeffs_gpu(lr,5))
          kf = kinf*(rpres/(1._rkind + rpres))*foff
        endselect

        if (isRev_gpu(lr) .eq. 1) then
         dkc = kc_tab_gpu(itt+1,lr)-kc_tab_gpu(itt,lr)
         kc  = kc_tab_gpu(itt,lr)+dkc*dtt
         kb = kf/kc
        else
         kb = 0._rkind
        endif

        q1  = 1._rkind
        q2  = 1._rkind
        do lsp=1,N_S
         rr = r_coeffs_gpu(lr,lsp)
         pp = p_coeffs_gpu(lr,lsp)
         if (rr == 1._rkind) then
          q1 = q1*conc(lsp)
         elseif (rr .ne. 0._rkind) then
          q1 = q1*conc(lsp)**rr
         endif
         if (pp == 1._rkind) then
          q2 = q2*conc(lsp)
         elseif (pp .ne. 0._rkind) then
          q2 = q2*conc(lsp)**pp
         endif
        enddo
        tb = 1._rkind
        if (reac_ty_gpu(lr) == 1) then
         tb = 0._rkind
         do lsp=1,N_S
           tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
         enddo
        endif
        qlr = tb*(kf*q1-kb*q2)
        
        do lsp=1,N_S
         wdtkj = (p_coeffs_gpu(lr,lsp)-r_coeffs_gpu(lr,lsp))*qlr*mw_gpu(lsp)
         w_aux_gpu(i,j,k,J_WDOT_START+lsp) = w_aux_gpu(i,j,k,J_WDOT_START+lsp) + wdtkj ! wdot
!
        enddo
       enddo

  !    PaSR model
       if (enable_pasr > 0) then
        tau_cf = 0._rkind 
        tau_cs = huge(1._rkind)
        do lsp=1,N_S
         if (abs(w_aux_gpu(i,j,k,J_WDOT_START+lsp)*ri) > 1.D-08) then
          tau_lsp = w_aux_gpu(i,j,k,lsp)/abs(w_aux_gpu(i,j,k,J_WDOT_START+lsp)*ri)
          tau_cf  = max(tau_cf,tau_lsp)
          if (tau_lsp > 0._rkind) tau_cs = min(tau_cs,tau_lsp)
         endif
        enddo
        tau_c = sqrt(tau_cs*tau_cf)  
        if (tau_k > tiny(1._rkind)) then
         gam_pasr = tau_c/(tau_c + tau_k)
        else
         gam_pasr = 1._rkind
        endif
        do lsp=1,N_S
         w_aux_gpu(i,j,k,J_WDOT_START+lsp) = gam_pasr*w_aux_gpu(i,j,k,J_WDOT_START+lsp)
        enddo
        w_aux_gpu(i,j,k,J_GAM_PASR) = gam_pasr
       endif

       w_aux_gpu(i,j,k,J_HRR) = 0._rkind
       do lsp=1,N_S
        w_aux_gpu(i,j,k,J_HRR) = w_aux_gpu(i,j,k,J_HRR) - h298_gpu(lsp)*w_aux_gpu(i,j,k,J_WDOT_START+lsp)/mw_gpu(lsp) ! hrr
       enddo
!
!       wdot_t = 0._rkind
!       do lsp=1,N_S
!        wdot = 0._rkind
!        do lr=1,nreactions
!         wdot = wdot + (p_coeffs_gpu(lr,lsp)-r_coeffs_gpu(lr,lsp))*q(lr)
!        enddo
!        h0lsp = get_species_h_from_temperature_dev(298.15_rkind,indx_cp_l,indx_cp_r,cp_coeff_gpu,&
!                                                nsetcv,trange_gpu,lsp)
!        wdot_t = wdot_t-wdot*h0lsp
!
!        fl_gpu(i,j,k,lsp) = fl_gpu(i,j,k,lsp) - wdot*mw_gpu(lsp)
!       enddo
       !fl_gpu(i,j,k,I_E) = fl_gpu(i,j,k,I_E) - wdot_t
      enddo
     enddo
    enddo
    !@cuf iercuda=cudaDeviceSynchronize()

    endsubroutine eval_chem_aux_cuf 

    subroutine eval_velaux_cuf(nx,ny,nz,ng,nv,w_gpu,w_aux_gpu)
        integer, intent(in) :: nx, ny, nz, ng, nv
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv), intent(inout), device :: w_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: w_aux_gpu
        integer :: i, j, k, ll, lsp
        integer :: iercuda
        real(rkind) :: rho,ri,rhou,rhov,rhow,uu,vv,ww
!
        !$cuf kernel do(3) <<<*,*>>>
        do k=1-ng,nz+ng
         do j=1-ng,ny+ng
          do i=1-ng,nx+ng
            do lsp=1,N_S
             w_gpu(i,j,k,lsp) = max(w_gpu(i,j,k,lsp),0._rkind)
            enddo
            rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
            ri   = 1._rkind/rho
            do lsp=1,N_S
             w_aux_gpu(i,j,k,lsp) = w_gpu(i,j,k,lsp)*ri
!            w_aux_gpu(i,j,k,lsp) = max(w_aux_gpu(i,j,k,lsp),0._rkind)
!            w_aux_gpu(i,j,k,lsp) = min(w_aux_gpu(i,j,k,lsp),1._rkind)
            enddo
            rhou = w_gpu(i,j,k,I_U)
            rhov = w_gpu(i,j,k,I_V)
            rhow = w_gpu(i,j,k,I_W)
            uu   = rhou*ri
            vv   = rhov*ri
            ww   = rhow*ri
!            
            w_aux_gpu(i,j,k,J_R) = rho
            w_aux_gpu(i,j,k,J_U) = uu
            w_aux_gpu(i,j,k,J_V) = vv
            w_aux_gpu(i,j,k,J_W) = ww
            w_aux_gpu(i,j,k,J_LES1) = 0._rkind
          enddo
         enddo
        enddo
        !@cuf iercuda=cudaDeviceSynchronize()
!
    end subroutine eval_velaux_cuf
!
!   LES
!
    subroutine les_wale_mut_cuf(nx,ny,nz,ng,ep_order,w_aux_gpu,coeff_deriv1_gpu,dcsidx_gpu, &
                                detady_gpu,dzitdz_gpu,les_c_wale,eps_sensor,sensor_type)

        integer, intent(in) :: nx,ny,nz,ng,ep_order,sensor_type
        real(rkind), intent(in) ::eps_sensor
        real(rkind), intent(in) :: les_c_wale
        real(rkind), dimension(1-ng:,1-ng:,1-ng:, 1:), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: coeff_deriv1_gpu
        real(rkind), dimension(1:), intent(in), device :: dcsidx_gpu, detady_gpu, dzitdz_gpu
        real(rkind) :: ccl, delta2
        real(rkind) :: mu_sgs,tt,cploc,rho,nu_sgs
        real(rkind) :: normS, normSd
        real(rkind) :: dudx, dudy, dudz
        real(rkind) :: dvdx, dvdy, dvdz
        real(rkind) :: dwdx, dwdy, dwdz
        real(rkind) :: S_xx, S_xy, S_xz
        real(rkind) ::       S_yy, S_yz
        real(rkind) ::             S_zz
        real(rkind) :: Sd_xx, Sd_xy, Sd_xz
        real(rkind) ::        Sd_yy, Sd_yz
        real(rkind) ::               Sd_zz
        real(rkind) :: sqvg_xx, sqvg_xy, sqvg_xz
        real(rkind) :: sqvg_yx, sqvg_yy, sqvg_yz
        real(rkind) :: sqvg_zx, sqvg_zy, sqvg_zz
        real(rkind) :: sqvg_tce
        real(rkind) :: div,div3l,omegax,omegay,omegaz,omod2
        real(rkind) :: eps_small
!
        integer :: lmax
        integer :: i,j,k,l,ll,iercuda
!
        lmax = ep_order/2
        eps_small = tiny(1._rkind)
        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do i=1,nx
           ! Compute local grid scale
           delta2 = (dcsidx_gpu(i)*detady_gpu(j)*dzitdz_gpu(k))
           delta2 = delta2**(-2._rkind/3._rkind)
           ! Compute velocity gradient
           dudx = 0._rkind
           dudy = 0._rkind
           dudz = 0._rkind
           dvdx = 0._rkind
           dvdy = 0._rkind
           dvdz = 0._rkind
           dwdx = 0._rkind
           dwdy = 0._rkind
           dwdz = 0._rkind
           do l = 1,lmax
            ccl = coeff_deriv1_gpu(l,lmax)
            dudx = dudx + ccl*(w_aux_gpu(i+l,j,k,J_U) - w_aux_gpu(i-l,j,k,J_U))
            dudy = dudy + ccl*(w_aux_gpu(i,j+l,k,J_U) - w_aux_gpu(i,j-l,k,J_U))
            dudz = dudz + ccl*(w_aux_gpu(i,j,k+l,J_U) - w_aux_gpu(i,j,k-l,J_U))
            dvdx = dvdx + ccl*(w_aux_gpu(i+l,j,k,J_V) - w_aux_gpu(i-l,j,k,J_V))
            dvdy = dvdy + ccl*(w_aux_gpu(i,j+l,k,J_V) - w_aux_gpu(i,j-l,k,J_V))
            dvdz = dvdz + ccl*(w_aux_gpu(i,j,k+l,J_V) - w_aux_gpu(i,j,k-l,J_V))
            dwdx = dwdx + ccl*(w_aux_gpu(i+l,j,k,J_W) - w_aux_gpu(i-l,j,k,J_W))
            dwdy = dwdy + ccl*(w_aux_gpu(i,j+l,k,J_W) - w_aux_gpu(i,j-l,k,J_W))
            dwdz = dwdz + ccl*(w_aux_gpu(i,j,k+l,J_W) - w_aux_gpu(i,j,k-l,J_W))
           enddo
           dudx = dudx*dcsidx_gpu(i)
           dudy = dudy*detady_gpu(j)
           dudz = dudz*dzitdz_gpu(k)
           dvdx = dvdx*dcsidx_gpu(i)
           dvdy = dvdy*detady_gpu(j)
           dvdz = dvdz*dzitdz_gpu(k)
           dwdx = dwdx*dcsidx_gpu(i)
           dwdy = dwdy*detady_gpu(j)
           dwdz = dwdz*dzitdz_gpu(k)
           !
           div    = dudx+dvdy+dwdz
           div3l  = div/3._rkind
           omegax = dwdy-dvdz
           omegay = dudz-dwdx
           omegaz = dvdx-dudy
           omod2  = omegax*omegax+omegay*omegay+omegaz*omegaz
!          w_aux_gpu(i,j,k,J_DUC) = (max(-div/sqrt(omod2+div**2+(u0/l0)**2),0._rkind))**2
           if (sensor_type == 0) then
            w_aux_gpu(i,j,k,J_DUC) = (-div/sqrt(omod2+div**2+eps_sensor))**2
           else
            w_aux_gpu(i,j,k,J_DUC) = (max(-div/sqrt(omod2+div**2+eps_sensor),0._rkind))**2
           endif
           ! Compute strain rate S tensor
           S_xx =            dudx
           S_xy = 0.5_rkind*(dudy + dvdx)
           S_xz = 0.5_rkind*(dudz + dwdx)
           S_yy =            dvdy
           S_yz = 0.5_rkind*(dvdz + dwdy)
           S_zz =            dwdz
           ! Compute square of the velocity gradient
           sqvg_xx = dudx*dudx + dudy*dvdx + dudz*dwdx
           sqvg_xy = dudx*dudy + dudy*dvdy + dudz*dwdy
           sqvg_xz = dudx*dudz + dudy*dvdz + dudz*dwdz
           sqvg_yx = dvdx*dudx + dvdy*dvdx + dvdz*dwdx
           sqvg_yy = dvdx*dudy + dvdy*dvdy + dvdz*dwdy
           sqvg_yz = dvdx*dudz + dvdy*dvdz + dvdz*dwdz
           sqvg_zx = dwdx*dudx + dwdy*dvdx + dwdz*dwdx
           sqvg_zy = dwdx*dudy + dwdy*dvdy + dwdz*dwdy
           sqvg_zz = dwdx*dudz + dwdy*dvdz + dwdz*dwdz
           ! Sd tensor
           sqvg_tce = (sqvg_xx + sqvg_yy + sqvg_zz)/3._rkind
           Sd_xx =            sqvg_xx              - sqvg_tce
           Sd_xy = 0.5_rkind*(sqvg_xy + sqvg_yx)
           Sd_xz = 0.5_rkind*(sqvg_xz + sqvg_zx)
           Sd_yy =            sqvg_yy              - sqvg_tce
           Sd_yz = 0.5_rkind*(sqvg_yz + sqvg_zy)
           Sd_zz =            sqvg_zz              - sqvg_tce
           ! Compute norms
           normS = S_xx*S_xx+S_yy*S_yy+S_zz*S_zz+2._rkind*(S_xy*S_xy+S_xz*S_xz+S_yz*S_yz)
           normS = sqrt(normS)
           normSd = Sd_xx*Sd_xx+Sd_yy*Sd_yy+Sd_zz*Sd_zz+2._rkind*(Sd_xy*Sd_xy+Sd_xz*Sd_xz+Sd_yz*Sd_yz)
           normSd = sqrt(normSd)
           ! Compute SGS viscosity
           rho    = w_aux_gpu(i,j,k,J_R)
           mu_sgs = rho*les_c_wale*les_c_wale*delta2*normSd**3/(normS**5 + sqrt(normSd**5)+eps_small)
           nu_sgs = mu_sgs/rho
           w_aux_gpu(i,j,k,J_LES1) = mu_sgs 
           w_aux_gpu(i,j,k,J_DIV)  = div3l
          enddo
         enddo
        enddo
        !@cuf iercuda=cudadevicesynchronize()
!        
    end subroutine les_wale_mut_cuf
!
    subroutine eval_aux_les_cuf(nx, ny, nz, ng, nv, nv_aux, w_gpu, w_aux_gpu, &
            p0, t_min_tab,dt_tab, R_univ, rgas_gpu, &
            cv_coeff_gpu, cp_coeff_gpu, nsetcv, trange_gpu, indx_cp_l, indx_cp_r, tol_iter_nr, &
            mw_gpu, mwinv_gpu, visc_species_gpu, lambda_species_gpu, diffbin_species_gpu, num_t_tab, les_pr, les_sc)
        integer, intent(in) :: nx, ny, nz, ng, nv, nv_aux, nsetcv, num_t_tab
        integer, intent(in) :: indx_cp_l, indx_cp_r
        real(rkind), intent(in) :: tol_iter_nr, R_univ, p0, les_pr, les_sc
        real(rkind), intent(in) :: t_min_tab,dt_tab
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cp_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu,mw_gpu,mwinv_gpu
        real(rkind), dimension(num_t_tab+1,N_S), intent(in), device :: visc_species_gpu,lambda_species_gpu
        real(rkind), dimension(num_t_tab+1,N_S,N_S), intent(in), device :: diffbin_species_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv), intent(inout), device :: w_gpu  
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv_aux), intent(inout), device :: w_aux_gpu
        integer     :: i,j,k,lsp,msp,itt
        real(rkind) :: rho, rhoe, uu, vv, ww, qq, pp, tt, mu, ee, c, ri
        real(rkind) :: cploc,gamloc,rmixtloc
        real(rkind) :: dmu, mulsp, dtt, mu_den, mumsp, phi_lm
        real(rkind) :: mwmixt, k_cond, k_cond2, dlam, xlsp, klsp
        integer     :: iercuda
        real(rkind) :: xmsp, diff_den, ddiff, diff_ij, tloc
        real(rkind) :: mu_sgs, nu_sgs 
!
        !$cuf kernel do(3) <<<*,*>>>
        do k=1-ng,nz+ng
         do j=1-ng,ny+ng
          do i=1-ng,nx+ng
            rho    = w_aux_gpu(i,j,k,J_R)
            ri     = 1._rkind/rho
            uu     = w_aux_gpu(i,j,k,J_U)
            vv     = w_aux_gpu(i,j,k,J_V)
            ww     = w_aux_gpu(i,j,k,J_W)
            mu_sgs = w_aux_gpu(i,j,k,J_LES1)
            nu_sgs = mu_sgs*ri
            qq     = 0.5_rkind*(uu*uu+vv*vv+ww*ww)
            rhoe   = w_gpu(i,j,k,I_E)
            ee = rhoe*ri-qq
            tt = get_mixture_temperature_from_e_dev(ee, w_aux_gpu(i,j,k,J_T), cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, &
                                              tol_iter_nr,i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
!
            rmixtloc = get_rmixture(i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu,rgas_gpu)
            cploc    = get_cp_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,&
                       i,j,k,nx,ny,nz,ng,nv_aux,w_aux_gpu)
            gamloc   = cploc/(cploc-rmixtloc)
            pp       = rho*tt*rmixtloc ! EoS for a perfect gas
            c        = sqrt(gamloc*pp*ri)
!
            w_aux_gpu(i,j,k,J_H)  = (rhoe+pp)*ri
            w_aux_gpu(i,j,k,J_T)  = tt
            w_aux_gpu(i,j,k,J_P)  = pp
            w_aux_gpu(i,j,k,J_C)  = c
!
            itt = int((tt-t_min_tab)/dt_tab)+1
            itt = max(itt,1)
            itt = min(itt,num_t_tab)
            tloc = t_min_tab+(itt-1)*dt_tab
            dtt = (tt-tloc)/dt_tab
            ! interpola viscosity and conductivity

!           Mathur et al (1967) mixture thermal conductivity
            mwmixt   = R_univ/rmixtloc 
!           w_aux_gpu(i,j,k,J_MW) = mwmixt
            k_cond   = 0._rkind
            k_cond2  = 0._rkind
            do lsp = 1,N_S
             dlam    = lambda_species_gpu(itt+1,lsp)-lambda_species_gpu(itt,lsp)
             klsp    = lambda_species_gpu(itt,lsp)+dlam*dtt
             xlsp    = w_aux_gpu(i,j,k,lsp)*mwmixt*mwinv_gpu(lsp)
             k_cond  = k_cond  + xlsp*klsp
             k_cond2 = k_cond2 + xlsp/klsp
            enddo
            k_cond = 0.5_rkind*(k_cond + 1._rkind/k_cond2)
!
!           Wilke (1950) mixture dynamic viscosity            
            mu   = 0._rkind
            do lsp=1,N_S
             dmu   = visc_species_gpu(itt+1,lsp)-visc_species_gpu(itt,lsp)
             mulsp = visc_species_gpu(itt,lsp)+dmu*dtt
             mu_den = 0._rkind
             do msp=1,N_S 
              dmu    = visc_species_gpu(itt+1,msp)-visc_species_gpu(itt,msp)
              mumsp  = visc_species_gpu(itt,msp)+dmu*dtt
              phi_lm = 1._rkind/sqrt(1._rkind+mw_gpu(lsp)*mwinv_gpu(msp))*&
                      (1._rkind + sqrt(mumsp/mulsp)*(mw_gpu(msp)*mwinv_gpu(lsp))**0.25_rkind)**2
              mu_den = mu_den + w_aux_gpu(i,j,k,msp)*phi_lm*mwinv_gpu(msp)
             enddo    
             mu = mu + mulsp*w_aux_gpu(i,j,k,lsp)*mwinv_gpu(lsp)/mu_den
            enddo    
            mu = sqrt(8._rkind)*mu
!
            w_aux_gpu(i,j,k,J_MU)     = mu     + mu_sgs
            w_aux_gpu(i,j,k,J_K_COND) = k_cond + mu_sgs*cploc/les_pr

!           Bird (1960) species' diffusion into mixture    
            do lsp = 1,N_S
              diff_den = 0._rkind
              do msp = 1,N_S
               if (msp /= lsp) then
                ddiff = diffbin_species_gpu(itt+1,lsp,msp)-diffbin_species_gpu(itt,lsp,msp)
                diff_ij = diffbin_species_gpu(itt,lsp,msp)+ddiff*dtt
                xmsp = w_aux_gpu(i,j,k,msp)*mwmixt*mwinv_gpu(msp)
                diff_den = diff_den + xmsp/diff_ij
               endif
              enddo
              if (diff_den > 1.0D-015) then
               w_aux_gpu(i,j,k,J_D_START+lsp) = (1._rkind-w_aux_gpu(i,j,k,lsp))/diff_den*p0/pp*rho ! rho * diffusion
              else
               w_aux_gpu(i,j,k,J_D_START+lsp) = 0._rkind
              endif
              w_aux_gpu(i,j,k,J_D_START+lsp) = w_aux_gpu(i,j,k,J_D_START+lsp) + mu_sgs/les_sc
            enddo
 
          enddo
         enddo
        enddo
    endsubroutine eval_aux_les_cuf
!
!    attributes(device) function get_species_rhodiff_dev(lsp,tt,i,j,k,nx,ny,nz,ng,nv_aux,w_aux_gpu,&
!                                mwmixt,p0,mwinv_gpu,num_t_tab,t_min_tab,dt_tab,diffbin_species_gpu,&
!                                enable_les,les_sc)
!    real(rkind) :: get_species_rhodiff_dev
!    integer, value :: lsp,i,j,k,nv_aux,nx,ny,nz,ng,num_t_tab,enable_les
!    real(rkind), value :: tt,p0,t_min_tab,dt_tab,mwmixt,les_sc
!    real(rkind), dimension(N_S) :: mwinv_gpu
!    real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
!    real(rkind), dimension(num_t_tab+1,N_S,N_S) :: diffbin_species_gpu
!    real(rkind) :: tloc,dtt,rho,pp,diff_den,ddiff,diff_ij,xmsp,rhodiff
!    integer :: msp,itt
!!
!    itt = int((tt-t_min_tab)/dt_tab)+1
!    tloc = t_min_tab+(itt-1)*dt_tab
!    dtt = (tt-tloc)/dt_tab
!!
!    rho = w_aux_gpu(i,j,k,J_R)
!    pp  = w_aux_gpu(i,j,k,J_P)
!!
!    diff_den = 0._rkind
!    do msp = 1,N_S
!     if (msp /= lsp) then
!      ddiff = diffbin_species_gpu(itt+1,lsp,msp)-diffbin_species_gpu(itt,lsp,msp)
!      diff_ij = diffbin_species_gpu(itt,lsp,msp)+ddiff*dtt
!      xmsp = w_aux_gpu(i,j,k,msp)*mwmixt*mwinv_gpu(msp)
!      diff_den = diff_den + xmsp/diff_ij
!     endif
!    enddo
!    if (diff_den > 0._rkind) then
!     rhodiff = (1._rkind-w_aux_gpu(i,j,k,lsp))/diff_den*p0/pp*rho ! rho * diffusion
!    else
!     rhodiff = 0._rkind
!    endif
!!
!    if (enable_les > 0) rhodiff = rhodiff + w_aux_gpu(i,j,k,J_LES1)/les_sc
!    !do lsp=1,N_S
!    ! w_aux_gpu(i,j,k,J_D_START+lsp) = rhodiff
!    !enddo
!!
!    get_species_rhodiff_dev = rhodiff
!!
!    endfunction get_species_rhodiff_dev
!
    subroutine tripping_cuf(nx,ny,nz,ng,nv,pi,xtr1,xtr2,x0tr,lamx,lamy,lamz,lamz1,phiz,phiz1,asl,bt, &
                                       x_gpu,y_gpu,z_gpu,w_gpu,fl_gpu)

      real(rkind),intent(in) :: pi,xtr1,xtr2,x0tr,lamx,lamy,lamz,lamz1,phiz,phiz1,asl,bt
      integer, intent(in) :: nx,ny,nz,nv,ng
      real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(in), device :: w_gpu
      real(rkind), dimension(1:,1:,1:,1:), intent(inout), device :: fl_gpu
      real(rkind), dimension(1-ng:), intent(in), device :: x_gpu, y_gpu, z_gpu
      real(rkind) :: xx,yy,zz,hzi,hzi1,gzt,fz,fzx,fzy,rho
      integer :: i,j,k,iercuda
      !Tripping application on the pressure side 
      !$cuf kernel do(3) <<<*,*>>>
      do k=1,nz
       do j=1,ny
        do i=1,nx
         !if (yc2_gpu(i,1)<0._rkind.and.xc2_gpu(i,1)<xtr2.and.xc2_gpu(i,1)>xtr1) then
         if (x_gpu(i) < xtr2 .and. x_gpu(i) > xtr1) then
          rho  = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
          xx = x_gpu(i)
          yy = y_gpu(j)
          zz = z_gpu(k)
          hzi  = sin(2._rkind*pi*lamz *zz+phiz )
          hzi1 = sin(2._rkind*pi*lamz1*zz+phiz1)
          gzt  = asl*((1-bt)*hzi+bt*hzi1)
          fz   = exp(-((xx-x0tr)/lamx)**2-(yy/lamy)**2)
          fz   = fz*gzt*rho ! force per unit mass -> force per unit volume
          fzx  = 0._rkind ! x component
          fzy  = fz       ! y component
          fl_gpu(i,j,k,I_U) = fl_gpu(i,j,k,I_U) - fzx
          fl_gpu(i,j,k,I_V) = fl_gpu(i,j,k,I_V) - fzy
          fl_gpu(i,j,k,I_E) = fl_gpu(i,j,k,I_E) -(fzx*w_gpu(i,j,k,I_U)+fzy*w_gpu(i,j,k,I_V))
         endif
        enddo
       enddo
      enddo
      !@cuf iercuda=cudaDeviceSynchronize()
    end subroutine tripping_cuf

    subroutine limiter_cuf(nx,ny,nz,ng,nv,nv_aux,w_gpu,w_aux_gpu,iblock,kblock,&
                           indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu, &
                           rho_lim,tem_lim,rho_lim_rescale,tem_lim_rescale)

     integer, intent(in) :: nx,ny,nz,ng,iblock,kblock,indx_cp_l,indx_cp_r,nsetcv,nv,nv_aux
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng, 1:nv), intent(inout), device :: w_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
     real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
     real(rkind), intent(in) :: rho_lim, tem_lim, rho_lim_rescale, tem_lim_rescale
     real(rkind) :: rho, tem, eei, uu, vv, ww, rho_old
     integer :: i,j,k,iercuda, n_limited_rho, n_limited_tem, lsp

     n_limited_rho = 0
     n_limited_tem = 0
     !$cuf kernel do(3) <<<*,*>>> reduction(+: n_limited_rho, n_limited_tem)
      do k=1,nz
       do j=1,ny
        do i=1,nx
         rho = w_aux_gpu(i,j,k,J_R)
         tem = w_aux_gpu(i,j,k,J_T)
         if((rho < rho_lim).or.(tem < tem_lim)) then
          if(rho < rho_lim) then
            !rho = rho_lim
            !rho = rho_lim*rho_lim_rescale
            n_limited_rho = n_limited_rho + 1
          endif
          if(tem < tem_lim) then
            !tem = tem_lim
            !tem = tem_lim*tem_lim_rescale
            n_limited_tem = n_limited_tem + 1
          endif
          !w_gpu(i,j,k,1) = rho
          !w_gpu(i,j,k,2) = w_aux_gpu(i,j,k,2)*rho
          !w_gpu(i,j,k,3) = w_aux_gpu(i,j,k,3)*rho
          !w_gpu(i,j,k,4) = w_aux_gpu(i,j,k,4)*rho
          !eei   = get_e_from_temperature_dev(tem, t0, indx_cp_l, indx_cp_r, cv_coeff_gpu, calorically_perfect)
          !w_gpu(i,j,k,5) = rho*eei + 0.5_rkind*(w_gpu(i,j,k,2)**2+w_gpu(i,j,k,3)**2+w_gpu(i,j,k,4)**2)/rho
         endif
        enddo
       enddo
      enddo
      !@cuf iercuda=cudaDeviceSynchronize()
      if(n_limited_rho > 0) print*,'warning! n_limited_rho :',n_limited_rho
      if(n_limited_tem > 0) print*,'warning! n_limited_tem :',n_limited_tem

      if(n_limited_rho > 0 .or. n_limited_tem > 0) then
      !$cuf kernel do(3) <<<*,*>>>
      do k=1,nz
       do j=1,ny
        do i=1,nx
         rho = get_rho_from_w_dev(i,j,k,nv,nx,ny,nz,ng,w_gpu)
         uu  = w_gpu(i,j,k,I_U)/rho
         vv  = w_gpu(i,j,k,I_V)/rho
         ww  = w_gpu(i,j,k,I_W)/rho
         tem = w_aux_gpu(i,j,k,J_T)
         rho_old = rho
         if((rho < rho_lim).or.(tem < tem_lim)) then
          if(rho < rho_lim) then
            rho = rho_lim*rho_lim_rescale
            w_aux_gpu(i,j,k,J_R) = rho
            !print*,'limiter rhofix: ',iblock,kblock,i,j,k
          endif
          if(tem < tem_lim) then
            tem = tem_lim*tem_lim_rescale
            w_aux_gpu(i,j,k,J_T) = tem
            !print*,'limiter temfix: ',iblock,kblock,i,j,k
          endif
          do lsp=1,N_S
           w_gpu(i,j,k,lsp) = rho*w_gpu(i,j,k,lsp)/rho_old
          enddo
          w_gpu(i,j,k,I_U) = rho*uu
          w_gpu(i,j,k,I_V) = rho*vv
          w_gpu(i,j,k,I_W) = rho*ww
          eei = get_mixture_e_from_temperature_dev(tem,indx_cp_l,indx_cp_r,cv_coeff_gpu,&
                                    nsetcv,trange_gpu,i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
          w_gpu(i,j,k,I_E) = rho*eei + 0.5_rkind*(w_gpu(i,j,k,I_U)**2+w_gpu(i,j,k,I_V)**2+w_gpu(i,j,k,I_W)**2)/rho
         endif
        enddo
       enddo
      enddo
      !@cuf iercuda=cudaDeviceSynchronize()
      endif
    endsubroutine limiter_cuf

    subroutine find_boom_cuf(nx,ny,nz,ng,nv,fln_gpu,dt,fluid_mask_gpu,ncoords1,ncoords3)
        integer, intent(in)  :: nx, ny, nz, ng, nv, ncoords1, ncoords3
        real(rkind), intent(in) :: dt
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: fluid_mask_gpu
        real(rkind), dimension(1:nx, 1:ny, 1:nz, nv), intent(in), device :: fln_gpu

        integer :: i,j,k,iercuda, atomex
        real(rkind) :: residual

        !$cuf kernel do(3) <<<*,*>>>
        do k=1,nz
         do j=1,ny
          do i=1,nx
           if (fluid_mask_gpu(i,j,k)==0) then
            residual = (fln_gpu(i,j,k,2)/dt)**2
            if (residual /= residual) then ! NaN is never equal to itself
              write(*,*) 'BOOM at ig, jg, kg = ', i+ncoords1*nx, j, k+ncoords3*nz
            endif
           endif
          enddo
         enddo
       enddo
       !@cuf iercuda=cudaDeviceSynchronize()

    endsubroutine find_boom_cuf

    attributes(global) launch_bounds(512) subroutine rosenbrock_krylov_kernel(nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,&
                                                     nreactions,arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,&
                                                     r_coeffs_gpu,p_coeffs_gpu,kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,&
                                                     cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l,indx_cp_r,dttry,time_start,time_end,&
                                                     simpler_splitting_,fl_sav_gpu,maxsteps,enable_pasr,les_c_yoshi,les_c_mix,les_c_eps,&
                                                     dcsidx_gpu,detady_gpu,dzitdz_gpu)
     implicit none
     ! passed arguments
     integer, value     :: nv,nv_aux,nx,ny,nz,ng,nreactions,num_t_tab,simpler_splitting_,enable_pasr
     integer, value     :: indx_cp_l,indx_cp_r,nsetcv,maxsteps    !,maxtry
     integer, dimension(nreactions) :: reac_ty_gpu,isReV_gpu
     real(rkind), value :: dttry,t_min_tab,dt_tab,tol_iter_nr,R_univ,time_start,time_end,les_c_yoshi,les_c_mix,les_c_eps
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv    ) :: w_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ) :: fl_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ) :: fl_sav_gpu
     real(rkind), dimension(N_S)          :: mw_gpu
     real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
     real(rkind), dimension(nreactions,2)           :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
     real(rkind), dimension(nreactions,5)           :: falloff_coeffs_gpu 
     real(rkind), dimension(nreactions,N_S)         :: tb_eff_gpu,r_coeffs_gpu,p_coeffs_gpu
     real(rkind), dimension(num_t_tab+1,nreactions) :: kc_tab_gpu
     real(rkind), dimension(1:) :: dcsidx_gpu, detady_gpu, dzitdz_gpu

     ! local variables
     integer :: i,j,k,iter,jtry,m,finish,CALL_ARN,row
     integer :: kk, n, nn, ii, ll, jj, mA, lsp
     real(rkind) :: time,dt,errmax,dtnext,t0,remaining,itol
     real(rkind), dimension(4+N_S)    :: w_sav, cc, rhs_sav, err, yscal, VEC, ysav, ys, yt
     real(rkind), dimension(4+N_S) :: jacobian
     real(rkind), dimension(4+N_S, M_max) :: Vb
     real(rkind), dimension(M_max,4+N_S)  :: Vt
     real(rkind), dimension(M_max, M_max) :: H
!          
     real(rkind), dimension(M_max, M_max) :: inv  !device
!          
     real(rkind), dimension(N_S) :: w_gpu_l
     real(rkind), dimension(N_S) :: w_aux
     real(rkind) :: T_iter, T_N, T_aux, errn, rm, p_if
!
     real(rkind), dimension(N_S)          :: rgas_gpu
     real(rkind) :: rho,rmixtloc,pp,tt,ee,qq,uu,vv,ww,ri,rhou,rhov,rhow,rhoe
!
     real(rkind) :: gam
     real(rkind), dimension(4,3) :: alpha, gamma_c
     real(rkind), dimension(4) :: bs, bt, Ec
     real(rkind), dimension(M_max,M_max) :: IMAT
     real(rkind), dimension(4+N_S) :: alfa, GK
     integer :: ns,it,s
     real(rkind), dimension(4+N_S) :: rhs_s
     real(rkind), dimension(4+N_S) :: F
     real(rkind), dimension(4+N_S,M_max) :: k_s
     real(rkind), dimension(4+N_S) :: errmax_i

     real(rkind), dimension(4+N_S, M_max) :: A
     real(rkind), dimension(4+N_S, 4+N_S) :: B
     real(rkind), dimension(4+N_S) :: C
     real(rkind), dimension(4+N_S) :: D
     real(rkind), dimension(M_max, 4+N_S) :: E
     real(rkind), dimension(M_max, M_max)  :: IDIFF
     real(rkind) :: delta, nu_sgs, eps_sgs, tau_k, nu_eff

     real(rkind) :: Zbil
     real(rkind), parameter :: eps_Z = 5E-4_rkind

     gam = 0.572816062482135_rkind
     
     ! Coefficienti alpha(i,j)
     alpha(2,1) =  0.432364435748567_rkind
     alpha(3,1) = -0.514211316876170_rkind
     alpha(3,2) =  1.382271144617360_rkind
     alpha(4,1) = -0.514211316876170_rkind
     alpha(4,2) =  1.382271144617360_rkind
     alpha(4,3) =  0._rkind
     
     ! Coefficienti bs(i)
     bs(1) = 0.194335256262729_rkind
     bs(2) = 0.483167813989227_rkind
     bs(3) = 0.0_rkind
     bs(4) = 0.322496929748044_rkind
     
     ! Coefficienti gamma(i,j)
     gamma_c(2,1) = -0.602765307997356_rkind
     gamma_c(3,1) = -1.389195789724843_rkind
     gamma_c(3,2) =  1.072950969011413_rkind
     gamma_c(4,1) =  0.992356412977094_rkind
     gamma_c(4,2) = -1.390032613873701_rkind
     gamma_c(4,3) = -0.440875890223325_rkind
     
     ! Gamma-weighted b coefficients (if needed)
     ! b_gamma(i) = b(i) * gamma_c(...)
     bt(1) = -0.217819895945721_rkind
     bt(2) = 1.03130847478467_rkind
     bt(3) = 0.186511421161047_rkind
     bt(4) = 0.0_rkind
     
     Ec(:) = bs(:) - bt(:)

     i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
     j = blockDim%y * (blockIdx%y - 1) + threadIdx%y
     k = blockDim%z * (blockIdx%z - 1) + threadIdx%z
     if (i > nx .or. j > ny .or. k > nz) return

     Zbil = w_aux_gpu(i,j,k,J_Z)
     if (Zbil>eps_Z .and. Zbil<1_rkind-eps_Z) then
      VEC(1:4+N_S) = w_gpu(i,j,k,1:4+N_S)
      T_iter = w_aux_gpu(i,j,k,J_T)
      time   = time_start
      it = 0
      dt = dttry  !/2._rkind
      CALL_ARN = 1
      errn = 1._rkind

!     preparaing variables for PaSR model
      if (enable_pasr > 0) then
       ri    = 1._rkind/w_aux_gpu(i,j,k,J_R)
       delta = (dcsidx_gpu(i)*detady_gpu(j)*dzitdz_gpu(k))**(-1._rkind/3._rkind)
       nu_sgs  = w_aux_gpu(i,j,k,J_LES1)*ri
       eps_sgs = les_c_eps*les_c_yoshi**(3._rkind/2._rkind)*nu_sgs**3/delta**4
       nu_eff  = w_aux_gpu(i,j,k,J_MU)*ri
       tau_k   = les_c_mix*sqrt(max(nu_eff/(eps_sgs+tiny(1._rkind)),0._rkind))
      else
       tau_k = 0._rkind
      endif

      do while (time.lt.(time_end-1.d-12))
       dt = min(dt,time_end-time)
       it = it + 1
       IF (CALL_ARN==1) call arnoldi(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                        arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                        kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                        indx_cp_l,indx_cp_r,simpler_splitting_,&
                                        VEC,dt,H,Vb,inv,T_iter,rhs_sav,mA,fl_sav_gpu,enable_pasr,tau_k)

       IMAT = 0._rkind
       DO n = 1, M_max
          IMAT(n,n) = 1._rkind
       ENDDO
       IDIFF = IMAT - inv
       do kk = 1, M_max
        do ll = 1, 4+N_S
         A(ll,kk) = Vb(ll,1) * IDIFF(1,kk)
         do jj = 2, M_max
          A(ll,kk) = A(ll,kk) + Vb(ll,jj) * IDIFF(jj,kk)
         enddo
        enddo
       enddo

       do ll = 1, 4+N_S
        do jj = 1, M_max
         E(jj,ll) = Vb(ll,jj)
        enddo
       enddo

       do kk = 1, 4+N_S
        do ll = 1, 4+N_S
         B(ll,kk) = A(ll,1) * E(1,kk)
         do jj = 2, M_max
          B(ll,kk) = B(ll,kk) + A(ll,jj) * E(jj,kk)
         enddo
        enddo
       enddo

       do ii=1, nstep
        ys(:) = VEC(:)

        do jj = 1, ii-1
         ys(:) = ys(:) + dt*alpha(ii,jj)*K_s(:,jj)
        enddo

        call rosenbrock_krylov_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                   arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isReV_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                   kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                   indx_cp_l,indx_cp_r,ys,T_iter,F,enable_pasr,tau_k)
        GK(:) = 0._rkind
        do jj = 1, ii-1
         GK(:) = GK(:) + gamma_c(ii,jj)/gam*K_s(:,jj)
        enddo

        F(:) = F(:) + GK(:)

        if (simpler_splitting_ == 1) then
         F(:) = F(:) - fl_sav_gpu(i,j,k,:)
        endif
        do ll = 1, 4+N_S
         C(ll) = B(ll,1) * F(1)
         do jj = 2, 4+N_S
          C(ll) = C(ll) + B(ll,jj) * F(jj)
         enddo
        enddo

        K_s(:,ii) = F(:) - C(:) - GK(:)

       enddo  !nstep

       errmax_i = 0._rkind
       do ll = 1, nv
        ys(ll) = VEC(ll)
        yt(ll) = VEC(ll)
        do ii = 1, nstep
         ys(ll) = ys(ll) + dt*bs(ii)*K_s(ll,ii) !- dt*fl_sav_gpu(i,j,k,ll)
         yt(ll) = yt(ll) + dt*bt(ii)*K_s(ll,ii) !- dt*fl_sav_gpu(i,j,k,ll)
        enddo
        errmax_i(ll) = (yt(ll)-ys(ll))/((1.d-4)*ys(ll)+1.d-8)
!       errmax_i(ll) = (yt(ll)-ys(ll))/( 1.d-4 * max(ys(ll),VEC(ll)) + 1.d-8) 
       enddo

       errmax = NORM2(errmax_i)
       dtnext = dt*min(5._rkind,max(0.2_rkind,0.625_rkind*(errn**(0.1_rkind))/(errmax**(0.175_rkind))))
       if (errmax < 1._rkind) then
        time = time + dt
        VEC(:) = ys(:)
!       calcolo densità:            
        rho = VEC(1)
        do n=2,N_S
         rho = rho+VEC(n)
        enddo
        ri   = 1._rkind/rho
        w_aux(1:N_S) = VEC(1:N_S)*ri
        rhou = VEC(N_S+1)
        rhov = VEC(N_S+2)
        rhow = VEC(N_S+3)
        rhoe = VEC(N_S+4)
        uu   = rhou*ri
        vv   = rhov*ri
        ww   = rhow*ri
        qq   = 0.5_rkind*(uu*uu + vv*vv + ww*ww)
        ee   = rhoe*ri - qq
        T_N  = get_mixture_temperature_from_e_krylov_dev(ee,T_iter, cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, tol_iter_nr,i,j,k,nv_aux,nx,ny,nz,ng,w_aux)
        T_iter = T_N
        CALL_ARN = 1
        dt = dtnext
        errn = errmax
       else
        dt = dtnext
        CALL_ARN = 1
        errn = errmax
       endif
       if (it > maxsteps) exit
      enddo !while

!     update variables
      do lsp=1,N_S
       w_gpu(i,j,k,lsp) = VEC(lsp)
      enddo
      w_gpu(i,j,k,I_U) = VEC(N_S+1)
      w_gpu(i,j,k,I_V) = VEC(N_S+2)
      w_gpu(i,j,k,I_W) = VEC(N_S+3)
      w_gpu(i,j,k,I_E) = VEC(N_S+4)
     endif
    endsubroutine rosenbrock_krylov_kernel
!
    attributes(device) subroutine rosenbrock_krylov_jacobian(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                  arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                  kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                  indx_cp_l,indx_cp_r,simpler_splitting_,rhs_sav,w_sav,m,V_j,T_iter,jacobian,fl_sav_gpu,&
                                  enable_pasr,tau_k)
     implicit none
     ! passed arguments
     integer, intent(in) :: i,j,k,nv,nv_aux,nx,ny,nz,ng,nreactions,num_t_tab,simpler_splitting_,enable_pasr
     integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
     integer, dimension(nreactions), intent(in) :: reac_ty_gpu,isRev_gpu
     real(rkind), intent(in) :: t_min_tab,dt_tab,tol_iter_nr,R_univ,tau_k
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv    ), intent(inout) :: w_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout) :: w_aux_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ), intent(in)    :: fl_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ) :: fl_sav_gpu
     real(rkind), dimension(N_S)         , intent(in) :: mw_gpu
     real(rkind), dimension(N_S,nsetcv+1), intent(in) :: trange_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in) :: cv_coeff_gpu
     real(rkind), dimension(nreactions,2)          , intent(in) :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
     real(rkind), dimension(nreactions,5)          , intent(in) :: falloff_coeffs_gpu 
     real(rkind), dimension(nreactions,N_S)        , intent(in) :: tb_eff_gpu,r_coeffs_gpu,p_coeffs_gpu
     real(rkind), dimension(num_t_tab+1,nreactions), intent(in) :: kc_tab_gpu
     real(rkind), dimension(4+N_S), intent(in) :: rhs_sav
     real(rkind), dimension(4+N_S), intent(in) :: w_sav
     real(rkind), dimension(4+N_S) :: w_gpu_l
     real(rkind), intent(in) :: T_iter
     ! output arguments
     real(rkind), dimension(4+N_S), intent(out) :: jacobian
     ! local variables
     integer :: n,nn
     integer, intent(in) :: m
     real(rkind), dimension(4+N_S) :: wtmp
     real(rkind) :: eps,delta,h
     real(rkind), dimension(4+N_S) :: rhsp,rhsm
     real(rkind), dimension (4+N_S,M_max+1), intent(in) :: V_j

     eps   = 1.0D-08

     w_gpu_l(:) = w_sav(:) + (eps*V_j(:,m))
     call rosenbrock_krylov_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                indx_cp_l,indx_cp_r,w_gpu_l,T_iter,rhsp,enable_pasr,tau_k)

     if (simpler_splitting_ == 1) then
      rhsp(:) = rhsp(:) - fl_sav_gpu(i,j,k,:)
     endif
     jacobian(:) = (rhsp(:) - rhs_sav(:))/eps

    endsubroutine rosenbrock_krylov_jacobian
!
    attributes(device) subroutine arnoldi(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                       arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                       kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                       indx_cp_l,indx_cp_r,simpler_splitting_,&
                       w_sav,dt,H,Vb,inv,T_iter,rhs_sav,mA,fl_sav_gpu,enable_pasr,tau_k)

    implicit none
     ! passed arguments
     integer, intent(in)     :: nv,nv_aux,nx,ny,nz,ng,nreactions,num_t_tab,simpler_splitting_,enable_pasr
     integer, intent(in)     :: indx_cp_l,indx_cp_r,nsetcv
     integer, intent(out)     :: mA
     integer, dimension(nreactions), intent(in) :: reac_ty_gpu,isRev_gpu
     real(rkind), intent(in) :: t_min_tab,dt_tab,tol_iter_nr,R_univ,tau_k
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux) :: w_aux_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv    ) :: w_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ) :: fl_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ) :: fl_sav_gpu
     real(rkind), dimension(N_S)          :: mw_gpu
     real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cv_coeff_gpu
     real(rkind), dimension(nreactions,2)           :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
     real(rkind), dimension(nreactions,5)           :: falloff_coeffs_gpu 
     real(rkind), dimension(nreactions,N_S)         :: tb_eff_gpu,r_coeffs_gpu,p_coeffs_gpu
     real(rkind), dimension(num_t_tab+1,nreactions) :: kc_tab_gpu
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!------ARGOMENTI RHS---------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     integer :: i,j,k,s,errorflag
     integer :: ii,jj,ll,kk,lim_min
     real(rkind), dimension(4+N_S), intent(in) :: w_sav
     real(rkind), intent(in) :: dt
     real(rkind), dimension(4+N_S), intent(out) :: rhs_sav
     real(rkind), dimension(4+N_S,M_max), intent(out) :: Vb
     integer :: m,b,mm,tt,t,n,nn,nnn,rr,ss,vv,vvv,d
     real(rkind), dimension(M_max, M_max), intent(out) :: H
     real(rkind), dimension(4+N_S) :: jacobian
     real(rkind), dimension(M_max, M_max), intent(out) :: inv
!    integer, dimension(M_max) :: indx1
     real(rkind), parameter :: gam = 0.572816062482135_rkind
     real(rkind), dimension(M_max+1,M_max) :: HH
!    real(rkind), dimension(M_max,M_max) :: inv_alloc
     real(rkind), dimension(4+N_S,M_max+1) :: V
     real(rkind), dimension(M_max,4+N_S) :: Vt
     real(rkind), dimension(M_max,M_max) :: HH_inv
     real(rkind), dimension(M_max) :: rhs_sav_pr,lambda
     real(rkind) :: TOL, tol_1, tol_2
     real(rkind), intent(in) :: T_iter

     TOL = 1.d-6
     HH = 0._rkind
     V = 0._rkind
     Vt = 0._rkind
     lambda = 0._rkind
     HH_inv = 0._rkind
     H = 0._rkind
     Vb = 0._rkind

     call rosenbrock_krylov_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                indx_cp_l,indx_cp_r,w_sav,T_iter,rhs_sav,enable_pasr,tau_k)

     if (simpler_splitting_ == 1) then
      rhs_sav(:) = rhs_sav(:) - fl_sav_gpu(i,j,k,:)
     endif

     V(:,1) = rhs_sav/norm2(rhs_sav)
     do kk =1, M_max
!      calcolo il prodotto Jac*V(:m):
       call rosenbrock_krylov_jacobian(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                       arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                       kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                       indx_cp_l,indx_cp_r,simpler_splitting_,rhs_sav,w_sav,kk,V,T_iter,jacobian,fl_sav_gpu,&
                                       enable_pasr,tau_k)

       do jj=1,kk
          HH(jj,kk) = DOT_PRODUCT(V(:,jj),jacobian(:))
 !         print *, "HH(jj,kk)", HH(jj,kk)
          jacobian(:) = jacobian(:) - (HH(jj,kk)*V(:,jj))
       enddo
       HH(kk+1,kk) = norm2(jacobian(:))
       V(:,kk+1) = jacobian(:)/HH(kk+1,kk)
       lim_min = min(kk,M_max)
       do ll=1,lim_min
         do jj=1,nv
          Vt(ll,jj) = V(jj,ll)
         enddo
        enddo

       !calcolo prodotto Vt*rhs_sav:
       do ll = 1,M_max   !kk
        rhs_sav_pr(ll) = 0._rkind
        do jj = 1, nv
         rhs_sav_pr(ll) = rhs_sav_pr(ll) + Vt(ll,jj) * rhs_sav(jj)
        enddo
       enddo

       do n=1, lim_min
        do nn=1,lim_min
         HH_inv(n,nn) = -(dt*gam*HH(n,nn))
         if (n == nn) HH_inv(n,nn) = HH_inv(n,nn) + 1._rkind
        enddo
       enddo
       if (kk < M_max) then
        do n=kk+1,M_max
         HH_inv(n,n) = 1._rkind
        enddo
       endif

       !inversione matrice:
       call inv_matrix_krylov(HH_inv,kk)
       !calcolo lambda:
       do ll = 1, M_max
        lambda(ll) = 0._rkind
        do jj = 1, M_max
         lambda(ll) = lambda(ll) + HH_inv(ll,jj) * rhs_sav_pr(jj)
        enddo
       enddo

       tol_1 = abs(dt*gam*HH(kk+1,kk))
       tol_2 = abs(lambda(kk))
       if ( abs(dt*gam*HH(kk+1,kk)) * abs(lambda(kk)) < TOL ) then
        mA = kk
        H(1:kk,1:kk) = HH(1:kk,1:kk)
        Vb(1:nv,1:kk) = V(1:nv,1:kk)
        inv(1:M_max,1:M_max) = HH_inv(1:M_max,1:M_max)
        exit
       endif
     enddo !M_max

    endsubroutine arnoldi
!
     attributes(device) subroutine rosenbrock_krylov_rhs(i,j,k,nv,nv_aux,nx,ny,nz,ng,w_gpu,w_aux_gpu,fl_gpu,mw_gpu,nreactions,&
                                   arr_a_gpu,arr_b_gpu,arr_ea_gpu,tb_eff_gpu,falloff_coeffs_gpu,reac_ty_gpu,isRev_gpu,r_coeffs_gpu,p_coeffs_gpu,&
                                   kc_tab_gpu,num_t_tab,t_min_tab,dt_tab,R_univ,tol_iter_nr,cv_coeff_gpu,nsetcv,trange_gpu,&
                                   indx_cp_l,indx_cp_r,w_sav,T_iter,rhs,enable_pasr,tau_k)
     implicit none
     ! passed arguments
     integer, intent(in) :: i,j,k,nv,nv_aux,nx,ny,nz,ng,nreactions,num_t_tab,enable_pasr
     integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
     real(rkind), intent(in) :: T_iter
     integer, dimension(nreactions), intent(in) :: reac_ty_gpu,isRev_gpu
     real(rkind), intent(in) :: t_min_tab,dt_tab,tol_iter_nr,R_univ,tau_k
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv    ), intent(inout) :: w_gpu
     real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout) :: w_aux_gpu
     real(rkind), dimension(1   :nx   ,1   :ny   ,1   :nz   ,1:nv    ), intent(in)    :: fl_gpu
     real(rkind), dimension(N_S)         , intent(in) :: mw_gpu
     real(rkind), dimension(N_S,nsetcv+1), intent(in) :: trange_gpu
     real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in) :: cv_coeff_gpu
     real(rkind), dimension(nreactions,2)          , intent(in) :: arr_a_gpu,arr_b_gpu, arr_ea_gpu
     real(rkind), dimension(nreactions,5)          , intent(in) :: falloff_coeffs_gpu 
     real(rkind), dimension(nreactions,N_S)        , intent(in) :: tb_eff_gpu,r_coeffs_gpu,p_coeffs_gpu
     real(rkind), dimension(num_t_tab+1,nreactions), intent(in) :: kc_tab_gpu
     real(rkind), dimension(4+N_S), intent(in) :: w_sav
     ! output arguments
     real(rkind), dimension(4+N_S), intent(out) :: rhs
     ! local arguments
     integer :: lsp,m,itt,lr,n
     real(rkind) :: rho,ri,ee,tt,ttleft,dtt,dkc,kc,kf,kb,q1,q2,rr,pp,tb,qlr,wdtkj
     real(rkind) :: rhou,rhov,rhow,rhoe,uu,vv,ww,qq
     real(rkind) :: arr_a,arr_b,arr_ea,Rtti
     real(rkind), dimension(N_S) :: w_gpu_l
     real(rkind), dimension(N_S) :: w_aux_l
     real(rkind) :: tau_lsp,tau_cf,tau_cs,tau_c,gam_pasr
     real(rkind) :: k0,kinf,rpres,fcent,coff,noff,doff,foff,xoff
     real(rkind),dimension(N_S) :: conc

     do n=1,N_S
        w_gpu_l(n) = max(w_sav(n),0._rkind)
     enddo
     rho  = w_gpu_l(1)
     do lsp=2,N_S
      rho = rho + w_gpu_l(lsp)
     enddo 
     ri   = 1._rkind/rho

     w_aux_l = 0._rkind
     do lsp=1,N_S
        w_aux_l(lsp) = w_gpu_l(lsp)*ri
        conc(lsp) = rho*w_aux_l(lsp)/mw_gpu(lsp) 
     enddo
     uu = w_sav(I_U)*ri
     vv = w_sav(I_V)*ri
     ww = w_sav(I_W)*ri
     qq = 0.5_rkind*(uu*uu + vv*vv + ww*ww)
     ee = (w_sav(I_E)*ri) - qq
     tt = get_mixture_temperature_from_e_krylov_dev(ee,T_iter, cv_coeff_gpu,nsetcv,trange_gpu,indx_cp_l, indx_cp_r, tol_iter_nr,i,j,k,nv_aux,nx,ny,nz,ng,w_aux_l)
     Rtti = 1._rkind/(R_univ*tt)

     itt = int((tt-t_min_tab)/dt_tab)+1
     itt = max(itt,1)
     itt = min(itt,num_t_tab)
     ttleft = (itt-1)*dt_tab + t_min_tab
     dtt = (tt-ttleft)/dt_tab
     rhs = 0._rkind
     do lr=1,nreactions
      ! compute kf
      select case(reac_ty_gpu(lr))
       case(0,1) !0 => Arrhenius, 1 => Three-body
        kf = arr_a_gpu(lr,1)
        if (arr_b_gpu(lr,1)  .ne. 0._rkind) kf = kf*(tt**arr_b_gpu(lr,1))
        if (arr_ea_gpu(lr,1) .ne. 0._rkind) kf = kf*exp(-arr_ea_gpu(lr,1)*Rtti)
       case(2) !falloff-Lindemann
        k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
        kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
        tb = 0._rkind
        do lsp=1,N_S
         tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
        enddo
        rpres = k0*tb/kinf
        kf = kinf*(rpres/(1._rkind + rpres))
       case(3) !falloff-Troe
        k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
        kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
        tb = 0._rkind
        do lsp=1,N_S
         tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
        enddo
        rpres = k0*tb/kinf
        if (rpres .lt. 1E-30_rkind) then
         kf = k0*tb
        else
         fcent = (1._rkind - falloff_coeffs_gpu(lr,1))*exp(-tt/falloff_coeffs_gpu(lr,2)) + &
                 falloff_coeffs_gpu(lr,1)*exp(-tt/falloff_coeffs_gpu(lr,3))
         if (falloff_coeffs_gpu(lr,4) .ne. -3.14_rkind) then
          fcent = fcent + exp(-falloff_coeffs_gpu(lr,4)/tt)
         end if
         fcent = log10(fcent)
         coff = -0.4_rkind - 0.67_rkind*fcent
         noff = 0.75_rkind - 1.27_rkind*fcent
         doff = 0.14_rkind
         foff = 10_rkind**((((1._rkind + ((log10(rpres) + coff)/(noff -doff*(log10(rpres) + &
         coff)))**2._rkind)**(-1._rkind))*fcent))
         kf = kinf*(rpres/(1._rkind + rpres))*foff
        endif
       case(4) !falloff-SRI
        k0   = arr_a_gpu(lr,1)*(tt**arr_b_gpu(lr,1))*exp(-arr_ea_gpu(lr,1)*Rtti)
        kinf = arr_a_gpu(lr,2)*(tt**arr_b_gpu(lr,2))*exp(-arr_ea_gpu(lr,2)*Rtti)
        tb = 0._rkind
        do lsp=1,N_S
         tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
        enddo
        rpres = k0*tb/kinf
        xoff = 1._rkind/(1._rkind + log10(rpres)**2._rkind)
        foff = falloff_coeffs_gpu(lr,4)*((falloff_coeffs_gpu(lr,1)*exp(-falloff_coeffs_gpu(lr,2)/tt) + &
               exp(-tt/falloff_coeffs_gpu(lr,3)))**xoff)*(tt**falloff_coeffs_gpu(lr,5))
        kf = kinf*(rpres/(1._rkind + rpres))*foff
      endselect

      if (isRev_gpu(lr) .eq. 1) then
       dkc = kc_tab_gpu(itt+1,lr)-kc_tab_gpu(itt,lr)
       kc  = kc_tab_gpu(itt,lr)+dkc*dtt
       kb = kf/kc
      else
       kb = 0._rkind
      endif

      q1  = 1._rkind
      q2  = 1._rkind
      do lsp=1,N_S
       rr = r_coeffs_gpu(lr,lsp)
       pp = p_coeffs_gpu(lr,lsp)
       if (rr == 1._rkind) then
        q1 = q1*conc(lsp)
       elseif (rr .ne. 0._rkind) then
        q1 = q1*conc(lsp)**rr
       endif
       if (pp == 1._rkind) then
        q2 = q2*conc(lsp)
       elseif (pp .ne. 0._rkind) then
        q2 = q2*conc(lsp)**pp
       endif
      enddo
      tb = 1._rkind
      if (reac_ty_gpu(lr) == 1) then
       tb = 0._rkind
       do lsp=1,N_S
         tb = tb + tb_eff_gpu(lr,lsp)*conc(lsp)
       enddo
      endif
      qlr = tb*(kf*q1-kb*q2)

      do lsp=1,N_S
       rr = r_coeffs_gpu(lr,lsp)
       pp = p_coeffs_gpu(lr,lsp)
       if (rr .ne. 0._rkind .or. pp .ne. 0._rkind) then
        wdtkj = (pp-rr)*qlr
        rhs(lsp) = rhs(lsp) + wdtkj*mw_gpu(lsp)
       endif
      enddo
     enddo

!    PaSR model
     if (enable_pasr > 0) then
      tau_cf = 0._rkind
      tau_cs = huge(1._rkind)
      do lsp=1,N_S
       if (abs(rhs(lsp)*ri) > 1.D-08) then
        tau_lsp = w_aux_l(lsp)/abs(rhs(lsp)*ri)
        tau_cf  = max(tau_cf,tau_lsp)
        if (tau_lsp > 0._rkind) tau_cs = min(tau_cs,tau_lsp)
       endif
      enddo
      tau_c = sqrt(tau_cs*tau_cf)
      if (tau_k > tiny(1._rkind)) then
       gam_pasr = tau_c/(tau_c + tau_k)
      else
       gam_pasr = 1._rkind
      endif
      do lsp=1,N_S
       rhs(lsp) = gam_pasr*rhs(lsp)
      enddo
     endif

     endsubroutine rosenbrock_krylov_rhs
    attributes(device) function get_mixture_temperature_from_e_krylov_dev(ee,T_start,cv_coeff_gpu,nsetcv,trange_gpu, &
                                indx_cp_l, indx_cp_r, tol_iter_nr,i,j,k,nv_aux,nx,ny,nz,ng,w_aux)
    real(rkind) :: get_mixture_temperature_from_e_krylov_dev
    real(rkind), value :: ee, T_start, tol_iter_nr,us_rho
    integer, value :: i,j,k,nv_aux,nx,ny,nz,ng,nsetcv,indx_cp_l,indx_cp_r
    real(rkind), dimension(N_S), device :: w_aux
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), device :: cv_coeff_gpu
    real(rkind), dimension(N_S,nsetcv+1), device :: trange_gpu
    real(rkind) :: tt,T_old,ebar,den,num,T_pow,T_powp,tden,tnum,sumb,sum0,tprod,cv_l
    integer :: nrangeloc,nmax,l,iter,max_iter,lsp,jl,jm,ju
    integer, dimension(N_S) :: nrange 
    max_iter = 10

    nmax = 100000
    do lsp=1,N_S
     nrange(lsp) = 1
    enddo
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do l=1,nmax
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (T_start >= trange_gpu(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
      nrange(lsp) = max(nrange(lsp),1)
      nrange(lsp) = min(nrange(lsp),nsetcv)
     enddo
    endif
!
    T_old = T_start
    do iter=1,max_iter
     if (nsetcv>1) then
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
       if (T_old<trange_gpu(lsp,nrangeloc)) then
        nrange(lsp) = nrange(lsp)-1
       elseif(T_old>trange_gpu(lsp,nrangeloc+1)) then
        nrange(lsp) = nrange(lsp)+1
       endif
       nrange(lsp) = max(nrange(lsp),1)
       nrange(lsp) = min(nrange(lsp),nsetcv)
      enddo
     endif
!
     sumb = 0._rkind
     do lsp=1,N_S
      nrangeloc = nrange(lsp)
      sumb = sumb+cv_coeff_gpu(indx_cp_r+1,lsp,nrangeloc)*w_aux(lsp)
     enddo
     ebar = ee-sumb
!
     den = 0._rkind
     num = 0._rkind
     do l=indx_cp_l,indx_cp_r
      T_pow  = T_old**l
      tden   = T_pow
      if (l==-1) then
       tnum   = log(T_old)
      else
       T_powp = T_old*T_pow
       tnum   = T_powp/(l+1._rkind)
      endif
      cv_l = 0._rkind
      do lsp=1,N_S
       nrangeloc = nrange(lsp)
       cv_l = cv_l+cv_coeff_gpu(l,lsp,nrangeloc)*w_aux(lsp)
      enddo
      num = num+cv_l*tnum
      den = den+cv_l*tden
     enddo
     tt = T_old+(ebar-num)/den
     if (abs(tt-T_old) < tol_iter_nr) exit
     T_old = tt
    enddo
    get_mixture_temperature_from_e_krylov_dev = tt
    endfunction get_mixture_temperature_from_e_krylov_dev
!
    attributes(device) subroutine inv_matrix_krylov(A,d)
     implicit none
     real(rkind), dimension(M_max,M_max), intent(inout) :: A
     integer, dimension(M_max) :: indx
     real(rkind), dimension(M_max,M_max) :: y
     integer :: r,s,n 
     integer, intent(in) :: d
!      
     if (d==1) then
       
       A(d,d) = 1._rkind/A(d,d)
       
     else
       do r=1,d  !M_max
          do s=1,d  !M_max
             y(r,s) = 0._rkind
          enddo   
       enddo
       do r=1,d   !M_max
          y(r,r) = 1._rkind
       enddo
       call ludcmp_krylov(A,indx,d)
       do s=1,d   !M_max
          call lubksb_krylov(A,indx,y(1,s),d)
       enddo
       do r=1,d   !M_max
          do s=1,d   !M_max
             A(r,s) = y(r,s)
          enddo
       enddo
     endif
     
     return
     endsubroutine inv_matrix_krylov
!
   attributes(device) subroutine ludcmp_krylov(a,indx,d)
!    
       implicit none
       real(rkind), parameter :: tiny = 1.5D-16
    
       real(rkind), intent(inout), dimension(M_max,M_max) :: A
       integer, intent(out), dimension(M_max) :: INDX
       integer, intent(in) :: d
       !f2py depend(N) A, indx

       real(rkind), dimension(M_max) :: VV
       real(rkind)  :: AMAX, DUM, SUMM
       integer :: i, j, k, imax

       DO I=1, d   !M_max
        AMAX=0._rkind 
        DO J=1, d  !M_max
        IF (DABS(A(I,J)).GT.AMAX) AMAX=DABS(A(I,J))
        END DO ! j loop
       IF(AMAX.LT.TINY) THEN
         RETURN 
        END IF
        VV(I) = 1._rkind / AMAX
       END DO ! i loop

       DO J=1, d
        DO I=1,J-1
         SUMM = A(I,J) 
         DO K=1,I-1
          SUMM = SUMM - A(I,K)*A(K,J)
         END DO ! k loop
         A(I,J) = SUMM
        END DO ! i loop
        AMAX = 0._rkind
        DO I=J, d
         SUMM = A(I,J)
         DO K=1,J-1
          SUMM = SUMM - A(I,K)*A(K,J)
         END DO ! k loop
         A(I,J) = SUMM
         DUM = VV(I)*DABS(SUMM)
         IF(DUM.GE.AMAX) THEN
          IMAX = I
          AMAX = DUM
         END IF
        END DO ! i loop  

        IF(J.NE.IMAX) THEN
         DO K=1, d  !M_max
           DUM = A(IMAX,K)
           A(IMAX,K) = A(J,K)
           A(J,K) = DUM
         END DO ! k loop
         VV(IMAX) = VV(J)
        END IF

        INDX(J) = IMAX
        IF(DABS(A(J,J)) < TINY) A(J,J) = TINY

        IF(J.NE.d) THEN !M_max
         DUM = 1._rkind / A(J,J)
         DO I=J+1, d  !M_max
           A(I,J) = A(I,J)*DUM
           END DO
        END IF
       END DO ! j loop

     RETURN

    endsubroutine ludcmp_krylov
!
    attributes(device) subroutine lubksb_krylov(a,indx,b,d)
       implicit none
       real(rkind), dimension(M_max,M_max), intent(in) :: A
       real(rkind), dimension(M_max), intent(out)   :: b
       integer, dimension(M_max), intent(in) :: INDX
       integer, intent(in) :: d
       !f2py depend(M_max) A, indx

       integer :: i,ii,j,ll
       real(rkind)  :: SUMM

       II = 0

       DO I=1, d  !M_max
         LL = INDX(I)
         SUMM = B(LL)
         B(LL) = B(I)
         IF(II.NE.0) THEN
           DO J=II,I-1
             SUMM = SUMM - A(I,J)*B(J)
           END DO ! j loop
         ELSE IF(SUMM.NE.0._rkind) THEN
           II = I
         END IF
         B(I) = SUMM
         END DO ! i loop

       DO I=d,1,-1  !M_max
         SUMM = B(I)
         IF(I < d) THEN  !M_max
           DO J=I+1, d    !M_max
             SUMM = SUMM - A(I,J)*B(J)
           END DO ! j loop
         END IF
         B(I) = SUMM / A(I,I)
       END DO ! i loop

       RETURN
    endsubroutine lubksb_krylov
!
!Energy Deposition
    subroutine energy_deposition_cuf(nx,ny,nz,ng,nv,nv_aux,nreactions,w_aux_gpu,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,&
               trange_gpu,x_gpu,y_gpu,z_gpu,time,w_gpu,endepo_param_gpu)
    !Passed variables
    integer, intent(in) :: nx,ny,nz,ng,nv,nv_aux,nreactions
    integer, intent(in) :: indx_cp_l,indx_cp_r,nsetcv
    real(rkind), intent(in) :: time
    real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
    real(rkind), dimension(nsetcv+1), intent(in), device :: trange_gpu
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), device :: cp_coeff_gpu
    real(rkind), dimension(1-ng:), intent(in), device :: x_gpu, y_gpu, z_gpu
    real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_gpu
    real(rkind), dimension(11), intent(in), device :: endepo_param_gpu
    !Local variables
    real(rkind) :: rho,tt,cploc,radius
    integer :: i,j,k,iercuda
    !real(rkind) :: xp,yp,zp,rint,rext,nit,A,sigmat,ign_time,sigmax,ign_temp
    real(rkind) :: prefact,expfact,torfact,endepo

    !endepo_param = xp, yp, zp, rint, rext, nit, A, sigmat, ign_time, sigmax, ign_temp
    !endepo_shape = A*time*rho*cploc*(ign_temp-tt)*exp(-sigmat*(time - ign_time)**2)*exp(-sigmax*(x_gpu(i) - xp)**2)*(tanh(nit*(radius - rint)) - tanh(nit*(radius - rext))) 

    !$cuf kernel do(3) <<<*,*>>>
    do k=1,nz
     do j=1,ny
      do i=1,nx

       tt  = w_aux_gpu(i,j,k,J_T)
       if (tt .lt. endepo_param_gpu(11)) then
        rho = w_aux_gpu(i,j,k,J_R)
        cploc = get_cp_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu,nsetcv,trange_gpu,&
               i,j,k,nx,ny,nz,ng,nv_aux,w_aux_gpu)
        radius = ((y_gpu(j) - endepo_param_gpu(2))**2_rkind + (z_gpu(k) - endepo_param_gpu(3))**2_rkind)**0.5_rkind
       
        prefact = endepo_param_gpu(7)*time*rho*cploc*(endepo_param_gpu(11)-tt)
        expfact = exp(-endepo_param_gpu(8)*(time-endepo_param_gpu(9)**2))
        expfact = expfact*exp(-endepo_param_gpu(10)*(x_gpu(i)-endepo_param_gpu(1))**2)
        torfact = tanh(endepo_param_gpu(6)*(radius-endepo_param_gpu(4)))-tanh(endepo_param_gpu(6)*(radius-endepo_param_gpu(5)))
        w_gpu(i,j,k,I_E) = w_gpu(i,j,k,I_E) + prefact*expfact*torfact
       endif

       !xp = endepo_param_gpu(1)
       !yp = endepo_param_gpu(2)
       !zp = endepo_param_gpu(3)
       !rint = endepo_param_gpu(4)
       !rext = endepo_param_gpu(5)
       !nit = endepo_param_gpu(6)
       !A = endepo_param_gpu(7)
       !sigmat = endepo_param_gpu(8)
       !ign_time = endepo_param_gpu(9)
       !sigmax = endepo_param_gpu(10)
       !ign_temp = endepo_param_gpu(11)
       !radius = ((y_gpu(j) - yp)**2_rkind + (z_gpu(k) - zp)**2_rkind)**0.5_rkind
!       if (radius .gt. rint .and. radius .lt. rext) then
        !en_depo = A*time*rho*cploc*(ign_temp-tt)*exp(-sigmat*(time - ign_time)**2)*exp(-sigmax*(x_gpu(i) - xp)**2)*&
        !       (tanh(nit*(radius - rint)) - tanh(nit*(radius - rext)))
        !w_gpu(i,j,k,I_E) = w_gpu(i,j,k,I_E) + en_depo
!       endif

      enddo
     enddo
    enddo
    endsubroutine energy_deposition_cuf

    subroutine ibm_vega_old_cuf(nx,ny,nz,ng,nv,nv_aux,indx_cp_l,indx_cp_r,w_gpu,ibm_inside_moving_gpu, &
                            cv_coeff_gpu,nsetcv,trange_gpu,ibm_aero_rad,ibm_aero_pp,ibm_aero_tt, ibm_vega_vel, &
                            ibm_aero_modvel,x_gpu,y_gpu,z_gpu,ibm_bc_relax_factor,rgas_gpu, &
                            w_aux_gpu,ibm_dw_aux_vega_gpu,ibm_vega_dist_gpu,ibm_vega_species)
        integer :: nx,ny,nz,ng,indx_cp_l,indx_cp_r,nv,nv_aux,ibm_vega_species, nsetcv
        real(rkind), intent(in) :: ibm_aero_rad,ibm_aero_pp,ibm_aero_tt,ibm_aero_modvel,ibm_bc_relax_factor
        real(rkind), intent(in) :: ibm_vega_vel
        integer, dimension(1:,1:,1:), intent(in), device :: ibm_inside_moving_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
        real(rkind), dimension(1-ng:), intent(in), device :: x_gpu,y_gpu,z_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(inout), device :: w_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: ibm_dw_aux_vega_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:), intent(inout), device :: ibm_vega_dist_gpu
        integer :: i,j,k,iercuda
        real(rkind) :: rho,rhou,rhov,rhow,rhoe,uu,vv,ww,qq,pp,tt,ee,rmixtloc
        real(rkind) :: rad
        integer :: indx_eikonal,lsp,isp
        real(rkind) :: normx,normy,normz,nmod,dxi,dyi,dzi,dtime
        real(rkind) :: dt,drho,dysp
!
!       Steady-state solution of convection along normal (towards inside of the body)
!       (Di Mascio and Zaghi, 2021)
!
        do indx_eikonal=1,3
        !$cuf kernel do(3) <<<*,*>>>
         do k=1,nz
          do j=1,ny
           do i=1,nx
            if (ibm_inside_moving_gpu(i,j,k)==1) then
             normx = ibm_vega_dist_gpu(i+1,j,k)-ibm_vega_dist_gpu(i-1,j,k)
             normy = ibm_vega_dist_gpu(i,j+1,k)-ibm_vega_dist_gpu(i,j-1,k)
             normz = ibm_vega_dist_gpu(i,j,k+1)-ibm_vega_dist_gpu(i,j,k-1)
!            dxi = 1._rkind/(x_gpu(i+1)-x_gpu(i-1))
!            dyi = 1._rkind/(y_gpu(j+1)-y_gpu(j-1))
!            dzi = 1._rkind/(z_gpu(k+1)-z_gpu(k-1))
!            normx = normx*dxi
!            normy = normy*dyi
!            normz = normz*dzi
!            nmod  = normx*normx+normy*normy+normz*normz
!            nmod  = sqrt(nmod)
!            normx = normx/nmod
!            normy = normy/nmod
!            normz = normz/nmod
!            dtime = 0.9_rkind/max(abs(normx*dxi),abs(normy*dyi),abs(normz*dzi))
!            normx = normx*dtime
!            normy = normy*dtime
!            normz = normz*dtime
             nmod = abs(normx) + abs(normy) + abs(normz)
             nmod = 0.9_rkind/nmod ! cfl = 0.9
             normx = normx*nmod
             normy = normy*nmod
             normz = normz*nmod
             dt    = 0._rkind
             drho  = 0._rkind
             if (normx > 0._rkind) then
                dxi  = 1._rkind!/(x_gpu(i)-x_gpu(i-1))
                dt   = dt   + normx*dxi*(w_aux_gpu(i,j,k,J_T)-w_aux_gpu(i-1,j,k,J_T))
                drho = drho + normx*dxi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i-1,j,k,J_R))
             else
                dxi  = 1._rkind!/(x_gpu(i+1)-x_gpu(i))
                dt   = dt   + normx*dxi*(w_aux_gpu(i+1,j,k,J_T)-w_aux_gpu(i,j,k,J_T))
                drho = drho + normx*dxi*(w_aux_gpu(i+1,j,k,J_R)-w_aux_gpu(i,j,k,J_R))
             endif
             if (normy > 0._rkind) then
                dyi = 1._rkind!/(y_gpu(j)-y_gpu(j-1))
                dt   = dt   + normy*dyi*(w_aux_gpu(i,j,k,J_T)-w_aux_gpu(i,j-1,k,J_T))
                drho = drho + normy*dyi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i,j-1,k,J_R))
             else
                dyi  = 1._rkind!/(y_gpu(j+1)-y_gpu(j))
                dt   = dt   + normy*dyi*(w_aux_gpu(i,j+1,k,J_T)-w_aux_gpu(i,j,k,J_T))
                drho = drho + normy*dyi*(w_aux_gpu(i,j+1,k,J_R)-w_aux_gpu(i,j,k,J_R))
             endif
             if (normz > 0._rkind) then
                dzi  = 1._rkind!/(z_gpu(k)-z_gpu(k-1))
                dt   = dt   + normz*dzi*(w_aux_gpu(i,j,k,J_T)-w_aux_gpu(i,j,k-1,J_T))
                drho = drho + normz*dzi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i,j,k-1,J_R))
             else
                dzi = 1._rkind!/(z_gpu(k+1)-z_gpu(k))
                dt   = dt   + normz*dzi*(w_aux_gpu(i,j,k+1,J_T)-w_aux_gpu(i,j,k,J_T))
                drho = drho + normz*dzi*(w_aux_gpu(i,j,k+1,J_R)-w_aux_gpu(i,j,k,J_R))
             endif
             ibm_dw_aux_vega_gpu(i,j,k,N_S+1) = drho
             ibm_dw_aux_vega_gpu(i,j,k,N_S+2) = dt
!
             do lsp=1,N_S
              dysp = 0._rkind
              if (normx > 0._rkind) then
                 dxi  = 1._rkind!/(x_gpu(i)-x_gpu(i-1))
                 dysp = dysp + normx*dxi*(w_aux_gpu(i,j,k,lsp)-w_aux_gpu(i-1,j,k,lsp))
              else
                 dxi  = 1._rkind!/(x_gpu(i+1)-x_gpu(i))
                 dysp = dysp + normx*dxi*(w_aux_gpu(i+1,j,k,lsp)-w_aux_gpu(i,j,k,lsp))
              endif
              if (normy > 0._rkind) then
                 dyi = 1._rkind!/(y_gpu(j)-y_gpu(j-1))
                 dysp = dysp + normy*dyi*(w_aux_gpu(i,j,k,lsp)-w_aux_gpu(i,j-1,k,lsp))
              else
                 dyi  = 1._rkind!/(y_gpu(j+1)-y_gpu(j))
                 dysp = dysp + normy*dyi*(w_aux_gpu(i,j+1,k,lsp)-w_aux_gpu(i,j,k,lsp))
              endif
              if (normz > 0._rkind) then
                 dzi  = 1._rkind!/(z_gpu(k)-z_gpu(k-1))
                 dysp = dysp + normz*dzi*(w_aux_gpu(i,j,k,lsp)-w_aux_gpu(i,j,k-1,lsp))
              else
                 dzi = 1._rkind!/(z_gpu(k+1)-z_gpu(k))
                 dysp = dysp + normz*dzi*(w_aux_gpu(i,j,k+1,lsp)-w_aux_gpu(i,j,k,lsp))
              endif
              ibm_dw_aux_vega_gpu(i,j,k,lsp) = dysp
             enddo
            endif
           enddo
          enddo
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()
         !$cuf kernel do(3) <<<*,*>>>
          do k=1,nz
           do j=1,ny
            do i=1,nx
             if (ibm_inside_moving_gpu(i,j,k)==1) then
              w_aux_gpu(i,j,k,J_R) = w_aux_gpu(i,j,k,J_R)-ibm_dw_aux_vega_gpu(i,j,k,N_S+1)
              w_aux_gpu(i,j,k,J_T) = w_aux_gpu(i,j,k,J_T)-ibm_dw_aux_vega_gpu(i,j,k,N_S+2)
              do lsp=1,N_S
               w_aux_gpu(i,j,k,lsp) = w_aux_gpu(i,j,k,lsp)-ibm_dw_aux_vega_gpu(i,j,k,lsp)
              enddo
             endif
            enddo
           enddo
          enddo
         !@cuf iercuda=cudaDeviceSynchronize()
        enddo
!
        !$cuf kernel do(3) <<<*,*>>>
         do k=1,nz
          do j=1,ny
           do i=1,nx

            if (ibm_inside_moving_gpu(i,j,k)==9) then
!
             rad = sqrt(x_gpu(i)**2+z_gpu(k)**2)
             if (rad<ibm_aero_rad) then
              isp = ibm_vega_species
              pp = ibm_aero_pp
              tt = ibm_aero_tt
!             uu = ibm_aero_modvel*ibm_nxyz_interface_gpu(1,l)
!             vv = ibm_aero_modvel*ibm_nxyz_interface_gpu(2,l)
!             ww = ibm_aero_modvel*ibm_nxyz_interface_gpu(3,l)
              uu =  0._rkind
              vv = -ibm_aero_modvel+ibm_vega_vel
              ww =  0._rkind
              rmixtloc = rgas_gpu(isp)
              rho  = pp/tt/rmixtloc
              rhou = rho*uu
              rhov = rho*vv
              rhow = rho*ww
              ee = get_species_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,isp)
              do lsp=1,N_S
               w_gpu(i,j,k,lsp) = 0._rkind
               if (lsp==isp) w_gpu(i,j,k,lsp) = rho*1._rkind
              enddo
              w_gpu(i,j,k,I_U) = rhou
              w_gpu(i,j,k,I_V) = rhov
              w_gpu(i,j,k,I_W) = rhow
              w_gpu(i,j,k,I_E) = rho*ee+0.5_rkind*(rhou*rhou+rhov*rhov+rhow*rhow)/rho
             endif
!
            elseif (ibm_inside_moving_gpu(i,j,k)==1) then
!
             rho  = w_aux_gpu(i,j,k,J_R)
             tt   = w_aux_gpu(i,j,k,J_T)
             uu   = 0._rkind
             vv   = ibm_vega_vel
             ww   = 0._rkind
             rhou = rho*uu
             rhov = rho*vv
             rhow = rho*ww
             ee   = get_mixture_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu, &
                                                  i,j,k,nv_aux,nx,ny,nz,ng,w_aux_gpu)
             do lsp=1,N_S
              w_gpu(i,j,k,lsp) = rho*w_aux_gpu(i,j,k,lsp)
             enddo
             w_gpu(i,j,k,I_U) = rhou
             w_gpu(i,j,k,I_V) = rhov
             w_gpu(i,j,k,I_W) = rhow
             w_gpu(i,j,k,I_E) = rho*ee+0.5_rkind*(rhou*rhou+rhov*rhov+rhow*rhow)/rho
!
            endif
!
           enddo
          enddo
         enddo
        !@cuf iercuda=cudaDeviceSynchronize()
!
    endsubroutine ibm_vega_old_cuf

    subroutine ibm_eikonal_old_cuf(nx,ny,nz,ng,indx_cp_l,indx_cp_r,w_gpu,cv_coeff_gpu,nsetcv,trange_gpu,&
                            x_gpu,y_gpu,z_gpu,ibm_bc_relax_factor, w_aux_gpu,ibm_dw_aux_eikonal_gpu,&
                            ibm_body_dist_gpu,ibm_is_interface_node_gpu,rgas_gpu,ibm_parbc_gpu)
        integer :: nx,ny,nz,ng,indx_cp_l,indx_cp_r,nsetcv
        real(rkind), intent(in) :: ibm_bc_relax_factor
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(1-ng:), intent(in), device :: x_gpu,y_gpu,z_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: ibm_dw_aux_eikonal_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:), intent(inout), device :: ibm_body_dist_gpu
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: ibm_is_interface_node_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_parbc_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
        real(rkind) :: rho,rhou,rhov,rhow,rhoe,uu,vv,ww,qq,pp,tt,ee
        real(rkind) :: rad
        real(rkind) :: normx,normy,normz,nmod,dxi,dyi,dzi,dtime
        real(rkind) :: dt,drho,du,dv,dw,dysp
        integer :: i,j,k,iercuda
        integer :: indx_eikonal,lsp,isp
!
!       Steady-state solution of convection along normal (towards inside of the body)
!       (Di Mascio and Zaghi, 2021)
!
        do indx_eikonal=1,3
        !$cuf kernel do(3) <<<*,*>>>
         do k=1,nz
          do j=1,ny
           do i=1,nx
!           if (ibm_body_dist_gpu(i,j,k)>0._rkind) then
            if (ibm_is_interface_node_gpu(i,j,k)==1) then
             normx = ibm_body_dist_gpu(i+1,j,k)-ibm_body_dist_gpu(i-1,j,k)
             normy = ibm_body_dist_gpu(i,j+1,k)-ibm_body_dist_gpu(i,j-1,k)
             normz = ibm_body_dist_gpu(i,j,k+1)-ibm_body_dist_gpu(i,j,k-1)
             nmod = abs(normx) + abs(normy) + abs(normz)
             nmod = 0.9_rkind/nmod ! cfl = 0.9
             normx = normx*nmod
             normy = normy*nmod
             normz = normz*nmod
             drho  = 0._rkind
             du    = 0._rkind
             dv    = 0._rkind
             dw    = 0._rkind
             dt    = 0._rkind
             if (normx > 0._rkind) then
                dxi  = 1._rkind!/(x_gpu(i)-x_gpu(i-1))
                drho = drho + normx*dxi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i-1,j,k,J_R))
                du   = du   + normx*dxi*(w_aux_gpu(i,j,k,J_U)-w_aux_gpu(i-1,j,k,J_U))
                dv   = dv   + normx*dxi*(w_aux_gpu(i,j,k,J_V)-w_aux_gpu(i-1,j,k,J_V))
                dw   = dw   + normx*dxi*(w_aux_gpu(i,j,k,J_W)-w_aux_gpu(i-1,j,k,J_W))
                dt   = dt   + normx*dxi*(w_aux_gpu(i,j,k,J_T)-w_aux_gpu(i-1,j,k,J_T))
             else
                dxi  = 1._rkind!/(x_gpu(i+1)-x_gpu(i))
                drho = drho + normx*dxi*(w_aux_gpu(i+1,j,k,J_R)-w_aux_gpu(i,j,k,J_R))
                du   = du   + normx*dxi*(w_aux_gpu(i+1,j,k,J_U)-w_aux_gpu(i,j,k,J_U))
                dv   = dv   + normx*dxi*(w_aux_gpu(i+1,j,k,J_V)-w_aux_gpu(i,j,k,J_V))
                dw   = dw   + normx*dxi*(w_aux_gpu(i+1,j,k,J_W)-w_aux_gpu(i,j,k,J_W))
                dt   = dt   + normx*dxi*(w_aux_gpu(i+1,j,k,J_T)-w_aux_gpu(i,j,k,J_T))
             endif
             if (normy > 0._rkind) then
                dyi = 1._rkind!/(y_gpu(j)-y_gpu(j-1))
                drho = drho + normy*dyi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i,j-1,k,J_R))
                du   = du   + normy*dyi*(w_aux_gpu(i,j,k,J_U)-w_aux_gpu(i,j-1,k,J_U))
                dv   = dv   + normy*dyi*(w_aux_gpu(i,j,k,J_V)-w_aux_gpu(i,j-1,k,J_V))
                dw   = dw   + normy*dyi*(w_aux_gpu(i,j,k,J_W)-w_aux_gpu(i,j-1,k,J_W))
                dt   = dt   + normy*dyi*(w_aux_gpu(i,j,k,J_T)-w_aux_gpu(i,j-1,k,J_T))
             else
                dyi  = 1._rkind!/(y_gpu(j+1)-y_gpu(j))
                drho = drho + normy*dyi*(w_aux_gpu(i,j+1,k,J_R)-w_aux_gpu(i,j,k,J_R))
                du   = du   + normy*dyi*(w_aux_gpu(i,j+1,k,J_U)-w_aux_gpu(i,j,k,J_U))
                dv   = dv   + normy*dyi*(w_aux_gpu(i,j+1,k,J_V)-w_aux_gpu(i,j,k,J_V))
                dw   = dw   + normy*dyi*(w_aux_gpu(i,j+1,k,J_W)-w_aux_gpu(i,j,k,J_W))
                dt   = dt   + normy*dyi*(w_aux_gpu(i,j+1,k,J_T)-w_aux_gpu(i,j,k,J_T))
             endif
             if (normz > 0._rkind) then
                dzi  = 1._rkind!/(z_gpu(k)-z_gpu(k-1))
                drho = drho + normz*dzi*(w_aux_gpu(i,j,k,J_R)-w_aux_gpu(i,j,k-1,J_R))
                du   = du   + normz*dzi*(w_aux_gpu(i,j,k,J_U)-w_aux_gpu(i,j,k-1,J_U))
                dv   = dv   + normz*dzi*(w_aux_gpu(i,j,k,J_V)-w_aux_gpu(i,j,k-1,J_V))
                dw   = dw   + normz*dzi*(w_aux_gpu(i,j,k,J_W)-w_aux_gpu(i,j,k-1,J_W))
                dt   = dt   + normz*dzi*(w_aux_gpu(i,j,k,J_T)-w_aux_gpu(i,j,k-1,J_T))
             else
                dzi = 1._rkind!/(z_gpu(k+1)-z_gpu(k))
                drho = drho + normz*dzi*(w_aux_gpu(i,j,k+1,J_R)-w_aux_gpu(i,j,k,J_R))
                du   = du   + normz*dzi*(w_aux_gpu(i,j,k+1,J_U)-w_aux_gpu(i,j,k,J_U))
                dv   = dv   + normz*dzi*(w_aux_gpu(i,j,k+1,J_V)-w_aux_gpu(i,j,k,J_V))
                dw   = dw   + normz*dzi*(w_aux_gpu(i,j,k+1,J_W)-w_aux_gpu(i,j,k,J_W))
                dt   = dt   + normz*dzi*(w_aux_gpu(i,j,k+1,J_T)-w_aux_gpu(i,j,k,J_T))
             endif
             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+1) = drho
             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+2) = du
             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+3) = dv
             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+4) = dw
             ibm_dw_aux_eikonal_gpu(i,j,k,N_S+5) = dt
!
             do lsp=1,N_S
              dysp = 0._rkind
              if (normx > 0._rkind) then
                 dxi  = 1._rkind!/(x_gpu(i)-x_gpu(i-1))
                 dysp = dysp + normx*dxi*(w_aux_gpu(i,j,k,lsp)-w_aux_gpu(i-1,j,k,lsp))
              else
                 dxi  = 1._rkind!/(x_gpu(i+1)-x_gpu(i))
                 dysp = dysp + normx*dxi*(w_aux_gpu(i+1,j,k,lsp)-w_aux_gpu(i,j,k,lsp))
              endif
              if (normy > 0._rkind) then
                 dyi = 1._rkind!/(y_gpu(j)-y_gpu(j-1))
                 dysp = dysp + normy*dyi*(w_aux_gpu(i,j,k,lsp)-w_aux_gpu(i,j-1,k,lsp))
              else
                 dyi  = 1._rkind!/(y_gpu(j+1)-y_gpu(j))
                 dysp = dysp + normy*dyi*(w_aux_gpu(i,j+1,k,lsp)-w_aux_gpu(i,j,k,lsp))
              endif
              if (normz > 0._rkind) then
                 dzi  = 1._rkind!/(z_gpu(k)-z_gpu(k-1))
                 dysp = dysp + normz*dzi*(w_aux_gpu(i,j,k,lsp)-w_aux_gpu(i,j,k-1,lsp))
              else
                 dzi = 1._rkind!/(z_gpu(k+1)-z_gpu(k))
                 dysp = dysp + normz*dzi*(w_aux_gpu(i,j,k+1,lsp)-w_aux_gpu(i,j,k,lsp))
              endif
              ibm_dw_aux_eikonal_gpu(i,j,k,lsp) = dysp
             enddo
!
            endif
           enddo
          enddo
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()
        enddo
!
!        !$cuf kernel do(3) <<<*,*>>>
!         do k=1,nz
!          do j=1,ny
!           do i=1,nx
!!
!            if (ibm_is_interface_node_gpu(i,j,k)==1) then
!             rho  =  w_aux_gpu(i,j,k,J_R)
!             uu   = -w_aux_gpu(i,j,k,J_U)
!             vv   = -w_aux_gpu(i,j,k,J_V)
!             ww   = -w_aux_gpu(i,j,k,J_W)
!             tt   =  w_aux_gpu(i,j,k,J_T)
!             rhou = rho*uu
!             rhov = rho*vv
!             rhow = rho*ww
!             isp  = 1
!             ee   = get_species_e_from_temperature_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,&
!                    isp)
!             do lsp=1,N_S
!              w_gpu(i,j,k,lsp) = rho*w_aux_gpu(i,j,k,lsp)
!             enddo
!             w_gpu(i,j,k,I_U) = rhou
!             w_gpu(i,j,k,I_V) = rhov
!             w_gpu(i,j,k,I_W) = rhow
!             w_gpu(i,j,k,I_E) = rho*ee+0.5_rkind*(rhou*rhou+rhov*rhov+rhow*rhow)/rho
!            endif
!!
!           enddo
!          enddo
!         enddo
!        !@cuf iercuda=cudaDeviceSynchronize()
!
    end subroutine ibm_eikonal_old_cuf
    
    subroutine ibm_interpolation_old_cuf(ibm_num_interface,nx,ny,nz,ng,nv,nv_aux,indx_cp_l,indx_cp_r,ibm_ijk_refl_gpu, &
               ibm_refl_type_gpu, w_gpu,w_aux_gpu,ibm_is_interface_node_gpu,ibm_coeff_d_gpu,ibm_coeff_n_gpu, &
               ibm_bc_gpu,cv_coeff_gpu,nsetcv,trange_gpu,ibm_w_refl_gpu,ibm_nxyz_interface_gpu, &
               rgas_gpu,ibm_parbc_gpu)
!
        integer, intent(in) :: ibm_num_interface,nx,ny,nz,ng,indx_cp_l,indx_cp_r,nv,nv_aux,nsetcv
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv), intent(in), device :: w_gpu
        real(rkind), dimension(1-ng:nx+ng,1-ng:ny+ng,1-ng:nz+ng,1:nv_aux), intent(in), device :: w_aux_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: ibm_coeff_d_gpu
        real(rkind), dimension(1:,1:,1:,1:), intent(in), device :: ibm_coeff_n_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_nxyz_interface_gpu
        integer, dimension(1-ng:,1-ng:,1-ng:), intent(in), device :: ibm_is_interface_node_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_refl_gpu
        integer, dimension(1:), intent(in), device :: ibm_refl_type_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_parbc_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
        real(rkind), dimension(1:,1:), intent(inout), device :: ibm_w_refl_gpu
!
        integer :: indx_patch
        integer :: l,i,j,k,ii,jj,kk,lsp,iercuda
        real(rkind) :: wi1,wi2,wi3,wi4,wi5,wi6,wsp
        real(rkind) :: rho,uu,vv,ww,pp,tt
        real(rkind) :: un,ut1,ut2,ut3
        real(rkind) :: rmixtloc
        real(rkind) :: twall_ibm
!
        if (ibm_num_interface>0) then
         !$cuf kernel do(1) <<<*,*>>>
         do l=1,ibm_num_interface
!
          i = ibm_ijk_refl_gpu(1,l)
          j = ibm_ijk_refl_gpu(2,l)
          k = ibm_ijk_refl_gpu(3,l)
!
          wi1 = 0._rkind
          wi2 = 0._rkind
          wi3 = 0._rkind
          wi4 = 0._rkind
          wi5 = 0._rkind
          wi6 = 0._rkind
!
          type_refl: if (ibm_refl_type_gpu(l)==0) then
!
           do kk=1,2
            do jj=1,2
             do ii=1,2
              wi1 = wi1 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
              wi2 = wi2 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
              wi3 = wi3 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
              wi4 = wi4 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
              wi5 = wi5 + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
             enddo
            enddo
           enddo
           do lsp=1,N_S
            wsp = 0._rkind
            do kk=1,2
             do jj=1,2
              do ii=1,2
               wsp = wsp + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,lsp)
              enddo
             enddo
            enddo
            ibm_w_refl_gpu(l,lsp) = wsp
           enddo
!
          else
!
           rmixtloc = 0._rkind
           do lsp=1,N_S ! Apply Neumann bc for species
            wsp = 0._rkind
            do kk=1,2
             do jj=1,2
              do ii=1,2
!              wsp = wsp + ibm_coeff_d_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,lsp)
               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
               else
                wsp = wsp + ibm_coeff_n_gpu(ii,jj,kk,l)*w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,lsp)
               endif
              enddo
             enddo
            enddo
            ibm_w_refl_gpu(l,lsp) = wsp
            rmixtloc = rmixtloc+rgas_gpu(lsp)*wsp
           enddo
!
           type_bc: select case(ibm_bc_gpu(1,l))
!
           case(5) ! Inviscid wall (un=> D, p,T,ut => N)
!
            do kk=1,2
             do jj=1,2
              do ii=1,2
!
               rho = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
               uu  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
               vv  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
               ww  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
               tt  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
               pp  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_P)
               un  = uu*ibm_nxyz_interface_gpu(1,l)+vv*ibm_nxyz_interface_gpu(2,l)+ww*ibm_nxyz_interface_gpu(3,l)
               ut1 = uu-un*ibm_nxyz_interface_gpu(1,l)
               ut2 = vv-un*ibm_nxyz_interface_gpu(2,l)
               ut3 = ww-un*ibm_nxyz_interface_gpu(3,l)
!
               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
                wi1 = wi1!+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
               else
                wi1 = wi1 + ibm_coeff_d_gpu(ii,jj,kk,l)*un
                wi2 = wi2 + ibm_coeff_n_gpu(ii,jj,kk,l)*ut1
                wi3 = wi3 + ibm_coeff_n_gpu(ii,jj,kk,l)*ut2
                wi4 = wi4 + ibm_coeff_n_gpu(ii,jj,kk,l)*ut3
                wi5 = wi5 + ibm_coeff_n_gpu(ii,jj,kk,l)*tt
                wi6 = wi6 + ibm_coeff_n_gpu(ii,jj,kk,l)*pp
               endif
!
              enddo
             enddo
            enddo
!
            wi2 = wi2 + wi1*ibm_nxyz_interface_gpu(1,l)
            wi3 = wi3 + wi1*ibm_nxyz_interface_gpu(2,l)
            wi4 = wi4 + wi1*ibm_nxyz_interface_gpu(3,l)
            wi1 = wi6/wi5/rmixtloc ! rho = p/tt/R
!
           case(6) ! Viscous isothermal wall (u,v,w,T => D, p => N)

           indx_patch = ibm_bc_gpu(2,l)
           twall_ibm  = ibm_parbc_gpu(indx_patch,1)

            do kk=1,2
             do jj=1,2
              do ii=1,2
!
               rho = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
               uu  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
               vv  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
               ww  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
               tt  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
               pp  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_P)
!
               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
                wi2 = wi2 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
                wi3 = wi3 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
                wi4 = wi4 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
                wi5 = wi5  + ibm_coeff_d_gpu(ii,jj,kk,l)*twall_ibm
               else
                wi2 = wi2 + ibm_coeff_d_gpu(ii,jj,kk,l)*uu
                wi3 = wi3 + ibm_coeff_d_gpu(ii,jj,kk,l)*vv
                wi4 = wi4 + ibm_coeff_d_gpu(ii,jj,kk,l)*ww
                wi5 = wi5 + ibm_coeff_d_gpu(ii,jj,kk,l)*tt
                wi6 = wi6 + ibm_coeff_n_gpu(ii,jj,kk,l)*pp
               endif
!
              enddo
             enddo
            enddo
!
            wi1 = wi6/wi5/rmixtloc ! rho = p/tt/R
!
           case(3,4,8) ! Subsonic inflow & Viscous adiabatic wall (u,v,w => D, p,T => N)

            do kk=1,2
             do jj=1,2
              do ii=1,2
!
!              rho = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_R)
               uu  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_U)
               vv  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_V)
               ww  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_W)
               tt  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_T)
               pp  = w_aux_gpu(i+ii-1,j+jj-1,k+kk-1,J_P)
!
               if (ibm_is_interface_node_gpu(i+ii-1,j+jj-1,k+kk-1)==1) then
                wi2 = wi2 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
                wi3 = wi3 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
                wi4 = wi4 !+ ibm_coeff_d_gpu(ii,jj,kk,l)*0._rkind
               else
                wi2 = wi2 + ibm_coeff_d_gpu(ii,jj,kk,l)*uu
                wi3 = wi3 + ibm_coeff_d_gpu(ii,jj,kk,l)*vv
                wi4 = wi4 + ibm_coeff_d_gpu(ii,jj,kk,l)*ww
                wi5 = wi5 + ibm_coeff_n_gpu(ii,jj,kk,l)*tt
                wi6 = wi6 + ibm_coeff_n_gpu(ii,jj,kk,l)*pp
               endif
!
              enddo
             enddo
            enddo
!
            wi1 = wi6/wi5/rmixtloc ! rho = p/tt/R
!
           case(1,2,9) ! Supersonic inflow (no interpolation needed because ibm_w_refl_gpu not used in forcing)
!
           endselect type_bc
!
          endif type_refl
!
          do lsp=1,N_S
           ibm_w_refl_gpu(l,lsp) = ibm_w_refl_gpu(l,lsp)*wi1
          enddo
          ibm_w_refl_gpu(l,I_U) = wi2
          ibm_w_refl_gpu(l,I_V) = wi3
          ibm_w_refl_gpu(l,I_W) = wi4
          ibm_w_refl_gpu(l,I_E) = wi5
!
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()
        endif
!
    endsubroutine ibm_interpolation_old_cuf

    subroutine ibm_forcing_old_cuf(ibm_num_interface,nx,ny,nz,ng,indx_cp_l,indx_cp_r,ibm_ijk_interface_gpu,w_gpu,w_aux_gpu,ibm_bc_gpu, &
             cp_coeff_gpu,cv_coeff_gpu,nsetcv,trange_gpu,ibm_w_refl_gpu,ibm_nxyz_interface_gpu,ibm_aero_rad,ibm_aero_pp,&
             ibm_aero_tt,ibm_aero_modvel,x_gpu,y_gpu,z_gpu,ibm_bc_relax_factor,rgas_gpu,ibm_parbc_gpu,ibm_eikonal,time,&
             randvar_a_gpu,randvar_p_gpu)
!
        integer, intent(in) :: ibm_num_interface,nx,ny,nz,ng,indx_cp_l,indx_cp_r,nsetcv,ibm_eikonal
        real(rkind), intent(in) :: ibm_aero_rad,ibm_aero_pp,ibm_aero_tt,ibm_aero_modvel,ibm_bc_relax_factor,time
        integer, dimension(1:,1:), intent(in), device :: ibm_ijk_interface_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_nxyz_interface_gpu
        integer, dimension(1:,1:), intent(in), device :: ibm_bc_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cp_coeff_gpu
        real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv), intent(in), device :: cv_coeff_gpu
        real(rkind), dimension(N_S,nsetcv+1), intent(in), device :: trange_gpu
        real(rkind), dimension(1:,1:), intent(inout), device :: ibm_w_refl_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_gpu
        real(rkind), dimension(1-ng:,1-ng:,1-ng:,1:), intent(inout), device :: w_aux_gpu
        real(rkind), dimension(1-ng:), intent(in), device :: x_gpu,y_gpu,z_gpu
        real(rkind), dimension(1:,1:), intent(in), device :: ibm_parbc_gpu
        real(rkind), dimension(N_S), intent(in), device :: rgas_gpu
        real(rkind), dimension(8), intent(in),device :: randvar_a_gpu,randvar_p_gpu
!
        integer :: l,m,i,j,k,lsp,iercuda,isp,modes
        integer :: indx_patch
        integer :: jl,ju,jm
        integer, dimension(N_S), device :: nrange
        real(rkind) :: rho,uu,vv,ww,pp,tt
        real(rkind) :: rhoi,uui,vvi,wwi,ppi,tti,eei,rhoui,rhovi,rhowi
        real(rkind) :: rad,umin
        real(rkind) :: unorm2
        real(rkind) :: twall_ibm,pp_patch,tt_patch,vel_patch
        real(rkind) :: rmixtloc,cv_m,tprod
        real(rkind) :: rhou,rhov,rhow
        real(rkind) :: massfrac, yc_patch, zc_patch
        real(rkind) :: ptotal,ttotal,cvloc,gamloc,vel_mod,del,rmfac,rml
        real(rkind), dimension(N_S),device :: yy
        real(rkind) :: inlet_phi,upert,utmp,avar_gpu
!
        if (ibm_num_interface>0) then
         !$cuf kernel do(1) <<<*,*>>>
         do l=1,ibm_num_interface
!
          i = ibm_ijk_interface_gpu(1,l)
          j = ibm_ijk_interface_gpu(2,l)
          k = ibm_ijk_interface_gpu(3,l)
!
!         Storing momentum for force computation
!         w2   = w_gpu(i,j,k,I_U)
!         w3   = w_gpu(i,j,k,I_V)
!         w4   = w_gpu(i,j,k,I_W)
!
!         Quantities at reflected node
!
          if (ibm_eikonal>0) then
           rho = w_aux_gpu(i,j,k,J_R)
           uu  = w_aux_gpu(i,j,k,J_U)
           vv  = w_aux_gpu(i,j,k,J_V)
           ww  = w_aux_gpu(i,j,k,J_W)
           tt  = w_aux_gpu(i,j,k,J_T)
           rmixtloc = 0._rkind
           do lsp=1,N_S
            rmixtloc = rmixtloc+rgas_gpu(lsp)*w_aux_gpu(i,j,k,lsp)
           enddo
           pp = rho*tt*rmixtloc
          else
           rho = ibm_w_refl_gpu(l,1)
           do lsp=2,N_S
            rho  = rho+ibm_w_refl_gpu(l,lsp)
           enddo
           uu   = ibm_w_refl_gpu(l,I_U)
           vv   = ibm_w_refl_gpu(l,I_V)
           ww   = ibm_w_refl_gpu(l,I_W)
           tt   = ibm_w_refl_gpu(l,I_E)
           rmixtloc = 0._rkind
           do lsp=1,N_S
            yy(lsp) = ibm_w_refl_gpu(l,lsp)/rho
            ibm_w_refl_gpu(l,lsp) = yy(lsp)
            rmixtloc = rmixtloc+rgas_gpu(lsp)*yy(lsp)
           enddo
           pp = rho*tt*rmixtloc
          endif
          cvloc = get_cp_yy_old_dev(tt,indx_cp_l,indx_cp_r,cv_coeff_gpu,nsetcv,trange_gpu,yy)
          gamloc = (cvloc+rmixtloc)/cvloc
!
          type_bc: select case(ibm_bc_gpu(1,l))
!
          case(1) ! Supersonic inflow
!
           indx_patch = ibm_bc_gpu(2,l)
           rmixtloc = 0._rkind
           do lsp=1,N_S
            massfrac = ibm_parbc_gpu(indx_patch,3+lsp)
            ibm_w_refl_gpu(l,lsp) = massfrac
            rmixtloc = rmixtloc+rgas_gpu(lsp)*massfrac
           enddo
           pp_patch  = ibm_parbc_gpu(indx_patch,1)
           tt_patch  = ibm_parbc_gpu(indx_patch,2)
           vel_patch = ibm_parbc_gpu(indx_patch,3)
           ppi = pp_patch
           tti = tt_patch
           uui = vel_patch*ibm_nxyz_interface_gpu(1,l)
           vvi = vel_patch*ibm_nxyz_interface_gpu(2,l)
           wwi = vel_patch*ibm_nxyz_interface_gpu(3,l)
!
          case(2) ! Supersonic Turbulent inflow
!
           indx_patch = ibm_bc_gpu(2,l)
           rmixtloc = 0._rkind
           do lsp=1,N_S
            massfrac = ibm_parbc_gpu(indx_patch,5+lsp)
            ibm_w_refl_gpu(l,lsp) = massfrac
            rmixtloc = rmixtloc+rgas_gpu(lsp)*massfrac
           enddo
           yc_patch  = ibm_parbc_gpu(indx_patch,1)
           zc_patch  = ibm_parbc_gpu(indx_patch,2)
           pp_patch  = ibm_parbc_gpu(indx_patch,3)
           tt_patch  = ibm_parbc_gpu(indx_patch,4)
           vel_patch = ibm_parbc_gpu(indx_patch,5)
           ppi = pp_patch
           tti = tt_patch

           rad = (((y_gpu(j) - yc_patch)**2_rkind + (z_gpu(k) - zc_patch)**2_rkind)**0.5_rkind)*436.681222707_rkind
           inlet_phi = datan2(z_gpu(k)-zc_patch,y_gpu(j)-yc_patch)
           utmp = vel_patch*((1-rad)**0.151515151515_rkind)*ibm_nxyz_interface_gpu(1,l)
           avar_gpu = 0.5_rkind*0.05_rkind*(utmp-0.7_rkind)
           upert = 0._rkind
           do modes=1,8
            if (modes .le. 6) then
             upert = upert + avar_gpu*sin(randvar_a_gpu(modes) + modes*inlet_phi+randvar_p_gpu(modes)*time)
            else
             upert = upert + avar_gpu*sin(randvar_p_gpu(modes)*time)
            endif
           enddo

           uui = utmp + upert
           vvi = vel_patch*ibm_nxyz_interface_gpu(2,l)
           wwi = vel_patch*ibm_nxyz_interface_gpu(3,l)
!
          case(3) ! Subsonic inflow
!
           indx_patch = ibm_bc_gpu(2,l)
           ptotal = ibm_parbc_gpu(indx_patch,1)
           ttotal = ibm_parbc_gpu(indx_patch,2)
           rmixtloc = 0._rkind
           do lsp=1,N_S
            massfrac = ibm_parbc_gpu(indx_patch,2+lsp)
            ibm_w_refl_gpu(l,lsp) = massfrac
            rmixtloc = rmixtloc+rgas_gpu(lsp)*massfrac
           enddo
           del = 0.5_rkind*(gamloc-1._rkind)
           rmfac = (pp/ptotal)**(-(gamloc-1._rkind)/gamloc)
           !if (pp/ptotal .lt. 1._rkind) print *, "Error: Negative mach!"
           rml = sqrt((rmfac-1._rkind)/del)
           ppi = pp
           tti = ttotal/rmfac
           vel_mod = rml*sqrt(gamloc*rmixtloc*tti)
           uui = vel_mod*ibm_nxyz_interface_gpu(1,l)
           vvi = vel_mod*ibm_nxyz_interface_gpu(2,l)
           wwi = vel_mod*ibm_nxyz_interface_gpu(3,l)
!
          case (4) !Subsonic Turbulent Inflow
!
           indx_patch = ibm_bc_gpu(2,l)
           yc_patch  = ibm_parbc_gpu(indx_patch,1)
           zc_patch  = ibm_parbc_gpu(indx_patch,2)
           ptotal = ibm_parbc_gpu(indx_patch,3)
           ttotal = ibm_parbc_gpu(indx_patch,4)
           rmixtloc = 0._rkind
           do lsp=1,N_S
            massfrac = ibm_parbc_gpu(indx_patch,4+lsp)
            ibm_w_refl_gpu(l,lsp) = massfrac
            rmixtloc = rmixtloc+rgas_gpu(lsp)*massfrac
           enddo

           del = 0.5_rkind*(gamloc-1._rkind)
           rmfac = (pp/ptotal)**(-(gamloc-1._rkind)/gamloc)
           rml = sqrt((rmfac-1._rkind)/del)
           ppi = pp
           tti = ttotal/rmfac
           vel_mod = rml*sqrt(gamloc*rmixtloc*tti)

           rad = (((y_gpu(j) - yc_patch)**2_rkind + (z_gpu(k) - zc_patch)**2_rkind)**0.5_rkind)*436.681222707_rkind
           inlet_phi = datan2(z_gpu(k)-zc_patch,y_gpu(j)-yc_patch)
           utmp = vel_mod*((1-rad)**0.151515151515_rkind)*ibm_nxyz_interface_gpu(1,l)
           avar_gpu = 0.5_rkind*0.05_rkind*(utmp-0.7_rkind)
           !print *, vel_mod,utmp,tti
           upert = 0._rkind
           do modes=1,8
            if (modes .le. 6) then
             upert = upert + avar_gpu*sin(randvar_a_gpu(modes) + modes*inlet_phi+randvar_p_gpu(modes)*time)
            else
             upert = upert + avar_gpu*sin(randvar_p_gpu(modes)*time)
            endif
           enddo

           uui = utmp + upert
           vvi = vel_mod*ibm_nxyz_interface_gpu(2,l)
           wwi = vel_mod*ibm_nxyz_interface_gpu(3,l)

          case (5) ! Inviscid adiabatic wall
!
           unorm2 = 2._rkind*(uu*ibm_nxyz_interface_gpu(1,l)+vv*ibm_nxyz_interface_gpu(2,l)+ww*ibm_nxyz_interface_gpu(3,l))
           uui  = uu-unorm2*ibm_nxyz_interface_gpu(1,l)
           vvi  = vv-unorm2*ibm_nxyz_interface_gpu(2,l)
           wwi  = ww-unorm2*ibm_nxyz_interface_gpu(3,l)
           ppi  = pp ! extrapolate from interior
           tti  = tt ! extrapolate from interior
!
          case (6) ! Viscous isothermal wall
!
           indx_patch = ibm_bc_gpu(2,l)
           twall_ibm  = ibm_parbc_gpu(indx_patch,1)
           ppi  = pp ! extrapolate from interior
           tti  = 2._rkind*twall_ibm-tt
           uui  = -uu
           vvi  = -vv
           wwi  = -ww
!
          case (8) ! Adiabatic wall
!
           ppi  = pp ! extrapolate from interior
           tti  = tt ! extrapolate from interior
           uui  = -uu
           vvi  = -vv
           wwi  = -ww
!
          case(9)
!
           rad = sqrt(x_gpu(i)**2+z_gpu(k)**2)
!          rad = sqrt(y_gpu(j)**2+z_gpu(k)**2)
           if (rad<ibm_aero_rad) then
            ppi = ibm_aero_pp
            tti = ibm_aero_tt
            uui = ibm_aero_modvel*ibm_nxyz_interface_gpu(1,l)
            vvi = ibm_aero_modvel*ibm_nxyz_interface_gpu(2,l)
            wwi = ibm_aero_modvel*ibm_nxyz_interface_gpu(3,l)
           else
            rhoi = w_aux_gpu(i,j,k,J_R)
            uui  = w_aux_gpu(i,j,k,J_U)
            vvi  = w_aux_gpu(i,j,k,J_V)
            wwi  = w_aux_gpu(i,j,k,J_W)
            tti  = w_aux_gpu(i,j,k,J_T)
            ppi  = w_aux_gpu(i,j,k,J_P)
           endif
!
          end select type_bc
!
          pp  = w_aux_gpu(i,j,k,J_P)
          tt  = w_aux_gpu(i,j,k,J_T)
          uu  = w_aux_gpu(i,j,k,J_U)
          vv  = w_aux_gpu(i,j,k,J_V)
          ww  = w_aux_gpu(i,j,k,J_W)
          ppi = pp+ibm_bc_relax_factor*(ppi-pp)
          uui = uu+ibm_bc_relax_factor*(uui-uu)
          vvi = vv+ibm_bc_relax_factor*(vvi-vv)
          wwi = ww+ibm_bc_relax_factor*(wwi-ww)
          tti = tt+ibm_bc_relax_factor*(tti-tt)
!
          rhoi  = ppi/tti/rmixtloc
          rhoui = rhoi*uui
          rhovi = rhoi*vvi
          rhowi = rhoi*wwi
!         Compute energy mixture
          nrange = 1
          if (nsetcv>1) then ! Replicate locate function of numerical recipes 
           do lsp=1,N_S
            jl = 0
            ju = nsetcv+1+1
            do
             if (ju-jl <= 1) exit
             jm = (ju+jl)/2
             if (tt >= trange_gpu(lsp,jm)) then
              jl=jm
             else
              ju=jm
             endif
            enddo
            nrange(lsp) = jl
            nrange(lsp) = max(nrange(lsp),1)
            nrange(lsp) = min(nrange(lsp),nsetcv)
           enddo
          endif
!          
          eei = 0._rkind
          do lsp=1,N_S
           eei = eei+cv_coeff_gpu(indx_cp_r+1,lsp,nrange(lsp))*ibm_w_refl_gpu(l,lsp)
          enddo
          do m=indx_cp_l,indx_cp_r
           if (m==-1) then
            tprod = log(tti)
           else
            tprod = tti**(m+1)/(m+1._rkind)
           endif
           cv_m = 0._rkind
           do lsp=1,N_S
            cv_m = cv_m+cv_coeff_gpu(m,lsp,nrange(lsp))*ibm_w_refl_gpu(l,lsp)
           enddo
           eei = eei+cv_m*tprod
          enddo
!
          do lsp=1,N_S
           w_gpu(i,j,k,lsp) = rhoi*ibm_w_refl_gpu(l,lsp)
          enddo
          w_gpu(i,j,k,I_U) = rhoui
          w_gpu(i,j,k,I_V) = rhovi
          w_gpu(i,j,k,I_W) = rhowi
          w_gpu(i,j,k,I_E) = rhoi*eei+0.5_rkind*(rhoui*rhoui+rhovi*rhovi+rhowi*rhowi)/rhoi
!
         enddo
         !@cuf iercuda=cudaDeviceSynchronize()
!
        endif
!
    endsubroutine ibm_forcing_old_cuf

    attributes(device) function get_cp_yy_old_dev(tt,indx_cp_l,indx_cp_r,cp_coeff_gpu, &
                       nsetcv,trange_gpu,yy)
    real(rkind) :: get_cp_yy_old_dev
    integer, value :: nsetcv,indx_cp_l,indx_cp_r
    real(rkind), value :: tt
    real(rkind), dimension(indx_cp_l:indx_cp_r+2,N_S,nsetcv) :: cp_coeff_gpu
    real(rkind), dimension(N_S) :: yy
    real(rkind), dimension(N_S,nsetcv+1) :: trange_gpu
    real(rkind) :: cploc,cp_l,tpow
    integer, dimension(N_S) :: nrange
    integer :: l,lsp,jl,jm,ju
!
    nrange = 1
    if (nsetcv>1) then ! Replicate locate function of numerical recipes
     do lsp=1,N_S
      jl = 0
      ju = nsetcv+1+1
      do
       if (ju-jl <= 1) exit
       jm = (ju+jl)/2
       if (tt>= trange_gpu(lsp,jm)) then
        jl=jm
       else
        ju=jm
       endif
      enddo
      nrange(lsp) = jl
      nrange(lsp) = max(nrange(lsp),1)
      nrange(lsp) = min(nrange(lsp),nsetcv)
     enddo
    endif
!    
    cploc = 0._rkind
    do l=indx_cp_l,indx_cp_r
     tpow = tt**l
     cp_l = 0._rkind
     do lsp=1,N_S
      cp_l = cp_l+cp_coeff_gpu(l,lsp,nrange(lsp))*yy(lsp)
     enddo
     cploc = cploc+cp_l*tpow
    enddo
    get_cp_yy_old_dev = cploc
!
    endfunction get_cp_yy_old_dev
    
endmodule streams_kernels_gpu

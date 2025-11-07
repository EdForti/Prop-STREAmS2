module budget_TKE
 use parameters
 use global_variables
 use reader
 use derivatives
 use comp_transform
!
 contains

  subroutine compute_budget
       implicit none
       integer :: i,j,m,nv_budget
       real(rkind) :: rho,ri,pp,uu,vv,ww,ufav,vfav,wfav,ufav2,vfav2,wfav2,uvfav2,uprime2,vprime2,wprime2
       real(rkind) :: sig11,sig12,sig13,sig22,sig23,sig33,tau11,tau12,tau22,ufav_x,ufav_y,vfav_x,vfav_y
       real(rkind) :: u_x,u_y,v_x,v_y,p_x,p_y,sig11_x,sig12_x,sig12_y,sig13_x,sig22_y,sig23_y
       real(rkind) :: C_bud,T1_bud,T2_bud,P_bud,PI_bud,D_bud,PHI_bud,K_bud,ST_bud,sum1,sum2
       real(rkind), dimension(nx,ny)   :: dudx,dudy,dvdx,dvdy,dufavdx,dufavdy,dvfavdx,dvfavdy,dpdx,dpdy
       real(rkind), dimension(nx,ny)   :: tke,term_p,term_phi,term_st,term_k,tmp,app
       real(rkind), dimension(nx,ny,2) :: dterm_c,dterm_t,dterm_pi,dterm_d 
       real(rkind), dimension(nx,ny,2) :: term_c,term_t,term_d
       real(rkind), dimension(nx,ny,3) :: term_pi
       real(rkind), dimension(nx,ny,6) :: dsigdx,dsigdy
       real(rkind) :: delta99,deltav,rhow,utau,muw,tauw,starfac,norm,yy
       character(6)  :: chstat

       call getderivative(wstat(:,:,2),dudx,dudy)
       call getderivative(wstat(:,:,3),dvdx,dvdy)
       call getderivative(wstat(:,:,13)/wstat(:,:,1),dufavdx,dufavdy)
       call getderivative(wstat(:,:,14)/wstat(:,:,1),dvfavdx,dvfavdy)

       call getderivative(wstat(:,:,5),dpdx,dpdy)
       do m=0,5
        call getderivative(wstat(:,:,43+m),dsigdx(:,:,m+1),dsigdy(:,:,m+1))
       enddo

       do j=1,ny
        do i=1,nx
         rho = wstat(i,j,1)
         ri  = 1._rkind/rho
         pp  = wstat(i,j,5)
         uu    = wstat(i,j,2)
         vv    = wstat(i,j,3)
         ww    = wstat(i,j,4)
         ufav  = wstat(i,j,13)*ri
         vfav  = wstat(i,j,14)*ri
         wfav  = wstat(i,j,15)*ri
         ufav2 = wstat(i,j,16)*ri
         vfav2 = wstat(i,j,17)*ri
         wfav2 = wstat(i,j,18)*ri
         uvfav2 = wstat(i,j,19)*ri
         uprime2  = ufav2 - ufav**2
         vprime2  = vfav2 - vfav**2
         wprime2  = wfav2 - wfav**2
         ! stress tensor     
         sig11 = wstat(i,j,43)
         sig12 = wstat(i,j,44)
         sig13 = wstat(i,j,45)
         sig22 = wstat(i,j,46)
         sig23 = wstat(i,j,47)
         sig33 = wstat(i,j,48)
         ! Reynolds stresses
         tau11 = -rho*(ufav2 -ufav**2  )
         tau12 = -rho*(uvfav2-ufav*vfav)
         tau22 = -rho*(vfav2 -vfav**2  )
         ! some derivatives
         u_x = dudx(i,j)
         u_y = dudy(i,j)
         v_x = dvdx(i,j)
         v_y = dvdy(i,j)
         ufav_x = dufavdx(i,j)
         ufav_y = dufavdy(i,j)
         vfav_x = dvfavdx(i,j)
         vfav_y = dvfavdy(i,j)
         p_x = dpdx(i,j)
         p_y = dpdy(i,j)
         sig11_x = dsigdx(i,j,1)
         sig12_x = dsigdx(i,j,2)
         sig12_y = dsigdy(i,j,2)
         sig13_x = dsigdx(i,j,3)
         sig22_y = dsigdy(i,j,4)
         sig23_y = dsigdy(i,j,5)

         ! SEE COGO JFM 2022
         ! TKE 
         tke(i,j) = 0.5_rkind*(uprime2+vprime2+wprime2)
         ! convective term
         term_c(i,j,1) = rho*ufav*tke(i,j)
         term_c(i,j,2) = rho*vfav*tke(i,j)
         ! production of TKE
         term_p(i,j) = tau11*ufav_x+tau12*(ufav_y+vfav_x)+tau22*vfav_y
         ! turbulent transport (check the sign!!!!!!) 
         term_t(i,j,1) = ri*(wstat(i,j,35)+wstat(i,j,37)+wstat(i,j,39))
         term_t(i,j,2) = ri*(wstat(i,j,36)+wstat(i,j,38)+wstat(i,j,40))
         term_t(i,j,1) = 0.5_rkind*rho*(term_t(i,j,1)                            &
                        -3._rkind*ufav*ufav2+2._rkind*ufav**3                    &
                        -2._rkind*vfav*uvfav2-ufav*vfav2+2._rkind*ufav*vfav**2   &
                        -ufav*wfav2)
         term_t(i,j,2) = 0.5_rkind*rho*(term_t(i,j,2)                            &
                        -2._rkind*ufav*uvfav2-vfav*ufav2+2._rkind*vfav*ufav**2   &
                        -3._rkind*vfav*vfav2+2._rkind*vfav**3                    &
                        -vfav*wfav2)
         ! pressure transport (1,2) and pressure dilatation (3)
         term_pi(i,j,1) = wstat(i,j,41)
         term_pi(i,j,2) = wstat(i,j,42)
         term_pi(i,j,1) = -term_pi(i,j,1)+pp*uu
         term_pi(i,j,2) = -term_pi(i,j,2)+pp*vv
         term_pi(i,j,3) = wstat(i,j,61)+wstat(i,j,62)+wstat(i,j,63)-pp*(u_x+v_y)
         ! molecular transport 
         term_d(i,j,1) = wstat(i,j,49)+wstat(i,j,51)+wstat(i,j,53)
         term_d(i,j,2) = wstat(i,j,50)+wstat(i,j,52)+wstat(i,j,54)
         term_d(i,j,1) = term_d(i,j,1)-uu*sig11-vv*sig12-ww*sig13
         term_d(i,j,2) = term_d(i,j,2)-uu*sig12-vv*sig22-ww*sig23
         ! dissipation of TKE
         term_phi(i,j) = wstat(i,j,57)+wstat(i,j,58)+wstat(i,j,59)
         term_phi(i,j) = -term_phi(i,j)+(sig11*u_x+sig12*(u_y+v_x)+sig22*v_y)
         ! additional terms + pressure work
         term_st(i,j) = -rho*tke(i,j)*(ufav_x+vfav_y)          &
                        + (uu-ufav)*(sig11_x+sig12_y-p_x) &
                        + (vv-vfav)*(sig12_x+sig22_y-p_y) &
                        + (ww     )*(sig13_x+sig23_y    )
         term_k(i,j)  = wstat(i,j,61)+wstat(i,j,62)+wstat(i,j,63)-pp*(u_x+v_y) &
                        + (uu-ufav)*(sig11_x+sig12_y-p_x) &
                        + (vv-vfav)*(sig12_x+sig22_y-p_y) &
                        + (ww     )*(sig13_x+sig23_y    )

        enddo
       enddo

       ! compute derivatives of term_c, term_t, term_pi, term_d
       call getderivative(term_c(:,:,1),dterm_c(:,:,1),tmp)
       call getderivative(term_c(:,:,2),tmp,dterm_c(:,:,2))

       call getderivative(term_t(:,:,1),dterm_t(:,:,1),tmp)
       call getderivative(term_t(:,:,2),tmp,dterm_t(:,:,2))

       call getderivative(term_pi(:,:,1),dterm_pi(:,:,1),tmp)
       call getderivative(term_pi(:,:,2),tmp,dterm_pi(:,:,2))

       call getderivative(term_d(:,:,1),dterm_d(:,:,1),tmp)
       call getderivative(term_d(:,:,2),tmp,dterm_d(:,:,2))

       nv_budget = 12
       allocate(wstat_budget(nx,ny,nv_budget))

       do j=1,ny
        do i=1,nx
         C_bud   = -(dterm_c(i,j,1)+dterm_c(i,j,2))
         T1_bud  = -(dterm_t(i,j,1)+dterm_t(i,j,2))+dterm_pi(i,j,1)+dterm_pi(i,j,2)
         T2_bud  = -(dterm_t(i,j,1)+dterm_t(i,j,2))
         PI_bud  = dterm_pi(i,j,1)+dterm_pi(i,j,2)+term_pi(i,j,3)
         P_bud   = term_p(i,j)
         D_bud   = dterm_d(i,j,1)+dterm_d(i,j,2)
         PHI_bud = term_phi(i,j)
         K_bud   = term_k(i,j)
         ST_bud  = term_st(i,j)
         sum1    = C_bud+T1_bud+P_bud +D_bud+PHI_bud+K_bud
         sum2    = P_bud+T2_bud+PI_bud+D_bud+PHI_bud+ST_bud
         deltav  = statprop(i,2)
         rhow    = statprop(i,3)
         utau    = statprop(i,4)
         tauw    = rhow*utau**2
         norm    = rhow*utau**3/deltav
         wstat_budget(i,j,1)  = tke(i,j)/norm
         wstat_budget(i,j,2)  = C_bud/norm
         wstat_budget(i,j,3)  = P_bud/norm
         wstat_budget(i,j,4)  = T1_bud/norm
         wstat_budget(i,j,5)  = T2_bud/norm
         wstat_budget(i,j,6)  = PI_bud/norm
         wstat_budget(i,j,7)  = D_bud/norm
         wstat_budget(i,j,8)  = PHI_bud/norm
         wstat_budget(i,j,9)  = K_bud/norm
         wstat_budget(i,j,10) = ST_bud/norm
         wstat_budget(i,j,11) = sum1/norm
         wstat_budget(i,j,12) = sum2/norm

        enddo
       enddo

       do ii=1,nstatloc
        i = ixstat(ii)
        write(chstat,1006) i
        write(*,*)'writing budget of profile,',i
        delta99 = statprop(i,1)
        deltav  = statprop(i,2)
        rhow    = statprop(i,3)
        utau    = statprop(i,4)
        muw     = statprop(i,5)
        tauw    = rhow*utau**2
        open(unit=25,file='POSTPRO/budget_'//chstat//'.prof')
        do j=1,ny
         starfac = sqrt(tauw*wstat(i,j,1))/wstat(i,j,20)
         yy = y(j) 
         write(25,100) yy, yy/delta99, yy/deltav, yy*starfac,&
                       (wstat_budget(i,j,m),m=1,nv_budget)
        enddo
        close(25)
       enddo
       100  format(200ES20.10)
       1006 format(I6.6)

       call save_budget_p3d(nv_budget,wstat_budget)

   endsubroutine compute_budget

   subroutine getderivative(f,dfdx,dfdy)
       implicit none
       integer :: i,j
       real(rkind), dimension(nx,ny), intent(in) :: f
       real(rkind), dimension(nx,ny), intent(out) :: dfdx,dfdy

       do j=1,ny
        call ddx(nx,f(:,j),x(1:nx),6,dfdx(:,j))
       enddo
       do i=1,nx
        call ddy(ny,f(i,:),y(1:ny),6,dfdy(i,:))
       enddo

   endsubroutine getderivative

   subroutine save_budget_p3d(nv_budget,wstat_budget)
       implicit none
       integer, intent(in) :: nv_budget
       real(rkind), dimension(nx,ny,nv_budget) :: wstat_budget
       real(rkind), dimension(:,:), allocatable :: wbudtemp
       integer :: m
       open(unit=124, file="POSTPRO/budget_fields.q", form="unformatted", access="stream")
       write(124) nx,ny,1,nv_budget
       allocate(wbudtemp(nx,ny))
       do m=1,nv_budget
           wbudtemp(:,:) = wstat_budget(:,:,m)
           write(124) wbudtemp
       enddo
       close(124)
   endsubroutine save_budget_p3d

   end module


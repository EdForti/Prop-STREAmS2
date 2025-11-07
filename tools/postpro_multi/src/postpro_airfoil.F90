module postpro_airfoil
 use parameters
 use global_variables 
 use reader
 use derivatives
 use comp_transform

 contains 

 subroutine forces_airfoil
 implicit none

 integer     :: i
 real(rkind) :: cf1,costh,sinth,dudy,dudy1,dudyw,dudyw1
 real(rkind) :: al,ds,dN,dA,dpN,dpA,dtN,dtA,N,A,pN,pA,tN,tA
 real(rkind) :: lift,drag,pres,fric,pdyn,tauw,tauw1,tauwf,cf,cp,pw,pw1,pwf

 call read_grid_airfoil_ramp()

 open(unit=12,file='POSTPRO/avg_coeff.dat',form='formatted')
  N  = 0._rkind
  A  = 0._rkind
  pN = 0._rkind
  pA = 0._rkind
  tN = 0._rkind
  tA = 0._rkind
  do i=ite,itu
   dudy   = -22._rkind*wstat(i,1,2)+36._rkind*wstat(i,2,2)-18._rkind*wstat(i,3,2)+ 4._rkind*wstat(i,4,2)
   dudyw  = dudy*meta(i,1)/12._rkind
   tauw   = wstat(i,1,20)*dudyw
   pw     = wstat(i,1,5)
   dudy1  = -22._rkind*wstat(i+1,1,2)+36._rkind*wstat(i+1,2,2)-18._rkind*wstat(i+1,3,2)+ 4._rkind*wstat(i+1,4,2)
   dudyw1 = dudy1*meta(i+1,1)/12._rkind
   tauw1  = wstat(i+1,1,20)*dudyw1
   pw1    = wstat(i+1,1,5)

   ! cf without sign
   cf = tauw/(0.5_rkind*u0*u0)
   ! cf with sign
   !if (i<ile) cf = -tauw/(0.5_rkind*u0*u0)
   !if (i>=ile) cf = tauw/(0.5_rkind*u0*u0)
   !if (i==ite) cf1 = cf
   !if (i==itu) cf  = cf1

   cp = (pw-p0)/(0.5_rkind*u0*u0)
   pwf   = .5_rkind*(pw+pw1)
   tauwf = .5_rkind*(tauw+tauw1)
   ds    = .5_rkind*(csimod (i,1)+csimod (i+1,1)) ! csimod  id the module of the wall-tangent
   costh = .5_rkind*(dxdcsin(i,1)+dxdcsin(i+1,1)) ! dxdcsin is the x-component of the wall-ta
   sinth = .5_rkind*(dydcsin(i,1)+dydcsin(i+1,1)) ! dydcsin is the y-component of the wall-ta
   if (i<itu) then
    dN  = -costh*pwf+sinth*tauwf
    dA  = +sinth*pwf+costh*tauwf
    dpN = -costh*pwf
    dpA = +sinth*pwf
    dtN = sinth*tauwf
    dtA = costh*tauwf
    N  = N + dN*ds
    A  = A + dA*ds
    pN = pN + dpN*ds
    pA = pA + dpA*ds
    tN = tN + dtN*ds
    tA = tA + dtA*ds
   endif
   write(12,100) xg(i,1),cf,cp,pw,tauw
  enddo
 close(12)
!
 al = alpha_airfoil*pi/180._rkind
 pdyn = .5_rkind*u0*u0 
 lift = ( N*cos(al)- A*sin(al))/pdyn 
 drag = ( N*sin(al)+ A*cos(al))/pdyn
 pres = (pN*sin(al)+pA*cos(al))/pdyn
 fric = (tN*sin(al)+tA*cos(al))/pdyn
 print*, 'Lift     =',lift,'Drag     =',drag
 print*, 'Pressure =',pres,'Friction =',fric
 open(unit=30,file='POSTPRO/avg_forces.dat')
  write(30,100) lift,drag,pres,fric
 close(30)
!  
 100  format(200ES20.10)
 end subroutine forces_airfoil

 subroutine stats2d_airfoil
 implicit none
 integer, parameter :: naux = 6
 integer :: i,j,m,j99,ii
 real(rkind), dimension(nx,ny) :: ufav,vfav,wfav,tfav
 real(rkind), dimension(npoints_bl) :: ufav_bl,vfav_bl,wfav_bl,tfav_bl,utfav_bl,vtfav_bl,rhout2_bl,rhovt2_bl,rhoutvt_bl
 real(rkind), dimension(nx) :: d99_vec,deltav_vec,utau_vec,rhow_vec,ttw_vec,dudyw_vec,rmuw_vec,jvortical
 real(rkind), dimension(naux,npoints_bl) :: vaux
 real(rkind), dimension(npoints_bl) :: yt,yvd,yv,uv,ut,uvd,yh,uh,uinv,dutfav_bl_dy
 real(rkind) :: uwall, dudyw, d2udyw, rmuw, rnuw,tauw,rhow,ttw,deltav,pdyn,dthre
 real(rkind) :: cf,dely,utau,uu,unum,uden,delta99,retau,prmsp 
 real(rkind) :: dstarinc,thetainc,rhoe,pe,ue,tte 
 real(rkind) :: rho,rhop,up,dy,dyh,dstar,theta,uup,gg,ry,py,udel,d_bl,uref,vref,al
 real(rkind) :: shapef,shapefinc,rmue,ff,aa,bb,alpha,beta 
 real(rkind) :: fc,ftheta,cfinc,rethetainc,rethetawall,redelta99,redelta2,retheta
 real(rkind) :: y99,uvdp,u2p,v2p,w2p,uvp,rhofac,utp,uvpl,yvp,ytp,Mtau,mum,deltaj99,ch,ppm,ppw,qw
 real(rkind) :: yp , cpcoeff, pwall, yyy, Bq, ttm, trmsp, uhp,ystar,rlamw,trec,rhormsp,rhom
 character(20) :: tname
 character(6)  :: chstat
 real(rkind) :: dtdyw, qwall, vortx2, vorty2, vortz2, omag, omag_factor, eta
 integer :: jvort

 !call read_grid_airfoil_ramp()
!
  do j=1,ny
   do i=1,nx
    ufav(i,j) = wstat(i,j,13)/wstat(i,j,1)
    vfav(i,j) = wstat(i,j,14)/wstat(i,j,1)
    wfav(i,j) = wstat(i,j,15)/wstat(i,j,1)
    tfav(i,j)  = wstat(i,j,25)/wstat(i,j,1)
   enddo
  enddo
!
! Mean boundary layer properties
!
  udel = 0.99_rkind*u0 ! this is an approximation, to do: compute udel as utilde at BL edge  
  gg    = 2*gam/(gam-1)
  al = alpha_airfoil*pi/180._rkind
  uref  = u0*cos(al)
  vref  = u0*sin(al)
!
  open(10,file='POSTPRO/bl_pressure.dat',form='formatted')
  do i=ite+1,ile-1,ix_out
   print*,'Analyzing wall i-index: ',i
   ! get wstat_bl(j,iv)
   call extract_bl(i)

   do j=1,npoints_bl
       ufav_bl(j)  = wstat_bl(j,13)/wstat_bl(j,1)
       vfav_bl(j)  = wstat_bl(j,14)/wstat_bl(j,1)
       wfav_bl(j)  = wstat_bl(j,15)/wstat_bl(j,1)
       tfav_bl(j)  = wstat_bl(j,25)/wstat_bl(j,1)
       utfav_bl(j) = (ufav_bl(j)*dxdcsi(i,1)+vfav_bl(j)*dydcsi(i,1))/csimod(i,1) ! csi tangent wall
       vtfav_bl(j) = (ufav_bl(j)*detadx(i,1)+vfav_bl(j)*detady(i,1))/meta(i,1) ! csi normal wall
   enddo

   dudyw  = -22._rkind*utfav_bl(1)+36._rkind*utfav_bl(2)-18._rkind*utfav_bl(3)+ 4._rkind*utfav_bl(4)
   dudyw  = dudyw * meta(i,1) / 12._rkind
   !print*,'dudyw: ',dudyw
   dtdyw  = -22._rkind*tfav_bl(1)+36._rkind*tfav_bl(2)-18._rkind*tfav_bl(3)+4._rkind*tfav_bl(4)
   dtdyw  = dtdyw * meta(i,1) / 12._rkind
   !print*,'dtdyw: ',dtdyw

   rhow   = wstat_bl(1,1)
   ttw    = tfav_bl(1)
   rmuw   = wstat_bl(1,20)
   rnuw   = rmuw/rhow

   ttw    = Twall
   rlamw  = rmuw*gam/(gam-1._rkind)/Prandtl
   tauw   = rmuw*dudyw
   qw     = -rlamw*dtdyw
   utau   = sqrt(abs(tauw)/rhow)
   deltav = rnuw/utau
   pdyn   = 0.5_rkind*rho0*u0**2
   cf     = tauw/pdyn
   trec   = t0 * (1.+0.5*(gam-1.)*rfac*Mach**2)
   Bq     = qw/((gam/(gam-1.))*rhow*utau*ttw)
   if (theta_wall==1._rkind) then
    ch = 0._rkind
   else
    ch = qw/u0/(gam/(gam-1.))/(ttw-trec)
   endif

   prmsp = sqrt(abs(wstat_bl(1,11)-wstat_bl(1,5)**2))/tauw
   ppw   = rhow*ttw 

   cpcoeff = (ppw-p0)/pdyn

   !-----------------------------------------------------------
   ! VORTICAL CRTIERIUM
   !-----------------------------------------------------------
   !TEORICO omag_factor = 0.05_rkind
   omag_factor = 0.25_rkind
   !omag_factor = 0.95_rkind
   jvort = 1
   do j=1,npoints_bl-1
    vortx2 = wstat_bl(j,22)
    vorty2 = wstat_bl(j,23)
    vortz2 = wstat_bl(j,24)
    omag = sqrt(vortx2+vorty2+vortz2)
    !print*,'omag: ',omag,vortx2,vorty2,vortz2,omag_factor*u0/l0
    if (omag < omag_factor*u0/l0) then
     jvort = j
     exit
    endif
   enddo

   udel = 0.99_rkind*abs(utfav_bl(jvort))

   j99 = 1
   do j=1,npoints_bl-1
    if (abs(utfav_bl(j))>udel) then
     j99 = j-1
     exit
    endif
   enddo

   print*,'u0, l0: ',u0, l0
   print*,'j: ',jvort, j99
   !-----------------------------------------------------------
   !-----------------------------------------------------------
   ! INVISCID CRTIERIUM
   !-----------------------------------------------------------
   ! Computation of delta_99 (see Griffin et al, PRF 2021)
   !uinv=0._rkind
   !gg = 2*gam/(gam-1)
   !do j=1,npoints_bl-1
   ! ry=wstat_bl(j,1)
   ! py=wstat_bl(j,5)
   ! uinv(j)=sqrt(gg*(p0/rho0-py/ry)+uref**2+vref**2-vtfav_bl(j)**2) ! inviscid velocity profile
   !enddo
   !j99   = 1
   !do j=1,npoints_bl-1
   ! uu = utfav_bl(j)
   ! udel = 0.99_rkind*uinv(j)
   ! if (uu>udel) then
   !  j99 = j-1
   !  exit
   ! endif
   !enddo
   !jvort = j99
   !print*,'j: ',jvort, j99
   !-----------------------------------------------------------
   jvortical(i) = jvort

   deltaj99 = delta_bl(j99-1)
   dely = delta_bl(j99) - delta_bl(j99-1)
   unum = udel-abs(utfav_bl(j99))
   uden = abs(utfav_bl(j99+1))-abs(utfav_bl(j99))
   delta99 = deltaj99 + dely*(unum/uden) ! b._rkindl._rkind thickness

   retau = delta99/deltav
   d99_vec(i)    = delta99
   deltav_vec(i) = deltav
   utau_vec(i)   = utau
   rhow_vec(i)   = rhow
   ttw_vec(i)    = ttw
   rmuw_vec(i)   = rmuw
!
!  Integral boundary layer thicknesses
!
   dstar     = 0._rkind
   theta     = 0._rkind
   dstarinc  = 0._rkind
   thetainc  = 0._rkind
   rhoe      = wstat_bl(j99,1)
   pe        = wstat_bl(j99,5)
   ue        = ufav_bl(j99)
   do j=1,j99
    rho  = wstat_bl(j,1)/rhoe
    rhop = wstat_bl(j+1,1)/rhoe
    uu   = ufav_bl(j)/ue
    uup  = ufav_bl(j+1)/ue
    dy   = delta_bl(j+1)-delta_bl(j)
    dyh  = 0.5_rkind*dy
!   Trapezoidal rule
    dstar = dstar       + dyh*((1._rkind-rho*uu)+(1._rkind-rhop*uup))
    theta = theta       + dyh*((rho*uu*(1._rkind-uu))+(rhop*uup*(1._rkind-uup)))
    dstarinc = dstarinc + dyh*((1._rkind-uu)+(1._rkind-uup))
    thetainc = thetainc + dyh*((uu*(1._rkind-uu))+(uup*(1._rkind-uup)))
   enddo
   shapef    = dstar/theta ! Shape factor H
   shapefinc = dstarinc/thetainc ! Incompressible Shape factor H_i
!
!  VAN DRIEST II
!
   tte  = pe/rhoe
   rmue = wstat_bl(j99,20) 
   ff   = ttw/tte
   aa   = ((rfac*0.2_rkind*Mach**2)/ff)**0.5_rkind
   bb   = (1+rfac*0.2_rkind*Mach**2-ff)/ff
   alpha = 2*aa**2-bb
   alpha = alpha/(4*aa**2+bb**2)**0.5_rkind
   beta = bb/(4*aa**2+bb**2)**0.5_rkind
   fc   = rfac*0.2_rkind*Mach**2
   fc   = fc/(asin(alpha)+asin(beta))**2
   ftheta = rmue/rmuw
   cfinc = cf*fc

   redelta99  = ue*rhoe*delta99/rmue
   retheta    = ue*rhoe*theta/rmue
   rethetainc = ue*rhoe*thetainc/rmue
   redelta2   = ue*rhoe*theta/rmuw ! = retheta*ftheta

   !              1        2        3      4              5       6       7       8     9       10
   write(10,100) xg(i,1),yg(i,1),delta99,delta_bl(jvort),dstar,theta,dstarinc,thetainc,shapef,shapefinc, &
   !                   11   12 13    14   15    16    17  18  19     20      21   22
                      rhow,ttw,ppw,prmsp,deltav,utau,tauw,cf,cfinc,cpcoeff,ue/u0,ufav_bl(jvort)/u0, &
   !                     23        24      25      26  27 28
                      redelta99,retheta,redelta2,retau,Bq,ch

  enddo
  close(10)
!
  open(10,file='POSTPRO/bl_suction.dat',form='formatted')
  do i=ile+2,itu-1,ix_out
   print*,'Analyzing wall i-index: ',i
   ! get wstat_bl(j,iv)
   call extract_bl(i)

   do j=1,npoints_bl
       ufav_bl(j)  = wstat_bl(j,13)/wstat_bl(j,1)
       vfav_bl(j)  = wstat_bl(j,14)/wstat_bl(j,1)
       wfav_bl(j)  = wstat_bl(j,15)/wstat_bl(j,1)
       tfav_bl(j)  = wstat_bl(j,25)/wstat_bl(j,1)
       utfav_bl(j) = (ufav_bl(j)*dxdcsi(i,1)+vfav_bl(j)*dydcsi(i,1))/csimod(i,1) ! csi tangent wall
       vtfav_bl(j) = (ufav_bl(j)*detadx(i,1)+vfav_bl(j)*detady(i,1))/meta(i,1) ! csi normal wall
   enddo

   dudyw  = -22._rkind*utfav_bl(1)+36._rkind*utfav_bl(2)-18._rkind*utfav_bl(3)+ 4._rkind*utfav_bl(4)
   dudyw  = dudyw * meta(i,1) / 12._rkind
   !print*,'dudyw: ',dudyw
   dtdyw  = -22._rkind*tfav_bl(1)+36._rkind*tfav_bl(2)-18._rkind*tfav_bl(3)+4._rkind*tfav_bl(4)
   dtdyw  = dtdyw * meta(i,1) / 12._rkind
   !print*,'dtdyw: ',dtdyw

   rhow   = wstat_bl(1,1)
   ttw    = tfav_bl(1)
   rmuw   = wstat_bl(1,20)
   rnuw   = rmuw/rhow

   ttw    = Twall
   rlamw  = rmuw*gam/(gam-1._rkind)/Prandtl
   tauw   = rmuw*dudyw
   qw     = -rlamw*dtdyw
   utau   = sqrt(abs(tauw)/rhow)
   deltav = rnuw/utau
   pdyn   = 0.5_rkind*rho0*u0**2
   cf     = tauw/pdyn
   trec   = t0 * (1.+0.5*(gam-1.)*rfac*Mach**2)
   Bq     = qw/((gam/(gam-1.))*rhow*utau*ttw)
   if (theta_wall==1._rkind) then
    ch = 0._rkind
   else
    ch = qw/u0/(gam/(gam-1.))/(ttw-trec)
   endif

   prmsp = sqrt(abs(wstat_bl(1,11)-wstat_bl(1,5)**2))/tauw
   ppw   = rhow*ttw 

   cpcoeff = (ppw-p0)/pdyn

   !-----------------------------------------------------------
   ! VORTICAL CRTIERIUM
   !-----------------------------------------------------------
   !TEORICO omag_factor = 0.05_rkind
   omag_factor = 0.25_rkind
   !omag_factor = 0.95_rkind
   jvort = 1
   do j=1,npoints_bl-1
    vortx2 = wstat_bl(j,22)
    vorty2 = wstat_bl(j,23)
    vortz2 = wstat_bl(j,24)
    omag = sqrt(vortx2+vorty2+vortz2)
    !print*,'omag: ',omag,vortx2,vorty2,vortz2,omag_factor*u0/l0
    if (omag < omag_factor*u0/l0) then
     jvort = j
     exit
    endif
   enddo

   udel = 0.99_rkind*utfav_bl(jvort)

   j99 = 1
   do j=1,npoints_bl-1
    if (utfav_bl(j)>udel) then
     j99 = j-1
     exit
    endif
   enddo

   print*,'u0, l0: ',u0, l0
   print*,'j: ',jvort, j99
   !-----------------------------------------------------------
   !-----------------------------------------------------------
   ! INVISCID CRTIERIUM
   !-----------------------------------------------------------
   ! Computation of delta_99 (see Griffin et al, PRF 2021)
   !uinv=0._rkind
   !gg = 2*gam/(gam-1)
   !do j=1,npoints_bl-1
   ! ry=wstat_bl(j,1)
   ! py=wstat_bl(j,5)
   ! uinv(j)=sqrt(gg*(p0/rho0-py/ry)+uref**2+vref**2-vtfav_bl(j)**2) ! inviscid velocity profile
   !enddo
   !j99   = 1
   !do j=1,npoints_bl-1
   ! uu = utfav_bl(j)
   ! udel = 0.99_rkind*uinv(j)
   ! if (uu>udel) then
   !  j99 = j-1
   !  exit
   ! endif
   !enddo
   !jvort = j99
   !print*,'j: ',jvort, j99
   !-----------------------------------------------------------
   jvortical(i) = jvort

   deltaj99 = delta_bl(j99-1)
   dely = delta_bl(j99) - delta_bl(j99-1)
   unum = udel-utfav_bl(j99)
   uden = utfav_bl(j99+1)-utfav_bl(j99)
   if(abs(uden) > 1e-6) then
       delta99 = deltaj99 + dely*(unum/uden) ! b._rkindl._rkind thickness
   else
       delta99 = deltaj99
   endif
   retau = delta99/deltav
   d99_vec(i)    = delta99
   deltav_vec(i) = deltav
   utau_vec(i)   = utau
   rhow_vec(i)   = rhow
   ttw_vec(i)    = ttw
   rmuw_vec(i)   = rmuw
!
!  Integral boundary layer thicknesses
!
   dstar     = 0._rkind
   theta     = 0._rkind
   dstarinc  = 0._rkind
   thetainc  = 0._rkind
   rhoe      = wstat_bl(j99,1)
   pe        = wstat_bl(j99,5)
   ue        = ufav_bl(j99)
   do j=1,j99
    rho  = wstat_bl(j,1)/rhoe
    rhop = wstat_bl(j+1,1)/rhoe
    uu   = ufav_bl(j)/ue
    uup  = ufav_bl(j+1)/ue
    dy   = delta_bl(j+1)-delta_bl(j)
    dyh  = 0.5_rkind*dy
!   Trapezoidal rule
    dstar = dstar       + dyh*((1._rkind-rho*uu)+(1._rkind-rhop*uup))
    theta = theta       + dyh*((rho*uu*(1._rkind-uu))+(rhop*uup*(1._rkind-uup)))
    dstarinc = dstarinc + dyh*((1._rkind-uu)+(1._rkind-uup))
    thetainc = thetainc + dyh*((uu*(1._rkind-uu))+(uup*(1._rkind-uup)))
   enddo
   shapef    = dstar/theta ! Shape factor H
   shapefinc = dstarinc/thetainc ! Incompressible Shape factor H_i
!
!  VAN DRIEST II
!
   tte  = pe/rhoe
   rmue = wstat_bl(j99,20) 
   ff   = ttw/tte
   aa   = ((rfac*0.2_rkind*Mach**2)/ff)**0.5_rkind
   bb   = (1+rfac*0.2_rkind*Mach**2-ff)/ff
   alpha = 2*aa**2-bb
   alpha = alpha/(4*aa**2+bb**2)**0.5_rkind
   beta = bb/(4*aa**2+bb**2)**0.5_rkind
   fc   = rfac*0.2_rkind*Mach**2
   fc   = fc/(asin(alpha)+asin(beta))**2
   ftheta = rmue/rmuw
   cfinc = cf*fc

   redelta99  = ue*rhoe*delta99/rmue
   retheta    = ue*rhoe*theta/rmue
   rethetainc = ue*rhoe*thetainc/rmue
   redelta2   = ue*rhoe*theta/rmuw ! = retheta*ftheta

   !              1        2        3      4              5       6       7       8     9       10
   write(10,100) xg(i,1),yg(i,1),delta99,delta_bl(jvort),dstar,theta,dstarinc,thetainc,shapef,shapefinc, &
   !                   11   12 13    14   15    16    17  18  19     20      21   22
                      rhow,ttw,ppw,prmsp,deltav,utau,tauw,cf,cfinc,cpcoeff,ue/u0,ufav_bl(jvort)/u0, &
   !                     23        24      25      26  27 28
                      redelta99,retheta,redelta2,retau,Bq,ch, real(jvort), real(j99)

  enddo
  close(10)

  print*, 'Start/end indices of suction side =',ile,itu
!
  do ii = 1,nstatloc
   i = ixstat(ii)
   write(chstat,1006) i
   write(*,*)'analyzing profile,',i
   call extract_bl(i)

   do j=1,npoints_bl
       ufav_bl(j)    = wstat_bl(j,13)/wstat_bl(j,1)
       vfav_bl(j)    = wstat_bl(j,14)/wstat_bl(j,1)
       wfav_bl(j)    = wstat_bl(j,15)/wstat_bl(j,1)
       tfav_bl(j)    = wstat_bl(j,25)/wstat_bl(j,1)
       utfav_bl(j)   = (ufav_bl(j)*dxdcsi(i,1)+vfav_bl(j)*dydcsi(i,1))/csimod(i,1) ! csi tangent wall
       vtfav_bl(j)   = (ufav_bl(j)*detadx(i,1)+vfav_bl(j)*detady(i,1))/meta(i,1)   ! csi normal
       rhout2_bl(j)  = (wstat_bl(j,16)*dxdcsi(i,1)**2+wstat_bl(j,17)*dydcsi(i,1)**2+ &
                       2._rkind*wstat_bl(j,19)*dxdcsi(i,1)*dydcsi(i,1))/csimod(i,1)**2
       rhovt2_bl(j)  = (wstat_bl(j,16)*detadx(i,1)**2+wstat_bl(j,17)*detady(i,1)**2+ &
                       2._rkind*wstat_bl(j,19)*detadx(i,1)*detady(i,1))/meta(i,1)**2
       rhoutvt_bl(j) = (wstat_bl(j,16)*dxdcsi(i,1)*detadx(i,1)+wstat_bl(j,17)*dydcsi(i,1)*detady(i,1)+ &
                       wstat_bl(j,19)*(dxdcsi(i,1)*detady(i,1)+dydcsi(i,1)*detadx(i,1)))/ &
                       csimod(i,1)/meta(i,1)
   enddo
!
   jvort = jvortical(i)

   rhow    = rhow_vec(i)
   rmuw    = rmuw_vec(i)
   rnuw    = rmuw/rhow
   delta99 = d99_vec(i)
   deltav  = deltav_vec(i)
   utau    = utau_vec(i)
   tauw    = rhow*utau**2
   ttw     = ttw_vec(i)
!
! Compute compressibility transformations
!
   vaux(1,:) = wstat_bl(:,1)/rhow
   vaux(2,:) = wstat_bl(:,21)/rnuw
   vaux(3,:) = wstat_bl(:,20)/rmuw
   vaux(4,:) = delta_bl(1:npoints_bl)/deltav ! yplus
   vaux(5,:) = delta_bl(1:npoints_bl)*sqrt(tauw*wstat_bl(:,1))/wstat_bl(:,20) ! ystar
   vaux(6,:) = utau/sqrt(gam*ttw) ! Mach_tau constant
!
   tname = 'TrettelLarsson'
   call transform(0,npoints_bl,delta_bl,utfav_bl,naux,vaux,yt,ut,tname)
!
   tname = 'vanDriest'
   call transform(0,npoints_bl,delta_bl,utfav_bl,naux,vaux,yvd,uvd,tname)
!
   tname = 'Volpiani'
   call transform(0,npoints_bl,delta_bl,utfav_bl,naux,vaux,yv,uv,tname)

   tname = 'Hasan'
   call transform(0,npoints_bl,delta_bl,utfav_bl,naux,vaux,yh,uh,tname)
!
   open(unit=15,file='POSTPRO/stat_'//chstat//'.prof')
   do j=1,npoints_bl
    if(j == 1) then
      d_bl = 0._rkind
    else
      d_bl = delta_bl(j-1)
    endif
    y99    = d_bl/delta99
    yp     = d_bl/deltav
    ystar  = vaux(5,j)
    up     = utfav_bl(j)/utau
    uvdp   = uvd(j)/utau

    u2p  = abs(rhout2_bl(j)-wstat_bl(j,1)*utfav_bl(j)*utfav_bl(j))/tauw
    v2p  = abs(rhovt2_bl(j)-wstat_bl(j,1)*vtfav_bl(j)*vtfav_bl(j))/tauw
    w2p  = abs(wstat_bl(j,18)-wstat_bl(j,1)*wfav_bl(j)*wfav_bl(j))/tauw
    uvp  = abs(rhoutvt_bl(j)-wstat_bl(j,1)*utfav_bl(j)*vtfav_bl(j))/tauw

    rhofac = sqrt(wstat_bl(j,1)/rhow)
    prmsp  = sqrt(abs(wstat_bl(j,11)-wstat_bl(j,5)**2))/tauw
    ytp    = yt(j)/deltav
    yvp    = yv(j)/deltav
    utp    = ut(j)/utau
    uhp    = uh(j)/utau
    uvpl   = uv(j)/utau

    rhom    = wstat_bl(j,1)
    ttm     = tfav_bl(j)
    ppm     = ttm*rhom
    Mtau    = utau/sqrt(gam*ttw)
    rhormsp = (sqrt(wstat_bl(j, 7)-wstat_bl(j,1)**2))/(rhow*gam*Mtau**2)
    trmsp   = (sqrt(wstat_bl(j,26)/wstat_bl(j,1)-tfav_bl(j)**2))/(ttw*gam*Mtau**2)
    prmsp    = sqrt(abs(wstat_bl(j,11)-wstat_bl(j,5)**2))/tauw
    mum     = wstat_bl(j,20)

    !             1     2    3            4          5           6         7   8   9
    write(15,100) d_bl,rhom,ufav_bl(j),vfav_bl(j),utfav_bl(j),vtfav_bl(j),ttm,ppm,mum, &
    !             10  11 12    13 
                  y99,yp,ystar,yvp, &
    !             14  15   16  17  18
                  up,uvdp,utp,uvpl,uhp, &
    !              19  20  21  22
                  u2p,v2p,w2p,uvp, &
    !              23      24     25
                  rhormsp,trmsp,prmsp
   enddo
   close(15)
  enddo
 100  format(200ES20.10)
 1006 format(I6.6)
 end subroutine stats2d_airfoil

end module postpro_airfoil

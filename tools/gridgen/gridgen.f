      PROGRAM GRIDGEN
*
*     Generates a multi-block grid for IB calculations
*
      PARAMETER (nxmax=20000,nymax=20000,nzmax=20000) ! Maximum number of points
      PARAMETER (ngmax=3,nbmax=256) ! Maximum ghost nodes and blocks
      PARAMETER (nstretchmax=10)
*     Coordinates and metrics (for each block)
      DIMENSION x(1-ngmax:nxmax+ngmax,nbmax),
     .          y(1-ngmax:nymax+ngmax,nbmax),
     .          z(1-ngmax:nzmax+ngmax,nbmax) 
      DIMENSION csix(1-ngmax:nxmax+ngmax,nbmax),
     .          etay(1-ngmax:nymax+ngmax,nbmax),
     .          zitz(1-ngmax:nzmax+ngmax,nbmax) 
*     Global coordinates and metrics
      DIMENSION xg(1-ngmax:nxmax+ngmax)
      DIMENSION yg(1-ngmax:nymax+ngmax)
      DIMENSION zg(1-ngmax:nzmax+ngmax)
      DIMENSION csixg(1-ngmax:nxmax+ngmax)
      DIMENSION etayg(1-ngmax:nymax+ngmax)
      DIMENSION zitzg(1-ngmax:nzmax+ngmax)
      DIMENSION xapp(nxmax)
      DIMENSION yapp(nymax)
      DIMENSION zapp(nzmax)
      DIMENSION csig(nxmax)
      DIMENSION etag(nymax)
      DIMENSION zitg(nzmax)
*     Number of points of each block
      DIMENSION nxb(nbmax),nyb(nbmax),nzb(nbmax)
*     Coefficients to compute metrics
      DIMENSION apadei(nxmax),bpadei(nxmax),cpadei(nxmax)
      DIMENSION apadej(nymax),bpadej(nymax),cpadej(nymax)
      DIMENSION apadek(nzmax),bpadek(nzmax),cpadek(nzmax)
      DIMENSION rx(nxmax)
      DIMENSION ry(nymax)
      DIMENSION rz(nzmax)
*     Coefficients for coordinates stretching
      DIMENSION astrx(  nstretchmax),
     .          bstrx(  nstretchmax),
     .          cstrx(0:nstretchmax)
      DIMENSION astry(  nstretchmax),
     .          bstry(  nstretchmax),
     .          cstry(0:nstretchmax)
      DIMENSION astrz(  nstretchmax),
     .          bstrz(  nstretchmax),
     .          cstrz(0:nstretchmax)
      CHARACTER(6) nab
* 
*     Leggo il file di input
*
      open(unit=11,file='input',form='formatted')
      read(11,*)
      read(11,*) ibcleft,ibcright,ibcbot,ibctop,ibcback,ibcfore ! boundary conditions
      read(11,*)
      read(11,*) rlx, rly, rlz ! Size of the computational box
      read(11,*)
      read(11,*) xshift, yshift, zshift  ! Coordinates shift
      read(11,*)
      read(11,*) nib, njb, nkb    ! Numero di blocchi in direzione x, y and z
      read(11,*)
      read(11,*) nxt, nyt, nzt    ! Number of points (for each block)
      read(11,*)
      read(11,*) iunif, junif, kunif ! Flag for uniform spacing
      read(11,*)
      read(11,*) irefl, jrefl, krefl ! Flag for uniform spacing
      read(11,*)
      read(11,*) nstretchx        ! Number of stretchin point in x direction
      read(11,*)
      do l=1,nstretchx
       read(11,*) astrx(l),bstrx(l),cstrx(l)   ! Streching parameters
      enddo
      read(11,*)
      read(11,*) nstretchy        ! Number of stretchin point in y direction
      read(11,*)
      do l=1,nstretchy
       read(11,*) astry(l),bstry(l),cstry(l)   ! Streching parameters
      enddo
      read(11,*)
      read(11,*) nstretchz        ! Number of stretchin point in z direction
      read(11,*)
      do l=1,nstretchz
       read(11,*) astrz(l),bstrz(l),cstrz(l)   ! Streching parameters
      enddo
      close(11)
*
      nxb = nxt
      nyb = nyt
      nzb = nzt
*
      nxt = nxt * nib ! total number of points in x
      nyt = nyt * njb ! total number of points in y
      nzt = nzt * nkb ! total number of points in z
      nx  = nxt
      ny  = nyt
      nz  = nzt
      if (nx.eq.1) iunif = 1
      if (ny.eq.1) junif = 1
      if (nz.eq.1) kunif = 1
      if (irefl.eq.1) then
       if (mod(nx,2).ne.0) then
        write(*,*) 'Number of points in x is odd'
        write(*,*) 'This is not a good choice if you force reflection'
       endif
      endif
      if (jrefl.eq.1) then
       if (mod(ny,2).ne.0) then
        write(*,*) 'Number of points in y is odd'
        write(*,*) 'This is not a good choice if you force reflection'
       endif
      endif
      if (krefl.eq.1) then
       if (mod(nz,2).ne.0) then
        write(*,*) 'Number of points in z is odd'
        write(*,*) 'This is not a good choice if you force reflection'
       endif
      endif
*
*     Uniform spacing in x,y,z
*
      if (iunif.eq.1) then
       if (nx.ne.1) then
c       dx = rlx/(nx-1)
        dx = rlx/nx
       else
        dx = rlx/nx
        xshift = 0.
       endif
       do i=1-ngmax,nx+ngmax
        xg(i)    = (i-1)*dx
       enddo
       xg = xg+xshift
      endif
      if (junif.eq.1) then
       if (ny.ne.1) then
c       dy = rly/(ny-1)
        dy = rly/ny
       else
        dy = rly/ny
        yshift = 0.
       endif
       do j=1-ngmax,ny+ngmax
        yg(j)    = (j-1)*dy
       enddo
       yg = yg+yshift
      endif
      if (kunif.eq.1) then
       if (nz.ne.1) then
c       dz = rlz/(nz-1)
        dz = rlz/nz
       else
        dz = rlz/nz
        zshift = 0.
       endif
       do k=1-ngmax,nz+ngmax
        zg(k)    = (k-1)*dz
       enddo
       zg = zg+zshift
      endif
*
*     Stretching in x-direction
*
      if (iunif.eq.1) goto 24
      csig     = 0.
      xapp(1)  = xshift
      dx       = rlx/(nxmax-1)
      cstrx(0) = 1.
      do i=2,nxmax
       xx = (i-1)*dx+xshift
       xapp(i) = xx
       sumx = 1.
       do l=1,nstretchx
        sumx = sumx + 0.5*(cstrx(l)-cstrx(l-1))
     .          *(1.+tanh((xx-astrx(l))/bstrx(l)))
       enddo
       sumx = 1./sumx
       csig(i) = csig(i-1)+dx*sumx
      enddo
*
      csig = csig/csig(nxmax)
*
*     Interpolate to find xg
*
      xg(1)  = xshift
      xg(nx) = xshift+rlx
      dcsi = 1./(nx-1)
      do i=2,nx-1
       csi = (i-1)*dcsi
       call locate(csig,nxmax,csi,ii) ! csi is between csig(ii) and csig(ii+1)
       call polint(csig(ii),xapp(ii),2,csi,xxint,ddx) ! 2 is the order of interpolation
       xg(i) = xxint
      enddo
*
*     Forcing exact reflection
*
      if (irefl.eq.1) then
       do i=nx/2+1,nx
        xg(i) = -xg(nx-i+1)
       enddo
      endif 
*     Writing ghost nodes
      do i=1,ngmax
       xg(1-i)  =  -(xg(1+i)-xshift)+xshift
       xg(nx+i) = 2*(xg(nx)-xshift)-(xg(nx-i)-xshift)+xshift
      enddo
 24   continue
*
*     Stretching in y-direction
*
      if (junif.eq.1) goto 25
      etag     = 0.
      yapp(1)  = yshift
      dy       = rly/(nymax-1)
      cstry(0) = 1.
      do j=2,nymax
       yy = (j-1)*dy+yshift
       yapp(j) = yy
       sumy = 1.
       do l=1,nstretchy
        sumy = sumy + 0.5*(cstry(l)-cstry(l-1))
     .          *(1.+tanh((yy-astry(l))/bstry(l)))
       enddo
       sumy = 1./sumy
       etag(j) = etag(j-1)+dy*sumy
      enddo
*
      etag = etag/etag(nymax)
*
*     Interpolate to find yg
*
      yg(1)  = yshift
      yg(ny) = yshift+rly
      deta = 1./(ny-1)
      do j=2,ny-1
       eta = (j-1)*deta
       call locate(etag,nymax,eta,jj) ! eta is between etag(jj) and etag(jj+1)
       call polint(etag(jj),yapp(jj),2,eta,yyint,ddy) ! 2 is the order of interpolation
       yg(j) = yyint
      enddo
*
*     Forcing exact reflection
*
      if (jrefl.eq.1) then
       do j=1,ny/2
        yapp(j) = 0.5*(yg(ny-j+1)-yg(j))
        yg(j) = -yapp(j)
        yg(ny-j+1) = yapp(j)
       enddo
      endif 
*     Writing ghost nodes
      do j=1,ngmax
       yg(1-j)  =  -(yg(1+j)-yshift)+yshift
       yg(ny+j) = 2*(yg(ny)-yshift)-(yg(ny-j)-yshift)+yshift
      enddo
 25   continue
*
*     Stretching in z-direction
*
      if (kunif.eq.1) goto 26
      zitg     = 0.
      zapp(1)  = zshift
      dz       = rlz/(nzmax-1)
      cstrz(0) = 1.
      do k=2,nzmax
       zz = (k-1)*dz+zshift
       zapp(k) = zz
       sumz = 1.
       do l=1,nstretchz
        sumz = sumz + 0.5*(cstrz(l)-cstrz(l-1))
     .          *(1.+tanh((zz-astrz(l))/bstrz(l)))
       enddo
       sumz = 1./sumz
       zitg(k) = zitg(k-1)+dz*sumz
      enddo
*
      zitg = zitg/zitg(nzmax)
*
*     Interpolate to find zg
*
      zg(1)  = zshift
      zg(nz) = zshift+rlz
      dzit = 1./(nz-1)
      do k=2,nz-1
       zit = (k-1)*dzit
       call locate(zitg,nzmax,zit,kk) ! zit is between zitg(kk) and zitg(kk+1)
       call polint(zitg(kk),zapp(kk),2,zit,zzint,ddz) ! 2 is the order of interpolation
       zg(k) = zzint
      enddo
*
*     Forcing exact reflection
*
      if (krefl.eq.1) then
       do k=nz/2+1,nz
        zg(k) = -zg(nz-k+1)
       enddo
      endif
*     Writing ghost nodes
      do k=1,ngmax
       zg(1-k)  =  -(zg(1+k)-zshift)+zshift
       zg(nz+k) = 2*(zg(nz)-zshift)-(zg(nz-k)-zshift)+zshift
      enddo
 26   continue
*
*     Computing metrics
*
      csixg = 0.
      do i=2,nxt-1
       apadei(i) = 1./4.
       bpadei(i) = 1.
       cpadei(i) = 1./4.
      enddo
      apadei(1) = 0.
      bpadei(1) = 1.
      cpadei(1) = 2.
      apadei(nxt) = 2.
      bpadei(nxt) = 1.
      cpadei(nxt) = 0.
      do i=2,nxt-1 ! forming rhs
       rx(i) = 3./4.*(xg(i+1)-xg(i-1))
      enddo
      rx(1)      = -5./2.*xg(1)
     .                +2.*xg(2)
     .               +0.5*xg(3)
      rx(nxt)  =  5./2.*xg(nxt)
     .              -2.*xg(nxt-1)
     .             -0.5*xg(nxt-2)
      call tridag(apadei(1:nxt),
     .            bpadei(1:nxt),
     .            cpadei(1:nxt),
     .            rx    (1:nxt),
     .            csixg (1:nxt),
     .                     nxt)
      do i=1,nxt
       csixg(i) = 1./csixg(i)
      enddo
      if (nxt.eq.1) csixg = 1.
*
      etayg = 0.
      do j=2,nyt-1
       apadej(j) = 1./4.
       bpadej(j) = 1.
       cpadej(j) = 1./4.
      enddo
      apadej(1) = 0.
      bpadej(1) = 1.
      cpadej(1) = 2.
      apadej(nyt) = 2.
      bpadej(nyt) = 1.
      cpadej(nyt) = 0.
*
      do j=2,nyt-1 ! forming rhs
       ry(j) = 3./4.*(yg(j+1)-yg(j-1))
      enddo
      ry(1)      = -5./2.*yg(1)
     .                +2.*yg(2)
     .               +0.5*yg(3)
      ry(nyt)  =  5./2.*yg(nyt)
     .              -2.*yg(nyt-1)
     .             -0.5*yg(nyt-2)
      call tridag(apadej(1:nyt),
     .            bpadej(1:nyt),
     .            cpadej(1:nyt),
     .            ry    (1:nyt),
     .            etayg (1:nyt),
     .                     nyt)
      do j=1,nyt
       etayg(j) = 1./etayg(j)
      enddo
      if (nyt.eq.1) etayg = 1.
*
      zitzg = 0.
      do k=2,nzt-1
       apadek(k) = 1./4.
       bpadek(k) = 1.
       cpadek(k) = 1./4.
      enddo
      apadek(1) = 0.
      bpadek(1) = 1.
      cpadek(1) = 2.
      apadek(nzt) = 2.
      bpadek(nzt) = 1.
      cpadek(nzt) = 0.
      do k=2,nzt-1 ! forming rhs
       rz(k) = 3./4.*(zg(k+1)-zg(k-1))
      enddo
      rz(1)      = -5./2.*zg(1)
     .                +2.*zg(2)
     .               +0.5*zg(3)
      rz(nzt)  =  5./2.*zg(nzt)
     .              -2.*zg(nzt-1)
     .             -0.5*zg(nzt-2)
      call tridag(apadek(1:nzt),
     .            bpadek(1:nzt),
     .            cpadek(1:nzt),
     .            rz    (1:nzt),
     .            zitzg (1:nzt),
     .                     nzt)
      do k=1,nzt
       zitzg(k) = 1./zitzg(k)
      enddo
      if (nzt.eq.1) zitzg = 1.
*
*     Partitioning
*
      ii = 1-ngmax-1
      do ib=1,nib
       do i=1-ngmax,nxb(ib)+ngmax
        ii = ii + 1
        x(i,ib) = xg(ii)
        csix(i,ib) = csixg(ii)
       enddo
       ii = ii - 2*ngmax
      enddo
      jj = 1-ngmax-1
      do jb=1,njb
       do j=1-ngmax,nyb(jb)+ngmax
        jj = jj + 1
        y(j,jb) = yg(jj)
        etay(j,jb) = etayg(jj)
       enddo
       jj = jj - 2*ngmax
      enddo
      kk = 1-ngmax-1
      do kb=1,nkb
       do k=1-ngmax,nzb(kb)+ngmax
        kk = kk + 1
        z(k,kb) = zg(kk)
        zitz(k,kb) = zitzg(kk)
       enddo
       kk = kk - 2*ngmax
      enddo
*
      open(11,file='x.dat')
      do i=1,nx
       write(11,200) xg(i), 1./csixg(i)
      enddo
      close(11)
      open(11,file='y.dat')
      do j=1,ny
       write(11,200) yg(j), 1./etayg(j)
 200  format(2F30.15)
      enddo
      close(11)
      open(11,file='z.dat')
      do k=1,nz
       write(11,200) zg(k), 1./zitzg(k)
      enddo
      close(11)
      open(11,file='dz.dat')
      do k=1,nz
       write(11,*) zg(k), 1./zitzg(k)
      enddo
      close(11)
*
*     Writing grid
*
      open(unit=11,file='grid',form='formatted')
      write(11,*) nib*njb*nkb, nib, njb, nkb
      write(11,*) nxt, nyt, nzt
      ibt = 0
      do ib=1,nib
       do jb=1,njb
        do kb=1,nkb
*
         ibt = ibt + 1
*
*        ibc codes
         ileft  = 10
         iright = 10
         jdown  = 10
         jup    = 10
         kback  = 10
         kfore  = 10
         if (ib.eq.1)   ileft  = ibcleft
         if (ib.eq.nib) iright = ibcright
         if (jb.eq.1)   jdown  = ibcbot
         if (jb.eq.njb) jup    = ibctop
         if (kb.eq.1)   kback  = ibcback
         if (kb.eq.nkb) kfore  = ibcfore
*
         write(11,*) 'Block #', ibt
         write(11,*) nxb(ib),nyb(jb),nzb(kb)
         write(11,*) ileft,iright,jdown,jup,kback,kfore
         do i=1-ngmax,nxb(ib)+ngmax
          write(11,100) x(i,ib),csix(i,ib)
         enddo
         do j=1-ngmax,nyb(jb)+ngmax
          write(11,100) y(j,jb),etay(j,jb)
         enddo
         do k=1-ngmax,nzb(kb)+ngmax
          write(11,100) z(k,kb),zitz(k,kb)
         enddo
        enddo
       enddo
      enddo
      close(11)
*
      open(unit=11,file='grid.dat',form='formatted')
      write(11,*) nx,ny
      write(11,*) ((xg(i),i=1,nx),j=1,ny),
     .            ((yg(j),i=1,nx),j=1,ny)
      close(11)

*
*     Writing 3D grid for tecplot
*
c     ibt = 0
c     do kb=1,nkb
c      do jb=1,njb
c       do ib=1,nib
c        ibt = ibt+1
c        write(nab,1006) ibt
c        open(unit=10,file='grid'//nab//'.dat',form='formatted')
c        write(10,*) 'zone i=',nxb(ib),', j=',nyb(jb),', k=',nzb(kb)
c        do k=1,nzb(kb)
c         do j=1,nyb(jb)
c          do i=1,nxb(ib)
c           write(10,100) x(i,ib),y(j,jb),z(k,kb)
c          enddo
c         enddo
c        enddo
c        close(10)
c       enddo
c      enddo
c     enddo
*
 1006 format(I6.6)
  100 format(4ES30.15)
  101 format(I6.6,2ES30.15)
*
      STOP
      END
*
      subroutine tridag(a,b,c,r,u,n)
      dimension a(n),b(n),c(n),r(n),u(n),g(n)
      bet=b(1)
      u(1)=r(1)/bet
      do j=2,n
        g(j)=c(j-1)/bet
        bet=b(j)-a(j)*g(j)
        u(j)=(r(j)-a(j)*u(j-1))/bet
      end do
      do j=n-1,1,-1
        u(j)=u(j)-g(j+1)*u(j+1)
      end do
      return
      end
*
      SUBROUTINE LOCATE(XX,N,X,J)
      DIMENSION XX(N)
      JL=0
      JU=N+1
10    IF(JU-JL.GT.1)THEN
        JM=(JU+JL)/2
        IF((XX(N).GT.XX(1)).EQV.(X.GT.XX(JM)))THEN
          JL=JM
        ELSE
          JU=JM
        ENDIF
      GO TO 10
      ENDIF
      J=JL
      RETURN
      END
*
      SUBROUTINE POLINT(XA,YA,N,X,Y,DY)
      PARAMETER (NMAX=10)
      DIMENSION XA(N),YA(N),C(NMAX),D(NMAX)
      NS=1
      DIF=ABS(X-XA(1))
      DO 11 I=1,N
        DIFT=ABS(X-XA(I))
        IF (DIFT.LT.DIF) THEN
          NS=I
          DIF=DIFT
        ENDIF
        C(I)=YA(I)
        D(I)=YA(I)
11    CONTINUE
      Y=YA(NS)
      NS=NS-1
      DO 13 M=1,N-1
        DO 12 I=1,N-M
          HO=XA(I)-X
          HP=XA(I+M)-X
          W=C(I+1)-D(I)
          DEN=HO-HP
c         IF(DEN.EQ.0.)PAUSE
          DEN=W/DEN
          D(I)=HP*DEN
          C(I)=HO*DEN
12      CONTINUE
        IF (2*NS.LT.N-M)THEN
          DY=C(NS+1)
        ELSE
          DY=D(NS)
          NS=NS-1
        ENDIF
        Y=Y+DY
13    CONTINUE
      RETURN
      END

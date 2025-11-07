!< STREAmS, general parameters.
module streams_parameters

    use, intrinsic :: iso_fortran_env
    use, intrinsic :: ieee_arithmetic
    use mpi
    use iso_c_binding
    use tcp

    implicit none
    private
    public :: ikind, ikind64, rkind, mpi_prec
    public :: tol_iter, tol_iter_nr
    public :: int2str_o, int2str
    public :: get_mpi_basic_info
    public :: mpi_initialize 
    public :: rename_wrapper
    public :: pi,R_univ
    public :: c_rkind
    public :: pol_int, locateval
    public :: invmat, detmat, fail_input_any
    public :: ieee_is_nan
    public :: error_unit
    public :: mpi_cart_shift_general
    public :: create_folder
    public :: get_abcstar_tables
    public :: polyfit
    public :: quadInterp
    public :: poly6
!
!   INSITU
    public :: insitu_start, insitu_end
    public :: REAL64
    public :: INT64
    public :: byte_size

    integer, parameter :: ikind = INT32
    integer, parameter :: ikind64 = INT64
#ifdef SINGLE_PRECISION
     integer, parameter :: rkind = REAL32
     integer, parameter :: mpi_prec = mpi_real4
     real(rkind), parameter :: tol_iter    = 0.00001_rkind
     real(rkind), parameter :: tol_iter_nr = 0.00001_rkind
     integer, parameter :: c_rkind = C_FLOAT
#else
     integer, parameter :: rkind = REAL64
     integer, parameter :: mpi_prec = mpi_real8
     real(rkind), parameter :: tol_iter = 0.000000001_rkind
     real(rkind), parameter :: tol_iter_nr = 0.000000000001_rkind
     integer, parameter :: c_rkind = C_DOUBLE
#endif

    real(rkind) :: pi = acos(-1._rkind)
    real(rkind) :: R_univ = 8314.46261815324_rkind

    interface
        function rename_wrapper(filein, fileout) bind(C, name="rename")
        import :: c_char, c_int
        integer(c_int) :: rename_wrapper
        character(kind=c_char) :: filein(*), fileout(*)
        endfunction rename_wrapper
    endinterface

    interface
        function mkdir_wrapper(foldername, perm) bind(C, name="mkdir")
          import :: c_char, c_int, c_ptr
          integer(c_int) :: mkdir_wrapper
          integer(c_int), value :: perm
          character(kind=c_char) :: foldername(*)
        endfunction mkdir_wrapper
    endinterface

    interface byte_size
            module procedure byte_size_int32, &
#ifdef SINGLE_PRECISION
                             byte_size_real32
#else
                             byte_size_real64
#endif
    endinterface

contains

    subroutine mpi_initialize()
        integer :: mpi_err
        call mpi_init(mpi_err)
    endsubroutine
   
    subroutine get_mpi_basic_info(nprocs, myrank, masterproc, mpi_err)
        integer :: nprocs, myrank, mpi_err
        logical :: masterproc
        call mpi_comm_size(mpi_comm_world, nprocs, mpi_err)
        call mpi_comm_rank(mpi_comm_world, myrank, mpi_err)
        masterproc = .false.
        if (myrank == 0) masterproc = .true.
    endsubroutine get_mpi_basic_info

    function int2str(int_num)
        implicit none
        integer :: int_num
        character(len=16) :: int2str, ret_value
        write(ret_value, "(I0)") int_num
        int2str = ret_value
    endfunction int2str

    function int2str_o(int_num)
        use mpi
        implicit none
        integer(KIND=MPI_OFFSET_KIND) :: int_num
        character(len=32) :: int2str_o, ret_value
        write(ret_value, "(I0)") int_num
        int2str_o = ret_value
    endfunction int2str_o

    subroutine pol_int(x,y,n,xs,ys)
!
!     Polynomial interpolation using Neville's algorithm  
!     Order of accuracy of the interpolation is n-1
!
     integer, intent(in) :: n
     real(rkind), dimension(n), intent(in) :: x,y
     real(rkind), intent(in) :: xs
     real(rkind), intent(out) :: ys
!
     integer :: i,m
     real(rkind), dimension(n)  :: v,vold
!
     v = y
!
     do m=2,n ! Tableu columns
      vold = v
      do i=1,n+1-m
       v(i) = (xs-x(m+i-1))*vold(i)+(x(i)-xs)*vold(i+1)
       v(i) = v(i)/(x(i)-x(i+m-1))
      enddo
     enddo
     ys = v(1)
    end subroutine pol_int
!
    subroutine locateval(xx,n,x,ii)
! 
     integer, intent(in) :: n
     integer, intent(out) :: ii
     real(rkind), dimension(1:n), intent(in) :: xx
     real(rkind) :: x
     integer :: il,jm,juu
!
     il=0
     juu=n+1
     do while (juu-il.gt.1)
      jm=(juu+il)/2
      if ((xx(n).gt.xx(1)).eqv.(x.gt.xx(jm))) then
       il=jm
      else
       juu=jm
      endif
     end do
     ii=il
    end subroutine locateval
!
    subroutine fail_input_any(msg)
    integer :: mpi_err
    character(len=*) :: msg
    write(error_unit,*) "Input Error! ", msg
    call mpi_abort(mpi_comm_world,15,mpi_err)
    endsubroutine fail_input_any
!
    subroutine invmat(mat,n)
!
    integer, intent(in) :: n
    real(rkind), dimension(n,n), intent(inout) :: mat
    integer :: i,j
    integer, dimension(n) :: indx
    real(rkind), dimension(n,n) :: y
    real(rkind) :: d
!
    y = 0._rkind
    do i=1,n
     y(i,i) = 1._rkind
    enddo
    call ludcmp(mat,n,n,indx,d)
    do j=1,n
     call lubksb(mat,n,n,indx,y(1,j))
    enddo
    mat = y
!
    end subroutine invmat
!
      SUBROUTINE LUDCMP(A,N,NP,INDX,D)
      integer, parameter :: NMAX = 100
      real(rkind), parameter :: TINY = 1.0D-20
      integer, intent(in) :: N,NP
      integer, dimension(N), intent(out) :: INDX
      real(rkind), intent(out) :: D
      real(rkind), dimension(NP,NP), intent(inout) :: A
      real(rkind), dimension(NMAX) :: VV
      integer :: I,J,K,IMAX
      real(rkind) :: SUM,DUM,AAMAX
!
      D=1._rkind
      DO 12 I=1,N
        AAMAX=0._rkind
        DO 11 J=1,N
          IF (ABS(A(I,J)).GT.AAMAX) AAMAX=ABS(A(I,J))
11      CONTINUE
        IF (AAMAX.EQ.0._rkind) write(*,*) 'Error, singular matrix.'
        VV(I)=1./AAMAX
12    CONTINUE
      DO 19 J=1,N
        IF (J.GT.1) THEN
          DO 14 I=1,J-1
            SUM=A(I,J)
            IF (I.GT.1)THEN
              DO 13 K=1,I-1
                SUM=SUM-A(I,K)*A(K,J)
13            CONTINUE
              A(I,J)=SUM
            ENDIF
14        CONTINUE
        ENDIF
        AAMAX=0.
        DO 16 I=J,N
          SUM=A(I,J)
          IF (J.GT.1)THEN
            DO 15 K=1,J-1
              SUM=SUM-A(I,K)*A(K,J)
15          CONTINUE
            A(I,J)=SUM
          ENDIF
          DUM=VV(I)*ABS(SUM)
          IF (DUM.GE.AAMAX) THEN
            IMAX=I
            AAMAX=DUM
          ENDIF
16      CONTINUE
        IF (J.NE.IMAX)THEN
          DO 17 K=1,N
            DUM=A(IMAX,K)
            A(IMAX,K)=A(J,K)
            A(J,K)=DUM
17        CONTINUE
          D=-D
          VV(IMAX)=VV(J)
        ENDIF
        INDX(J)=IMAX
        IF(J.NE.N)THEN
          IF(A(J,J).EQ.0._rkind)A(J,J)=TINY
          DUM=1./A(J,J)
          DO 18 I=J+1,N
            A(I,J)=A(I,J)*DUM
18        CONTINUE
        ENDIF
19    CONTINUE
      IF(A(N,N).EQ.0._rkind)A(N,N)=TINY
      RETURN
      END
!
      SUBROUTINE LUBKSB(A,N,NP,INDX,B)
      integer, intent(in) :: N,NP
      integer, dimension(N), intent(in) :: INDX
      real(rkind), dimension(NP,NP), intent(in) :: A
      real(rkind), dimension(N), intent(inout) :: B
      integer :: II,I,LL,J
      real(rkind) :: SUM
      II=0
      DO 12 I=1,N
        LL=INDX(I)
        SUM=B(LL)
        B(LL)=B(I)
        IF (II.NE.0)THEN
          DO 11 J=II,I-1
            SUM=SUM-A(I,J)*B(J)
11        CONTINUE
        ELSE IF (SUM.NE.0.) THEN
          II=I
        ENDIF
        B(I)=SUM
12    CONTINUE
      DO 14 I=N,1,-1
        SUM=B(I)
        IF(I.LT.N)THEN
          DO 13 J=I+1,N
            SUM=SUM-A(I,J)*B(J)
13        CONTINUE
        ENDIF
        B(I)=SUM/A(I,I)
14    CONTINUE
      RETURN
      END
!
      subroutine detmat(mat,n,det)
!
       integer, intent(in) :: n
       real(rkind), dimension(n,n), intent(inout) :: mat
       real(rkind), intent(out) :: det
       integer :: j
       integer, dimension(n) :: indx
       real(rkind) :: d
!
       call ludcmp(mat,n,n,indx,d)
       do j=1,n
        d = d*mat(j,j)
       enddo
       det = d
!
      end subroutine detmat
!
      subroutine insitu_start(fcoproc,vtkpipeline,masterproc)
       character(len=*), intent(in) :: vtkpipeline
       logical, intent(in) :: masterproc
       logical, intent(inout) :: fcoproc
!
       inquire(file=vtkpipeline, exist=fcoproc)
      
       if (fcoproc) then
        if (masterproc) print*, 'Connecting to Catalyst...'
        if (masterproc) print*, 'Adding '//vtkpipeline
        CALL coprocessorinitializewithpython(vtkpipeline,len(vtkpipeline)) ! Initialize Catalyst
       else
        if (masterproc) print*, 'WARNING: ', vtkpipeline, ' is missing'
       endif
!
      end subroutine insitu_start
      
      subroutine insitu_end(fcoproc)
       logical, intent(in) :: fcoproc
!
       if (fcoproc) CALL coprocessorfinalize() ! Finalize Catalyst
!
      end subroutine insitu_end

      elemental function byte_size_int32(x) result(bytes)
         integer, intent(in) :: x
         integer(ikind64)    :: bytes

         bytes = storage_size(x) / 8
      endfunction byte_size_int32
#ifdef SINGLE_PRECISION
      elemental function byte_size_real32(x) result(bytes)
         real(rkind), intent(in) :: x
         integer(ikind64)        :: bytes

         bytes = storage_size(x) / 8
      endfunction byte_size_real32
#else
      elemental function byte_size_real64(x) result(bytes)
         real(rkind), intent(in) :: x
         integer(ikind64)        :: bytes

         bytes = storage_size(x) / 8
      endfunction byte_size_real64
#endif

    subroutine mpi_cart_shift_general(mp_cart, shifts, rank_source, rank_dest)
        integer, intent(in) :: mp_cart, rank_source
        integer, intent(out) :: rank_dest
        integer, dimension(3), intent(in) :: shifts
        integer, dimension(3) :: ncoords_source, ncoords_dest, dims, coords
        logical, dimension(3) :: pbc
        integer :: ierr, i

        ! get communicator features
        call mpi_cart_get(mp_cart, 3, dims, pbc, coords, ierr)
!       print*,'Periodicity active on directions: ',pbc
        ! get coordinates of rank_source
        call mpi_cart_coords(mp_cart, rank_source, 3, ncoords_source, ierr)
!       print*,'coords/my_coords should be equal: ',coords, ncoords_source

        ncoords_dest(1:3) = ncoords_source(1:3) + shifts(1:3)
        do i=1,3
            if (pbc(i)) then
                if (ncoords_dest(i)==-1) ncoords_dest(i) = dims(i)-1
                if (ncoords_dest(i)==dims(i)) ncoords_dest(i) = 0
            endif
        enddo
        if(any(ncoords_dest(1:3)==-1) .or. any(ncoords_dest(1:3)==dims(1:3))) then
            rank_dest = mpi_proc_null
        else
            call mpi_cart_rank(mp_cart,ncoords_dest,rank_dest,ierr)
        endif
    endsubroutine mpi_cart_shift_general

    function create_folder(foldername)
      character(len=*) :: foldername
      character(len=256, kind=c_char) :: foldername_c
      integer, dimension(3) :: permissions=[7,5,5]
      integer :: create_folder, mode_io
      foldername_c = foldername//c_null_char
      mode_io = permissions(1)*8**2+permissions(2)*8+permissions(3)
      create_folder = mkdir_wrapper(foldername_c, mode_io)
    endfunction create_folder
!
    subroutine get_abcstar_tables(delta,tstar,astar_table,bstar_table,cstar_table)
        real(rkind), intent(out), dimension(8)    :: delta
        real(rkind), dimension(37)   :: tstar22
        real(rkind), intent(out), dimension(39)   :: tstar
        real(rkind), dimension(37,8) :: omega22_table
        real(rkind), intent(out), dimension(39,8) :: astar_table,bstar_table,cstar_table

        delta = (/0.0_rkind,0.25_rkind,0.50_rkind,0.75_rkind,1.0_rkind,1.5_rkind,2.0_rkind,2.5_rkind/)

        tstar22 = (/ &
          0.1_rkind,0.2_rkind,0.3_rkind,0.4_rkind,0.5_rkind,0.6_rkind,0.7_rkind,0.8_rkind,0.9_rkind,1.0_rkind, &
          1.2_rkind,1.4_rkind,1.6_rkind,1.8_rkind,2.0_rkind,2.5_rkind,3.0_rkind,3.5_rkind,4.0_rkind, &
          5.0_rkind,6.0_rkind,7.0_rkind,8.0_rkind,9.0_rkind,10.0_rkind,12.0_rkind,14.0_rkind,16.0_rkind, &
          18.0_rkind,20.0_rkind,25.0_rkind,30.0_rkind,35.0_rkind,40.0_rkind, 50.0_rkind,75.0_rkind,100.0_rkind /)

        omega22_table = reshape((/ &
         4.1005_rkind,  3.2626_rkind,  2.8399_rkind,   2.531_rkind,  2.2837_rkind,  2.0838_rkind,   1.922_rkind,  1.7902_rkind,  &
         1.6823_rkind,  1.5929_rkind,  1.4551_rkind,  1.3551_rkind,    1.28_rkind,  1.2219_rkind,  1.1757_rkind,  1.0933_rkind,  &
         1.0388_rkind, 0.99963_rkind, 0.96988_rkind, 0.92676_rkind, 0.89616_rkind, 0.87272_rkind, 0.85379_rkind, 0.83795_rkind,  &
        0.82435_rkind, 0.80184_rkind, 0.78363_rkind, 0.76834_rkind, 0.75518_rkind, 0.74364_rkind, 0.71982_rkind, 0.70097_rkind,  &
        0.68545_rkind, 0.67232_rkind, 0.65099_rkind, 0.61397_rkind,  0.5887_rkind,   4.266_rkind,   3.305_rkind,   2.836_rkind,  &
          2.522_rkind,   2.277_rkind,   2.081_rkind,   1.924_rkind,   1.795_rkind,   1.689_rkind,   1.601_rkind,   1.465_rkind,  &
          1.365_rkind,   1.289_rkind,   1.231_rkind,   1.184_rkind,     1.1_rkind,   1.044_rkind,   1.004_rkind,  0.9732_rkind,  &
         0.9291_rkind,  0.8979_rkind,  0.8741_rkind,  0.8549_rkind,  0.8388_rkind,  0.8251_rkind,  0.8024_rkind,   0.784_rkind,  &
         0.7687_rkind,  0.7554_rkind,  0.7438_rkind,    0.72_rkind,  0.7011_rkind,  0.6855_rkind,  0.6724_rkind,   0.651_rkind,  &
         0.6141_rkind,  0.5889_rkind,   4.833_rkind,   3.516_rkind,   2.936_rkind,   2.586_rkind,   2.329_rkind,    2.13_rkind,  &
           1.97_rkind,    1.84_rkind,   1.733_rkind,   1.644_rkind,   1.504_rkind,     1.4_rkind,   1.321_rkind,   1.259_rkind,  &
          1.209_rkind,   1.119_rkind,   1.059_rkind,   1.016_rkind,   0.983_rkind,   0.936_rkind,   0.903_rkind,   0.878_rkind,  &
          0.858_rkind,  0.8414_rkind,  0.8273_rkind,  0.8039_rkind,  0.7852_rkind,  0.7696_rkind,  0.7562_rkind,  0.7445_rkind,  &
         0.7204_rkind,  0.7014_rkind,  0.6858_rkind,  0.6726_rkind,  0.6512_rkind,  0.6143_rkind,  0.5894_rkind,   5.742_rkind,  &
          3.914_rkind,   3.168_rkind,   2.749_rkind,    2.46_rkind,   2.243_rkind,   2.072_rkind,   1.934_rkind,    1.82_rkind,  &
          1.725_rkind,   1.574_rkind,   1.461_rkind,   1.374_rkind,   1.306_rkind,   1.251_rkind,    1.15_rkind,   1.083_rkind,  &
          1.035_rkind,  0.9991_rkind,  0.9473_rkind,  0.9114_rkind,  0.8845_rkind,  0.8632_rkind,  0.8456_rkind,  0.8308_rkind,  &
         0.8065_rkind,  0.7872_rkind,  0.7712_rkind,  0.7575_rkind,  0.7455_rkind,  0.7211_rkind,  0.7019_rkind,  0.6861_rkind,  &
         0.6728_rkind,  0.6513_rkind,  0.6145_rkind,    0.59_rkind,   6.729_rkind,   4.433_rkind,   3.511_rkind,   3.004_rkind,  &
          2.665_rkind,   2.417_rkind,   2.225_rkind,    2.07_rkind,   1.944_rkind,   1.838_rkind,    1.67_rkind,   1.544_rkind,  &
          1.447_rkind,    1.37_rkind,   1.307_rkind,   1.193_rkind,   1.117_rkind,   1.062_rkind,   1.021_rkind,  0.9628_rkind,  &
          0.923_rkind,  0.8935_rkind,  0.8703_rkind,  0.8515_rkind,  0.8356_rkind,  0.8101_rkind,  0.7899_rkind,  0.7733_rkind,  &
         0.7592_rkind,   0.747_rkind,  0.7221_rkind,  0.7026_rkind,  0.6867_rkind,  0.6733_rkind,  0.6516_rkind,  0.6147_rkind,  &
         0.5903_rkind,   8.624_rkind,    5.57_rkind,   4.329_rkind,    3.64_rkind,   3.187_rkind,   2.862_rkind,   2.614_rkind,  &
          2.417_rkind,   2.258_rkind,   2.124_rkind,   1.913_rkind,   1.754_rkind,    1.63_rkind,   1.532_rkind,   1.451_rkind,  &
          1.304_rkind,   1.204_rkind,   1.133_rkind,   1.079_rkind,   1.005_rkind,  0.9545_rkind,  0.9181_rkind,  0.8901_rkind,  &
         0.8678_rkind,  0.8493_rkind,  0.8201_rkind,  0.7976_rkind,  0.7794_rkind,  0.7642_rkind,  0.7512_rkind,   0.725_rkind,  &
         0.7047_rkind,  0.6883_rkind,  0.6743_rkind,  0.6524_rkind,  0.6148_rkind,  0.5901_rkind,   10.34_rkind,   6.637_rkind,  &
          5.126_rkind,   4.282_rkind,   3.727_rkind,   3.329_rkind,   3.028_rkind,   2.788_rkind,   2.596_rkind,   2.435_rkind,  &
          2.181_rkind,   1.989_rkind,   1.838_rkind,   1.718_rkind,   1.618_rkind,   1.435_rkind,    1.31_rkind,    1.22_rkind,  &
          1.153_rkind,   1.058_rkind,  0.9955_rkind,  0.9505_rkind,  0.9164_rkind,  0.8895_rkind,  0.8676_rkind,  0.8337_rkind,  &
         0.8081_rkind,  0.7878_rkind,  0.7711_rkind,  0.7569_rkind,  0.7289_rkind,  0.7076_rkind,  0.6905_rkind,  0.6762_rkind,  &
         0.6534_rkind,  0.6148_rkind,  0.5895_rkind,   11.89_rkind,   7.618_rkind,   5.874_rkind,   4.895_rkind,   4.249_rkind,  &
          3.786_rkind,   3.435_rkind,   3.156_rkind,   2.933_rkind,   2.746_rkind,   2.451_rkind,   2.228_rkind,   2.053_rkind,  &
          1.912_rkind,   1.795_rkind,   1.578_rkind,   1.428_rkind,   1.319_rkind,   1.236_rkind,   1.121_rkind,   1.044_rkind,  &
         0.9893_rkind,  0.9482_rkind,   0.916_rkind,  0.8901_rkind,  0.8504_rkind,  0.8212_rkind,  0.7983_rkind,  0.7797_rkind,  &
         0.7642_rkind,  0.7339_rkind,  0.7112_rkind,  0.6932_rkind,  0.6784_rkind,  0.6546_rkind,  0.6147_rkind,  0.5885_rkind &
        /), (/37,8/))


        tstar = (/ &
            0.0_rkind, 0.1_rkind, 0.2_rkind, 0.3_rkind, 0.4_rkind, 0.5_rkind, 0.6_rkind, 0.7_rkind, 0.8_rkind, 0.9_rkind, 1.0_rkind, &
            1.2_rkind, 1.4_rkind, 1.6_rkind, 1.8_rkind, 2.0_rkind, 2.5_rkind, 3.0_rkind, 3.5_rkind, 4.0_rkind, &
            5.0_rkind, 6.0_rkind, 7.0_rkind, 8.0_rkind, 9.0_rkind, 10.0_rkind, 12.0_rkind, 14.0_rkind, 16.0_rkind, &
            18.0_rkind, 20.0_rkind, 25.0_rkind, 30.0_rkind, 35.0_rkind, 40.0_rkind, 50.0_rkind, 75.0_rkind, 100.0_rkind, 500.0_rkind /)

       astar_table = reshape((/ &
        1.0065_rkind,  1.0231_rkind,  1.0424_rkind,  1.0719_rkind,  1.0936_rkind,  1.1053_rkind,  1.1104_rkind,  1.1114_rkind,  &
        1.1104_rkind,  1.1086_rkind,  1.1063_rkind,  1.1020_rkind,  1.0985_rkind,  1.0960_rkind,  1.0943_rkind,  1.0934_rkind,  &
        1.0926_rkind,  1.0934_rkind,  1.0948_rkind,  1.0965_rkind,  1.0997_rkind,  1.1025_rkind,  1.1050_rkind,  1.1072_rkind,  &
        1.1091_rkind,  1.1107_rkind,  1.1133_rkind,  1.1154_rkind,  1.1172_rkind,  1.1186_rkind,  1.1199_rkind,  1.1223_rkind,  &
        1.1243_rkind,  1.1259_rkind,  1.1273_rkind,  1.1297_rkind,  1.1339_rkind,  1.1364_rkind, 1.14187_rkind,  1.0840_rkind,  &
        1.0660_rkind,  1.0450_rkind,  1.0670_rkind,  1.0870_rkind,  1.0980_rkind,  1.1040_rkind,  1.1070_rkind,  1.1070_rkind,  &
        1.1060_rkind,  1.1040_rkind,  1.1020_rkind,  1.0990_rkind,  1.0960_rkind,  1.0950_rkind,  1.0940_rkind,  1.0940_rkind,  &
        1.0950_rkind,  1.0960_rkind,  1.0970_rkind,  1.1000_rkind,  1.1030_rkind,  1.1050_rkind,  1.1070_rkind,  1.1090_rkind,  &
        1.1110_rkind,  1.1140_rkind,  1.1150_rkind,  1.1170_rkind,  1.1190_rkind,  1.1200_rkind,  1.1220_rkind,  1.1240_rkind,  &
        1.1260_rkind,  1.1270_rkind,  1.1300_rkind,  1.1340_rkind,  1.1370_rkind, 1.14187_rkind,  1.0840_rkind,  1.0380_rkind,  &
        1.0480_rkind,  1.0600_rkind,  1.0770_rkind,  1.0880_rkind,  1.0960_rkind,  1.1000_rkind,  1.1020_rkind,  1.1020_rkind,  &
        1.1030_rkind,  1.1030_rkind,  1.1010_rkind,  1.0990_rkind,  1.0990_rkind,  1.0970_rkind,  1.0970_rkind,  1.0970_rkind,  &
        1.0980_rkind,  1.0990_rkind,  1.1010_rkind,  1.1040_rkind,  1.1060_rkind,  1.1080_rkind,  1.1090_rkind,  1.1110_rkind,  &
        1.1130_rkind,  1.1160_rkind,  1.1170_rkind,  1.1190_rkind,  1.1200_rkind,  1.1220_rkind,  1.1240_rkind,  1.1260_rkind,  &
        1.1270_rkind,  1.1300_rkind,  1.1340_rkind,  1.1370_rkind, 1.14187_rkind,  1.0840_rkind,  1.0400_rkind,  1.0520_rkind,  &
        1.0550_rkind,  1.0690_rkind,  1.0800_rkind,  1.0890_rkind,  1.0950_rkind,  1.0990_rkind,  1.1010_rkind,  1.1030_rkind,  &
        1.1050_rkind,  1.1040_rkind,  1.1030_rkind,  1.1020_rkind,  1.1020_rkind,  1.0990_rkind,  1.0990_rkind,  1.1000_rkind,  &
        1.1010_rkind,  1.1020_rkind,  1.1050_rkind,  1.1070_rkind,  1.1080_rkind,  1.1100_rkind,  1.1110_rkind,  1.1140_rkind,  &
        1.1160_rkind,  1.1180_rkind,  1.1190_rkind,  1.1200_rkind,  1.1220_rkind,  1.1240_rkind,  1.1260_rkind,  1.1270_rkind,  &
        1.1300_rkind,  1.1350_rkind,  1.1380_rkind, 1.14187_rkind,  1.0840_rkind,  1.0430_rkind,  1.0560_rkind,  1.0580_rkind,  &
        1.0680_rkind,  1.0780_rkind,  1.0860_rkind,  1.0930_rkind,  1.0980_rkind,  1.1010_rkind,  1.1040_rkind,  1.1070_rkind,  &
        1.1080_rkind,  1.1080_rkind,  1.1080_rkind,  1.1070_rkind,  1.1050_rkind,  1.1040_rkind,  1.1030_rkind,  1.1040_rkind,  &
        1.1050_rkind,  1.1060_rkind,  1.1080_rkind,  1.1090_rkind,  1.1110_rkind,  1.1120_rkind,  1.1140_rkind,  1.1160_rkind,  &
        1.1180_rkind,  1.1190_rkind,  1.1200_rkind,  1.1220_rkind,  1.1240_rkind,  1.1260_rkind,  1.1270_rkind,  1.1300_rkind,  &
        1.1350_rkind,  1.1390_rkind, 1.14187_rkind,  1.0840_rkind,  1.0500_rkind,  1.0650_rkind,  1.0680_rkind,  1.0750_rkind,  &
        1.0820_rkind,  1.0890_rkind,  1.0950_rkind,  1.1000_rkind,  1.1050_rkind,  1.1080_rkind,  1.1120_rkind,  1.1150_rkind,  &
        1.1160_rkind,  1.1170_rkind,  1.1160_rkind,  1.1150_rkind,  1.1130_rkind,  1.1120_rkind,  1.1100_rkind,  1.1100_rkind,  &
        1.1100_rkind,  1.1110_rkind,  1.1120_rkind,  1.1130_rkind,  1.1140_rkind,  1.1150_rkind,  1.1170_rkind,  1.1180_rkind,  &
        1.1190_rkind,  1.1210_rkind,  1.1230_rkind,  1.1240_rkind,  1.1260_rkind,  1.1270_rkind,  1.1300_rkind,  1.1340_rkind,  &
        1.1380_rkind, 1.14187_rkind,  1.0840_rkind,  1.0520_rkind,  1.0660_rkind,  1.0710_rkind,  1.0780_rkind,  1.0840_rkind,  &
        1.0900_rkind,  1.0960_rkind,  1.1000_rkind,  1.1050_rkind,  1.1090_rkind,  1.1150_rkind,  1.1190_rkind,  1.1210_rkind,  &
        1.1230_rkind,  1.1230_rkind,  1.1230_rkind,  1.1220_rkind,  1.1190_rkind,  1.1180_rkind,  1.1160_rkind,  1.1150_rkind,  &
        1.1150_rkind,  1.1150_rkind,  1.1150_rkind,  1.1160_rkind,  1.1170_rkind,  1.1180_rkind,  1.1190_rkind,  1.1200_rkind,  &
        1.1210_rkind,  1.1230_rkind,  1.1250_rkind,  1.1260_rkind,  1.1270_rkind,  1.1300_rkind,  1.1340_rkind,  1.1370_rkind,  &
       1.14187_rkind,  1.0840_rkind,  1.0510_rkind,  1.0640_rkind,  1.0710_rkind,  1.0780_rkind,  1.0840_rkind,  1.0900_rkind,  &
        1.0950_rkind,  1.0990_rkind,  1.1040_rkind,  1.1080_rkind,  1.1150_rkind,  1.1200_rkind,  1.1240_rkind,  1.1260_rkind,  &
        1.1280_rkind,  1.1300_rkind,  1.1290_rkind,  1.1270_rkind,  1.1260_rkind,  1.1230_rkind,  1.1210_rkind,  1.1200_rkind,  &
        1.1190_rkind,  1.1190_rkind,  1.1190_rkind,  1.1190_rkind,  1.1200_rkind,  1.1200_rkind,  1.1210_rkind,  1.1220_rkind,  &
        1.1240_rkind,  1.1250_rkind,  1.1260_rkind,  1.1280_rkind,  1.1290_rkind,  1.1320_rkind,  1.1350_rkind, 1.14187_rkind &
       /), (/39,8/))

       bstar_table = reshape((/ &
        1.1852_rkind,  1.1960_rkind,  1.2451_rkind,  1.2900_rkind,  1.2986_rkind,  1.2865_rkind,  1.2665_rkind,  1.2455_rkind,  &
        1.2253_rkind,  1.2078_rkind,  1.1919_rkind,  1.1678_rkind,  1.1496_rkind,  1.1366_rkind,  1.1270_rkind,  1.1197_rkind,  &
        1.1080_rkind,  1.1016_rkind,  1.0980_rkind,  1.0958_rkind,  1.0935_rkind,  1.0925_rkind,  1.0922_rkind,  1.0922_rkind,  &
        1.0923_rkind,  1.0923_rkind,  1.0927_rkind,  1.0930_rkind,  1.0933_rkind,  1.0937_rkind,  1.0939_rkind,  1.0943_rkind,  &
        1.0944_rkind,  1.0944_rkind,  1.0943_rkind,  1.0941_rkind,  1.0947_rkind,  1.0957_rkind, 1.10185_rkind,  1.2963_rkind,  &
         1.216_rkind,   1.257_rkind,   1.294_rkind,   1.291_rkind,   1.281_rkind,   1.264_rkind,   1.244_rkind,   1.225_rkind,  &
         1.210_rkind,   1.192_rkind,   1.172_rkind,   1.155_rkind,   1.141_rkind,   1.130_rkind,   1.122_rkind,   1.110_rkind,  &
         1.103_rkind,   1.099_rkind,   1.097_rkind,   1.094_rkind,   1.092_rkind,   1.092_rkind,   1.092_rkind,   1.092_rkind,  &
         1.092_rkind,   1.093_rkind,   1.093_rkind,   1.094_rkind,   1.093_rkind,   1.094_rkind,   1.094_rkind,   1.095_rkind,  &
         1.094_rkind,   1.095_rkind,   1.094_rkind,   1.095_rkind,   1.095_rkind, 1.10185_rkind,  1.2963_rkind,   1.237_rkind,  &
         1.340_rkind,   1.272_rkind,   1.284_rkind,   1.276_rkind,   1.261_rkind,   1.248_rkind,   1.234_rkind,   1.216_rkind,  &
         1.205_rkind,   1.181_rkind,   1.161_rkind,   1.147_rkind,   1.138_rkind,   1.129_rkind,   1.116_rkind,   1.107_rkind,  &
         1.102_rkind,   1.099_rkind,   1.095_rkind,   1.094_rkind,   1.093_rkind,   1.093_rkind,   1.093_rkind,   1.092_rkind,  &
         1.093_rkind,   1.093_rkind,   1.093_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind,   1.095_rkind,  &
         1.094_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind, 1.10185_rkind,  1.2963_rkind,   1.269_rkind,   1.389_rkind,  &
         1.258_rkind,   1.278_rkind,   1.272_rkind,   1.263_rkind,   1.255_rkind,   1.240_rkind,   1.227_rkind,   1.216_rkind,  &
         1.195_rkind,   1.174_rkind,   1.159_rkind,   1.148_rkind,   1.140_rkind,   1.122_rkind,   1.112_rkind,   1.106_rkind,  &
         1.102_rkind,   1.097_rkind,   1.095_rkind,   1.094_rkind,   1.093_rkind,   1.093_rkind,   1.093_rkind,   1.093_rkind,  &
         1.093_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind,  &
         1.094_rkind,   1.094_rkind,   1.093_rkind, 1.10185_rkind,  1.2963_rkind,   1.285_rkind,   1.366_rkind,   1.262_rkind,  &
         1.277_rkind,   1.277_rkind,   1.269_rkind,   1.262_rkind,   1.252_rkind,   1.242_rkind,   1.230_rkind,   1.209_rkind,  &
         1.189_rkind,   1.174_rkind,   1.162_rkind,   1.149_rkind,   1.132_rkind,   1.120_rkind,   1.112_rkind,   1.107_rkind,  &
         1.100_rkind,   1.098_rkind,   1.096_rkind,   1.095_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind,   1.094_rkind,  &
         1.094_rkind,   1.094_rkind,   1.094_rkind,   1.095_rkind,   1.094_rkind,   1.094_rkind,   1.095_rkind,   1.094_rkind,  &
         1.093_rkind,   1.092_rkind, 1.10185_rkind,  1.2963_rkind,   1.290_rkind,   1.327_rkind,   1.282_rkind,   1.288_rkind,  &
         1.286_rkind,   1.284_rkind,   1.278_rkind,   1.271_rkind,   1.264_rkind,   1.256_rkind,   1.237_rkind,   1.221_rkind,  &
         1.202_rkind,   1.191_rkind,   1.178_rkind,   1.154_rkind,   1.138_rkind,   1.127_rkind,   1.119_rkind,   1.109_rkind,  &
         1.104_rkind,   1.100_rkind,   1.098_rkind,   1.097_rkind,   1.096_rkind,   1.095_rkind,   1.094_rkind,   1.095_rkind,  &
         1.094_rkind,   1.095_rkind,   1.095_rkind,   1.095_rkind,   1.095_rkind,   1.095_rkind,   1.094_rkind,   1.093_rkind,  &
         1.093_rkind, 1.10185_rkind,  1.2963_rkind,   1.297_rkind,   1.314_rkind,   1.290_rkind,   1.294_rkind,   1.292_rkind,  &
         1.292_rkind,   1.289_rkind,   1.284_rkind,   1.281_rkind,   1.273_rkind,   1.261_rkind,   1.246_rkind,   1.231_rkind,  &
         1.218_rkind,   1.205_rkind,   1.180_rkind,   1.160_rkind,   1.145_rkind,   1.135_rkind,   1.120_rkind,   1.112_rkind,  &
         1.106_rkind,   1.103_rkind,   1.101_rkind,   1.099_rkind,   1.098_rkind,   1.096_rkind,   1.096_rkind,   1.096_rkind,  &
         1.095_rkind,   1.096_rkind,   1.095_rkind,   1.096_rkind,   1.095_rkind,   1.094_rkind,   1.094_rkind,   1.093_rkind,  &
       1.10185_rkind,  1.2963_rkind,   1.294_rkind,   1.278_rkind,   1.299_rkind,   1.297_rkind,   1.298_rkind,   1.298_rkind,  &
         1.296_rkind,   1.295_rkind,   1.292_rkind,   1.287_rkind,   1.277_rkind,   1.266_rkind,   1.256_rkind,   1.242_rkind,  &
         1.231_rkind,   1.205_rkind,   1.183_rkind,   1.165_rkind,   1.153_rkind,   1.134_rkind,   1.122_rkind,   1.115_rkind,  &
         1.110_rkind,   1.106_rkind,   1.103_rkind,   1.101_rkind,   1.099_rkind,   1.098_rkind,   1.097_rkind,   1.097_rkind,  &
         1.096_rkind,   1.096_rkind,   1.096_rkind,   1.095_rkind,   1.096_rkind,   1.095_rkind,   1.094_rkind, 1.10185_rkind &
       /), (/39,8/))

       cstar_table = reshape((/ &
        0.8889_rkind, 0.88575_rkind, 0.87268_rkind, 0.85182_rkind, 0.83542_rkind, 0.82629_rkind, 0.82299_rkind, 0.82357_rkind,  &
       0.82657_rkind,  0.8311_rkind,  0.8363_rkind, 0.84762_rkind, 0.85846_rkind,  0.8684_rkind, 0.87713_rkind, 0.88479_rkind,  &
       0.89972_rkind, 0.91028_rkind, 0.91793_rkind, 0.92371_rkind, 0.93135_rkind, 0.93607_rkind, 0.93927_rkind, 0.94149_rkind,  &
       0.94306_rkind, 0.94419_rkind, 0.94571_rkind, 0.94662_rkind, 0.94723_rkind, 0.94764_rkind,  0.9479_rkind, 0.94827_rkind,  &
       0.94842_rkind, 0.94852_rkind, 0.94861_rkind, 0.94872_rkind, 0.94881_rkind, 0.94863_rkind, 0.94444_rkind, 0.77778_rkind,  &
        0.8988_rkind,  0.8692_rkind,  0.8525_rkind,  0.8362_rkind,  0.8278_rkind,  0.8249_rkind,  0.8257_rkind,   0.828_rkind,  &
        0.8234_rkind,  0.8366_rkind,  0.8474_rkind,  0.8583_rkind,  0.8674_rkind,  0.8755_rkind,  0.8831_rkind,  0.8986_rkind,  &
        0.9089_rkind,  0.9166_rkind,  0.9226_rkind,  0.9304_rkind,  0.9353_rkind,  0.9387_rkind,  0.9409_rkind,  0.9426_rkind,  &
        0.9437_rkind,  0.9455_rkind,  0.9464_rkind,  0.9471_rkind,  0.9474_rkind,  0.9478_rkind,  0.9481_rkind,  0.9484_rkind,  &
        0.9484_rkind,  0.9487_rkind,  0.9486_rkind,  0.9488_rkind,  0.9487_rkind, 0.94444_rkind, 0.77778_rkind,  0.8378_rkind,  &
        0.8647_rkind,  0.8366_rkind,  0.8306_rkind,  0.8252_rkind,   0.823_rkind,  0.8241_rkind,  0.8264_rkind,  0.8295_rkind,  &
        0.8342_rkind,  0.8438_rkind,   0.853_rkind,  0.8619_rkind,  0.8709_rkind,  0.8779_rkind,  0.8936_rkind,  0.9043_rkind,  &
        0.9125_rkind,  0.9189_rkind,  0.9274_rkind,  0.9329_rkind,  0.9366_rkind,  0.9393_rkind,  0.9412_rkind,  0.9425_rkind,  &
        0.9445_rkind,  0.9456_rkind,  0.9464_rkind,  0.9469_rkind,  0.9474_rkind,   0.948_rkind,  0.9481_rkind,  0.9483_rkind,  &
        0.9484_rkind,  0.9486_rkind,  0.9489_rkind,  0.9489_rkind, 0.94444_rkind, 0.77778_rkind,  0.8029_rkind,  0.8479_rkind,  &
        0.8198_rkind,  0.8196_rkind,  0.8169_rkind,  0.8165_rkind,  0.8178_rkind,  0.8199_rkind,  0.8228_rkind,  0.8267_rkind,  &
        0.8358_rkind,  0.8444_rkind,  0.8531_rkind,  0.8616_rkind,  0.8695_rkind,  0.8846_rkind,  0.8967_rkind,  0.9058_rkind,  &
        0.9128_rkind,  0.9226_rkind,  0.9291_rkind,  0.9334_rkind,  0.9366_rkind,  0.9388_rkind,  0.9406_rkind,   0.943_rkind,  &
        0.9444_rkind,  0.9455_rkind,  0.9462_rkind,  0.9465_rkind,  0.9472_rkind,  0.9478_rkind,   0.948_rkind,  0.9481_rkind,  &
        0.9483_rkind,   0.949_rkind,  0.9491_rkind, 0.94444_rkind, 0.77778_rkind,  0.7876_rkind,  0.8237_rkind,  0.8054_rkind,  &
        0.8076_rkind,  0.8074_rkind,  0.8072_rkind,  0.8084_rkind,  0.8107_rkind,  0.8136_rkind,  0.8168_rkind,   0.825_rkind,  &
        0.8336_rkind,  0.8423_rkind,  0.8504_rkind,  0.8578_rkind,  0.8742_rkind,  0.8869_rkind,   0.897_rkind,   0.905_rkind,  &
        0.9164_rkind,   0.924_rkind,  0.9292_rkind,  0.9331_rkind,  0.9357_rkind,   0.938_rkind,  0.9409_rkind,  0.9428_rkind,  &
        0.9442_rkind,   0.945_rkind,  0.9457_rkind,  0.9467_rkind,  0.9472_rkind,  0.9475_rkind,  0.9479_rkind,  0.9482_rkind,  &
        0.9487_rkind,  0.9493_rkind, 0.94444_rkind, 0.77778_rkind,  0.7805_rkind,  0.7975_rkind,  0.7903_rkind,  0.7918_rkind,  &
        0.7916_rkind,  0.7922_rkind,  0.7927_rkind,  0.7939_rkind,   0.796_rkind,  0.7986_rkind,  0.8041_rkind,  0.8118_rkind,  &
        0.8186_rkind,  0.8265_rkind,  0.8338_rkind,  0.8504_rkind,  0.8649_rkind,  0.8768_rkind,  0.8861_rkind,  0.9006_rkind,  &
        0.9109_rkind,  0.9162_rkind,  0.9236_rkind,  0.9276_rkind,  0.9308_rkind,  0.9353_rkind,  0.9382_rkind,  0.9405_rkind,  &
        0.9418_rkind,   0.943_rkind,  0.9447_rkind,  0.9458_rkind,  0.9465_rkind,  0.9468_rkind,  0.9475_rkind,  0.9482_rkind,  &
        0.9491_rkind, 0.94444_rkind, 0.77778_rkind,  0.7799_rkind,  0.7881_rkind,  0.7839_rkind,  0.7842_rkind,  0.7838_rkind,  &
        0.7839_rkind,  0.7839_rkind,  0.7842_rkind,  0.7854_rkind,  0.7864_rkind,  0.7904_rkind,  0.7957_rkind,  0.8011_rkind,  &
        0.8072_rkind,  0.8133_rkind,  0.8294_rkind,  0.8438_rkind,  0.8557_rkind,  0.8664_rkind,  0.8833_rkind,  0.8958_rkind,  &
         0.905_rkind,  0.9122_rkind,  0.9175_rkind,  0.9219_rkind,  0.9283_rkind,  0.9325_rkind,  0.9355_rkind,  0.9378_rkind,  &
        0.9394_rkind,  0.9422_rkind,  0.9437_rkind,  0.9449_rkind,  0.9455_rkind,  0.9464_rkind,  0.9476_rkind,  0.9483_rkind,  &
       0.94444_rkind, 0.77778_rkind,  0.7801_rkind,  0.7784_rkind,   0.782_rkind,  0.7806_rkind,  0.7802_rkind,  0.7798_rkind,  &
        0.7794_rkind,  0.7796_rkind,  0.7798_rkind,  0.7805_rkind,  0.7822_rkind,  0.7854_rkind,  0.7898_rkind,  0.7939_rkind,  &
         0.799_rkind,  0.8125_rkind,  0.8253_rkind,  0.8372_rkind,  0.8484_rkind,  0.8662_rkind,  0.8802_rkind,  0.8911_rkind,  &
        0.8997_rkind,  0.9065_rkind,  0.9119_rkind,  0.9201_rkind,  0.9258_rkind,  0.9298_rkind,  0.9328_rkind,  0.9352_rkind,  &
        0.9391_rkind,  0.9415_rkind,   0.943_rkind,   0.943_rkind,  0.9452_rkind,  0.9468_rkind,  0.9476_rkind, 0.94444_rkind &
       /), (/39,8/))

   endsubroutine get_abcstar_tables

    real(rkind) function poly6(x,c)
        real(rkind), intent(in) :: x
        real(rkind), dimension(7), intent(in) :: c

        poly6 = ((((((c(7)*x + c(6))*x + c(5))*x + c(4))*x + c(3))*x + c(2))*x + c(1))
    endfunction poly6
!
!--------------------------------------------------------
! Quadratic interpolation
! x0 : point to interpolate
! x(:), y(:) : 3 points arrays
!--------------------------------------------------------
    real(rkind) function quadInterp(x0, x, y)
        real(rkind), intent(in) :: x0
        real(rkind), intent(in) :: x(3), y(3)
        real(rkind) :: dx21, dx32, dx31
        real(rkind) :: dy32, dy21
        real(rkind) :: a

        dx21 = x(2) - x(1)
        dx32 = x(3) - x(2)
        dx31 = dx21 + dx32

        dy32 = y(3) - y(2)
        dy21 = y(2) - y(1)

        a = (dx21*dy32 - dy21*dx32)/(dx21*dx31*dx32)

        quadInterp = a*(x0 - x(1))*(x0 - x(2)) + (dy21/dx21)*(x0 - x(2)) + y(2)
    end function quadInterp
!
! Computes the coefficients of a polynomial of a given degree that fits 
! a set of data points (x, y) in a least-squares sense.
!
! INPUTS:
!   x       - real array of length n, independent variable data points
!   y       - real array of length n, dependent variable data points
!   n       - integer, number of data points
!   degree  - integer, desired polynomial degree (must be < n)
!
! OUTPUTS:
!   coeff   - real array of length degree+1, polynomial coefficients
!             coeff(1) + coeff(2)*x + coeff(3)*x^2 + ... + coeff(degree+1)*x^degree
!
    subroutine polyfit(x,y,n,degree,coeff)
        integer, intent(in) :: n       ! number of points
        integer, intent(in) :: degree  ! polynomial degree
        real(rkind), dimension(n), intent(in) :: x, y
        real(rkind), dimension(degree+1), intent(out) :: coeff

        real(rkind), dimension(n, degree+1) :: V
        real(rkind), dimension(degree+1, degree+1) :: A
        real(rkind), dimension(degree+1) :: b
        integer, dimension(degree+1) :: indx
        real(rkind) :: D
        integer :: i, j, k

        if (degree >= n) then
         print *, 'Error in polyfit: Polynomial degree (', degree, &
                  ') must be less than number of input data points (', n, ')'
         stop
        end if

        ! Vandermonde matrix ( n x (degree+1) )
        do i=1,n
         V(i,1) = 1._rkind
         do j=2,degree+1
          V(i,j) = x(i)**(j-1)
         enddo
        enddo
        ! build A=V^T * V, b=V^T * y
        A = matmul(transpose(V),V) ! (degree+1) x (degree+1)
        b = matmul(transpose(V),y) ! (degree+1)

        call ludcmp(A,degree+1,degree+1,indx,D)
        call lubksb(A,degree+1,degree+1,indx,b)

        coeff = b

    endsubroutine polyfit

endmodule streams_parameters

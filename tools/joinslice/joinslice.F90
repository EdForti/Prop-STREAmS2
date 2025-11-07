program slicejoin

      use, intrinsic :: iso_fortran_env
      use mpi
      implicit none

      integer, parameter :: rkind  = REAL64
      integer, parameter :: ikind = INT32
      integer, parameter :: rkinds = REAL32
      integer            :: nsl
      integer, parameter :: nxtot = 5120
      integer, parameter :: nytot =  300
      integer, parameter :: nztot =  320
      integer, parameter :: nbx =  4
      integer, parameter :: nby =  1
      integer, parameter :: nbz =  4
      integer, parameter :: nxb =  nxtot/nbx
      integer, parameter :: nyb =  nytot
      integer, parameter :: nzb =  nztot/nbz
      integer, parameter :: ng  = 3
      integer, dimension(:), allocatable :: numlist
      integer, dimension(:), allocatable :: goodnumlist
      character(3) :: chx,chy,chz
      character(6) :: chnumsl
      character(7) :: chnumsl7
      character(10) :: str
      character(12) :: chsl
      character(13) :: chsl13
      character(100)  :: filename
      integer :: ksize,kslice_num
      integer :: num_slicexy_record
      integer :: num_slicexz_record
      integer :: num_sliceyz_record
      integer :: nxmax,nymax,nzmax,jslicenum,jsize,nx,ny,nz,nv
      integer :: l,istatus,num_records
      integer :: i,j,k,ib,kb,js,ks,it
      real(rkinds) :: record_marker
      real(rkind) :: time, timeold
      real(rkind), dimension(1-ng:nxb+ng,1-ng:nyb+ng,1,17) :: slicexy_aux
      real(rkind), dimension(1-ng:nxb+ng,2,1-ng:nzb+ng,17) :: slicexz_aux
      integer, dimension(:), allocatable :: igslice,jgslice,kgslice 
      integer :: ierr,nrank,nproc,counter,icyc,fu_num,knum
      logical :: record_donexy, record_donexz, record_doneyz
      logical :: slicexy_exists, sliceyz_exists, slicexz_exists

      CALL MPI_INIT(ierr)
      CALL MPI_COMM_RANK(mpi_comm_world,nrank,ierr)
      CALL MPI_COMM_SIZE(mpi_comm_world,nproc,ierr)

      nrank = nrank+(nbx*nbz-nproc)
      ib    = nrank/nbz
      kb    = mod(nrank,nbz)
      write(chx,1003) ib
      write(chz,1003) kb


!     Use output of this command and copy paste to the numlist below:
!     du -hs SLICE* | awk '{print substr($2,7,12)" ,&"}' | grep -v FINAL | sed '$ s/,//' 
      numlist = [ &
        1331063, &
        1589934  &
                ]

      nsl = size(numlist)
      if (nrank==0) print*,'Processing ',nsl,' slices'
      allocate(goodnumlist(nsl))

      allocate(jgslice(nytot))
      allocate(kgslice(nztot))

 1006 format(I6.6)
 1007 format(I7.7)

!     do kb=0,nbz-1
!      do ib=nbx/2,nbx-1
!       write(chx,1003) ib
!
       slicexy_exists = .false.
       do l=1,nsl
        if (numlist(l)<1000000) then
           write(chnumsl,1006) numlist(l)
           chsl = 'SALVA_'//chnumsl
           filename = '../'//chsl//'/slicexy_'//chx//'_000_'//chz//'.bin'
           filename = trim(filename)
        else
           write(chnumsl7,1007) numlist(l)
           chsl13 = 'SALVA_'//chnumsl7
           filename = '../'//chsl13//'/slicexy_'//chx//'_000_'//chz//'.bin'
           filename = trim(filename)
        endif
        inquire(file=filename,exist=slicexy_exists)
       enddo
       if (slicexy_exists) then
        open(15,file='SLICE_FINAL/slicexy_'//chx//'_000_'//chz//'.bin', form='unformatted')
        do l=1,nsl
         timeold = 0.
         slicexy_exists = .false.
         if (nrank==0) write(*,*) '--------------------Start reading SALVA: ',numlist(l),'------------------------'
         if (numlist(l)<1000000) then
            write(chnumsl,1006) numlist(l)
            chsl = 'SALVA_'//chnumsl
            filename = '../'//chsl//'/slicexy_'//chx//'_000_'//chz//'.bin'
            filename = trim(filename)
         else
            write(chnumsl7,1007) numlist(l)
            chsl13 = 'SALVA_'//chnumsl7
            filename = '../'//chsl13//'/slicexy_'//chx//'_000_'//chz//'.bin'
            filename = trim(filename)
         endif
         inquire(file=filename,exist=slicexy_exists)
         if (slicexy_exists) then
          open(14,file=filename, action='read', form='unformatted')
          num_records = 0
          do 
           read(14,iostat=istatus) icyc, time
           if (istatus.ne.0) exit
           read(14,iostat=istatus) nxmax,nymax,nzmax
           read(14,iostat=istatus) ksize,(kgslice(ks),ks=1,ksize)
           read(14,iostat=istatus) nx,ny,ksize,nv
           read(14,iostat=istatus) slicexy_aux
           if (istatus.ne.0) exit
           if (time>timeold) then
            write(15) icyc, time
            write(15) nxmax,nymax,nzmax
            write(15) ksize,(kgslice(ks),ks=1,ksize)
            write(15) nx,ny,ksize,nv
            write(15) slicexy_aux
            num_records = num_records+1
            timeold = time
           endif
          enddo
          write(*,*) filename,numlist(l), num_records
          close(14)
         endif
         if (nrank==0) write(*,*) '--------------------End reading SALVA: ',numlist(l),'------------------------'
        enddo
        close(15)
       endif

       slicexz_exists = .false.
       do l=1,nsl
        if (numlist(l)<1000000) then
           write(chnumsl,1006) numlist(l)
           chsl = 'SALVA_'//chnumsl
           filename = '../'//chsl//'/slicexz_'//chx//'_000_'//chz//'.bin'
           filename = trim(filename)
        else
           write(chnumsl7,1007) numlist(l)
           chsl13 = 'SALVA_'//chnumsl7
           filename = '../'//chsl13//'/slicexz_'//chx//'_000_'//chz//'.bin'
           filename = trim(filename)
        endif
        inquire(file=filename,exist=slicexz_exists)
       enddo
       if (slicexz_exists) then
        open(15,file='SLICE_FINAL/slicexz_'//chx//'_000_'//chz//'.bin', form='unformatted')
        do l=1,nsl
         timeold = 0._rkind
         slicexz_exists = .false.
         if (nrank==0) write(*,*) '--------------------Start reading SALVA: ',numlist(l),'------------------------'
         if (numlist(l)<1000000) then
            write(chnumsl,1006) numlist(l)
            chsl = 'SALVA_'//chnumsl
            filename = '../'//chsl//'/slicexz_'//chx//'_000_'//chz//'.bin'
            filename = trim(filename)
         else
            write(chnumsl7,1007) numlist(l)
            chsl13 = 'SALVA_'//chnumsl7
            filename = '../'//chsl13//'/slicexz_'//chx//'_000_'//chz//'.bin'
            filename = trim(filename)
         endif
         inquire(file=filename,exist=slicexz_exists)
         if (slicexz_exists) then
          open(14,file=filename, action='read', form='unformatted')
          num_records = 0
          do 
           read(14,iostat=istatus) icyc, time
           if (istatus.ne.0) exit
           read(14,iostat=istatus) nxmax,nymax,nzmax
           read(14,iostat=istatus) jsize,(jgslice(ks),js=1,jsize)
           read(14,iostat=istatus) nx,jsize,nz,nv
           read(14,iostat=istatus) slicexz_aux
           if (istatus.ne.0) exit
           if (time>timeold) then
            write(15) icyc, time
            write(15) nxmax,nymax,nzmax
            write(15) jsize,(jgslice(js),js=1,jsize)
            write(15) nx,jsize,nz,nv
            write(15) slicexz_aux
            num_records = num_records+1
            timeold = time
           endif
          enddo
          if (nrank==0) write(*,*) numlist(l), num_records
          close(14)
         endif
         if (nrank==0) write(*,*) '--------------------End reading SALVA: ',numlist(l),'------------------------'
        enddo
        close(15)
       endif

     call mpi_finalize(ierr)
     1003 format(I3.3)
end program

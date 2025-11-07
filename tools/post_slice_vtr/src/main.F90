program joinslice
use parameters
use cfgio_mod, only: cfg_t, parse_cfg
use join_utils
implicit none
!
integer :: istatus
integer :: nxmax,nymax,nzmax
integer :: nx,ny,nz,ng,nv, enable_les
integer :: i,j,k,is,js,ks,m,ii,jj,kk
integer :: nbx,nby,nbz
integer :: islice_num,jslice_num,kslice_num
integer, dimension(:), allocatable :: igslice,jgslice,kgslice
integer, dimension(:), allocatable :: list_aux_slice
integer :: icyc,nxmax_tmp,nymax_tmp,nzmax_tmp
integer :: isize,jsize,ksize
integer :: inum,jnum,knum
integer :: num_slicexy_record
integer :: num_slicexz_record
integer :: num_sliceyz_record
integer :: it,itskip,fu_num
!
real(rkind) :: tmpvar
real(rkind) :: time
real(rkind), dimension(:), allocatable :: xg,yg,zg,dxg,dyg,dzg
real(rkind), dimension(:,:,:,:), allocatable :: slicexy,gslicexy
real(rkind), dimension(:,:,:,:), allocatable :: slicexz,gslicexz
real(rkind), dimension(:,:,:,:), allocatable :: sliceyz,gsliceyz
!
character(3) :: chx,chy,chz
!
logical :: slicexy_exists, sliceyz_exists, slicexz_exists
logical :: record_donexy, record_donexz, record_doneyz
!
type(cfg_t)               :: cfg
integer, dimension(:), allocatable :: exp_iters
integer, dimension(:), allocatable :: exp_xy
integer, dimension(:), allocatable :: exp_xz
integer, dimension(:), allocatable :: exp_yz
integer                            :: exp_vtr
integer                            :: exp_tec

logical :: exp_ind, exp_time
character(128) :: filename 
character(32)  :: file_prefix 

!
record_donexy = .false.
record_donexz = .false.
record_doneyz = .false.
!
nxmax = 0
open(18,file='../dxg.dat')
do
  read(18,*,iostat=istatus) tmpvar
  if (istatus==0) nxmax = nxmax+1
  if (istatus/=0) exit
enddo
close(18)
nymax = 0
open(18,file='../dyg.dat')
do
  read(18,*,iostat=istatus) tmpvar
  if (istatus==0) nymax = nymax+1
  if (istatus/=0) exit
enddo
close(18)
nzmax = 0
open(18,file='../dzg.dat')
do
  read(18,*,iostat=istatus) tmpvar
  if (istatus==0) nzmax = nzmax+1
  if (istatus/=0) exit
enddo
close(18)
!
write(*,*) 'Total grid size in x,y,z:', nxmax, nymax, nzmax
allocate(xg(nxmax),dxg(nxmax),igslice(nxmax))
allocate(yg(nymax),dyg(nymax),jgslice(nymax))
allocate(zg(nzmax),dzg(nzmax),kgslice(nzmax))
!
open(18,file='../dxg.dat')
do i=1,nxmax
  read(18,*) xg(i),dxg(i)
enddo
close(18)
open(18,file='../dyg.dat')
do j=1,nymax
  read(18,*) yg(j),dyg(j)
enddo
close(18)
open(18,file='../dzg.dat')
do k=1,nzmax
  read(18,*) zg(k),dzg(k)
enddo
close(18)
!
filename = '../multideal.ini'
cfg = parse_cfg(filename)
write(*,*) 'Reading MPI split and number of ghost nodes'
call cfg%get("mpi","x_split",nbx)
call cfg%get("mpi","y_split",nby)
call cfg%get("mpi","z_split",nbz)
call cfg%get("grid","ng",ng)

if (cfg%has_key("output","list_aux_slice")) then                   
 call cfg%get("output","list_aux_slice",list_aux_slice)
else
 list_aux_slice = [1, 2, 3, 4, 5, 6]
endif
nv = size(list_aux_slice)

enable_les = 0
if (cfg%has_key("lespar","enable_les")) then                   
 call cfg%get("lespar","enable_les", enable_les)
endif

filename = 'joinslice.ini'
cfg = parse_cfg(filename)
write(*,*) 'Reading join slice inputs'
call cfg%get("joinslice", "iters", exp_iters)
call cfg%get("joinslice", "xy",    exp_xy)
call cfg%get("joinslice", "xz",    exp_xz)
call cfg%get("joinslice", "yz",    exp_yz)
call cfg%get("joinslice", "vtr",   exp_vtr)
call cfg%get("joinslice", "tec",   exp_tec)

nx = nxmax/nbx
ny = nymax/nby
nz = nzmax/nbz
!
kslice_num = 0
do k=0,nbz-1
  write(chx,'(I3.3)') 0
  write(chy,'(I3.3)') 0
  write(chz,'(I3.3)') k
  slicexy_exists = .false.
  inquire(file='../slicexy_'//chx//'_'//chy//'_'//chz//'.bin', &
  exist=slicexy_exists)
  if (slicexy_exists) then
    open(135,file='../slicexy_'//chx//'_'//chy//'_'//chz//'.bin', &
    form='unformatted')
    read(135) icyc,time
    read(135) nxmax_tmp, nymax_tmp, nzmax_tmp
    if (nxmax_tmp/=nxmax) then
      write(*,*) 'Grid size error:', nxmax_tmp
      stop
    endif
    if (nymax_tmp/=nymax) then
      write(*,*) 'Grid size error:', nymax_tmp
      stop
    endif
    if (nzmax_tmp/=nzmax) then
      write(*,*) 'Grid size error:', nzmax_tmp
      stop
    endif
    read(135) ksize,(kgslice(kslice_num+ks),ks=1,ksize)
    read(135)
    read(135)
    do ks=1,ksize
      kgslice(kslice_num+ks) = kgslice(kslice_num+ks) + k*nz
    enddo
    kslice_num = kslice_num+ksize
    if (.not.record_donexy) then
      record_donexy = .true.
      num_slicexy_record = 1
      do
        read(135,iostat=istatus) icyc,time
        if (istatus==0) then
          num_slicexy_record = num_slicexy_record+1
          read(135)
          read(135)
          read(135)
          read(135)
        else
          exit
        endif
      enddo
    endif
    close(135)
  endif
enddo
!
jslice_num = 0
do j=0,nby-1
  write(chx,'(I3.3)') 0
  write(chy,'(I3.3)') j
  write(chz,'(I3.3)') 0
  slicexz_exists = .false.
  inquire(file='../slicexz_'//chx//'_'//chy//'_'//chz//'.bin', &
  exist=slicexz_exists)
  if (slicexz_exists) then
    open(135,file='../slicexz_'//chx//'_'//chy//'_'//chz//'.bin', &
    form='unformatted')
    read(135) icyc,time
    read(135) nxmax_tmp, nymax_tmp, nzmax_tmp
    if (nxmax_tmp/=nxmax) then
      write(*,*) 'Grid size error:', nxmax_tmp
      stop
    endif
    if (nymax_tmp/=nymax) then
      write(*,*) 'Grid size error:', nymax_tmp
      stop
    endif
    if (nzmax_tmp/=nzmax) then
      write(*,*) 'Grid size error:', nzmax_tmp
      stop
    endif
    read(135) jsize,(jgslice(jslice_num+js),js=1,jsize)
    read(135)
    read(135)
    do js=1,jsize
      jgslice(jslice_num+js) = jgslice(jslice_num+js) + j*ny
    enddo
    jslice_num = jslice_num+jsize
    if (.not.record_donexz) then
      record_donexz = .true.
      num_slicexz_record = 1
      do
        read(135,iostat=istatus) icyc,time
        if (istatus==0) then
          num_slicexz_record = num_slicexz_record+1
          read(135)
          read(135)
          read(135)
          read(135)
        else
          exit
        endif
      enddo
    endif
    close(135)
  endif
enddo
!
islice_num = 0
do i=0,nbx-1
  write(chx,'(I3.3)') i
  write(chy,'(I3.3)') 0
  write(chz,'(I3.3)') 0
  sliceyz_exists = .false.
  inquire(file='../sliceyz_'//chx//'_'//chy//'_'//chz//'.bin', &
  exist=sliceyz_exists)
  if (sliceyz_exists) then
    open(135,file='../sliceyz_'//chx//'_'//chy//'_'//chz//'.bin', &
    form='unformatted')
    read(135) icyc,time
    read(135) nxmax_tmp, nymax_tmp, nzmax_tmp
    if (nxmax_tmp/=nxmax) then
      write(*,*) 'Grid size error:', nxmax_tmp
      stop
    endif
    if (nymax_tmp/=nymax) then
      write(*,*) 'Grid size error:', nymax_tmp
      stop
    endif
    if (nzmax_tmp/=nzmax) then
      write(*,*) 'Grid size error:', nzmax_tmp
      stop
    endif
    read(135) isize,(igslice(islice_num+is),is=1,isize)
    read(135)
    read(135)
    do is=1,isize
      igslice(islice_num+is) = igslice(islice_num+is) + i*nx
    enddo
    islice_num = islice_num+isize
    if (.not.record_doneyz) then
      record_doneyz = .true.
      num_sliceyz_record = 1
      do
        read(135,iostat=istatus) icyc,time
        if (istatus==0) then
          num_sliceyz_record = num_sliceyz_record+1
          read(135)
          read(135)
          read(135)
          read(135)
        else
          exit
        endif
      enddo
    endif
    close(135)
  endif
enddo
!
write(*,*) 'Total number of islice (sliceyz) and records:', &
  islice_num, num_sliceyz_record
if (islice_num>0) write(*,*) (igslice(is),is=1,islice_num)
write(*,*) 'Total number of jslice (slicexz) and records:', &
  jslice_num, num_slicexz_record
if (jslice_num>0) write(*,*) (jgslice(js),js=1,jslice_num)
write(*,*) 'Total number of kslice (slicexy) and records:', &
  kslice_num, num_slicexy_record
if (kslice_num>0) write(*,*) (kgslice(ks),ks=1,kslice_num)
!
if (islice_num>0) then
  allocate(gsliceyz(islice_num,nymax,nzmax,6))
endif
if (jslice_num>0) then
  allocate(gslicexz(nxmax,jslice_num,nzmax,6))
endif
if (kslice_num>0) then
  allocate(gslicexy(nxmax,nymax,kslice_num,6))
endif
!
!write slicexy
if (kslice_num==0) goto 51
open(10,file='gslicexy.bin',form='unformatted')
write(10) nxmax, nymax, nzmax
write(10) kslice_num, (kgslice(ks),ks=1,kslice_num)
write(10) (zg(kgslice(ks)),ks=1,kslice_num)
write(10) num_slicexy_record
do it=1,num_slicexy_record
  knum = 0
  do k=0,nbz-1
    if (allocated(slicexy)) deallocate(slicexy)
    write(chx,'(I3.3)') 0
    write(chy,'(I3.3)') 0
    write(chz,'(I3.3)') k
    slicexy_exists = .false.
    inquire(file='../slicexy_'//chx//'_'//chy//'_'//chz//'.bin', &
    exist=slicexy_exists)
    if (slicexy_exists) then
      do j=0,nby-1
        write(chy,'(I3.3)') j
        do i=0,nbx-1
          write(chx,'(I3.3)') i
          fu_num = 11+k*nbx*nby+j*nbx+i
          if (it==1) then
            open(fu_num, &
            file='../slicexy_'//chx//'_'//chy//'_'//chz//'.bin', &
            form='unformatted')
          endif
          read(fu_num) icyc,time
          read(fu_num)
          read(fu_num) ksize
          if (.not.allocated(slicexy)) then
            allocate(slicexy(1-ng:nx+ng,1-ng:ny+ng,ksize,6))
          endif
          read(fu_num)
          read(fu_num) slicexy
          do ks=1,ksize
            do jj=1,ny
              do ii=1,nx
                do m=1,6
                  gslicexy(i*nx+ii,j*ny+jj,knum+ks,m) = slicexy(ii,jj,ks,m)
                enddo
              enddo
            enddo
          enddo
!         
        enddo
      enddo
      if (it==num_slicexy_record) close(fu_num)
      knum = knum+ksize
      !print*,'knum, ksize: ',knum, ksize
    endif
  enddo
  write(10) icyc,time
  write(10) gslicexy

  exp_time = .false.
  if(any(exp_iters == it) .or. &
     (size(exp_iters) == 1 .and. exp_iters(1) == 0) .or. &
     (size(exp_iters) == 1 .and. exp_iters(1) == -1 .and. it == num_slicexy_record)) then
        exp_time = .true.
  endif

  do ks=1,knum
    exp_ind = .false.
    if(any(exp_xy == ks) .or.  (size(exp_xy) == 1 .and. exp_xy(1) == 0)) then
      exp_ind = .true.
    endif
    print*,'exp_time, exp_ind: ',exp_time, exp_ind
    if(exp_time .and. exp_ind) then
      print*,'printing time = ',it, ' ; ks = ',ks
      file_prefix = "s_xy_???????_???????"
      write(file_prefix(6:12) ,'(I7.7)') ks
      write(file_prefix(14:20),'(I7.7)') it
      call write_vtk(0._rkind, 1, gslicexy(:,:,ks:ks,:), xg, yg, zg(kgslice(ks):kgslice(ks)), & 
                     file_prefix, enable_les, list_aux_slice)
    endif
  enddo

enddo
close(10)
51   continue
!
!write slicexz
if (jslice_num==0) goto 52
open(10,file='gslicexz.bin',form='unformatted')
write(10) nxmax, nymax, nzmax
write(10) jslice_num, (jgslice(js),js=1,jslice_num)
write(10) (yg(jgslice(js)),js=1,jslice_num)
write(10) num_slicexz_record
do it=1,num_slicexz_record
  jnum = 0
  do j=0,nby-1
    if (allocated(slicexz)) deallocate(slicexz)
    write(chx,'(I3.3)') 0
    write(chy,'(I3.3)') j
    write(chz,'(I3.3)') 0
    slicexz_exists = .false.
    inquire(file='../slicexz_'//chx//'_'//chy//'_'//chz//'.bin', &
    exist=slicexz_exists)
    if (slicexz_exists) then
      do k=0,nbz-1
        write(chz,'(I3.3)') k
        do i=0,nbx-1
          write(chx,'(I3.3)') i
          fu_num = 11+k*nbx*nby+j*nbx+i
          if (it==1) then
            open(fu_num, &
            file='../slicexz_'//chx//'_'//chy//'_'//chz//'.bin', &
            form='unformatted')
          endif
          read(fu_num) icyc,time
          read(fu_num)
          read(fu_num) jsize
          if (.not.allocated(slicexz)) then
            allocate(slicexz(1-ng:nx+ng,jsize,1-ng:nz+ng,6))
          endif
          read(fu_num)
          read(fu_num) slicexz
          do kk=1,nz
            do js=1,jsize
              do ii=1,nx
                do m=1,6
                  gslicexz(i*nx+ii,jnum+js,k*nz+kk,m) = slicexz(ii,js,kk,m)
                enddo
              enddo
            enddo
          enddo
!         
        enddo
      enddo
      if (it==num_slicexz_record) close(fu_num)
      jnum = jnum+jsize
    endif
  enddo
  write(10) icyc,time
  write(10) gslicexz

  exp_time = .false.
  if(any(exp_iters == it) .or. &
     (size(exp_iters) == 1 .and. exp_iters(1) == 0) .or. &
     (size(exp_iters) == 1 .and. exp_iters(1) == -1 .and. it == num_slicexz_record)) then
        exp_time = .true.
  endif

  do js=1,jnum
    exp_ind = .false.
    if(any(exp_xz == js) .or.  (size(exp_xz) == 1 .and. exp_xz(1) == 0)) then
      exp_ind = .true.
    endif
    print*,'exp_time, exp_ind: ',exp_time, exp_ind
    if(exp_time .and. exp_ind) then
      print*,'printing time = ',it, ' ; js = ',js
      file_prefix = "s_xz_???????_???????"
      write(file_prefix(6:12) ,'(I7.7)') js
      write(file_prefix(14:20),'(I7.7)') it
      call write_vtk(0._rkind, 1, gslicexz(:,js:js,:,:), xg, yg(jgslice(js):jgslice(js)), zg, & 
                    file_prefix, enable_les, list_aux_slice)
    endif
  enddo

enddo
close(10)
52   continue
!
!write sliceyz
if (islice_num==0) goto 53
open(10,file='gsliceyz.bin',form='unformatted')
write(10) nxmax, nymax, nzmax
write(10) islice_num, (igslice(is),is=1,islice_num)
write(10) (xg(igslice(is)),is=1,islice_num)
write(10) num_sliceyz_record
do it=1,num_sliceyz_record
  inum = 0
  do i=0,nbx-1
    if (allocated(sliceyz)) deallocate(sliceyz)
    write(chx,'(I3.3)') i
    write(chy,'(I3.3)') 0
    write(chz,'(I3.3)') 0
    sliceyz_exists = .false.
    inquire(file='../sliceyz_'//chx//'_'//chy//'_'//chz//'.bin', &
    exist=sliceyz_exists)
    if (sliceyz_exists) then
      do k=0,nbz-1
        write(chz,'(I3.3)') k
        do j=0,nby-1
          write(chy,'(I3.3)') j
          fu_num = 11+k*nbx*nby+j*nbx+i
          if (it==1) then
            open(fu_num,  &
            file='../sliceyz_'//chx//'_'//chy//'_'//chz//'.bin', &
            form='unformatted')
          endif
          read(fu_num) icyc,time
          read(fu_num)
          read(fu_num) isize
          if (.not.allocated(sliceyz)) then
            allocate(sliceyz(isize,1-ng:ny+ng,1-ng:nz+ng,6))
          endif
          read(fu_num)
          read(fu_num) sliceyz
          do kk=1,nz
            do jj=1,ny
              do is=1,isize
                do m=1,6
                  gsliceyz(inum+is,j*ny+jj,k*nz+kk,m) = sliceyz(is,jj,kk,m)
                enddo
              enddo
            enddo
          enddo
!         
        enddo
      enddo
      if (it==num_sliceyz_record) close(fu_num)
      inum = inum+isize
    endif
  enddo
  write(10) icyc,time
  write(10) gsliceyz

  exp_time = .false.
  if(any(exp_iters == it) .or. &
     (size(exp_iters) == 1 .and. exp_iters(1) == 0) .or. &
     (size(exp_iters) == 1 .and. exp_iters(1) == -1 .and. it == num_sliceyz_record)) then
        exp_time = .true.
  endif

  do is=1,inum
    exp_ind = .false.
    if(any(exp_yz == is) .or.  (size(exp_yz) == 1 .and. exp_yz(1) == 0)) then
      exp_ind = .true.
    endif
    print*,'exp_time, exp_ind: ',exp_time, exp_ind
    if(exp_time .and. exp_ind) then
      print*,'printing time = ',it, ' ; is = ',is
      file_prefix = "s_yz_???????_???????"
      write(file_prefix(6:12) ,'(I7.7)') is
      write(file_prefix(14:20),'(I7.7)') it
      call write_vtk(0._rkind, 1, gsliceyz(is:is,:,:,:), xg(igslice(is):igslice(is)), yg, zg, &
                    file_prefix, enable_les, list_aux_slice)
    endif
  enddo

enddo
close(10)
53   continue
!
if (kslice_num>0) then
  open(135,file='slicexy.dat')
  write(135,*) 'zone i=',nxmax,', j=',nymax
  do j=1,nymax
    do i=1,nxmax
      write(135,100) xg(i),yg(j),(gslicexy(i,j,1,m),m=1,6)
    enddo
  enddo
  close(135)
endif
!
if (jslice_num>0) then
  open(135,file='slicexz.dat')
  write(135,*) 'zone i=',nxmax,', j=',nzmax
  do k=1,nzmax
    do i=1,nxmax
      write(135,100) xg(i),zg(k),(gslicexz(i,1,k,m),m=1,6)
    enddo
  enddo
  close(135)
endif
!
if (islice_num>0) then
  open(135,file='sliceyz.dat')
  write(135,*) 'zone i=',nzmax,', j=',nymax
  do j=1,nymax
    do k=1,nzmax
      write(135,100) zg(k),yg(j),(gsliceyz(1,j,k,m),m=1,6)
    enddo
  enddo
  close(135)
endif
!
100  format(20ES20.10)
!
endprogram joinslice

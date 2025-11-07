#include "index_define.h"
module utils 
        use mpi
        use cfgio_mod, only: cfg_t, parse_cfg
        use parameters
        use global_variables
        use cantera

contains
        subroutine read_input
                implicit none
                !
                character(20)  :: ch
                character(100) :: filename
                type(cfg_t)               :: cfg
                type(cfg_t)               :: flow_params_cfg
                type(cfg_t)               :: post_cfg
                type(phase_t)             :: mixture
                integer :: k,lsp
                integer, dimension(:), allocatable  :: list_aux_slice

                filename = '../../multideal.ini'
                cfg = parse_cfg(filename)
                call cfg%get("grid","nxmax",nxmax)
                call cfg%get("grid","nymax",nymax)
                call cfg%get("grid","nzmax",nzmax)
                call cfg%get("grid","ng",ng)
                call cfg%get("grid","ystag",ystag)
                call cfg%get("fluid","enable_chemistry",enable_chemistry)
                call cfg%get("lespar","enable_les",enable_les)
                call cfg%get("mpi","x_split",mpi_split_x)
                call cfg%get("mpi","z_split",mpi_split_z)

                nx = nxmax/mpi_split_x
                ny = nymax
                nz = nzmax/mpi_split_z


                mixture = importphase('../../input_cantera.yaml')
                if(nspecies(mixture)/=N_S) print *, "Error! Check number of species in yaml or index_define.h"
                do lsp=1,N_S
                 call getSpeciesName(mixture, lsp, species_names(lsp))
                 if (species_names(lsp)=='H2') H2=lsp
                 if (species_names(lsp)=='O2') O2=lsp
                 if (species_names(lsp)=='N2') N2=lsp
                enddo
             
                if (cfg%has_key("output","list_aux_slice")) then
                 call cfg%get("output","list_aux_slice",list_aux_slice)
                 nv_slice = size(list_aux_slice)
                else
                 nv_slice = N_S+8
                endif

                call cfg%get("output","igslice",igslice)
                call cfg%get("output","jgslice",jgslice)
                call cfg%get("output","kgslice",kgslice)
                
        endsubroutine read_input

        subroutine read_grid_bl
            implicit none
            integer :: i,j,k

            allocate(xg(1-ng:nxmax+ng+1))
            allocate(yg(1-ng:nymax+ng))
            allocate(zg(1-ng:nzmax+ng))

            open(10,file='../../x.dat',form='formatted')
            do i=1-ng,nxmax+ng+1
            read(10,*) xg(i)
            enddo
            close(10)

            open(10,file='../../y.dat',form='formatted')
            do j=1-ng,nymax+ng
            read(10,*) yg(j)
            enddo
            close(10)

            open(10,file='../../z.dat',form='formatted')
            do k=1-ng,nzmax+ng
            read(10,*) zg(k)
            enddo
            close(10)

        end subroutine read_grid_bl

        subroutine find_slices
            implicit none
            integer :: ib,kb
            character(3) :: chx,chy,chz
            character(100)  :: filename

            slicexy_exists = .false.
            slicexz_exists = .false.
            sliceyz_exists = .false.
            write(chx,1003) ncoords(1)
            write(chz,1003) ncoords(3)
            if (.not. slicexy_exists) then
             filename = 'slicexy_'//chx//'_000_'//chz//'.bin'
             inquire(file=filename,exist=slicexy_exists)
            endif
            if (.not. slicexz_exists) then
             filename = 'slicexz_'//chx//'_000_'//chz//'.bin'
             inquire(file=filename,exist=slicexz_exists)
            endif
            if (.not. sliceyz_exists) then
             filename = 'sliceyz_'//chx//'_000_'//chz//'.bin'
             inquire(file=filename,exist=sliceyz_exists)
            endif
1003 format(I3.3)
        endsubroutine find_slices

        subroutine initialize_mpi
            implicit none
            integer :: mpi_err
            logical :: pbc(3)
            logical :: remain_dims(3)
            logical :: reord

            call mpi_init(mpi_err)
            call mpi_comm_size(mpi_comm_world,nprocs,mpi_err)
            call mpi_comm_rank(mpi_comm_world,myrank,mpi_err)

            if (myrank == 0) masterproc = .true.

            nblocks(1) = mpi_split_x
            nblocks(2) = 1
            nblocks(3) = mpi_split_z
            pbc(1) = .false.
            pbc(2) = .false.
            pbc(3) = .true.

            ! Create 3D topology
            reord = .false.
            call mpi_cart_create(mpi_comm_world,3,nblocks,pbc,reord,mp_cart,mpi_err)
            call mpi_cart_coords(mp_cart,myrank,3,ncoords,mpi_err)
            
            ! Create 1D communicators
            remain_dims(1) = .true.
            remain_dims(2) = .false.
            remain_dims(3) = .false.
            call mpi_cart_sub(mp_cart,remain_dims,mp_cartx,mpi_err)
            call mpi_comm_rank(mp_cartx,nrank_x,mpi_err)
            call mpi_cart_shift(mp_cartx,0,1,ileftx,irightx,mpi_err)
            remain_dims(2) = .true.
            remain_dims(1) = .false.
            remain_dims(3) = .false.
            call mpi_cart_sub(mp_cart,remain_dims,mp_carty,mpi_err)
            call mpi_comm_rank(mp_carty,nrank_y,mpi_err)
            call mpi_cart_shift(mp_carty,0,1,ilefty,irighty,mpi_err)
            remain_dims(3) = .true.
            remain_dims(1) = .false.
            remain_dims(2) = .false.
            call mpi_cart_sub(mp_cart,remain_dims,mp_cartz,mpi_err)
            call mpi_comm_rank(mp_cartz,nrank_z,mpi_err)
            call mpi_cart_shift(mp_cartz,0,1,ileftz,irightz,mpi_err)

        endsubroutine initialize_mpi

        subroutine slice_prepare
            implicit none
            integer :: i,j,k

            inum = 0
            do i = 1,size(igslice)
             if (igslice(i)>0) then
              icord = (igslice(i)-1)/nx
              if (ncoords(1)==icord) inum = inum + 1
             endif
            enddo
            jnum = 0
            do j = 1,size(jgslice)
             if (jgslice(j)>0) then
              jcord = (jgslice(j)-1)/ny
              if (ncoords(2)==jcord) jnum = jnum + 1
             endif
            enddo
            knum = 0
            do k = 1,size(kgslice)
             if (kgslice(k)>0) then
              kcord = (kgslice(k)-1)/nz
              if (ncoords(3)==kcord) knum = knum + 1
             endif
            enddo
!
            if (inum>0) then
             allocate(islice(inum))
             allocate(sliceyz_aux(inum,1-ng:ny+ng,1-ng:nz+ng,nv_slice))
            endif
            if (jnum>0) then
             allocate(jslice(jnum))
             allocate(slicexz_aux(1-ng:nx+ng,jnum,1-ng:nz+ng,nv_slice))
            endif
            if (knum>0) then
             allocate(kslice(knum))
             allocate(slicexy_aux(1-ng:nx+ng,1-ng:ny+ng,knum,nv_slice))
            endif  

            inum = 0
            do i = 1,size(igslice)
             if (igslice(i)>0) then
              icord = (igslice(i)-1)/nx
              if (ncoords(1)==icord) then
               inum = inum+1
               islice(inum) = igslice(i)-ncoords(1)*nx
              endif
             endif
            enddo
            jnum = 0
            do j = 1,size(jgslice)
             if (jgslice(j)>0) then
              jcord = (jgslice(j)-1)/ny
              if (ncoords(2)==jcord) then
               jnum = jnum + 1
               jslice(jnum) = jgslice(j)-ncoords(2)*ny
              endif
             endif
            enddo
            knum = 0
            do k = 1,size(kgslice)
             if (kgslice(k)>0) then
              kcord = (kgslice(k)-1)/nz
              if (ncoords(3)==kcord) then
               knum = knum + 1
               kslice(knum) = kgslice(k)-ncoords(3)*nz
              endif
             endif
            enddo 

        endsubroutine slice_prepare
       
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

endmodule utils 

#----------------------------------------------------------------------
# SUPPORTED COMPILE: nv-cuda, nv, intel, gnu, ibmxl, cray-cuda
# SUPPORTED MODES  : opt, debug [default=debug]
# SUPPORTED PRECISIONS (PREC) : single, double [default=double]
#----------------------------------------------------------------------
EQUATION = "multideal"
BACKEND  = "gpu"
COMPILE  = "nv-cuda"
MODE     = "opt"
PREC     = "double"
CUDA_VER = 11.8
CCA      = cc80
MPI_CMD  = 
CGAL     = "no"
CATALYST = "no"

define LOAD
module load profile/global ;
module load cmake/3.24.3 ;
module load gcc/11.3.0 ;
module load cuda/11.8 ;
module load python/3.10.8--gcc--11.3.0 ;
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 ;
module load nvhpc/25.3 ;
module load hpcx-mpi/2.19 ;
endef
export LOAD

CANTERA  = "yes"
CANTERA_ROOT    := /leonardo/home/userexternal/eforti00/CANTERA_NVHPC/install_cantera
CANTERA_INCLUDE := -I${CANTERA_ROOT}/include/cantera -I${CANTERA_ROOT}/include
CANTERA_LIB     := ${CANTERA_ROOT}/lib
CANTERA_LDFLAGS := -L${CANTERA_LIB}/ -lcantera_fortran -lcantera -Wl,-rpath,${CANTERA_LIB}  -ldl -lstdc++ -llapack -lblas -Wl,--copy-dt-needed-entries -Wl,-rpath,/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-8.5.0/gcc-11.3.0-tm6phj7wkcw7cuy6gjixemkvh5x2mhza/lib64 /leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-8.5.0/gcc-11.3.0-tm6phj7wkcw7cuy6gjixemkvh5x2mhza/lib64/libstdc++.so.6.0.29

CGAL = "yes"
CGAL_INCLUDE = -I/leonardo/home/userexternal/eforti00/INSTALL_CGAL-5.0.2/include/ -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-8.5.0/mpfr-4.1.0-w4ofd2ll4ftu53yawc4d3sssrmeo2huc/include -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/boost-1.80.0-wykvmwvatulay54p3phkqnyl4tjxuahr/include

CGAL_LDFLAGS = -Wl,-rpath,/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-8.5.0/gcc-11.3.0-tm6phj7wkcw7cuy6gjixemkvh5x2mhza/lib64 /leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-8.5.0/gcc-11.3.0-tm6phj7wkcw7cuy6gjixemkvh5x2mhza/lib64/libstdc++.so.6.0.29


program streams_multideal_gpu
!< STREAmS, STREAmS for Navier-Stokes equation, GPU backend.

use streams_equation_multideal_gpu_object
use streams_parameters
use mpi

implicit none
type(equation_multideal_gpu_object) :: multideal_gpu        !< Navier-Stokes equations system.
integer :: mpi_err

call mpi_initialize()

call multideal_gpu%run(filename='multideal.ini')

call MPI_Finalize(mpi_err)

endprogram streams_multideal_gpu

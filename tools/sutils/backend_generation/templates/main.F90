program streams_${equation_type}_${backend}

use streams_equation_${equation_type}_${backend}_object
use streams_parameters
use mpi

implicit none
type(equation_${equation_type}_${backend}_object) :: ${equation_type}_${backend}
integer :: mpi_err

call mpi_initialize()

call ${equation_type}_${backend}%run(filename='${equation_type}.ini')

call MPI_Finalize(mpi_err)

endprogram streams_${equation_type}_${backend}
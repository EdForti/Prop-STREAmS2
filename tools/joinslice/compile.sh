rm *.o *.mod *.exe
FILENAME=joinslice
mpif90 -g -c ${FILENAME}.F90
mpif90 -g ${FILENAME}.o -o ${FILENAME}.exe


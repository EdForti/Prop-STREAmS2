CXX=g++
FC=gfortran
CANTERA_ROOT=/home/marco/CANTERA_GCC/install_cantera
CANTERA_INCLUDE=${CANTERA_ROOT}/include
CANTERA_LIB=${CANTERA_ROOT}/lib

${CXX} cantera_utils_cxx.cxx -c -I${CANTERA_INCLUDE}
echo "#1 compiled"
${FC} cantera_utils.F90 -c -I${CANTERA_INCLUDE}
echo "#2 compiled"
${FC} test_cantera_utils.F90 -c -I${CANTERA_INCLUDE} -I${CANTERA_INCLUDE}/cantera
echo "#3 compiled"
#${FC} cantera_utils_cxx.o cantera_utils.o test_cantera_utils.o -o test.exe -I${CANTERA_INCLUDE} -L${CANTERA_LIB}/ -lcantera -Wl,-rpath,${CANTERA_LIB} -ldl -lstdc++
${FC} cantera_utils_cxx.o cantera_utils.o test_cantera_utils.o -o test.exe -L${CANTERA_LIB}/ -lcantera_fortran -lcantera -Wl,-rpath,${CANTERA_LIB} -ldl -lstdc++ -llapack -lblas -Wl,--copy-dt-needed-entries
echo "#4 compiled"

#-lstdc++

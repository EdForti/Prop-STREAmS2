# Compilers
CXX = g++
FC  = gfortran

# Cantera paths
CANTERA_ROOT    = /home/marco/CANTERA_GCC/install_cantera
CANTERA_INCLUDE = $(CANTERA_ROOT)/include
CANTERA_LIB     = $(CANTERA_ROOT)/lib

# Sources
CXXSRC  = cantera_utils_cxx.cxx
F90SRC1 = cantera_utils.F90
F90SRC2 = test_cantera_utils.F90

# Objects
OBJS = cantera_utils_cxx.o cantera_utils.o test_cantera_utils.o

# Executable
EXE = test.exe

# Flags
CXXFLAGS = -I$(CANTERA_INCLUDE)
FFLAGS   = -I$(CANTERA_INCLUDE) -I$(CANTERA_INCLUDE)/cantera
LDFLAGS  = -L$(CANTERA_LIB) -lcantera_fortran -lcantera -Wl,-rpath,$(CANTERA_LIB) -ldl -lstdc++ -llapack -lblas -Wl,--copy-dt-needed-entries

# Default target
all: $(EXE)

$(EXE): $(OBJS)
	$(FC) -g $(OBJS) -o $@ $(LDFLAGS)

cantera_utils_cxx.o: $(CXXSRC)
	$(CXX) -g -c $< -o $@ $(CXXFLAGS)

cantera_utils.o: $(F90SRC1)
	$(FC) -g -c $< -o $@ $(FFLAGS)

test_cantera_utils.o: $(F90SRC2)
	$(FC) -g -c $< -o $@ $(FFLAGS)

# Cleanup
clean:
	rm -f $(OBJS) $(EXE) *mod


#----------------------------------------------------------------------
# SUPPORTED COMPILE: gnu
# SUPPORTED PRECISIONS (PREC) : single, double [default=double]
#----------------------------------------------------------------------
COMPILE = "gnu"
PREC    = "double"
#----------------------------------------------------------------------

ifeq (${PREC},"single") 
    PRECFLAGS = -DSINGLE_PRECISION 
endif

ifeq (${COMPILE},"gnu")
    FC = mpif90
    #OLEVEL = -O1 -fbounds-check -g -fbacktrace -fdump-core -ffpe-trap=zero,invalid,overflow
    OLEVEL = -O3 -g
    MODULE_FLAG = -J
endif

RM = rm -f

#----------------------------------------------------------------------
#CANTERA
#----------------------------------------------------------------------
CANTERA_ROOT    := /home/sudr/sudr454508/CANTERA_GCC/install_cantera
CANTERA_INCLUDE := -I${CANTERA_ROOT}/include/cantera -I${CANTERA_ROOT}/include
CANTERA_LIB     := ${CANTERA_ROOT}/lib
CANTERA_LDFLAGS := -L${CANTERA_LIB}/ -lcantera_fortran -lcantera -Wl,-rpath,${CANTERA_LIB} -ldl -lstdc++ -llapack -lblas -Wl,--copy-dt-needed-entries
#----------------------------------------------------------------------
# Base code
#----------------------------------------------------------------------
MAIN = src/main.o
OBJ_FILES = src/parameters.o src/global_variables.o src/utils.o src/postpro_slice.o

OBJ_MODS = third-party/cfgio/string_conv_mod.o third-party/cfgio/cfgio_mod.o
OBJS = $(OBJ_MODS) $(OBJ_FILES) $(MAIN)
OBJS_BUILD = $(OBJS:%.o=build/%.o)
EXEC = post_slice_stats.exe
#----------------------------------------------------------------------

$(EXEC): $(OBJS_BUILD) Makefile 
	$(FC) $(PRECFLAGS) $(OLEVEL) -o $@ $(OBJS_BUILD) $(FCLIBS) $(LDFLAGS)

build/%.o:%.F90 Makefile
	$(FC) $(PRECFLAGS) $(INCLUDE) $(OLEVEL) -c $< -o $@ $(MODULE_FLAG) build/mods

.PHONY: clean
clean:
	$(RM) $(EXEC) $(MAIN) $(OBJS_BUILD) build/mods/*.mod
INCLUDE := $(INCLUDE) $(CANTERA_INCLUDE)
LDFLAGS := $(LDFLAGS) $(CANTERA_LDFLAGS)

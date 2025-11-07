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
# Base code
#----------------------------------------------------------------------
MAIN = joinslice.o
OBJ_FILES = 

OBJ_MODS = 
OBJS = $(OBJ_MODS) $(OBJ_FILES) $(MAIN)
OBJS_BUILD = $(OBJS:%.o=%.o)
INCLUDE= 
EXEC = joinslice.exe
#----------------------------------------------------------------------

$(EXEC): $(OBJS_BUILD) Makefile 
	$(FC) $(PRECFLAGS) $(OLEVEL) -o $@ $(OBJS_BUILD) $(FCLIBS) $(LDFLAGS)

%.o:%.F90 Makefile
	$(FC) $(PRECFLAGS) $(INCLUDE) $(OLEVEL) -c $< -o $@ $(MODULE_FLAG) #build/mods

.PHONY: clean
clean:
	$(RM) $(EXEC) $(MAIN) $(OBJS_BUILD) 
###INCLUDE := $(INCLUDE) 
###LDFLAGS := $(LDFLAGS) 

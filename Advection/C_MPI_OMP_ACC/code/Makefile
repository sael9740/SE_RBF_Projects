CC = mpicc -g

### MKL include/libs from https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
MKLFLAGS = -DMKL_ILP64 -I${MKLROOT}/include
MKLLIBS = -L$(NETCDF_C)/lib -lnetcdf -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
### for pgi openmpi
MKLLIBS += -lmkl_blacs_openmpi_ilp64
OMPFLAGS = -mp
### for impi
# MKLFLAGS += -std=c99
# MKLLIBS += -lmkl_blacs_intelmpi_ilp64
# OMPFLAGS = -fopenmp

### METIS
METISFLAGS = -I$(METISROOT)/include
METISLIBS = -L$(METISROOT)/lib -lmetis

CFLAGS = $(OMPFLAGS) $(MKLFLAGS) $(METISFLAGS)
LIBS = $(MKLLIBS) $(METISLIBS)


OBJS = main.o wrappers.o debug.o nodesets.o layers.o domains.o patches.o rbffd_DMs.o

EXE = adv_solver.exe

all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(LIBS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(EXE)

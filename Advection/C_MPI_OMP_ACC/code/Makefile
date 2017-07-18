CC = mpicc

# MKL include/libs from https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
CFLAGS = -std=c99 -DMKL_ILP64 -I${MKLROOT}/include
# for openmpi
# LIBS = -L$(NETCDF_C)/lib -lnetcdf -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_ilp64 -lpthread -lm -ldl
# for impi
LIBS = -L$(NETCDF_C)/lib -lnetcdf -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_ilp64 -lpthread -lm -ldl

OBJS = main.o io.o rbffd.o

EXE = adv_solver.exe

all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(LIBS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(EXE)

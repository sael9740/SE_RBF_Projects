CC = mpicc

CFLAGS = -DMKL_ILP64 -I${MKLROOT}/include
LIBS = -L$(NETCDF_C)/lib -lnetcdf -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_ilp64 -lpthread -lm -ldl

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

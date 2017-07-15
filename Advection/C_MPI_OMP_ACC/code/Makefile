CC = mpicc

CFLAGS = 
LIBS = -L$(NETCDF_C)/lib -lnetcdf

OBJS = main.o io.o

EXE = adv_solver.exe

all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(LIBS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(EXE)

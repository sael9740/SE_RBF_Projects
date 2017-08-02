#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "include/wrappers.h"
#include "include/config.h"
#include "include/debug.h"

int mpi_size;
int mpi_rank;

int main(int argc, char** argv) {

	/***** Initialize MPI *****/
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	double tstart;

	/***** Get Runtime Configruation *****/
	get_rt_config();

	/***** Initialize Physical Constants *****/
	init_phys_constants();

	/***** Initialize the Global Nodeset and Layer Structure *****/
	tstart = MPI_Wtime();
	init_global_nodeset();
	if (mpi_rank == 0) {
		printf("\ninit_global_nodeset() walltime -> %.2f seconds\n", MPI_Wtime() - tstart);
	}
	init_global_layers();

	/***** Partition the Global Domain *****/
	tstart = MPI_Wtime();
	partition_global_domain();
	if (mpi_rank == 0) {
		printf("\npartition_global_domain() walltime -> %.2f seconds\n", MPI_Wtime() - tstart);
	}

	/***** Initialize Local Patch *****/
	tstart = MPI_Wtime();
	init_local_patch();
	if (mpi_rank == 0) {
		printf("\ninit_local_patch() walltime -> %.2f seconds\n", MPI_Wtime() - tstart);
	}

	/***** Initialize RBFFD Matrices *****/
	tstart = MPI_Wtime();
	init_patch_rbffd_DMs();
	if (mpi_rank == 0) {
		printf("\ninit_patch_rbffd_DMs() walltime -> %.2f seconds\n", MPI_Wtime() - tstart);
	}

	/***** Set Initial Conditions *****/
	tstart = MPI_Wtime();
	get_model_ICs();
	if (mpi_rank == 0) {
		printf("\nget_model_ICs() walltime -> %.2f seconds\n", MPI_Wtime() - tstart);
	}

	/***** Perform Timestepping *****/
	get_solution();

	/***** Verification *****/
	verify_solution();

	/***** Finalize MPI *****/
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

}


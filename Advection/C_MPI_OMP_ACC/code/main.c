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
	
	/***** Get Runtime Configruation *****/
	get_rt_config();

	/***** Initialize Physical Constants *****/
	init_phys_constants();

	/***** Initialize the Global Nodeset and Layer Structure *****/	
	init_global_nodeset();
	init_global_layers();

	/***** Partition the Global Domain *****/
	partition_global_domain();

	/***** Initialize Local Patch *****/
	init_local_patch();

	/***** Initialize RBFFD Matrices *****/
	init_patch_rbffd_DMs();

	/***** Set Initial Conditions *****/
	get_model_ICs();

	/***** Perform Timestepping *****/
	get_solution();

	/***** Verification *****/
	verify_solution();

	/***** Finalize MPI *****/
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

}


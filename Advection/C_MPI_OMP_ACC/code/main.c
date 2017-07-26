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

	/***** *****/
	get_model_ICs();

	/***** Partitioning of Global Nodeset *****/

	/***** Global Vertical Layer Initialization *****/

//	global_domains.layers = get_layers(rt_config.model_height, rt_config.num_layers);

	
	/******************************** Local Patch Initialization **********************************/
	
	
	/*nodeset_struct nodeset[1];
	init_nodeset(nodeset);

	// reorder nodes in nodeset for contiguous patch data
	reorder_nodeset(nodeset);

	// initialize local rank's patch data
	patch_struct LP[1];
	init_patches(LP, nodeset);

	// ============================ Free Remaining Data Space ================================ //

	free(nodeset->x);
	free(nodeset->y);
	free(nodeset->z);
	free(nodeset->D_idx);
	free(nodeset->idx);
	free(nodeset->patch_ids);
	free(nodeset->patch_start_ids);
	free(nodeset->patch_sizes);*/

	// ============================ Finalize MPI ================================ //

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "include/wrappers.h"
#include "include/config.h"
#include "include/debug.h"
#include "include/nodesets.h"
#include "include/layers.h"
#include "include/domains.h"
#include "include/patches.h"
#include "include/rbffd_DMs.h"


/***** GLOBAL DATA *****/

extern int mpi_size;
extern int mpi_rank;

rt_config_struct rt_config[1];

phys_constants_struct phys_constants[1];

domains_struct global_domains[1];

layers_struct global_layers[1];

patch_struct local_patch[1];


/***** WRAPPER FUNCTIONS FOR MAIN *****/

void init_patch_rbffd_DMs() {

	get_part_rbffd_stencils(local_patch);

	init_rbffd_HH_rot_Ms(local_patch);

	get_rbffd_DMs_Dxyz(local_patch);

}

/* FUNCTION - INIT_LOCAL_PATCH
 * - Initializes the local patch using the global_domains struct which contains the global nodeset
 *   as well as the partitioning of the nodeset
 */
void init_local_patch() {

	init_part_utils(global_domains);
	
//	print_part_utils(global_domains->Nnodes);
	
	init_patch_nodeset(local_patch, global_domains);
	
//	print_nodeset(local_patch->nodeset);

	init_patch_halos(local_patch);
//	print_halos(local_patch);

}


/* FUNCTION - INIT_GLOBAL_LAYERS
 * - Initializes the global_layers structure
 */
void init_global_layers() {

	int Nv = rt_config->num_layers;
	double htop = rt_config->model_height;

	get_layers(global_layers, htop, Nv);

//	print_layers(global_layers);

}

/* FUNCTION - INIT_GLOBAL_NODESET
 * - reads/determines the global nodeset and initializes the relevant members of global_domains
 */
void init_global_nodeset() {

	int Nnodes;
	int n = rt_config->stencil_size;

	// Get the global nodeset
	if (rt_config->nodeset_from_file == TRUE) {
		global_domains->Nnodes = get_nodeset_from_file(rt_config->nodeset_file, global_domains->global_nodeset);
	}
	else {
		abort_solver("Nodeset creation not yet implemented. Please provide a NetCDF file containing a valid nodeset.");
	}

	// get stencils (idx) and weights (D) of the global nodeset
	get_stencils(global_domains->global_nodeset, n);

}

/* FUNCTION - PARTITION_GLOBAL_DOMAIN
 * - determines the MPI partitioning of the global domain and performs any necessary reordering
 *   of the nodeset for the partitioning
 */
void partition_global_domain() {

	// assign/get global_domains sizes/dimensions
	int Nparts = mpi_size;
	int Nnodes = global_domains->Nnodes;
	global_domains->Nparts = Nparts;

	if (rt_config->use_metis == TRUE) {

		// get the METIS partitioning
		get_metis_partitioning(global_domains, Nparts);
		
		// get the mappings for contiguous partitions based on the metis partitioning
		reorder_domain_contiguous_parts(global_domains);
	} 
	else {
		abort_solver("Only METIS partitioning is currently implemented. Please set ADV_USE_METIS=1 and rerun.");
	}

}

/* FUNCTION - INIT_PHYS_CONSTANTS
 * - Simply assigns the correct values to the global phys_constants struct members
 */
void init_phys_constants() {

	phys_constants->R = EARTH_RADIUS;
	phys_constants->g = GRAVITATIONAL_CONSTANT;

}

/* FUNCTION - GET_RUNTIME_CONFIG
 * - Reads all runtime environment parameterizations and assigns them to the associated members of 
 *   the global rt_config struct
 */
void get_rt_config() {

	// temporary pointer to hold the return value of getenv()
	char* result;

	// Have rank 0 set rt_config and broadcast to other ranks
	if (mpi_rank == 0) {

		result = getenv("ADV_NODESET_FROM_FILE");
		rt_config->nodeset_from_file = result == NULL ? DEFAULT_NODESET_FROM_FILE : atoi(result);

		result = getenv("ADV_NODESET_FILE");
		strcpy(rt_config->nodeset_file, result == NULL ? "NONE" : result);

		result = getenv("ADV_USE_METIS");
		rt_config->use_metis = result == NULL ? DEFAULT_USE_METIS : atoi(result);

		result = getenv("ADV_STENCIL_SIZE");
		rt_config->stencil_size = result == NULL ? DEFAULT_STENCIL_SIZE : atoi(result);

		result = getenv("ADV_NUM_LAYERS");
		rt_config->num_layers = result == NULL ? DEFAULT_NUM_LAYERS : atoi(result);

		result = getenv("ADV_MODEL_HEIGHT");
		rt_config->model_height = result == NULL ? DEFAULT_MODEL_HEIGHT : atof(result);

		result = getenv("ADV_TIMESTEP_LENGTH");
		rt_config->timestep_length = result == NULL ? DEFAULT_TIMESTEP : atof(result);

		// Print the resulting configuration to stdout
		printf("\n======================================= Advection Solver Parameterizations ====================================\n\n");

		printf("\tADV_NODESET_FROM_FILE: \t%d\n", rt_config->nodeset_from_file);
		printf("\tADV_NODESET_FILE:      \t%s\n", rt_config->nodeset_file);
		printf("\tADV_STENCIL_SIZE:      \t%d\n", rt_config->stencil_size);
		printf("\tADV_USE_METIS:         \t%d\n", rt_config->use_metis);
		printf("\tADV_NUM_LAYERS:        \t%d\n", rt_config->num_layers);
		printf("\tADV_MODEL_HEIGHT:      \t%.1f meters\n", rt_config->model_height);
		printf("\tADV_TIMESTEP_LENGTH:   \t%.1f seconds\n", rt_config->timestep_length);

		printf("\n===============================================================================================================\n\n");
	}

	// Broadcast resulting rt_config struct to the other ranks
	MPI_Bcast((void*) rt_config, sizeof(rt_config_struct), MPI_BYTE, 0, MPI_COMM_WORLD);

}



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "include/config.h"
#include "include/io.h"
#include "include/init_domains.h"
#include "include/patches.h"

int mpi_size;
int mpi_rank;

config_struct config;
global_params_struct global_params;

void get_config();
void set_global_params();

int main(int argc, char** argv) {

	// ============================ Startup MPI ================================ //
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	
	// ============================ Get Environment Parameters ================================ //
	
	//config_struct config[1];
	get_config();
	set_global_params();
	
	// ============================ Initialize Nodeset ================================ //
	
	// read unit sphere nodeset from file
	nodeset_struct nodeset[1];
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
	free(nodeset->patch_sizes);

	// ============================ Finalize MPI ================================ //
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	
}

void set_global_params() {

	// global domain dimensions
	int Nh = config.num_nodes;
	int Nv_compute = config.num_layers;
	int Nv = PAD_UP(Nv_compute + 2*GHOST_LAYER_DEPTH, MEM_ALIGNMENT);	// only pad total number of vertical layers to enforce cache alignment
	int n = config.stencil_size;

	// vertical layers
	double R_e = EARTH_RADIUS;
	double h_top = config.h_top;
	double dh = h_top / Nv_compute;
	
	// timestepping
	double dt = config.dt;

	// assign all values to global_params struct
	global_params.Nh = Nh;
	global_params.Nv_compute = Nv_compute;
	global_params.Nv = Nv;
	global_params.n = n;

	global_params.R_e = R_e;
	global_params.h_top = h_top;
	global_params.dh = dh;
	global_params.dt = dt;
}

void get_config() {

	char* result;

	if (mpi_rank == 0) {

		result = getenv("ADV_NODESET_FILE");
		if (result != NULL)
			strcpy(config.nodeset_input_file, result);
		else {
			printf("\n\nERROR: environment variable ADV_NODESET_FILE not set\n\n"); exit(0);
		}

		result = getenv("ADV_STENCIL_SIZE");
		config.stencil_size = result == NULL ? DEFAULT_STENCIL_SIZE : atoi(result);

		result = getenv("ADV_NUM_LAYERS");
		config.num_layers = result == NULL ? DEFAULT_NUM_LAYERS : atoi(result);

		result = getenv("ADV_NUM_NODES");
		if (result != NULL)
			config.num_nodes = atoi(result);
		else {
			printf("\n\nERROR: environment variable ADV_NUM_NODES not set\n\n"); exit(0);
		}

		result = getenv("ADV_MODEL_HEIGHT");
		config.h_top = result == NULL ? DEFAULT_MODEL_HEIGHT : atof(result);

		result = getenv("ADV_TIMESTEP");
		config.dt = result == NULL ? DEFAULT_TIMESTEP : atof(result);

		printf("\n======================================= Advection Solver Parameterizations ====================================\n\n");

		printf("\tADV_NODESET_FILE:\t%s\n", config.nodeset_input_file);
		printf("\tADV_NUM_NODES:   \t%d\n", config.num_nodes);
		printf("\tADV_STENCIL_SIZE:\t%d\n", config.stencil_size);
		printf("\tADV_NUM_LAYERS:  \t%d\n", config.num_layers);
		printf("\tADV_NUM_NODES:   \t%d\n", config.num_nodes);
		printf("\tADV_MODEL_HEIGHT:\t%.1f meters\n", config.h_top);
		printf("\tADV_TIMESTEP:    \t%.1f seconds\n", config.dt);

		printf("\n===============================================================================================================\n\n");
	}

	MPI_Bcast((void*) &config, sizeof(config_struct), MPI_BYTE, 0, MPI_COMM_WORLD);
}




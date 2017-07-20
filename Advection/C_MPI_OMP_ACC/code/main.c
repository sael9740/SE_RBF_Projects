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

void get_adv_params(adv_params_struct* adv_params);

int main(int argc, char** argv) {

	// ============================ Startup MPI ================================ //
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	
	// ============================ Get Environment Parameters ================================ //
	
	adv_params_struct adv_params[1];
	get_adv_params(adv_params);
	
	// ============================ Initialize Nodeset ================================ //
	
	// read unit sphere nodeset from file
	unit_nodeset_struct ns1[1];
	init_ns1(ns1, adv_params);

	// reorder nodes in ns1 for contiguous patch data
	reorder_ns1(ns1);

	// initialize local rank's patch data
	patch_struct LP[1];
	init_patches(LP, ns1, adv_params);

	// ============================ Free Remaining Data Space ================================ //

	free(ns1->x);
	free(ns1->y);
	free(ns1->z);
	free(ns1->D_idx);
	free(ns1->idx);
	free(ns1->patch_ids);
	free(ns1->patch_start_ids);
	free(ns1->patch_sizes);

	// ============================ Finalize MPI ================================ //
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	
}

void get_adv_params(adv_params_struct* adv_params) {

	strcpy(adv_params->nodesetFile, getenv("ADV_NODESET_FILE"));
	adv_params->n = atoi(getenv("ADV_STENCIL_SIZE"));
	adv_params->Nv = atoi(getenv("ADV_NVERT_LVLS"));

	if (mpi_rank == 0) {
		printf("\n======================================= Advection Solver Parameterizations ====================================\n\n");

		printf("ADV_NODESET_FILE:\t%s\n", adv_params->nodesetFile);
		printf("ADV_STENCIL_SIZE:\t%d\n", adv_params->n);
		printf("ADV_NVERT_LVLS:  \t%d\n", adv_params->Nv);

		printf("\n===============================================================================================================\n\n");
	}

}




#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "include/config.h"
#include "include/io.h"
#include "include/rbffd.h"


int mpi_size;
int mpi_rank;

adv_params_struct get_adv_params();

int main(int argc, char** argv) {

	// ============================ Startup MPI ================================ //
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	
	// ============================ Get Environment Parameters ================================ //
	
	adv_params_struct adv_params = get_adv_params();
	
	// ============================ Initialize Nodeset ================================ //
	
	// read unit sphere nodeset from file
	nodeset_struct ns1 = get_ns1(adv_params.nodesetFile);

	// get distance^2 quadrature weights
	double* D = get_D(ns1);

	// get n-nearest neighbor stencils
	int* idx = get_idx(D, ns1.Nh, adv_params.n);
	
	//if (mpi_rank == 0) {
	//	get_rbffd_DMs(ns1, adv_params.n);
	//}

	// ============================ Free Remaining Data Space ================================ //
	
	free(ns1.x);
	free(ns1.y);
	free(ns1.z);
	free(D);
	free(idx);

	// ============================ Finalize MPI ================================ //
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

adv_params_struct get_adv_params() {

	adv_params_struct adv_params;

	strcpy(adv_params.nodesetFile, getenv("ADV_NODESET_FILE"));
	adv_params.n = atoi(getenv("ADV_STENCIL_SIZE"));

	if (mpi_rank == 0) {
		printf("\n======================================= Advection Solver Parameterizations ====================================\n\n");

		printf("ADV_NODESET_FILE:\t%s\n", adv_params.nodesetFile);
		printf("ADV_STENCIL_SIZE:\t%d\n", adv_params.n);

		printf("\n===============================================================================================================\n\n");
	}

	return adv_params;
}




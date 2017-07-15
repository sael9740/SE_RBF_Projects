#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "include/config.h"
#include "include/io.h"


int mpi_size;
int mpi_rank;

adv_params_struct get_adv_params();

void print_ns1(nodeset_struct ns1);

int main(int argc, char** argv) {

	// ============================ Startup MPI ================================ //
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	
	// ============================ Get Environment Parameters ================================ //
	
	adv_params_struct adv_params = get_adv_params();
	
	// ============================ Initialize Nodeset ================================ //
	
	nodeset_struct ns1 = get_ns1(adv_params.nodesetFile);

	if (mpi_rank == 0)
		print_ns1(ns1);


	// ============================ Free Remaining Data Space ================================ //
	
	free(ns1.x);
	free(ns1.y);
	free(ns1.z);

	// ============================ Finalize MPI ================================ //
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}

adv_params_struct get_adv_params() {

	adv_params_struct adv_params;

	strcpy(adv_params.nodesetFile, getenv("ADV_NODESET_FILE"));

	if (mpi_rank == 0) {
		printf("\n======================================= Advection Solver Parameterizations ====================================\n\n");

		printf("ADV_NODESET_FILE:\t%s\n", adv_params.nodesetFile);

		printf("\n===============================================================================================================\n\n");
	}

	return adv_params;
}

void print_ns1(nodeset_struct ns1) {

	printf("Unit Nodeset:\n\tNh = %d\n\tNv = %d\n", ns1.Nh, ns1.Nv);

	for (int i = 0; i < ns1.Nh; i++) {
		printf("\t\tnodeid = %3d:  x = %4.2f,  y = %4.2f,  z = %e\n", i, ns1.x[i], ns1.y[i], ns1.z[i]);
	}
}

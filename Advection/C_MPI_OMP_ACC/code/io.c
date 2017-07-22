#include "include/io.h"

#include <stdio.h>

#include <netcdf.h>
#include <mpi.h>

extern int mpi_rank;
extern int mpi_size;

extern config_struct config;

// reads nodeset on unit sphere from netcdf file
void get_nodeset(nodeset_struct* nodeset) {

	// for unit nodeset Nv = 1
	nodeset->Nv = 1;
	size_t Nh_temp;

	// init netcdf variable ids
	int ncid;
	int ns_gid;
	int Nh_did;
	int x_vid, y_vid, z_vid, lambda_vid, phi_vid;

	// Only rank 0 reads nodeset
	if (mpi_rank == 0) {

		// open file and get nodeset groupid
		nc_open(config.nodeset_input_file, NC_NOWRITE, &ncid);
		nc_inq_ncid(ncid, "nodeset", &ns_gid);
		
		// get number of horizontal nodes
		nc_inq_dimid(ncid, "hid", &Nh_did);
		nc_inq_dimlen(ncid, Nh_did, &Nh_temp);
		nodeset->Nh = (int) Nh_temp;

	}

	// communicate Nh to all ranks
	MPI_Bcast((void*) &nodeset->Nh, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// allocate data space for x,y,z
	nodeset->x = (double*) malloc(sizeof(double) * nodeset->Nh);
	nodeset->y = (double*) malloc(sizeof(double) * nodeset->Nh);
	nodeset->z = (double*) malloc(sizeof(double) * nodeset->Nh);
	nodeset->lambda = (double*) malloc(sizeof(double) * nodeset->Nh);
	nodeset->phi = (double*) malloc(sizeof(double) * nodeset->Nh);

	// rank 0 read nodeset
	if (mpi_rank == 0) {

		// read x,y,z
		nc_inq_varid(ns_gid, "x", &x_vid);
		nc_get_var_double(ns_gid, x_vid, nodeset->x);
		nc_inq_varid(ns_gid, "y", &y_vid);
		nc_get_var_double(ns_gid, y_vid, nodeset->y);
		nc_inq_varid(ns_gid, "z", &z_vid);
		nc_get_var_double(ns_gid, z_vid, nodeset->z);
		nc_inq_varid(ns_gid, "lambda", &lambda_vid);
		nc_get_var_double(ns_gid, lambda_vid, nodeset->lambda);
		nc_inq_varid(ns_gid, "phi", &phi_vid);
		nc_get_var_double(ns_gid, phi_vid, nodeset->phi);

		// close file
		nc_close(ncid);
	}

	// communicate nodeset to all ranks
	MPI_Bcast((void*) nodeset->x, nodeset->Nh, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) nodeset->y, nodeset->Nh, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) nodeset->z, nodeset->Nh, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) nodeset->lambda, nodeset->Nh, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) nodeset->phi, nodeset->Nh, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Debugging: print each rank's nodeset
	/*for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\n\nPrinting rank %d nodeset\n\n",rank); fflush(stdout);
			print_nodeset(nodeset);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}*/

}

void print_nodeset(nodeset_struct* nodeset) {

	printf("Unit Nodeset:\n\tNh = %d\n\tNv = %d\n", nodeset->Nh, nodeset->Nv); fflush(stdout);
	
	for (int i = 0; i < nodeset->Nh; i++) {
		printf("\t\tnodeid = %3d:  x = %4.2f,  y = %4.2f,  z = %4.2f,  lambda = %4.2f,  phi = %4.2f\n",
				i, nodeset->x[i], nodeset->y[i], nodeset->z[i], nodeset->lambda[i], nodeset->phi[i]); fflush(stdout);
	}
}

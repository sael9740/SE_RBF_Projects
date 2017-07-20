#include "include/io.h"

#include <stdio.h>

#include <netcdf.h>
#include <mpi.h>

extern int mpi_rank;
extern int mpi_size;

// reads nodeset on unit sphere from netcdf file
void get_ns1_xyz(char* nodesetFile, unit_nodeset_struct* ns1) {

	// for unit nodeset Nv = 1
	ns1->Nv = 1;
	size_t Nh_temp;

	// init netcdf variable ids
	int ncid;
	int ns_gid;
	int Nh_did;
	int x_vid, y_vid, z_vid;

	// Only rank 0 reads nodeset
	if (mpi_rank == 0) {

		// open file and get nodeset groupid
		nc_open(nodesetFile, NC_NOWRITE, &ncid);
		nc_inq_ncid(ncid, "nodeset", &ns_gid);
		
		// get number of horizontal nodes
		nc_inq_dimid(ncid, "hid", &Nh_did);
		nc_inq_dimlen(ncid, Nh_did, &Nh_temp);
		ns1->Nh = (int) Nh_temp;

	}

	// communicate Nh to all ranks
	MPI_Bcast((void*) &ns1->Nh, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// allocate data space for x,y,z
	ns1->x = (double*) malloc(sizeof(double) * ns1->Nh);
	ns1->y = (double*) malloc(sizeof(double) * ns1->Nh);
	ns1->z = (double*) malloc(sizeof(double) * ns1->Nh);

	// rank 0 read nodeset
	if (mpi_rank == 0) {

		// read x,y,z
		nc_inq_varid(ns_gid, "x", &x_vid);
		nc_get_var_double(ns_gid, x_vid, ns1->x);
		nc_inq_varid(ns_gid, "y", &y_vid);
		nc_get_var_double(ns_gid, y_vid, ns1->y);
		nc_inq_varid(ns_gid, "z", &z_vid);
		nc_get_var_double(ns_gid, z_vid, ns1->z);

		// close file
		nc_close(ncid);
	}

	// communicate nodeset to all ranks
	MPI_Bcast((void*) ns1->x, ns1->Nh, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) ns1->y, ns1->Nh, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) ns1->z, ns1->Nh, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Debugging: print each rank's nodeset
	/*for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\n\nPrinting rank %d ns1\n\n",rank); fflush(stdout);
			print_ns1(ns1);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}*/

}

void print_ns1(unit_nodeset_struct* ns1) {

	printf("Unit Nodeset:\n\tNh = %d\n\tNv = %d\n", ns1->Nh, ns1->Nv); fflush(stdout);
	
	for (int i = 0; i < ns1->Nh; i++) {
		printf("\t\tnodeid = %3d:  x = %4.2f,  y = %4.2f,  z = %4.2f\n", i, ns1->x[i], ns1->y[i], ns1->z[i]); fflush(stdout);
	}
}

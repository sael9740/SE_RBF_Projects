#include "include/nodesets.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <netcdf.h>
#include <mpi.h>

extern int mpi_rank;
extern int mpi_size;

/* FUNCTION - GET_NODESET_FROM_FILE
 * - reads a nodeset from the netcdf input file
 * - Inputs:
 *		- nodeset_file -> path to input file
 *		- nodeset -> pointer to a nodeset_struct that the data will be assigned to
 * - returns the number of nodes in the nodeset
 */
int get_nodeset_from_file(char* nodeset_file, nodeset_struct* nodeset) {

	// use temp size_t variable for netcdf call before casting to int
	size_t Nnodes_temp;
	int Nnodes;

	// init netcdf variable ids
	int ncid;
	int ns_gid;
	int Nnodes_did;
	int x_vid, y_vid, z_vid, lambda_vid, phi_vid;

	// Only rank 0 reads nodeset
	if (mpi_rank == 0) {

		// open file and get nodeset groupid
		nc_open(nodeset_file, NC_NOWRITE, &ncid);
		nc_inq_ncid(ncid, "nodeset", &ns_gid);
		
		// get number of horizontal nodes
		nc_inq_dimid(ncid, "hid", &Nnodes_did);
		nc_inq_dimlen(ncid, Nnodes_did, &Nnodes_temp);
		Nnodes = (int) Nnodes_temp;

	}

	// communicate Nnodes to all ranks
	MPI_Bcast((void*) &Nnodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// allocate data space for coordinates
	double* x = (double*) malloc(sizeof(double) * Nnodes);
	double* y = (double*) malloc(sizeof(double) * Nnodes);
	double* z = (double*) malloc(sizeof(double) * Nnodes);
	double* lambda = (double*) malloc(sizeof(double) * Nnodes);
	double* phi = (double*) malloc(sizeof(double) * Nnodes);

	// rank 0 read nodeset
	if (mpi_rank == 0) {

		// read coordinates
		nc_inq_varid(ns_gid, "x", &x_vid);
		nc_get_var_double(ns_gid, x_vid, x);
		nc_inq_varid(ns_gid, "y", &y_vid);
		nc_get_var_double(ns_gid, y_vid, y);
		nc_inq_varid(ns_gid, "z", &z_vid);
		nc_get_var_double(ns_gid, z_vid, z);
		nc_inq_varid(ns_gid, "lambda", &lambda_vid);
		nc_get_var_double(ns_gid, lambda_vid, lambda);
		nc_inq_varid(ns_gid, "phi", &phi_vid);
		nc_get_var_double(ns_gid, phi_vid, phi);

		// close file
		nc_close(ncid);
	}

	// communicate coordinate subsets to all ranks
	MPI_Bcast((void*) x, Nnodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) y, Nnodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) z, Nnodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) lambda, Nnodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) phi, Nnodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Assign members of the nodeset
	nodeset->Nnodes = Nnodes;
	nodeset->x = x;
	nodeset->y = y;
	nodeset->z = z;
	nodeset->lambda = lambda;
	nodeset->phi = phi;

	// return the number of nodes in the nodeset
	return Nnodes;
}

/* FUNCTION - GET_STENCILS
 * - Determines and assigns the nearest neighbor stencils (idx) and distances (D) for a nodeset
 * - Inputs:
 *  	- nodeset -> nodeset to determine nearest neighbor stencils
 *  	- n -> number of nodes in each stencil
 * - Methodology: 
 * 		1) Calcualte the distance^2 matrix for ALL node pairings (DD2) in the nodeset
 * 			- O(Nnodes^2) operations
 * 		2) Find the n smallest weights in each row of DD2 to determine the nearest neighbor 
 * 			stencils (idx)
 * 			- O(n*Nnodes^2) operations 
 * 		3) Calculate the distance matrix for just the stencil node pairings (D) in idx
 * 			- O(n*Nnodes) operations
 * 		Notes on Parallelization:
 * 			- Calculations 1 & 2 in particular can be extremely time consuming
 * 			- Can easily parallelize these calculations
 * 				- Simple linear decomposition of outer loops results in each rank calculating a 
 * 					slice of rows at each step
 * 				- no dependencies between calculations for MPI
 * 				- resulting slices are then broadcast to neighboring ranks
 */
void get_stencils(nodeset_struct* nodeset, int n) {

	nodeset->n = n;

	// extract constants
	int Nnodes = nodeset->Nnodes;

	// get mpi DD2/idx compute slice bounds for current rank
	int start_id = (Nnodes/mpi_size) * mpi_rank;
	int size = mpi_rank == mpi_size-1 ? Nnodes : (Nnodes/mpi_size) * (mpi_rank+1); size -= start_id;

	// calculate slice of DD2
	double* DD2_slice = get_DD2_slice(nodeset, start_id, size);

	// allocate space for idx and D matrices
	int* idx = (int*) malloc(sizeof(int) * Nnodes * n);
	double* D = (double*) malloc(sizeof(double) * Nnodes * n);

	// calculate slices of idx and D matrices
	get_D_idx_slice(&D[start_id*n], &idx[start_id*n], DD2_slice, nodeset, start_id, size);

	// communicate partitions of idx to other ranks
	for (int rank = 0; rank < mpi_size; rank++) {

		// mpi get slice info for mpi_rank == rank
		int start_id_r = start_id;
		int size_r = size;
		MPI_Bcast((void*) &start_id_r, 1, MPI_INT, rank, MPI_COMM_WORLD);
		MPI_Bcast((void*) &size_r, 1, MPI_INT, rank, MPI_COMM_WORLD);

		// broadcast slice data to other ranks
		MPI_Bcast((void*) &idx[start_id_r*n], size_r*n, MPI_INT, rank, MPI_COMM_WORLD);
		MPI_Bcast((void*) &D[start_id_r*n], size_r*n, MPI_DOUBLE, rank, MPI_COMM_WORLD);
	}

	free(DD2_slice);

	nodeset->idx = idx;
	nodeset->D = D;

}

/* FUNCTION - GET_DD2_SLICE
 * - Calculates a slice of the DD2 matrix starting with row start_id with a size of size rows
 * - Inputs:
 *  	- nodeset -> nodeset to calculate slice for
 *   	- start_id -> starting node id of slice
 *   	- size -> size of slice
 */
double* get_DD2_slice(nodeset_struct* nodeset, int start_id, int size) {

	// extract constants
	int Nnodes = nodeset->Nnodes;

	// allocate space for DD2_slice matrix
	double* DD2_slice = (double*) malloc(sizeof(double) * Nnodes * size);

	// iterate through each row of slice
	for (int i = 0; i < size; i++) {	
		
		// iterate through each node in the nodeset
		#pragma omp parallel for simd
		for (int j = 0; j < Nnodes; j++) {		
			
			// calculate associated weight
			DD2_slice[(i*Nnodes) + j] = pow(nodeset->x[i+start_id] - nodeset->x[j], 2) 
						+ pow(nodeset->y[i+start_id] - nodeset->y[j], 2) 
						+ pow(nodeset->z[i+start_id] - nodeset->z[j], 2);
		}
	}

	return DD2_slice;
}


/* FUNCTION - GET_D_IDX_SLICE
 * - Calculates a slice of both the idx and D matrices (see get_stencils description)
 * - Inputs:
 *   	- D_slice -> allocated slice of D matrix
 *   	- idx_slice -> allocated slice of idx matrix
 *   	- DD2_slice -> initialized slice of matrix DD2
 *  	- nodeset -> nodeset pointer to calculate slice for
 *   	- start_id -> starting node id of slice
 *   	- size -> number of nodes in the slice
 */
void get_D_idx_slice(double* D_slice, int* idx_slice, double* DD2_slice, nodeset_struct* nodeset, int start_id, int size) {

	// extract nodeset data
	int Nnodes = nodeset->Nnodes;
	int n = nodeset->n;


	/***** Determining Nearest Neighbor Stencils *****/

	/* Initialize D_slice = 5.0 > max(DD2_ij) to ensure replacement on unassigned neighbor and 
	 * idx_slice to -1 to identify unassigned neighbors.
	 */
	#pragma omp parallel for simd
	for (int i = 0; i < size*n; i++) {
		D_slice[i] = 5.0;
		idx_slice[i] = -1;
	}

	// iterate through each stencil in the slice
	#pragma omp parallel for simd
	for (int i = 0; i < size; i++) {
		
		// iterate through each node in global domain
		for (int j = 0; j < Nnodes; j++) {
			
			// get associated weight (distance) and node id of potential neighbor
			double d_ij_k = DD2_slice[(i*Nnodes) + j];
			int idx_ij_k = j;
			
			// iterate through each neighbor of current stencil
			for (int k = 0; k < n; k++) {
				
				/* if weight is smaller than neighbor, assign new nearest neighbor and continue 
				 * iteration with old neighbors id/weight
				 */
				if (D_slice[(i*n)+k] > d_ij_k) {

					// preform swap of weights/neighbor node ids
					double d_temp = d_ij_k;
					int idx_temp = idx_ij_k;
					d_ij_k = D_slice[(i*n)+k];
					idx_ij_k = idx_slice[(i*n)+k];
					D_slice[(i*n)+k] = d_temp;
					idx_slice[(i*n)+k] = idx_temp;

					/* if old neighbor was not assigned break loop since old node shifting is not 
					 * necessary
					 */
					if (idx_temp == -1) {
						break;
					}
				}
			}
		}
	}


	/***** Getting Distances *****/

	// iterate through each stencil in the slice
	#pragma omp parallel for simd
	for (int i = 0; i < size; i++) {

		// iterate through each neighbor node
		for (int j = 0; j < n; j++) {

			// calculate distance (just take square root)
			D_slice[(i*n)+j] = sqrt(D_slice[(i*n)+j]);
		}
	}
}


/*void reorder_nodeset(nodeset_struct* nodeset_p, int* mapping, int* inv_mapping) {

	int Nnodes = nodeset_p->Nnodes;
	int n = nodeset_p->n;


	// reorder coordinates
	double* x = (double*) malloc(sizeof(double) * Nnodes);
	double* y = (double*) malloc(sizeof(double) * Nnodes);
	double* z = (double*) malloc(sizeof(double) * Nnodes);
	double* lambda = (double*) malloc(sizeof(double) * Nnodes);
	double* phi = (double*) malloc(sizeof(double) * Nnodes);

	for (int i = 0; i < Nnodes; i++) {
		x[i] = nodeset_p->x[inv_mapping[i]];
		y[i] = nodeset_p->y[inv_mapping[i]];
		z[i] = nodeset_p->z[inv_mapping[i]];
		lambda[i] = nodeset_p->lambda[inv_mapping[i]];
		phi[i] = nodeset_p->phi[inv_mapping[i]];
	}

	// reorder D_idx matrix
	double* D = (double*) malloc(sizeof(double) * Nnodes * n);
	
	for (int i = 0; i < Nnodes; i++) {
		for (int j = 0; j < n; j++) {
			D[(i*n) + j] = nodeset_p->D[(inv_mapping[i]*n) + j];
		}
	}
	

	// reorder idx and map old id values in idx to the new ordering
	int* idx = (int*) malloc(sizeof(int) * Nnodes * n);

	for (int i = 0; i < Nnodes; i++) {
		for (int j = 0; j < n; j++) {
			idx[(i*n) + j] = mapping[nodeset_p->idx[(inv_mapping[i]*n) + j]];
		}
	}

	free(nodeset_p->x);
	free(nodeset_p->y);
	free(nodeset_p->z);
	free(nodeset_p->lambda);
	free(nodeset_p->phi);
	free(nodeset_p->idx);
	free(nodeset_p->D);

	nodeset_p->idx = idx;
	nodeset_p->D = D;
	nodeset_p->x = x;
	nodeset_p->y = y;
	nodeset_p->z = z;
	nodeset_p->lambda = lambda;
	nodeset_p->phi = phi;


}*/

/* FUNCTION - REORDER_NODESET
 * - reorders all data in the input nodeset struct
 * - Selcted Variable Descriptions:
 * 		- mapping -> mapping from old to new node id
 * 		- inv_mapping -> mapping from new to old node id
 * 		- distributed -> binary switch indicating if reordering is shared between MPI ranks and 
 * 			the reordering should be parallelized
 */
void reorder_nodeset(nodeset_struct* nodeset, int* mapping, int* inv_mapping, int distributed) {

	// extract nodeset data
	int Nnodes = nodeset->Nnodes;
	int n = nodeset->n;

	// calculate bounds
	int start_id, end_id, size;

	/* For distributed MPI calculation -> get slice bounde
	 * If serial caclulation -> slice is full matrix
	 */
	if (distributed == TRUE) {
		start_id = (Nnodes/mpi_size) * mpi_rank;
		end_id = mpi_rank == mpi_size-1 ? Nnodes : (Nnodes/mpi_size) * (mpi_rank+1);
		size = end_id - start_id;
	}
	else {
		start_id = 0;
		end_id = Nnodes;
		size = end_id - start_id;
	}


	// reorder coordinates
	double* x = (double*) malloc(sizeof(double) * Nnodes);
	double* y = (double*) malloc(sizeof(double) * Nnodes);
	double* z = (double*) malloc(sizeof(double) * Nnodes);
	double* lambda = (double*) malloc(sizeof(double) * Nnodes);
	double* phi = (double*) malloc(sizeof(double) * Nnodes);

	for (int i = start_id; i < end_id; i++) {
		x[i] = nodeset->x[inv_mapping[i]];
		y[i] = nodeset->y[inv_mapping[i]];
		z[i] = nodeset->z[inv_mapping[i]];
		lambda[i] = nodeset->lambda[inv_mapping[i]];
		phi[i] = nodeset->phi[inv_mapping[i]];
	}

	// reorder D_idx matrix
	double* D = (double*) malloc(sizeof(double) * Nnodes * n);

	for (int i = start_id; i < end_id; i++) {
		for (int j = 0; j < n; j++) {
			D[(i*n) + j] = nodeset->D[(inv_mapping[i]*n) + j];
		}
	}

	// reorder idx and map old id values in idx to the new ordering
	int* idx = (int*) malloc(sizeof(int) * Nnodes * n);

	for (int i = start_id; i < end_id; i++) {
		for (int j = 0; j < n; j++) {
			idx[(i*n) + j] = mapping[nodeset->idx[(inv_mapping[i]*n) + j]];
		}
	}

	// For distributed caclulation -> broadcast each ranks slice
	if (distributed == TRUE) {
		for (int rank = 0; rank < mpi_size; rank++) {

			// mpi get slice info for mpi_rank == rank
			int start_id_r = start_id;
			int size_r = size;
			MPI_Bcast((void*) &start_id_r, 1, MPI_INT, rank, MPI_COMM_WORLD);
			MPI_Bcast((void*) &size_r, 1, MPI_INT, rank, MPI_COMM_WORLD);

			// broadcast slice data to other ranks
			MPI_Bcast((void*) &x[start_id_r], size_r, MPI_DOUBLE, rank, MPI_COMM_WORLD);
			MPI_Bcast((void*) &y[start_id_r], size_r, MPI_DOUBLE, rank, MPI_COMM_WORLD);
			MPI_Bcast((void*) &z[start_id_r], size_r, MPI_DOUBLE, rank, MPI_COMM_WORLD);
			MPI_Bcast((void*) &lambda[start_id_r], size_r, MPI_DOUBLE, rank, MPI_COMM_WORLD);
			MPI_Bcast((void*) &phi[start_id_r], size_r, MPI_DOUBLE, rank, MPI_COMM_WORLD);

			MPI_Bcast((void*) &D[start_id_r*n], size_r*n, MPI_DOUBLE, rank, MPI_COMM_WORLD);

			MPI_Bcast((void*) &idx[start_id_r*n], size_r*n, MPI_INT, rank, MPI_COMM_WORLD);
		}
	}

	free(nodeset->x);
	free(nodeset->y);
	free(nodeset->z);
	free(nodeset->lambda);
	free(nodeset->phi);
	free(nodeset->idx);
	free(nodeset->D);

	nodeset->idx = idx;
	nodeset->D = D;
	nodeset->x = x;
	nodeset->y = y;
	nodeset->z = z;
	nodeset->lambda = lambda;
	nodeset->phi = phi;

}



void print_stencils(nodeset_struct* nodeset) {

	int Nnodes = nodeset->Nnodes;
	int n = nodeset->n;
	int* idx = nodeset->idx;
	double* D = nodeset->D;

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\nRank %d idx:\n", mpi_rank); fflush(stdout);
			for (int i = 0; i < Nnodes; i++) {
				printf("\n\tnode id = %d:\t\t", i);
				for (int j = 0; j < n; j++) {
					printf("%6d,%6.4f\t", idx[(i*n)+j], D[(i*n)+j]); fflush(stdout);
				}
			}
			printf("\n");
		}
		usleep(1000);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}




























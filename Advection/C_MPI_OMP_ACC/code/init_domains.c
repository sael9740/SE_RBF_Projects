#include "include/init_domains.h"
#include "include/io.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>
#include <mkl_lapacke.h>
#include <metis.h>

extern int mpi_rank;
extern int mpi_size;

extern config_struct config;

void print_fp_matrix(double* A, int dim1, int dim2);
void print_int_matrix(int* A, int dim1, int dim2);

void init_nodeset(nodeset_struct* nodeset) {

	nodeset->n = config.stencil_size;
	get_nodeset(nodeset);
	
	int Nh = nodeset->Nh;

	// determine mpi loop partitioning for calculating D and determining idx
	int start_id = (Nh/mpi_size) * mpi_rank;
	int size = mpi_rank == mpi_size-1 ? Nh : (Nh/mpi_size) * (mpi_rank+1); size -= start_id;

	// get quadrature weights
	double* D_r = get_D_r(nodeset, start_id, size);

	// get n-nearest neighbor stencils
	get_idx(D_r, nodeset, start_id, size);
	
	// use metis to determine MPI patch partitioning
	get_partitions(nodeset);

	free(D_r);
}

// calculate quadrature weights (euclidian distances) for all node pairings 
// returns horizontal slice of D matrix for each mpi rank
double* get_D_r(nodeset_struct* nodeset, int start_id, int size) {

	int Nh = nodeset->Nh;

	// allocate space for D matrix
	double* D_r = (double*) malloc(sizeof(double) * Nh * size);

	// calculate weights
	#pragma omp parallel for simd
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < Nh; j++) {
			D_r[(i*Nh) + j] = sqrt(pow(nodeset->x[i+start_id] - nodeset->x[j], 2) 
						+ pow(nodeset->y[i+start_id] - nodeset->y[j], 2) 
						+ pow(nodeset->z[i+start_id] - nodeset->z[j], 2));
		}
	}

	return D_r;
}

// determine n-nearest neighbor stencils for each node
void get_idx(double* D_r, nodeset_struct* nodeset, int start_id, int size) {

	int Nh = nodeset->Nh;
	int n = nodeset->n;
	
	// allocate space for idx and D_idx
	int* idx = (int*) malloc(sizeof(int) * Nh * n);
	double* D_idx = (double*) malloc(sizeof(double) * Nh * n);

	// initialize D_idx = 3.0 > max(D_ij) to ensure replacement on unassigned neighbor
	// and idx to -1s to identify unassigned neighbors
	for (int i = 0; i < size*n; i++) {
		D_idx[(start_id * n) + i] = 3.0;
		idx[(start_id * n) + i] = -1;
	}

	// iterate through each stencil in rank's partition
	#pragma omp parallel for simd
	for (int i = 0; i < size; i++) {
		// iterate through each node in global domain
		for (int j = 0; j < Nh; j++) {
			// get associated weight (distance) and node id of potential neighbor
			double d = D_r[(i*Nh) + j];
			int idx_temp1 = j;
			// iterate through each neighbor of current stencil
			for (int k = 0; k < n; k++) {
				// if weight is smaller than neighbor, assign new nearest 
				// neighbor and continue iteration with old neighbors id/weight
				if (D_idx[((i+start_id)*n)+k] > d) {
					// preform swap of weights/neighbor node ids
					double d_temp = d;
					int idx_temp2 = idx_temp1;
					d = D_idx[((i+start_id)*n)+k];
					idx_temp1 = idx[((i+start_id)*n)+k];
					D_idx[((i+start_id)*n)+k] = d_temp;
					idx[((i+start_id)*n)+k] = idx_temp2;

					// if old neighbor was not assigned break loop since
					// old node shifting is not necessary
					if (idx_temp1 == -1) {
						break;
					}
				}
			}
		}
	}

	// communicate partitions of idx to other ranks
	for (int rank = 0; rank < mpi_size; rank++) {

		// mpi loop partitioning info for communication
		int start_id_r = (Nh/mpi_size) * rank;
		int size_r = rank == mpi_size-1 ? Nh : (Nh/mpi_size) * (rank+1); size_r -= start_id_r;

		MPI_Bcast((void*) &idx[start_id_r*n], size_r*n, MPI_INT, rank, MPI_COMM_WORLD);
		MPI_Bcast((void*) &D_idx[start_id_r*n], size_r*n, MPI_DOUBLE, rank, MPI_COMM_WORLD);
	}

	// Debugging: print each ranks resulting idx
	/*for (int rank = 0; rank < mpi_size; rank++) {
		if (mpi_rank == rank) {
			printf("\nRank %d's idx Matrix:\n", rank);
			fflush(stdout);
			print_int_matrix(idx, Nh, n);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}*/


	nodeset->idx = idx;
	nodeset->D_idx = D_idx;

}

// get mpi partitioning of the domain using metis
void get_partitions(nodeset_struct* nodeset) {

	int* idx = nodeset->idx;
	int Nh = nodeset->Nh;
	int n = nodeset->n;

	// variables for metis
	idx_t Nvert = Nh;
	idx_t Ncon = 1;
	idx_t Nparts = mpi_size;
	idx_t objval;

	// graph and partition data
	idx_t* xadj;
	idx_t* adjncy;
	idx_t* patch_ids = (idx_t*) malloc(sizeof(idx_t) * Nh);

	if (mpi_rank == 0) {
		// graph and partition data
		xadj = (idx_t*) malloc(sizeof(idx_t) * (Nh + 1));
		adjncy = (idx_t*) malloc(sizeof(idx_t) * Nh * (n - 1));

		// setup metis graph data
		for (int i = 0; i < Nh + 1; i++) {
			xadj[i] = i * (n - 1);
		}
		for (int i = 0; i < Nh; i++) {
			for (int j = 1; j < n; j++) {
				adjncy[xadj[i] + j - 1] = idx[(i * n) + j];
			}
		}
		// Debugging
		/*for (int i = 0; i < Nh; i++) {
			printf("\nxadj[%d] = %d:\t\t", i, xadj[i]); fflush(stdout);
			for (int j = 1; j < n; j++) {
				printf("%3d\t", adjncy[xadj[i] + j - 1]); fflush(stdout);
			}
		}*/

		METIS_PartGraphKway(&Nvert, &Ncon, xadj, adjncy, NULL,
				NULL, NULL, &Nparts, NULL, NULL, NULL, &objval, patch_ids);

		free(xadj);
		free(adjncy);
	}
	

	MPI_Bcast((void*) patch_ids, Nh, MPI_INT, 0, MPI_COMM_WORLD);

	// Debugging
	/*for (int rank = 0; rank < mpi_size; rank++) {
		if (mpi_rank == rank) {
			printf("\nRank %d's partition matrix:\n", rank);
			for (int i = 0; i < Nh; i++) {
				printf("\npart[%d] = %d", i, part[i]);
				fflush(stdout);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}*/

	nodeset->patch_ids = (int*) patch_ids;

}

void reorder_nodeset(nodeset_struct* nodeset) {

	int Nh = nodeset->Nh;
	int n = nodeset->n;
	
	// allocate dataspace
	int* patch_ids = (int*) malloc(sizeof(int) * Nh);
	int* patch_sizes = (int*) malloc(sizeof(int) * mpi_size);
	int* patch_start_ids = (int*) malloc(sizeof(int) * mpi_size);

	// mapping and inverse mapping arrays
	int* mapping = (int*) malloc(sizeof(int) * Nh);		// mapping[old_id] = new_id
	int* inv_mapping = (int*) malloc(sizeof(int) * Nh);	// inv_mapping[new_id] = old_id
	
	// determine mapping
	int counter1 = 0;
	for (int rank = 0; rank < mpi_size; rank++) {
		
		patch_start_ids[rank] = counter1;
		int counter2 = 0;

		for (int i = 0; i < Nh; i++) {

			if (nodeset->patch_ids[i] == rank) {
				patch_ids[counter1] = rank;
				mapping[i] = counter1;
				inv_mapping[counter1] = i;
				counter1++;
				counter2++;
			}
		}

		patch_sizes[rank] = counter2;
	}

	// reorder x,y,z
	double* x = (double*) malloc(sizeof(double) * Nh);
	double* y = (double*) malloc(sizeof(double) * Nh);
	double* z = (double*) malloc(sizeof(double) * Nh);
	double* lambda = (double*) malloc(sizeof(double) * Nh);
	double* phi = (double*) malloc(sizeof(double) * Nh);

	for (int i = 0; i < Nh; i++) {
		x[i] = nodeset->x[inv_mapping[i]];
		y[i] = nodeset->y[inv_mapping[i]];
		z[i] = nodeset->z[inv_mapping[i]];
		lambda[i] = nodeset->lambda[inv_mapping[i]];
		phi[i] = nodeset->phi[inv_mapping[i]];
	}

	// reorder D_idx matrix
	double* D_idx = (double*) malloc(sizeof(double) * Nh * n);
	
	for (int i = 0; i < Nh; i++) {
		for (int j = 0; j < n; j++) {
			D_idx[(i*n) + j] = nodeset->D_idx[(inv_mapping[i]*n) + j];
		}
	}
	

	// reorder idx and map old id values in idx to the new ordering
	int* idx = (int*) malloc(sizeof(int) * Nh * n);

	for (int i = 0; i < Nh; i++) {
		for (int j = 0; j < n; j++) {
			idx[(i*n) + j] = mapping[nodeset->idx[(inv_mapping[i]*n) + j]];
		}
	}

	// Debugging
	/*if (mpi_rank == 0) {
		printf("\n\npatch_ids: \n\t");
		for (int i = 0; i < Nh; i++)
			printf("%d\t",nodeset->patch_ids[i]);
		printf("\n\nOld idx matrix:\n\n");
		print_int_matrix(nodeset->idx, Nh, n);
		printf("\n\nNew idx matrix:\n\n");
		print_int_matrix(idx, Nh, n);
		printf("\n\nOld D_idx matrix:\n\n");
		print_fp_matrix(nodeset->D_idx, Nh, n);
		printf("\n\nNew D_idx matrix:\n\n");
		print_fp_matrix(D_idx, Nh, n);
	}*/

	// Assign reordered data to nodeset and free old data
	free(mapping);
	free(inv_mapping);
	
	free(nodeset->x);
	free(nodeset->y);
	free(nodeset->z);
	free(nodeset->lambda);
	free(nodeset->phi);
	free(nodeset->idx);
	free(nodeset->D_idx);
	free(nodeset->patch_ids);

	nodeset->patch_sizes = patch_sizes;
	nodeset->patch_start_ids = patch_start_ids;
	nodeset->patch_ids = patch_ids;
	nodeset->idx = idx;
	nodeset->D_idx = D_idx;
	nodeset->x = x;
	nodeset->y = y;
	nodeset->z = z;
	nodeset->lambda = lambda;
	nodeset->phi = phi;


}


void print_int_matrix(int* A, int dim1, int dim2) {

	for (int i = 0; i < dim1; i++) {
		printf("\n\t"); fflush(stdout);
		for (int j = 0; j < dim2; j++) {
			printf("%6d", A[(i*dim2) + j]); fflush(stdout);
		}
	}
	printf("\n\n"); fflush(stdout);
}

void print_fp_matrix(double* A, int dim1, int dim2) {

	for (int i = 0; i < dim1; i++) {
		printf("\n\t"); fflush(stdout);
		for (int j = 0; j < dim2; j++) {
			printf("%.3f\t", A[(i*dim2) + j]); fflush(stdout);
		}
	}
	printf("\n\n"); fflush(stdout);
}

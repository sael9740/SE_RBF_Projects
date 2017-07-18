#include "include/rbffd.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>
#include <mkl_lapacke.h>

extern int mpi_rank;
extern int mpi_size;


int* get_idx(double* D, size_t Nh, size_t n);
void print_fp_matrix(double* A, size_t dim1, size_t dim2);
void print_int_matrix(int* A, size_t dim1, size_t dim2);

void get_rbffd_DMs(nodeset_struct ns1, size_t n) {

	double tstart;

	tstart = MPI_Wtime();
	double* D = get_D(ns1);
	printf("\nget_D() walltime = %.3f seconds\n", MPI_Wtime() - tstart);

	tstart = MPI_Wtime();
	int* idx = get_idx(D, ns1.Nh, n);
	printf("\nget_idx() walltime = %.3f seconds\n", MPI_Wtime() - tstart);
	//print_idx(idx, Nh, n);

	//get_Dx(idx, D2, ns1, Nh, n);

}

// calculate quadrature weights (euclidian distances) for all node pairings
double* get_D(nodeset_struct ns1) {

	size_t Nh = ns1.Nh;

	// determine mpi partitioning
	int start_id = (Nh/mpi_size) * mpi_rank;
	int end_id = mpi_rank == mpi_size-1 ? Nh : (Nh/mpi_size) * (mpi_rank+1);
	int size = end_id - start_id;

	// allocate space for D2 matrix
	double* D = (double*) malloc(sizeof(double) * Nh * size);

	// calculate weights
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < Nh; j++) {
			D[(i*Nh) + j] = sqrt(pow(ns1.x[i+start_id] - ns1.x[j], 2) 
						+ pow(ns1.y[i+start_id] - ns1.y[j], 2) 
						+ pow(ns1.z[i+start_id] - ns1.z[j], 2));
		}
	}

	return D;
}

// determine n-nearest neighbor stencils for each node
int* get_idx(double* D, size_t Nh, size_t n) {

	// determine mpi partitioning
	int start_id = (Nh/mpi_size) * mpi_rank;
	int end_id = mpi_rank == mpi_size-1 ? Nh : (Nh/mpi_size) * (mpi_rank+1);
	int size = end_id - start_id;

	// allocate space for idx and D2_idx
	int* idx = (int*) malloc(sizeof(int) * size * n);
	double* D_idx = (double*) malloc(sizeof(double) * size * n);

	// initialize D_idx = 3.0 > max(D_ij) to ensure replacement on unassigned neighbor
	// and idx to -1s to identify unassigned neighbors
	for (int i = 0; i < size*n; i++) {
		D_idx[i] = 3.0;
		idx[i] = -1;
	}

	// iterate through each stencil in rank's partition
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		// iterate through each node in global domain
		for (int j = 0; j < Nh; j++) {
			// get associated weight (distance) and node id of potential neighbor
			double d = D[(i*Nh) + j];
			int idx_temp1 = j;
			// iterate through each neighbor of current stencil
			for (int k = 0; k < n; k++) {
				// if weight is smaller than neighbor, assign new nearest 
				// neighbor and continue iteration with old neighbors id/weight
				if (D_idx[(i*n)+k] > d) {
					// preform swap of weights/neighbor node ids
					double d_temp = d;
					int idx_temp2 = idx_temp1;
					d = D_idx[(i*n)+k];
					idx_temp1 = idx[(i*n)+k];
					D_idx[(i*n)+k] = d_temp;
					idx[(i*n)+k] = idx_temp2;

					// if old neighbor was not assigned break loop since
					// old node shifting is not necessary
					if (idx_temp1 == -1) {
						break;
					}
				}
			}
		}
	}

	// Debugging: print each ranks resulting idx
	/*for (int rank = 0; rank < mpi_size; rank++) {
		if (mpi_rank == rank) {
			printf("\nRank %d's idx Matrix:\n", rank);
			fflush(stdout);
			print_int_matrix(idx, size, n);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}*/

	return (idx);

}

void print_int_matrix(int* A, size_t dim1, size_t dim2) {
	
	for (int i = 0; i < dim1; i++) {
		printf("\n\t"); fflush(stdout);
		for (int j = 0; j < dim2; j++) {
			printf("%6d", A[(i*dim2) + j]); fflush(stdout);
		}
	}
	printf("\n\n"); fflush(stdout);
}

void print_fp_matrix(double* A, size_t dim1, size_t dim2) {
	
	for (int i = 0; i < dim1; i++) {
		printf("\n\t"); fflush(stdout);
		for (int j = 0; j < dim2; j++) {
			printf("%.3f\t", A[(i*dim2) + j]); fflush(stdout);
		}
	}
	printf("\n\n"); fflush(stdout);
}

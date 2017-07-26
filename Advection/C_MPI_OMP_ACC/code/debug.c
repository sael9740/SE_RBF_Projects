#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>

#include "include/debug.h"

extern int mpi_rank;
extern int mpi_size;

// if called prints message and exits program
void abort_solver(char* message) {
	printf("\n\nADVECTION SOLVER ERROR (RANK %d): %s\nABORTING SOLVER\n\n", mpi_rank, message);
	exit(0);
}

void checkpoint(int num) {

	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);
	if (mpi_rank == 0) {
		printf("\nCHECKPOINT %d\n", num);
		fflush(stdout);
	}
	usleep(100000);
	MPI_Barrier(MPI_COMM_WORLD);
}

void print_nodeset(nodeset_struct* nodeset) {

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\nRank %d nodeset:\n", mpi_rank); fflush(stdout);
			for (int i = 0; i < nodeset->Nnodes; i++) {
				printf("\t\tnodeid = %3d:\tx = %4.2f,\ty = %4.2f,\tz = %4.2f,\tlambda = %4.2f,\tphi = %4.2f\n",
						i, nodeset->x[i], nodeset->y[i], nodeset->z[i], nodeset->lambda[i], nodeset->phi[i]); fflush(stdout);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}


void print_DD2_slice(double* DD2_slice, int size, int Nnodes) {

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\nRank %d, size = %d, Nnodes = %d DD2_slice:\n", mpi_rank, size, Nnodes); fflush(stdout);
			for (int i = 0; i < size; i++) {
				printf("\n\ti = %6d:\t", i);
				for (int j = 0; j < Nnodes; j++) {
					printf("%4.2f\t", DD2_slice[(i*Nnodes)+j]); fflush(stdout);
				}
			}
			printf("\n");
		}
		usleep(1000);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void print_part_ids(domains_struct global_domains) {

	int Nnodes = global_domains.Nnodes;
	int* part_ids = global_domains.part_ids;

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\nRank %d, part_ids:\n\t", mpi_rank); fflush(stdout);
			for (int i = 0; i < Nnodes; i++) {
				printf("%3d\t", part_ids[i]); fflush(stdout);
			}
			printf("\n");
		}
		usleep(1000);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void print_generic_int_matrix(int* matrix, int Nrows, int Ncols) {

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\n\nRank %d Matrix:\n", rank); fflush(stdout);
			for (int row = 0; row < Nrows; row++) {
				for (int col = 0; col < Ncols; col++) {
					printf("\t%d", matrix[(row * Ncols) + col]); fflush(stdout);
				}
				printf("\n"); fflush(stdout);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void print_generic_fp_matrix(double* matrix, int Nrows, int Ncols, int all_ranks) {

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\n\nRank %d Matrix:\n", rank); fflush(stdout);
			for (int row = 0; row < Nrows; row++) {
				for (int col = 0; col < Ncols; col++) {
					printf("\t%4.2e", matrix[(row * Ncols) + col]); fflush(stdout);
				}
				printf("\n"); fflush(stdout);
			}
		}
		if (all_ranks == TRUE) {
			MPI_Barrier(MPI_COMM_WORLD);
			usleep(1000);
		}
	}
}

#include "include/patches.h"
#include "include/init_domains.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>

extern int mpi_rank;
extern int mpi_size;

extern config_struct config;

void exchange_halos_1D(double* var, patch_struct* LP);
void print_patch_nodeset(patch_struct* LP);

void init_patches(patch_struct* LP, nodeset_struct* nodeset) {

	// extract frequently used constants
	int Nh_global = nodeset->Nh;
	int n = config.stencil_size;
	int Nv = config.num_layers;
	int compute_size = nodeset->patch_sizes[mpi_rank];

	// matrix to hold halo sizes between each patch
	int* adjcny_mat = (int*) calloc(pow(mpi_size, 2), sizeof(int));

	// mask representiing dependent nodes for each patch
	int* dependency_mask = (int*) calloc(Nh_global, sizeof(int));

	// iterate through patch's idx to set adjcny_mat and dependency_mask
	int i_start = nodeset->patch_start_ids[mpi_rank];
	for (int i = i_start; i < i_start + compute_size; i++) {
		for (int j = 0; j < n; j++) {
			
			int gid_ij = nodeset->idx[(i * n) + j];
			int patch_id_ij = nodeset->patch_ids[gid_ij];
			
			if (dependency_mask[gid_ij] == 0) {
				dependency_mask[gid_ij] = 1;
				adjcny_mat[(mpi_rank*mpi_size) + patch_id_ij] += 1;
			}
		}
	}


	// communicate adjncy matrix subsets to other ranks
	for (int rank = 0; rank < mpi_size; rank++) {
		MPI_Bcast((void*) &adjcny_mat[rank*mpi_size], mpi_size, MPI_INT, rank, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// determine patch size Nh
	int Nh = 0;
	int l_halo_size = 0;
	for (int rank = 0; rank < mpi_size; rank++) {
		int N_r = adjcny_mat[(mpi_rank*mpi_size) + rank];
		Nh += N_r;
		if (rank < mpi_rank)
			l_halo_size += N_r;
	}

	// determine neighbor ranks
	int* nbr_ranks = (int*) malloc(sizeof(int) * mpi_size);
	int Nnbrs = 0;
	for (int rank = 0; rank < mpi_size; rank++) {
		if (mpi_rank == rank)
			continue;
		int N_rr = adjcny_mat[(mpi_rank*mpi_size) + rank] + adjcny_mat[(rank*mpi_size) + mpi_rank];
		if (N_rr != 0) {
			nbr_ranks[Nnbrs] = rank;
			Nnbrs += 1;
		}
	}

	// get mappings
	int* gid_map = (int*) malloc(sizeof(int) * Nh);
	int* pid_map = (int*) calloc(Nh_global, sizeof(int));

	int count = 0;
	
	for (int i = 0; i < Nh_global; i++) {
		if (dependency_mask[i] == 1) {
			gid_map[count] = i;
			pid_map[i] = count;
			count++;
		}
	}
	
	// assign compute nodes
	int* compute_pids = (int*) malloc(sizeof(int) * compute_size);
	for (int i = 0; i < compute_size; i++) {
		compute_pids[i] = i + l_halo_size;
	}

	// assign x,y,z for compute nodes
	double* x = (double*) malloc(sizeof(double) * Nh);
	double* y = (double*) malloc(sizeof(double) * Nh);
	double* z = (double*) malloc(sizeof(double) * Nh);
	double* lambda = (double*) malloc(sizeof(double) * Nh);
	double* phi = (double*) malloc(sizeof(double) * Nh);

	for (int i = 0; i < Nh; i++) {
		int gid = gid_map[i];
		x[i] = nodeset->x[gid];
		y[i] = nodeset->y[gid];
		z[i] = nodeset->z[gid];
		lambda[i] = nodeset->lambda[gid];
		phi[i] = nodeset->phi[gid];
	}

	// Debugging	
	for (int rank = 0; rank < mpi_size; rank++) {
		if (mpi_rank == rank) {
			//printf("Rank = %d,\t Nnbrs = %d\n",mpi_rank,Nnbrs);
			//print_int_matrix(dependency_mask, 1, Nh_global);
			//print_int_matrix(gid_map, 1, Nh);

			//print_int_matrix(adjcny_mat, mpi_size, mpi_size);
			//print_int_matrix(nodeset->patch_start_ids, 1, Nh_global);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// set up halos for each neighboring patch

	halo_struct* halos = (halo_struct*) malloc(sizeof(halo_struct) * Nnbrs);

	MPI_Request* request = (MPI_Request*) malloc(sizeof(MPI_Request) * Nnbrs * 2);
	MPI_Status* status = (MPI_Status*) malloc(sizeof(MPI_Status) * Nnbrs * 2);

	int halo_pid_s = 0;
	int right_T = 0;
	for (int nbr = 0; nbr < Nnbrs; nbr++) {

		int nbr_rank = nbr_ranks[nbr];
		halos[nbr].nbr_rank = nbr_rank;
		halos[nbr].halo_size = adjcny_mat[(mpi_rank*mpi_size) + nbr_rank];
		halos[nbr].nbr_halo_size = adjcny_mat[(nbr_rank*mpi_size) + mpi_rank];

		halos[nbr].halo_pids = (int*) malloc(sizeof(int) * halos[nbr].halo_size);
		halos[nbr].halo_gids = (int*) malloc(sizeof(int) * halos[nbr].halo_size);
		halos[nbr].nbr_halo_pids = (int*) malloc(sizeof(int) * halos[nbr].nbr_halo_size);

		if (nbr_rank > mpi_rank && right_T == 0) {
			halo_pid_s += compute_size;
			right_T = 1;
		}

		for (int i = 0; i < halos[nbr].halo_size; i++) {
			halos[nbr].halo_pids[i] = i + halo_pid_s;
		}
		halo_pid_s += halos[nbr].halo_size;

		for (int i = 0; i < halos[nbr].halo_size; i++) {
			halos[nbr].halo_gids[i] = gid_map[halos[nbr].halo_pids[i]];
		}

		MPI_Isend((void*) halos[nbr].halo_gids, halos[nbr].halo_size, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &request[nbr*2]);
		MPI_Irecv((void*) halos[nbr].nbr_halo_pids, halos[nbr].nbr_halo_size, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &request[(nbr*2)+1]);
	}

	MPI_Waitall(Nnbrs*2, request, status);

	free(request);
	free(status);

	for (int nbr = 0; nbr < Nnbrs; nbr++) {
		for (int i = 0; i < halos[nbr].nbr_halo_size; i++) {
			halos[nbr].nbr_halo_pids[i] = pid_map[halos[nbr].nbr_halo_pids[i]];
		}
	}

	free(nbr_ranks);
	free(adjcny_mat);
	free(dependency_mask);

	LP->Nh = Nh;
	LP->Nv = Nv;

	LP->x = x;
	LP->y = y;
	LP->z = z;
	LP->lambda = lambda;
	LP->phi = phi;

	LP->gid_map = gid_map;
	LP->pid_map = pid_map;

	LP->compute_size = compute_size;
	LP->compute_pids = compute_pids;

	LP->Nnbrs = Nnbrs;
	LP->halos = halos;


}

void exchange_halos_1D(double* var, patch_struct* LP) {

	int Nnbrs = LP->Nnbrs;

	MPI_Request* request = (MPI_Request*) malloc(sizeof(MPI_Request) * Nnbrs * 2);
	MPI_Status* status = (MPI_Status*) malloc(sizeof(MPI_Status) * Nnbrs * 2);
	
	double** send_buff_arr = (double**) malloc(sizeof(double*)*Nnbrs);

	for (int nbr = 0; nbr < Nnbrs; nbr++) {

		halo_struct halo = LP->halos[nbr];

		double* send_buff = (double*) malloc(sizeof(double) * halo.nbr_halo_size);
		send_buff_arr[nbr] = send_buff;

		for (int i = 0; i < halo.nbr_halo_size; i++) {
			send_buff[i] = var[halo.nbr_halo_pids[i]];
		}

		MPI_Isend((void*) send_buff, halo.nbr_halo_size, MPI_DOUBLE, halo.nbr_rank, 0, MPI_COMM_WORLD, &request[nbr*2]);
		MPI_Irecv((void*) &var[halo.halo_pids[0]], halo.halo_size, MPI_DOUBLE, halo.nbr_rank, 0, MPI_COMM_WORLD, &request[(nbr*2)+1]);
	}

	MPI_Waitall(Nnbrs * 2, request, status);
}

void print_patch_nodeset(patch_struct* LP) {

	for (int rank = 0; rank < mpi_size; rank++) {
		if (mpi_rank == rank) {
			printf("\nRank = %d:\n", rank); fflush(stdout);
			for (int i = 0; i < LP->Nh; i++) {
				printf("\tpid,gid -> %6d,%6d\tx = %4.3f\ty = %4.3f\tz = %4.3f\tl = %4.3f\tp = %4.3f\n",
						i, LP->gid_map[i], LP->x[i], LP->y[i], LP->z[i], LP->lambda[i], LP->phi[i]); fflush(stdout);
			}
			printf("\n"); fflush(stdout);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}


void checkpoint(int num) {

	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);
	if (mpi_rank == 0) {
		printf("\n\nCHECKPOINT %d\n\n", num);
		fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}





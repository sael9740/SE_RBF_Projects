#include "include/patches.h"
//#include "include/domains.h"
#include "include/debug.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>

extern int mpi_rank;
extern int mpi_size;

int* halo_size_matrix;
int* dependency_mask;

int l_halo_size;

/* FUNCTION - GET_PART_UTILS
 * - Determines the halo_size_matrix and dependency_mask for the MPI partitions of the global domain
 * 		- halo_size_matrix -> matrix holding the halo layer size between each patch pair
 * 		- dependency mask -> binary valued (size Nnodes_global) dependency indicator array for the
 * 			local partition
 */
void init_part_utils(domains_struct* global_domains) {

	// extract data from inputs
	nodeset_struct* global_nodeset = global_domains->global_nodeset;

	int Nnodes_global = global_nodeset->Nnodes;
	int n = global_nodeset->n;
	int Nnodes_part = global_domains->part_sizes[mpi_rank];
	int* part_ids = global_domains->part_ids;
	int* part_start_ids = global_domains->part_start_ids;
	int* idx = global_nodeset->idx;

	// matrix to hold halo sizes between each patch
	halo_size_matrix = (int*) calloc(mpi_size*mpi_size, sizeof(int));
	// mask representiing dependent nodes for each patch
	dependency_mask = (int*) calloc(Nnodes_global, sizeof(int));


	// iterate through patch's idx to set halo_size_matrix and dependency_mask
	int i_start = part_start_ids[mpi_rank];
	int i_end = i_start + Nnodes_part;
	for (int i = i_start; i < i_end; i++) {
		for (int j = 0; j < n; j++) {
			
			int gid = idx[(i * n) + j];
			int partid = part_ids[gid];
			
			if (dependency_mask[gid] == 0) {
				dependency_mask[gid] = 1;
				halo_size_matrix[(mpi_rank*mpi_size) + partid] += 1;
			}
		}
	}
	
	// communicate adjncy matrix subsets to other ranks
	for (int rank = 0; rank < mpi_size; rank++) {
		MPI_Bcast((void*) &halo_size_matrix[rank*mpi_size], mpi_size, MPI_INT, rank, MPI_COMM_WORLD);
	}

}



void init_patch_nodeset(patch_struct* local_patch, domains_struct* global_domains) {

	// extract data from inputs
	nodeset_struct* global_nodeset = global_domains->global_nodeset;

	int Nnodes_global = global_nodeset->Nnodes;
	int n = global_nodeset->n;
	int part_Nnodes = global_domains->part_sizes[mpi_rank];

	// determine patch size Nh
	int Nnodes = 0;
	l_halo_size = 0;
	for (int rank = 0; rank < mpi_size; rank++) {
		int halo_size = halo_size_matrix[(mpi_rank*mpi_size) + rank];
		Nnodes += halo_size;
		if (rank < mpi_rank)
			l_halo_size += halo_size;
	}

	// get mappings
	int* pid2gid = (int*) malloc(sizeof(int) * Nnodes);
	int* gid2pid = (int*) calloc(Nnodes_global, sizeof(int));

	int count = 0;
	
	for (int gid = 0; gid < Nnodes_global; gid++) {
		if (dependency_mask[gid] == 1) {
			pid2gid[count] = gid;
			gid2pid[gid] = count;
			count++;
		}
	}
	
	// assign compute nodes
	int* part_pids = (int*) malloc(sizeof(int) * part_Nnodes);
	for (int i = 0; i < part_Nnodes; i++) {
		part_pids[i] = i + l_halo_size;
	}

	// get coordinates for patch nodeset
	double* x = (double*) malloc(sizeof(double) * Nnodes);
	double* y = (double*) malloc(sizeof(double) * Nnodes);
	double* z = (double*) malloc(sizeof(double) * Nnodes);
	double* lambda = (double*) malloc(sizeof(double) * Nnodes);
	double* phi = (double*) malloc(sizeof(double) * Nnodes);

	for (int pid = 0; pid < Nnodes; pid++) {
		int gid = pid2gid[pid];
		x[pid] = global_nodeset->x[gid];
		y[pid] = global_nodeset->y[gid];
		z[pid] = global_nodeset->z[gid];
		lambda[pid] = global_nodeset->lambda[gid];
		phi[pid] = global_nodeset->phi[gid];
	}

	// get D for patch nodeset
	double* D = (double*) malloc(sizeof(double) * Nnodes * n);

	for (int pid = 0; pid < Nnodes; pid++) {
		int gid = pid2gid[pid];
		for (int nbr = 0; nbr < n; nbr++) {
			D[(pid * n) + nbr] = global_nodeset->D[(gid * n) + nbr];
		}
	}
	
	// get idx for patch nodeset
	int* idx = (int*) malloc(sizeof(int) * Nnodes * n);

	for (int pid = 0; pid < Nnodes; pid++) {
		int gid = pid2gid[pid];
		for (int nbr = 0; nbr < n; nbr++) {
			idx[(pid * n) + nbr] = gid2pid[global_nodeset->idx[(gid * n) + nbr]];
		}
	}

	// assign data to local_patch struct
	nodeset_struct patch_nodeset[1];

	patch_nodeset->Nnodes = Nnodes;
	patch_nodeset->n = n;
	patch_nodeset->x = x;
	patch_nodeset->y = y;
	patch_nodeset->z = z;
	patch_nodeset->lambda = lambda;
	patch_nodeset->phi = phi;
	patch_nodeset->idx = idx;
	patch_nodeset->D = D;

	local_patch->nodeset[0] = patch_nodeset[0];		// deep copy

	local_patch->Nnodes = Nnodes;
	local_patch->part_Nnodes = part_Nnodes;
	local_patch->pid2gid = pid2gid;
	local_patch->gid2pid = gid2pid;
	local_patch->part_pids = part_pids;

}

void init_patch_halos(patch_struct* local_patch) {

	int part_Nnodes = local_patch->part_Nnodes;

	// determine neighbor ranks and number of neighbors
	int* nbr_ranks = (int*) malloc(sizeof(int) * mpi_size);
	int Nnbrs = 0;
	for (int rank = 0; rank < mpi_size; rank++) {
		if (mpi_rank == rank)
			continue;
		int pair_halo_sum = halo_size_matrix[(mpi_rank*mpi_size) + rank] + halo_size_matrix[(rank*mpi_size) + mpi_rank];
		if (pair_halo_sum != 0) {
			nbr_ranks[Nnbrs] = rank;
			Nnbrs += 1;
		}
	}

	// determine halo sizes, neighbor halo sizes and related data
	int halo_size_sum = 0;
	int nbr_halo_size_sum = 0;

	int* halo_sizes = (int*) malloc(sizeof(int) * Nnbrs);
	int* nbr_halo_sizes = (int*) malloc(sizeof(int) * Nnbrs);
	int* halo_offsets = (int*) malloc(sizeof(int) * (Nnbrs+1));
	int* nbr_halo_offsets = (int*) malloc(sizeof(int) * (Nnbrs+1));

	for (int nbr = 0; nbr < Nnbrs; nbr++) {
		
		int nbr_rank = nbr_ranks[nbr];
		int halo_size = halo_size_matrix[(mpi_rank * mpi_size) + nbr_rank];
		int nbr_halo_size = halo_size_matrix[(nbr_rank * mpi_size) + mpi_rank];
		
		halo_sizes[nbr] = halo_size;
		nbr_halo_sizes[nbr] = nbr_halo_size;

		halo_offsets[nbr] = halo_size_sum;
		nbr_halo_offsets[nbr] = nbr_halo_size_sum;

		halo_size_sum += halo_size;
		nbr_halo_size_sum += nbr_halo_size;

	}
	halo_offsets[Nnbrs] = halo_size_sum;
	nbr_halo_offsets[Nnbrs] = nbr_halo_size_sum;

	int* hid2pid = (int*) malloc(sizeof(int) * halo_size_sum);
	int* hid2gid = (int*) malloc(sizeof(int) * halo_size_sum);

	for (int hid = 0; hid < halo_size_sum; hid++) {
		int pid = hid < l_halo_size ? hid : hid + part_Nnodes;
		int gid = local_patch->pid2gid[pid];
		hid2pid[hid] = pid;
		hid2gid[hid] = gid;
	}

	int* nbr_hid2pid = (int*) malloc(sizeof(int) * nbr_halo_size_sum);
	int* nbr_hid2gid = (int*) malloc(sizeof(int) * nbr_halo_size_sum);
	
	MPI_Request* request = (MPI_Request*) malloc(sizeof(MPI_Request) * Nnbrs * 2);
	MPI_Status* status = (MPI_Status*) malloc(sizeof(MPI_Status) * Nnbrs * 2);

	for (int nbr = 0; nbr < Nnbrs; nbr++) {
		
		int nbr_rank = nbr_ranks[nbr];
		int halo_size = halo_sizes[nbr];
		int nbr_halo_size = nbr_halo_sizes[nbr];
		int halo_offset = halo_offsets[nbr];
		int nbr_halo_offset = nbr_halo_offsets[nbr];

		MPI_Isend((void*) &hid2gid[halo_offset], halo_size, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &request[nbr*2]);
		MPI_Irecv((void*) &nbr_hid2gid[nbr_halo_offset], nbr_halo_size, MPI_INT, nbr_rank, 0, MPI_COMM_WORLD, &request[(nbr*2)+1]);

	}

	MPI_Waitall(Nnbrs*2, request, status);

	free(request);
	free(status);

	for (int nbr_hid = 0; nbr_hid < nbr_halo_size_sum; nbr_hid++) {
		int gid = nbr_hid2gid[nbr_hid];
		int pid = local_patch->gid2pid[gid];
		nbr_hid2pid[nbr_hid] = pid;
	}

	halos_struct* halos = local_patch->halos;

	halos->Nnbrs = Nnbrs;
	halos->halo_size_sum = halo_size_sum;
	halos->nbr_halo_size_sum = nbr_halo_size_sum;
	halos->nbr_ranks = nbr_ranks;
	halos->halo_sizes = halo_sizes;
	halos->nbr_halo_sizes = nbr_halo_sizes;
	halos->halo_offsets = halo_offsets;
	halos->nbr_halo_offsets = nbr_halo_offsets;
	halos->hid2pid = hid2pid;
	halos->hid2gid = hid2gid;
	halos->nbr_hid2pid = nbr_hid2pid;
	halos->nbr_hid2gid = nbr_hid2gid;

}

void print_halos(patch_struct* local_patch) {
	
	halos_struct* halos = local_patch->halos;

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\nRank %d\t -> \tNnbrs = %d\thalo_size_sum = %d\tnbr_halo_size_sum = %d\n\tNeighbor Halo Data:",
					rank, halos->Nnbrs, halos->halo_size_sum, halos->nbr_halo_size_sum); fflush(stdout);
			for (int nbr = 0; nbr < halos->Nnbrs; nbr++) {
				printf("\n\t\tnbr_rank = %4d,    halo_size = %4d,    halo_offset = %4d,    nbr_halo_size = %4d,    nbr_halo_offset = %4d",
						halos->nbr_ranks[nbr], halos->halo_sizes[nbr], halos->halo_offsets[nbr], halos->nbr_halo_sizes[nbr], halos->nbr_halo_offsets[nbr]); fflush(stdout);
			}
			printf("\n\n\tlocal halo mapping:\n"); fflush(stdout);
			int nbr = 0;
			for (int hid = 0; hid < halos->halo_size_sum; hid++) {
				if (hid == halos->halo_offsets[nbr+1])
					nbr += 1;
				printf("\t\thid = %4d,   pid = %4d,   gid = %4d,   nbr_rank = %4d\n", hid, halos->hid2pid[hid], halos->hid2gid[hid], halos->nbr_ranks[nbr]); fflush(stdout);
			}
			printf("\n\n\tneighbor halo mapping:\n"); fflush(stdout);
			nbr = 0;
			for (int hid = 0; hid < halos->nbr_halo_size_sum; hid++) {
				if (hid == halos->nbr_halo_offsets[nbr+1])
					nbr += 1;
				printf("\t\tnbr_hid = %4d,   pid = %4d,   gid = %4d,   nbr_rank = %4d\n", hid, halos->nbr_hid2pid[hid], halos->nbr_hid2gid[hid], halos->nbr_ranks[nbr]); fflush(stdout);
			}
			printf("\n"); fflush(stdout);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}


part_SV_struct* allocate_part_SV_data(patch_struct* local_patch, int Ndim) {

	layers_struct* layers = local_patch->layers;

	int part_Nnodes = local_patch->part_Nnodes;
	
	int padded_Nv = layers->padded_Nv;
	
	part_SV_struct* part_SV = (part_SV_struct*) malloc(sizeof(part_SV_struct));

	part_SV->Ndim = Ndim;

	part_SV->SV_data = (double*) calloc(sizeof(double), part_Nnodes * padded_Nv * Ndim);

	return part_SV;
}

patch_SV_struct* allocate_patch_SV_data(patch_struct* local_patch, int Ndim) {
	
	layers_struct* layers = local_patch->layers;
	halos_struct* halos = local_patch->halos;

	int Nnodes = local_patch->Nnodes;
	
	int padded_Nv = layers->padded_Nv;
	int padded_Nvt = layers->padded_Nvt;
	
	int Nnbrs = halos->Nnbrs;
	int halo_size_sum = halos->halo_size_sum;
	int nbr_halo_size_sum = halos->nbr_halo_size_sum;

	patch_SV_struct* patch_SV = (patch_SV_struct*) malloc(sizeof(patch_SV_struct));

	patch_SV->Ndim = Ndim;

	patch_SV->SV_data = (double*) calloc(sizeof(double), (Nnodes * padded_Nvt * Ndim) + CACHE_ALIGN) + CACHE_ALIGN;

		
		patch_SV->halo_buff = (double*) malloc(sizeof(double) * halo_size_sum * padded_Nvt * Ndim);
		patch_SV->nbr_halo_buff = (double*) malloc(sizeof(double) * nbr_halo_size_sum * padded_Nvt * Ndim);

		patch_SV->request = (MPI_Request*) malloc(sizeof(MPI_Request) * Nnbrs * 2);
		patch_SV->status = (MPI_Status*) malloc(sizeof(MPI_Status) * Nnbrs * 2);

	return patch_SV;

}

void patch_SV_diffs(patch_struct* local_patch, patch_SV_struct* SV1, patch_SV_struct* SV2) {

	int Ndim = SV1->Ndim;

	int Nnodes = local_patch->Nnodes;
	
	layers_struct* layers = local_patch->layers;
	int Nv = layers->Nv;
	int padded_Nvt = layers->padded_Nvt;

	double* var1 = SV1->SV_data;
	double* var2 = SV2->SV_data;

	double epsilon = 1.0e-16;

	for (int pid = 0; pid < Nnodes; pid++) {
		for (int vid = 0; vid < Nv; vid++) {
			double diff = fabs(var1[(pid*padded_Nvt) + vid] - var2[(pid*padded_Nvt) + vid]);
			if (diff > epsilon) {
				printf("\npid = %d, gid = %d, vid = %d, diff = %.1e, var1 = %.1e, var2 = %.1e",
						pid,local_patch->pid2gid[pid],vid,diff,var1[(pid*padded_Nvt) + vid],var2[(pid*padded_Nvt) + vid]); usleep(100);
			}
		}
	}
}

void exchange_halos(patch_struct* local_patch, patch_SV_struct* SV) {

	int Ndim = SV->Ndim;
	double* data = SV->SV_data;
	double* halo_buff = SV->halo_buff;
	double* nbr_halo_buff = SV->nbr_halo_buff;
	MPI_Request* request = SV->request;
	MPI_Status* status = SV->status;

	layers_struct* layers = local_patch->layers;
	int Nv = layers->Nv;
	int padded_Nv = layers->padded_Nv;
	int padded_Nvt = layers->padded_Nvt;

	halos_struct* halos = local_patch->halos;
	int Nnbrs = halos->Nnbrs;
	int halo_size_sum = halos->halo_size_sum;
	int nbr_halo_size_sum = halos->nbr_halo_size_sum;

	// pack neighbor halo buffer
	for (int nbr_hid = 0; nbr_hid < nbr_halo_size_sum; nbr_hid++) {
		int pid = halos->nbr_hid2pid[nbr_hid];
		for (int dimid = 0; dimid < Ndim; dimid++) {
			//if (mpi_rank == 0) 
			//printf("\nnbr_hid = %d, pid = %d, dimid = %d, linid dest = %d, linid source = %d, size = %d",nbr_hid,pid,dimid,((nbr_hid*Ndim)+dimid)*padded_Nv,((pid*Ndim)+dimid)*padded_Nvt,padded_Nv);
			memcpy((void*) &nbr_halo_buff[((nbr_hid*Ndim)+dimid)*padded_Nv], (void*) &data[((pid*Ndim)+dimid)*padded_Nvt], sizeof(double)*padded_Nv);
		}
	}

	// send/recv neighbor halos
	for (int nbr = 0; nbr < Nnbrs; nbr++) {		
		MPI_Isend((void*) &nbr_halo_buff[halos->nbr_halo_offsets[nbr] * Ndim * padded_Nv], halos->nbr_halo_sizes[nbr] * Ndim * padded_Nv, MPI_DOUBLE, halos->nbr_ranks[nbr], 0, MPI_COMM_WORLD, &request[nbr*2]);
		MPI_Irecv((void*) &halo_buff[halos->halo_offsets[nbr] * Ndim * padded_Nv], halos->halo_sizes[nbr] * Ndim * padded_Nv, MPI_DOUBLE, halos->nbr_ranks[nbr], 0, MPI_COMM_WORLD, &request[(nbr*2)+1]);
	}

	MPI_Waitall(Nnbrs*2, request, status);

	// unpack halo buffers
	for (int hid = 0; hid < halo_size_sum; hid++) {
		int pid = halos->hid2pid[hid];
		for (int dimid = 0; dimid < Ndim; dimid++) {
			//if (mpi_rank == 0) 
			//printf("\nhid = %d, pid = %d, dimid = %d, linid dest = %d, linid source = %d, size = %d",hid,pid,dimid,((hid*Ndim)+dimid)*padded_Nv,((pid*Ndim)+dimid)*padded_Nvt,padded_Nv);
			memcpy((void*) &data[((pid*Ndim)+dimid)*padded_Nvt], (void*) &halo_buff[((hid*Ndim)+dimid)*padded_Nv], sizeof(double)*padded_Nv);
		}
	}
}

/*
// set up halos for each neighboring patch

halo_struct* halos = (halo_struct*) malloc(sizeof(halo_struct) * Nnbrs);

MPI_Request* request = (MPI_Request*) malloc(sizeof(MPI_Request) * Nnbrs * 2);
MPI_Status* status = (MPI_Status*) malloc(sizeof(MPI_Status) * Nnbrs * 2);

int halo_pid_s = 0;
int right_T = 0;
for (int nbr = 0; nbr < Nnbrs; nbr++) {

int nbr_rank = nbr_ranks[nbr];
halos[nbr].nbr_rank = nbr_rank;
halos[nbr].halo_size = halo_size_matrix[(mpi_rank*mpi_size) + nbr_rank];
halos[nbr].nbr_halo_size = halo_size_matrix[(nbr_rank*mpi_size) + mpi_rank];

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
free(halo_size_matrix);
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



*/


/* FUNCTION - PRINT_PART_UTILS
 * - Debugging/Validation 
 * - Simply prints the contents of the adjacency matrix and dependency masks
 */
void print_part_utils(int Nnodes_global) {

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\nRank %d part utils:\n\tScalars ->  \tNnodes_global = %d\n\thalo_size_matrix:\n",
					mpi_rank, Nnodes_global); fflush(stdout);
			for (int i = 0; i < mpi_size; i++) {
				printf("\n\t\ti = %d: \t", i); fflush(stdout);
				for (int j = 0; j < mpi_size; j++) {
					printf("%6d,\t", halo_size_matrix[(i * mpi_size) + j]); fflush(stdout);
				}
			}
			printf("\n\tdependency_mask:\t");
			for (int i = 0; i < Nnodes_global; i++) {
				printf("%d,", dependency_mask[i]); fflush(stdout);
			}
			printf("\n"); fflush(stdout);
		}
		usleep(1000);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}


#include "include/domains.h"
#include "include/nodesets.h"
#include "include/debug.h"
#include <stdlib.h>
#include <mpi.h>
#include <metis.h>

extern int mpi_rank;
extern int mpi_size;

/* FUNCTION - GET_METIS_PARTITIONING
 * 	INPUTS:
 * 		- global_dmonais_p -> global_domains prointer with initialized nodeset to be partitioned
 * 			using METIS
 *	DESCRIPTION: Determines the METIS partitioning of the global domain and reorders the nodeset 
 *		for contiguous partitions
 */
/*void get_metis_partitioning(domains_struct* global_domains) {

	// assign/get global_domains sizes/dimensions
	int Nparts = mpi_size;
	int Nnodes = global_domains->Nnodes;
	global_domains->Nparts = Nparts;

	// get the METIS partitioning
	get_metis_part_ids(global_domains);

	// allocate space for the mapping and inverse mapping arrays
	int* mapping = (int*) malloc(sizeof(int) * Nnodes);
	int* inv_mapping = (int*) malloc(sizeof(int) * Nnodes);

	// get the mappings for contiguous partitions based on the metis partitioning
	get_contiguous_part_mapping(global_domains, mapping, inv_mapping);

	// reorder the global nodeset using these mappings
	reorder_nodeset_dist(&global_domains->nodeset, mapping, inv_mapping);

}*/

/* FUNCTION - GET_CONTIGUOUS_PART_MAPPING
 * 	INPUTS:
 * 		- global_dmonais_p -> global_domains prointer with initialized part_ids member
 * 		- mapping/inv_mapping -> allocated dataspaces to hold resulting mappings for reordering the
 * 			nodesets
 *	DESCRIPTION: Determines the mappings necessary to reorder the nodeset so as to obtain 
 *		contiguous partitions
 */
void reorder_domain_contiguous_parts(domains_struct* global_domains) {
	
	// extract members
	int Nparts = global_domains->Nparts;
	int Nnodes = global_domains->Nnodes;
	int* part_ids = global_domains->part_ids;

	// allocate space for the mapping and inverse mapping arrays
	int* mapping = (int*) malloc(sizeof(int) * Nnodes);
	int* inv_mapping = (int*) malloc(sizeof(int) * Nnodes);

	// allocate dataspace
	int* part_sizes = (int*) malloc(sizeof(int) * Nparts);
	int* part_start_ids = (int*) malloc(sizeof(int) * Nparts);
	int* part_ids_new = (int*) malloc(sizeof(int) * Nnodes);

	/* Determine Mapping 
	 * - Not parallelizable with mpi -> have rank 0 determine and broadcast results to other ranks
	 * - Assign new part_ids, part_start_ids, and part_sizes during the process
	 */
	if (mpi_rank == 0) {
		
		// holds current position of inv_mapping (how many nodes have been assigned a mapping)
		int counter1 = 0;
		
		for (int rank = 0; rank < mpi_size; rank++) {
			
			// holds size of current ranks partition
			int counter2 = 0;
			part_start_ids[rank] = counter1;

			for (int i = 0; i < Nnodes; i++) {
				
				if (part_ids[i] == rank) {
					
					part_ids_new[counter1] = rank;
					mapping[i] = counter1;
					inv_mapping[counter1] = i;
					counter1++;
					counter2++;
				}
			}
			part_sizes[rank] = counter2;
		}
	}

	// brodcast results to all ranks
	MPI_Bcast((void*) mapping, Nnodes, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) inv_mapping, Nnodes, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) part_ids_new, Nnodes, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) part_sizes, Nparts, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*) part_start_ids, Nparts, MPI_INT, 0, MPI_COMM_WORLD);

	// reorder the global nodeset using these mappings
	reorder_nodeset(global_domains->global_nodeset, mapping, inv_mapping, TRUE);

	free(mapping);
	free(inv_mapping);
	free(part_ids);


	global_domains->part_ids = part_ids_new;
	global_domains->part_sizes = part_sizes;
	global_domains->part_start_ids = part_start_ids;

}

void get_metis_partitioning(domains_struct* global_domains, int Nparts) {

	// extract data from nodeset
	nodeset_struct* nodeset = global_domains->global_nodeset;
	int Nnodes = nodeset->Nnodes;
	int n = nodeset->n;
	int* idx = nodeset->idx;

	// variables for metis
	idx_t Nvert = Nnodes;
	idx_t Ncon = 1;
	idx_t objval;

	// graph and partition data
	idx_t* xadj;
	idx_t* adjncy;
	idx_t* part_ids = (idx_t*) malloc(sizeof(idx_t) * Nnodes);

	if (mpi_rank == 0) {

		// graph and partition data
		xadj = (idx_t*) malloc(sizeof(idx_t) * (Nnodes + 1));
		adjncy = (idx_t*) malloc(sizeof(idx_t) * Nnodes * (n - 1));

		// setup metis graph data
		for (int i = 0; i < Nnodes + 1; i++) {
			xadj[i] = i * (n - 1);
		}
		for (int i = 0; i < Nnodes; i++) {
			for (int j = 1; j < n; j++) {
				adjncy[xadj[i] + j - 1] = idx[(i * n) + j];
			}
		}

		METIS_PartGraphKway(&Nvert, &Ncon, xadj, adjncy, NULL,
				NULL, NULL, (idx_t*) &Nparts, NULL, NULL, NULL, &objval, part_ids);

		free(xadj);
		free(adjncy);
	}

	MPI_Bcast((void*) part_ids, Nnodes, MPI_INT, 0, MPI_COMM_WORLD);

	global_domains->part_ids = (int*) part_ids;
}

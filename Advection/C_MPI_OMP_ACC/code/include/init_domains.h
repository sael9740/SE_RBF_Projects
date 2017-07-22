#ifndef INIT_DOMAINS_H
#define INIT_DOMAINS_H

#include "config.h"

void init_nodeset(nodeset_struct* nodeset);

// calculate quadrature weights (euclidian distances) for all node pairings 
// returns horizontal slice of D matrix for each mpi rank
double* get_D_r(nodeset_struct* nodeset, int start_id, int size);

// determine n-nearest neighbor stencils for each node
void get_idx(double* D_r, nodeset_struct* nodeset, int start_id, int size);

// get mpi partitioning of the domain using metis
void get_partitions(nodeset_struct* nodeset);

void reorder_nodeset(nodeset_struct* nodeset);

#endif

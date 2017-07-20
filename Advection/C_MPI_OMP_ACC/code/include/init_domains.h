#ifndef INIT_DOMAINS_H
#define INIT_DOMAINS_H

#include "config.h"

void init_ns1(unit_nodeset_struct* ns1, adv_params_struct* adv_params);

// calculate quadrature weights (euclidian distances) for all node pairings 
// returns horizontal slice of D matrix for each mpi rank
double* get_D_r(unit_nodeset_struct* ns1, int start_id, int size);

// determine n-nearest neighbor stencils for each node
void get_idx(double* D_r, unit_nodeset_struct* ns1, int start_id, int size);

// get mpi partitioning of the domain using metis
void get_partitions(unit_nodeset_struct* ns1);

void reorder_ns1(unit_nodeset_struct* ns1);

#endif

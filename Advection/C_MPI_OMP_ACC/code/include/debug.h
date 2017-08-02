#ifndef ADV_DEBUG_H
#define ADV_DEBUG_H

#include "config.h"

void abort_solver(char* message);

void checkpoint(int num);

void print_nodeset(nodeset_struct* nodeset);

void print_DD2_slice(double* DD2_slice, int size, int Nnodes);

void print_part_ids(domains_struct global_domains);

void print_generic_fp_matrix(double* matrix, int Nrows, int Ncols, int all_ranks);

void print_generic_int_matrix(int* matrix, int Nrows, int Ncols, int all_ranks);

#endif

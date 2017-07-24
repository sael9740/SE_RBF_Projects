#ifndef ADV_NODESETS_H
#define ADV_NODESETS_H

#include "config.h"

int get_nodeset_from_file(char* nodeset_file, nodeset_struct* nodeset);

void get_stencils(nodeset_struct* nodeset, int n);

double* get_DD2_slice(nodeset_struct* nodeset, int start_id, int size);

void get_D_idx_slice(double* D_slice, int* idx_slice, double* DD2_slice, nodeset_struct* nodeset, int start_id, int size);

//void reorder_nodeset(nodeset_struct* nodeset, int* mapping, int* inv_mapping);
void reorder_nodeset(nodeset_struct* nodeset, int* mapping, int* inv_mapping, int distributed);

//void reorder_nodeset_dist(nodeset_struct* nodeset, int* mapping, int* inv_mapping);

void print_stencils(nodeset_struct* nodeset);

#endif

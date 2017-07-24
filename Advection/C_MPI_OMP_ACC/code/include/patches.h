#ifndef ADV_PATCHES_H
#define ADV_PATCHES_H

#include "config.h"

void init_part_utils(domains_struct* global_domains);

void init_patch_nodeset(patch_struct* local_patch, domains_struct* global_domains);

void init_patch_halos(patch_struct* local_patch);

void print_halos(patch_struct* local_patch);

void print_part_utils(int Nnodes_global);

//void init_patches(patch_struct* LP, nodeset_struct* nodeset);

#endif

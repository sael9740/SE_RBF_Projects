#ifndef ADV_WRAPPERS_H
#define ADV_WRAPPERS_H

#include "config.h"

void get_model_ICs();

void get_rt_config();

void init_patch_rbffd_DMs();

void init_global_layers();

void init_local_patch();

void init_phys_constants();

void init_global_nodeset();

void partition_global_domain();

#endif

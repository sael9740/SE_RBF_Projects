#ifndef PATCHES_H
#define PATCHES_H

#include "config.h"

void init_patches(patch_struct* LP, unit_nodeset_struct* np1, adv_params_struct* adv_params);

void checkpoint(int num);
#endif

#ifndef ADV_RBFFD_DMS_H
#define ADV_RBFFD_DMS_H

#include "config.h"

void get_part_rbffd_stencils(patch_struct* local_patch);

void init_rbffd_HH_rot_Ms(patch_struct* local_patch);

void get_rbffd_DMs_Dxyz(patch_struct* local_patch);

double phs(double r, int k);

double ddri_phs(double r, double ri, int k);

double L2_norm(double x, double y, double z);

#endif

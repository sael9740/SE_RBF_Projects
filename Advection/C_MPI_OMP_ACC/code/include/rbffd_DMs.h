#ifndef ADV_RBFFD_DMS_H
#define ADV_RBFFD_DMS_H

#include "config.h"

void get_part_rbffd_stencils(patch_struct* local_patch);

void init_rbffd_HH_rot_Ms(patch_struct* local_patch);

void get_rbffd_DMs(patch_struct* local_patch, int k_phs, int k_L, double ep);

double phsrbf(double r, int k);

double d1_phsrbf(double r, double ri, int k);

double garbf(double r, double epsilon);

double L_garbf(double r, double epsilon, int k);

double L2_norm(double x, double y, double z);

void eval_partSVDotGradSV(patch_struct* local_patch, double alpha, part_SV_struct* SV_A, patch_SV_struct* SV_B, part_SV_struct* SV_C);

void eval_sumpatchSVpartSV(patch_struct* local_patch, patch_SV_struct* SV_A, double alpha, part_SV_struct* SV_B, patch_SV_struct* SV_C);

void eval_sumpartSVpartSV(patch_struct* local_patch, part_SV_struct* SV_A, double alpha, part_SV_struct* SV_B, part_SV_struct* SV_C);

void eval_sumpatchSVpatchSV(patch_struct* local_patch, patch_SV_struct* SV_A, double alpha, patch_SV_struct* SV_B, part_SV_struct* SV_C);

void update_Up(patch_struct* local_patch);

#endif

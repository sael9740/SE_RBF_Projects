#ifndef ADV_TEST_CASES_H
#define ADV_TEST_CASES_H

#include "config.h"

typedef struct TC2_params {
	
	double tau;
	double K;
	double u0;
	double w0;
	double h_1;
	double h_2;
	double htop;
	
} TC2_params_struct;

void init_TC2(patch_struct* local_patch);

double q0_TC2(double h);

double HS_isoT_rho(double h);
void set_TC2_U_t(patch_struct* local_patch, double t);

double TC2_u_r(double h, double phi);
double TC2_u_lambda(double phi);
double TC2_u_phi(double h, double phi);

#endif

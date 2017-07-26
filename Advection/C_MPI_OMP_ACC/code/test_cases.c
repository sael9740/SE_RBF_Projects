#include "include/test_cases.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern int mpi_size;
extern int mpi_rank;

extern phys_constants_struct phys_constants[1];

TC2_params_struct TC2_params[1];

void init_TC2(patch_struct* local_patch) {

	TC2_params->tau = 86400.0;
	TC2_params->K = 5.0;
	TC2_params->u0 = 40.0;
	TC2_params->w0 = .15;
	TC2_params->h_1 = 2000.0;
	TC2_params->h_2 = 5000.0;


	layers_struct* layers = local_patch->layers;
	nodeset_struct* nodeset = local_patch->nodeset;
	int* part_pids = local_patch->part_pids;

	int Nv = layers->Nv;
	int Ng = layers->Ng;
	int padded_Nv = layers->padded_Nv;
	int Nvt = layers->Nvt;
	int padded_Nvt = layers->padded_Nvt;
	int Nnodes = local_patch->Nnodes;
	int part_Nnodes = local_patch->part_Nnodes;

	double* h = layers->h;
	double* phi = nodeset->phi;

	double* q0 = local_patch->SV_q0->SV_data;
	double* qt = local_patch->SV_qt->SV_data;
	double* Usph = local_patch->SV_Usph->SV_data;

	for (int pid = 0; pid < Nnodes; pid++) {
		for (int vid = 0; vid < Nv; vid++) {
			double h_vid = h[vid];
			qt[(pid*padded_Nvt) + vid] = q0_TC2(h_vid);
		}
	}
	for (int partid = 0; partid < part_Nnodes; partid++) {
		int pid = part_pids[partid];
		double phi_partid = phi[pid];
		for (int vid = 0; vid < Nv; vid++) {
			double h_vid = h[vid];
			q0[(pid*padded_Nvt) + vid] = q0_TC2(h_vid);
			Usph[((3*partid+0)*padded_Nv) + vid] = TC2_u_r(h_vid, phi_partid);
			Usph[((3*partid+1)*padded_Nv) + vid] = TC2_u_lambda(phi_partid);
			Usph[((3*partid+2)*padded_Nv) + vid] = TC2_u_phi(h_vid, phi_partid);
		}
	}

}

void set_TC2_U_t(patch_struct* local_patch, double t) {

	layers_struct* layers = local_patch->layers;
	nodeset_struct* nodeset = local_patch->nodeset;
	int* part_pids = local_patch->part_pids;

	int Nv = layers->Nv;
	int Ng = layers->Ng;
	int padded_Nv = layers->padded_Nv;
	int Nvt = layers->Nvt;
	int padded_Nvt = layers->padded_Nvt;
	int Nnodes = local_patch->Nnodes;
	int part_Nnodes = local_patch->part_Nnodes;

	double* Usph = local_patch->SV_Usph->SV_data;
	double* U = local_patch->SV_U->SV_data;

	double* lambda = nodeset->lambda;
	double* phi = nodeset->phi;

	double tau = TC2_params->tau;
	double w_t = cos(PI * (t/tau));
	
	for (int partid = 0; partid < part_Nnodes; partid++) {
		int pid = part_pids[partid];
		double phi_pid = phi[pid];
		double lambda_pid = lambda[pid];
		for (int vid = 0; vid < Nv; vid++) {
			double u_r = Usph[((3*partid+0)*padded_Nv) + vid] * w_t;
			double u_lambda = Usph[((3*partid+1)*padded_Nv) + vid];
			double u_phi = Usph[((3*partid+2)*padded_Nv) + vid] * w_t;
			U[((3*partid+0)*padded_Nv) + vid] = cos(phi_pid)*cos(lambda_pid)*u_r - sin(lambda_pid)*u_lambda - sin(phi_pid)*cos(lambda_pid)*u_phi;
			U[((3*partid+1)*padded_Nv) + vid] = cos(phi_pid)*sin(lambda_pid)*u_r + cos(lambda_pid)*u_lambda - sin(phi_pid)*sin(lambda_pid)*u_phi;
			U[((3*partid+2)*padded_Nv) + vid] = sin(phi_pid)*u_r + cos(phi_pid)*u_phi;
		}
		if (mpi_rank == 0 && partid == 0) {
			//printf("\nNODE -> x = %.3e, y = %.3e, z = %.3e\n",nodeset->x[pid],nodeset->y[pid],nodeset->z[pid]);
			//print_generic_fp_matrix(U,3,padded_Nv,FALSE);
		}
	}
}

double HS_isoT_rho(double h) {

	double Rd = phys_constants->Rd;
	double T0 = phys_constants->T0;
	double g = phys_constants->g;
	double rho0 = phys_constants->rho0;

	return rho0 * exp(-(h*g)/(Rd * T0));
}

double q0_TC2(double h) {

	double h_1 = TC2_params->h_1;
	double h_2 = TC2_params->h_2;
	double h_0 = (h_0 + h_1)/2;

	return h < h_1 || h > h_2 ? 0.0 : (1.0 + cos((2.0*PI) * (h - h_0)/(h_2 - h_1)))/2.0;
}

double TC2_u_r(double h, double phi) {


	double tau = TC2_params->tau;
	double w0 = TC2_params->w0;
	double K = TC2_params->K;
	double htop = phys_constants->htop;
	double rho0 = phys_constants->rho0;

	return (w0 * rho0)/(K * HS_isoT_rho(h)) * (((-2) * sin(K * phi) * sin(phi)) + (K * cos(K * phi) * cos(phi))) * sin(PI * h/htop); // * cos(PI * (t/tau)));

}

double TC2_u_lambda(double phi) {
	
	double u0 = TC2_params->u0;

	return u0 * cos(phi);
}

double TC2_u_phi(double h, double phi) {
	
	double tau = TC2_params->tau;
	double w0 = TC2_params->w0;
	double K = TC2_params->K;
	double htop = phys_constants->htop;
	double rho0 = phys_constants->rho0;
	double R = phys_constants->R;

	return -((R*w0*PI*rho0)/(K*htop*HS_isoT_rho(h)))*cos(phi)*sin(K*phi)*cos(PI*h/htop); //cos(PI*t/tau);

}








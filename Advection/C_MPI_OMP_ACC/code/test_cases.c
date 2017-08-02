#include "include/test_cases.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TC2_TAU 86400.0
#define TC2_K 5.0
#define TC2_U0 40.0
#define TC2_W0 .15
#define TC2_H1 2000.0
#define TC2_H2 5000.0

extern int mpi_size;
extern int mpi_rank;

extern rt_config_struct rt_config[1];

extern phys_constants_struct phys_constants[1];

TC2_params_struct TC2_params[1];

void update_U(patch_struct* local_patch, double t) {

	if (rt_config->TC == 2) {
		set_TC2_U_t(local_patch, t);
	}

}


/* FUNCTION - INIT_TC2
 * - Inititalizes the following:
 *   1) TC2_params data -> test case parameterizations such as number of cells and characteristic 
 *   	velocities
 *   2) the initial mass tracer ratios q_0 and q_t for t = 0
 *   3) Usph -> the time independent spherical wind field terms without temporal scaling factors
 */
void init_TC2(patch_struct* local_patch) {

	/***** Initialize TC2_params data *****/
	TC2_params->tau = TC2_TAU;
	TC2_params->K = TC2_K;
	TC2_params->u0 = TC2_U0;
	TC2_params->w0 = TC2_W0;
	TC2_params->h1 = TC2_H1;
	TC2_params->h2 = TC2_H2;


	/***** Determine timestepping configuration *****/
	double dt = rt_config->timestep_length;
	double t_end = TC2_params->tau;
	int nsteps = (int) ceil(t_end/dt);
	dt = t_end/nsteps;
	rt_config->nsteps = nsteps;
	rt_config->dt = dt;

	printf("\nRank %d: nsteps = %d, dt = %f\n",mpi_rank,nsteps,dt);

	/***** Extract Relevant Values/Data *****/
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

	double* q_init = local_patch->SV_q_init->data;
	double* q_t = local_patch->SV_q_t->data;
	double* Usph = local_patch->SV_Usph->data;


	/***** Initialize q_0 and q_t at t = 0 for entire patch *****/
	for (int pid = 0; pid < Nnodes; pid++) {
		for (int vid = 0; vid < Nv; vid++) {
			double h_vid = h[vid];
			double q_vid = q_init_TC2(h_vid);
			q_init[(pid*padded_Nvt) + vid] = q_vid;
			q_t[(pid*padded_Nvt) + vid] = q_vid;
		}
	}

	/***** Initialize Usph *****/
	for (int partid = 0; partid < part_Nnodes; partid++) {
		int pid = part_pids[partid];
		double phi_partid = phi[pid];
		for (int vid = 0; vid < Nv; vid++) {
			double h_vid = h[vid];
			Usph[((3*partid+0)*padded_Nv) + vid] = TC2_u_r(h_vid, phi_partid);
			Usph[((3*partid+1)*padded_Nv) + vid] = TC2_u_lambda(phi_partid);
			Usph[((3*partid+2)*padded_Nv) + vid] = TC2_u_phi(h_vid, phi_partid);
		}
	}
}


/* FUNCTION - SET_TC2_U_T
 * - Determines and assigns the cartesian TC2 wind field at time t for the patch
 */
void set_TC2_U_t(patch_struct* local_patch, double t) {

	/***** Extract Relevant Values/Data *****/
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

	double* Usph = local_patch->SV_Usph->data;
	double* U = local_patch->SV_U->data;

	double* lambda = nodeset->lambda;
	double* phi = nodeset->phi;

	double tau = TC2_params->tau;


	/***** Temporal scaling factor *****/
	double w_t = cos(PI * (t/tau));
	

	/***** Calculate cartesian TC2 wind field using Usph and temporal scaling factor *****/
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
	}
}


/* FUNCTION - HS_ISOT_RHO
 * - Determines the air density (rho) at an atmospheric height (h) for a hydrostatic, isothermal 
 *   atmosphere
 */
double HS_isoT_rho(double h) {

	/***** Extract Relevant Values/Data *****/
	double Rd = phys_constants->Rd;
	double T0 = phys_constants->T0;
	double g = phys_constants->g;
	double rho0 = phys_constants->rho0;

	/***** Calculate and return the air density *****/
	return rho0 * exp(-(h*g)/(Rd * T0));
}


/* FUNCTION - Q0_TC2
 * - Determines the initial tracer mass ratio for TC2 at an atmospheric height h
 */
double q_init_TC2(double h) {

	/***** Extract Relevant Values/Data *****/
	double h1 = TC2_params->h1;
	double h2 = TC2_params->h2;
	double h0 = (h1 + h2)/2;

	/***** Calculate and return the initial tracer mass ratio *****/
	return h < h1 || h > h2 ? 0.0 : (1.0 + cos((2.0*PI) * (h - h0)/(h2 - h1)))/2.0;
}


/* FUNCTION - TC2_U_R
 * - Determines the radial wind velocity component for TC2 without the temporal scaling factor
 */
double TC2_u_r(double h, double phi) {

	/***** Extract Relevant Values/Data *****/
	double tau = TC2_params->tau;
	double w0 = TC2_params->w0;
	double K = TC2_params->K;
	double htop = phys_constants->htop;
	double rho0 = phys_constants->rho0;

	/***** Calculate and return the radial velocity *****/
	return (w0 * rho0)/(K * HS_isoT_rho(h)) * (((-2) * sin(K * phi) * sin(phi)) + (K * cos(K * phi) * cos(phi))) * sin(PI * h/htop); // * cos(PI * (t/tau)));
}


/* FUNCTION - TC2_U_LAMBDA
 * - Determines the meridonal wind velocity component for TC2 without the temporal scaling factor
 */
double TC2_u_lambda(double phi) {
	
	/***** Extract Relevant Values/Data *****/
	double u0 = TC2_params->u0;

	/***** Calculate and return the meridonal velocity *****/
	return u0 * cos(phi);
}


/* FUNCTION - TC2_U_PHI
 * - Determines the latitudinal wind velocity component for TC2 without the temporal scaling factor
 */
double TC2_u_phi(double h, double phi) {
	
	/***** Extract Relevant Values/Data *****/
	double tau = TC2_params->tau;
	double w0 = TC2_params->w0;
	double K = TC2_params->K;
	double htop = phys_constants->htop;
	double rho0 = phys_constants->rho0;
	double R = phys_constants->R;

	/***** Calculate and return the latitudinal velocity *****/
	return -((R*w0*PI*rho0)/(K*htop*HS_isoT_rho(h)))*cos(phi)*sin(K*phi)*cos(PI*h/htop); //cos(PI*t/tau);

}








#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "include/wrappers.h"
#include "include/config.h"
#include "include/debug.h"
#include "include/nodesets.h"
#include "include/layers.h"
#include "include/domains.h"
#include "include/patches.h"
#include "include/rbffd_DMs.h"
#include "include/test_cases.h"


/***** GLOBAL DATA *****/

extern int mpi_size;
extern int mpi_rank;

rt_config_struct rt_config[1];

phys_constants_struct phys_constants[1];

domains_struct global_domains[1];

layers_struct global_layers[1];

patch_struct local_patch[1];


/***** WRAPPER FUNCTIONS FOR MAIN *****/

/* FUNCTION - GET_SOLUTION
 * - Preforms timestepping of the solution
 */
void get_solution() {

	/***** Extract timestepping data *****/
	int nsteps = rt_config->nsteps;
	double dt = rt_config->dt;

	/***** Set U and Up for initial timestep *****/
	update_U(local_patch, 0.0);
	update_Up(local_patch);	

	int num = 0;
	int padded_Nv = local_patch->layers->padded_Nv;
	int padded_Nvt = local_patch->layers->padded_Nvt;

	/***** Timestepping *****/
	for (int tstep = 0; tstep < nsteps; tstep++) {
		
		double t = tstep * dt;
		if (mpi_rank == 0) {
			printf("\nSolver Status: \ttstep = %d -> t = %.1f s", tstep, t);
		}

		/***** RK4 Substep #1 *****/
		// F_1 = RHS of PDE for q = q_1 = q_t and U = U(t)
		eval_partSVDotGradSV(local_patch, -1.0, local_patch->SV_Up, local_patch->SV_q_t, local_patch->SV_F);


		/***** RK4 Substep #2 *****/
		update_U(local_patch, t + (dt/2.0));
		update_Up(local_patch);	
		// q_2 = q_t + (dt/2) * F_1
		eval_sumpatchSVpartSV(local_patch, local_patch->SV_q_t, dt/2.0, local_patch->SV_F, local_patch->SV_q_k);
		exchange_halos(local_patch, local_patch->SV_q_k);
		// F_2 = RHS of PDE for q = q_2 and U = U(t + dt/2)
		eval_partSVDotGradSV(local_patch, -1.0, local_patch->SV_Up, local_patch->SV_q_k, local_patch->SV_F_k);
		// F = F + 2 * F_2
		eval_sumpartSVpartSV(local_patch, local_patch->SV_F, 2.0, local_patch->SV_F_k, local_patch->SV_F);


		/***** RK4 Substep #3 *****/
		// q_3 = q_t + (dt/2) * F_2
		eval_sumpatchSVpartSV(local_patch, local_patch->SV_q_t, dt/2.0, local_patch->SV_F_k, local_patch->SV_q_k);
		exchange_halos(local_patch, local_patch->SV_q_k);
		// F_3 = RHS of PDE for q = q_3 and U = U(t + dt/2)
		eval_partSVDotGradSV(local_patch, -1.0, local_patch->SV_Up, local_patch->SV_q_k, local_patch->SV_F_k);
		// F = F + 2 * F_3
		eval_sumpartSVpartSV(local_patch, local_patch->SV_F, 2.0, local_patch->SV_F_k, local_patch->SV_F);


		/***** RK4 Substep #4 *****/
		update_U(local_patch, t + dt);	
		update_Up(local_patch);	
		// q_4 = q_t + dt * F_3
		eval_sumpatchSVpartSV(local_patch, local_patch->SV_q_t, dt, local_patch->SV_F_k, local_patch->SV_q_k);
		exchange_halos(local_patch, local_patch->SV_q_k);
		// F_4 = RHS of PDE for q = q_4 and U = U(t + dt)
		eval_partSVDotGradSV(local_patch, -1.0, local_patch->SV_Up, local_patch->SV_q_k, local_patch->SV_F_k);
		// F = F + 1 * F_4
		eval_sumpartSVpartSV(local_patch, local_patch->SV_F, 1.0, local_patch->SV_F_k, local_patch->SV_F);


		/***** Update Solution ****/
		// q_t = q_t + (dt/6) * F
		eval_sumpatchSVpartSV(local_patch, local_patch->SV_q_t, dt/6.0, local_patch->SV_F, local_patch->SV_q_t);
		exchange_halos(local_patch, local_patch->SV_q_t);
	}

}

void verify_solution() {

	/***** Extract Local Patch Data *****/
	layers_struct* layers = local_patch->layers;
	int* partid2pid = local_patch->part_pids;
	int part_Nnodes = local_patch->part_Nnodes;
	int Nnodes = local_patch->Nnodes;
	int global_Nnodes = global_domains->Nnodes;

	/***** Extract Layers Data *****/
	int Nv = layers->Nv;
	int padded_Nv = layers->padded_Nv;
	int Nvt = layers->Nvt;
	int padded_Nvt = layers->padded_Nvt;
	double dh = layers->dh;
	double* r = layers->r;
	double* h = layers->h;

	double* q_init = local_patch->SV_q_init->data;
	double* q_t = local_patch->SV_q_t->data;


	double errors[10] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0};
	double results[10] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0};

	for (int partid = 0; partid < part_Nnodes; partid++) {

		int pid = partid2pid[partid];

		for (int vid = 0; vid < Nv; vid++) {
			double q_real = q_init[(pid * padded_Nvt) + vid];
			double q_approx = q_t[(pid * padded_Nvt) + vid];
			double diff = fabs(q_real - q_approx);
			errors[0] += diff;
			errors[1] += fabs(q_real);
			errors[2] += pow(diff,2);
			errors[3] += pow(q_real,2);

			double rho = HS_isoT_rho(h[vid]);
			double V = (4 * PI * pow(r[vid],2) * dh)/global_Nnodes;
			errors[4] += rho*V*q_real;
			errors[5] += rho*V*q_approx;

			errors[6] = errors[6] > diff ? errors[6] : diff;
			errors[7] = errors[7] > fabs(q_approx) ? errors[7] : fabs(q_approx);

			errors[8] = errors[8] < q_approx ? q_approx : errors[8];
			errors[9] = errors[9] < q_approx ? errors[9] : q_approx;
		}
	}

	MPI_Reduce((const void *) &errors[0], (void*) &results[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce((const void *) &errors[6], (void*) &results[6], 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce((const void *) &errors[9], (void*) &results[9], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	if (mpi_rank == 0) {

		printf("\n\n======================================= Verification Results =======================================\n");
		printf("\nExtrema:\n\tMin(q) = %e\n\tMax(q) = %e\n", results[9], results[8]);
		printf("\nError Norms:\n\tL1(q) = %e\n\tL2(q) = %e\n\tLinf(q) = %e\n", 
				results[0]/results[1], sqrt(results[2]/results[3]), results[6]/results[7]);
		printf("\nTracer Mass Conservation:\n\tTotal Mass (real) -> %e kg\n\tTotal Mass (approximated) -> %e kg\n\tMass Difference -> %e kg\n\tMass Difference Ratio -> %e",
				results[4], results[5], results[5] - results[4], fabs(results[5] - results[4])/results[4]);
		printf("\n\n====================================================================================================\n\n");

	}

	MPI_Barrier(MPI_COMM_WORLD);
}


/* FUNCTION - GET_MODEL_ICS
 * - Assigns the initial conditions as well as any other relevant time-independent state variable 
 *   data that is used during the timestepping
 */
void get_model_ICs() {

	if (TEST_CASE == 2) {
		init_TC2(local_patch);
	}
	else {
		abort_solver("Only Test Case 2 is implemented. Please set TEST_CASE=2 and rerun.");
	}

}


/* FUNCTION - INIT_PATCH_RBFFD_DMS
 * - Determines and assigns all differentiation weights for the RBFFD DMs and sets up all other 
 *   data in the patch's RBFFD_DMs struct
 */
void init_patch_rbffd_DMs() {

	get_part_rbffd_stencils(local_patch);

	init_rbffd_HH_rot_Ms(local_patch);

	// determine correct RBF params
	int global_Nnodes = global_domains->Nnodes;
	int n = local_patch->nodeset->n;

	double c1 = 4.4e-2; //1.6e2*sqrt(n);
	double c2 = 1.4e-1; //5.0e2*sqrt(n);
	double epsilon = c1*sqrt(global_Nnodes) - c2;

	get_rbffd_DMs(local_patch, RBF_PHS_ORDER, HYPERVISCOSITY_ORDER, epsilon);

}


/* FUNCTION - INIT_LOCAL_PATCH
 * - Initializes the local patch using the global_domains struct which contains the global nodeset
 *   as well as the partitioning of the nodeset
 */
void init_local_patch() {


	init_part_utils(global_domains);

	//	print_part_utils(global_domains->Nnodes);

	init_patch_nodeset(local_patch, global_domains);

	//	print_nodeset(local_patch->nodeset);

	init_patch_halos(local_patch);
	//	print_halos(local_patch);

	int Nv = rt_config->num_layers;
	double htop = rt_config->model_height;

	get_layers(local_patch->layers, htop, Nv);

	local_patch->SV_q_init = allocate_patch_SV_data(local_patch, 1);
	local_patch->SV_q_t = allocate_patch_SV_data(local_patch, 1);
	local_patch->SV_q_k = allocate_patch_SV_data(local_patch, 1);

	local_patch->SV_U = allocate_part_SV_data(local_patch, 3);
	local_patch->SV_Up = allocate_part_SV_data(local_patch, 3);
	local_patch->SV_Usph = allocate_part_SV_data(local_patch, 3);

	local_patch->SV_F = allocate_part_SV_data(local_patch, 1);
	local_patch->SV_F_k = allocate_part_SV_data(local_patch, 1);

	local_patch->SV_3D_temp = allocate_part_SV_data(local_patch, 3);

}


/* FUNCTION - INIT_GLOBAL_LAYERS
 * - Initializes the global_layers structure
 */
void init_global_layers() {

	int Nv = rt_config->num_layers;
	double htop = rt_config->model_height;

	get_layers(global_layers, htop, Nv);

	//	print_layers(global_layers);

}

/* FUNCTION - INIT_GLOBAL_NODESET
 * - reads/determines the global nodeset and initializes the relevant members of global_domains
 */
void init_global_nodeset() {

	int Nnodes;
	int n = rt_config->stencil_size;
	double tstart;
	// Get the global nodeset
	if (rt_config->nodeset_from_file == TRUE) {
		tstart = MPI_Wtime();
		global_domains->Nnodes = get_nodeset_from_file(rt_config->nodeset_file, global_domains->global_nodeset);
		if (mpi_rank == 0) {
			printf("\nget_nodeset_from_file() walltime -> %.2f seconds\n", MPI_Wtime() - tstart);
		}
	}
	else {
		abort_solver("Nodeset creation not yet implemented. Please provide a NetCDF file containing a valid nodeset.");
	}

	// get stencils (idx) and weights (D) of the global nodeset
	tstart = MPI_Wtime();
	get_stencils(global_domains->global_nodeset, n);
	if (mpi_rank == 0) {
		printf("\nget_stencils() walltime -> %.2f seconds\n", MPI_Wtime() - tstart);
	}

}

/* FUNCTION - PARTITION_GLOBAL_DOMAIN
 * - determines the MPI partitioning of the global domain and performs any necessary reordering
 *   of the nodeset for the partitioning
 */
void partition_global_domain() {

	// assign/get global_domains sizes/dimensions
	int Nparts = mpi_size;
	int Nnodes = global_domains->Nnodes;
	global_domains->Nparts = Nparts;

	if (rt_config->use_metis == TRUE) {

		// get the METIS partitioning
		get_metis_partitioning(global_domains, Nparts);

		// get the mappings for contiguous partitions based on the metis partitioning
		reorder_domain_contiguous_parts(global_domains);
	} 
	else {
		abort_solver("Only METIS partitioning is currently implemented. Please set ADV_USE_METIS=1 and rerun.");
	}

}

/* FUNCTION - INIT_PHYS_CONSTANTS
 * - Simply assigns the correct values to the global phys_constants struct members
 */
void init_phys_constants() {

	phys_constants->R = EARTH_RADIUS;
	phys_constants->g = GRAVITATIONAL_CONSTANT;
	phys_constants->p0 = REFERENCE_SURFACE_PRESSURE;
	phys_constants->cp = DRY_AIR_CP;
	phys_constants->Rd = DRY_AIR_RD;
	phys_constants->kappa = DRY_AIR_CP/DRY_AIR_RD;
	phys_constants->T0 = ISOTHERMAL_TEMP;
	phys_constants->htop = rt_config->model_height;
	phys_constants->rho0 = phys_constants->p0 / (phys_constants->Rd * phys_constants->T0);

}

/* FUNCTION - GET_RUNTIME_CONFIG
 * - Reads all runtime environment parameterizations and assigns them to the associated members of 
 *   the global rt_config struct
 */
void get_rt_config() {

	// temporary pointer to hold the return value of getenv()
	char* result;

	// Have rank 0 set rt_config and broadcast to other ranks
	if (mpi_rank == 0) {

		result = getenv("ADV_NODESET_FROM_FILE");
		rt_config->nodeset_from_file = result == NULL ? DEFAULT_NODESET_FROM_FILE : atoi(result);

		result = getenv("ADV_NODESET_FILE");
		strcpy(rt_config->nodeset_file, result == NULL ? "NONE" : result);

		result = getenv("ADV_USE_METIS");
		rt_config->use_metis = result == NULL ? DEFAULT_USE_METIS : atoi(result);

		result = getenv("ADV_TEST_CASE");
		rt_config->TC = result == NULL ? DEFAULT_TEST_CASE : atoi(result);

		result = getenv("ADV_STENCIL_SIZE");
		rt_config->stencil_size = result == NULL ? DEFAULT_STENCIL_SIZE : atoi(result);

		result = getenv("ADV_NUM_LAYERS");
		rt_config->num_layers = result == NULL ? DEFAULT_NUM_LAYERS : atoi(result);

		result = getenv("ADV_MODEL_HEIGHT");
		rt_config->model_height = result == NULL ? DEFAULT_MODEL_HEIGHT : atof(result);

		result = getenv("ADV_TIMESTEP_LENGTH");
		rt_config->timestep_length = result == NULL ? DEFAULT_TIMESTEP : atof(result);

		// Print the resulting configuration to stdout
		printf("\n======================================= Advection Solver Parameterizations ====================================\n\n");

		printf("\tADV_NODESET_FROM_FILE: \t%d\n", rt_config->nodeset_from_file);
		printf("\tADV_NODESET_FILE:      \t%s\n", rt_config->nodeset_file);
		printf("\tADV_STENCIL_SIZE:      \t%d\n", rt_config->stencil_size);
		printf("\tADV_USE_METIS:         \t%d\n", rt_config->use_metis);
		printf("\tADV_TEST_CASE:         \t%d\n", rt_config->TC);
		printf("\tADV_NUM_LAYERS:        \t%d\n", rt_config->num_layers);
		printf("\tADV_MODEL_HEIGHT:      \t%.1f meters\n", rt_config->model_height);
		printf("\tADV_TIMESTEP_LENGTH:   \t%.1f seconds\n", rt_config->timestep_length);

		printf("\n===============================================================================================================\n\n");
	}

	// Broadcast resulting rt_config struct to the other ranks
	MPI_Bcast((void*) rt_config, sizeof(rt_config_struct), MPI_BYTE, 0, MPI_COMM_WORLD);

}



#ifndef ADV_CONFIG_H
#define ADV_CONFIG_H

#include <stdlib.h>

#define MAX_PATH_SIZE 200

#define GHOST_LAYER_DEPTH 3

#define MEM_ALIGNMENT 8

#ifndef SIMD_LENGTH
  #define SIMD_LENGTH 8
#endif

#define PAD_UP(N,M) (((((N)-1)/(M))+1)*(M))
#define PAD_DOWN(N,M) (((N)/(M))*(M))

// ================= Default Config Parameterizations =================== //

#define DEFAULT_NUM_LAYERS 32
#define DEFAULT_STENCIL_SIZE 55
#define DEFAULT_MODEL_HEIGHT 12000.0
#define DEFAULT_TIMESTEP 600.0

// ========================= Physical Constants ========================= //

#define EARTH_RADIUS 6.37122e6

/* CONFIG STRUCT
 * PURPOSE: holds all data associated with the runtime environment configuration 
 * options such as input files, rbf stencil size, number of levels, etc.
 */
typedef struct config_struct {

	int num_nodes;
	int num_layers;
	int stencil_size;

	double h_top;
	double dt;

	char nodeset_input_file[MAX_PATH_SIZE];

} config_struct;

/* GLOBAL PARAMETERS STRUCT
 * PURPOSE: holds all globally valid definitions and constants such as the 
 * global nodeset, vertical layer desciptions, physical constants, etc.
 */
typedef struct global_params_struct {

	int Nh;		// Number of node points in global domain
	
	int Nv;		// Number of vertical layers
	int Nv_compute;	// Number of vertical compute layers
	
	int n;		// RBF-FD stencil size

	double R_e;	// radius of earth (m)
	double h_top;	// model atmosphere height (m)
	double dh;	// height of vertical layers
	double dt;	// timestep 

} global_params_struct;

/* NODESET STRUCT
 * PURPOSE: Holds all data required to describe a particular nodeset as 
 * well as useful information closely associated with the dataset such
 * as distances and MPI partitioning information
 */
typedef struct nodeset_struct {
	
	int Nh;
	int Nv;
	int n;

	double* x;
	double* y;
	double* z;
	double* lambda;
	double* phi;

	double* D_idx;

	int* idx;
	int* patch_ids;

	int* patch_sizes;
	int* patch_start_ids;

} nodeset_struct;

//typedef struct global_domain_struct {} global_domain_struct;

/* HALO STRUCT
 * PURPOSE: Holds all data required for MPI halo layer communication with a single
 * neighboring patch such as the neghbors rank and mappings describing the halo 
 * layers in both domains.
 */
typedef struct halo_struct {
	
	// neighboring patch's mpi rank
	int nbr_rank;

	// current rank's halo size in neighboring domain
	int halo_size;		
	
	// neighbor's halo size in current rank's domain
	int nbr_halo_size;	

	// local patch ids of neighbor's halo layer in the current rank's domain
	int* nbr_halo_pids;

	// local patch ids of current rank's halo layer in its own domain
	int* halo_pids;

	// global ids of current rank's halo layer
	int* halo_gids;

} halo_struct;

/* PATCH STRUCT
 * PURPOSE: Holds ALL data associated with an MPI patch including 
 * the nodeset, state variables, and the halo layer descriptions
 * necessary for MPI communication (see halo_struct below)
 */
typedef struct patch_struct {

	// ==================== Patch Nodeset Description ========================= //

	// dimensions
	int Nh;		// Number of node points in local patch
	int Nv;		// Number of vertical layers

	// cartesian coordinates of horizontal nodeset (on unit sphere), shape -> (Nh)
	double* x;
	double* y;
	double* z;

	// sphereical angular coordinates of horizontal nodeset, shape -> (Nh)
	// meridonal -> lambda | latitudinal -> phi
	double* lambda;
	double* phi;

	// spherical radial coordinates of vertical layers, shape -> (Nv)
	// r -> radius | h -> height above surface
	double* r;
	double* h;
	
	
	// ================= Mappings to/from Global Domain ====================== //
	
	// pid -> local patch horizontal node id
	// gid -> global domain horizontal node id
	
	// mapping: pid -> gid, shape -> (Nh)
	int* gid_map;

	// mapping: gid -> qid, shape -> (Nh_global)
	int* pid_map;


	// ====================== Compute Domain Description ===================== //
	
	// compute domain size
	int compute_size;

	// pids of compute nodes of the local patch, shape -> (Nh)
	int* compute_pids;


	// ======================= Halo Layer Description ======================== //
	
	// Number of neighbor patches
	int Nnbrs;

	// halo layer information for each neighboring patch, shape -> (Nnbrs)
	halo_struct* halos;

} patch_struct;

#endif

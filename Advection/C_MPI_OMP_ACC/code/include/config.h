#ifndef ADV_CONFIG_H
#define ADV_CONFIG_H

#include <stdlib.h>

#define MAX_PATH_SIZE 200

typedef struct adv_params_struct {

	int Nv;
	int n;

	char nodesetFile[MAX_PATH_SIZE];

} adv_params_struct;

typedef struct unit_nodeset_struct {
	
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

} unit_nodeset_struct;

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

//typedef struct global_domain_struct {} global_domain_struct;

typedef struct patch_struct {

	// ==================== Patch Nodeset Description ========================= //

	// dimensions
	int Nh;		// Number of horizontal node points in local patch
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
	
	
	// ================ Mappings to/from Global Domain ====================== //
	
	// pid -> local patch horizontal node id
	// gid -> global domain horizontal node id
	
	// mapping: pid -> gid, shape -> (Nh)
	int* gid_map;

	// mapping: gid -> qid, shape -> (Nh_global)
	int* pid_map;


	// ===================== Compute Domain Description ===================== //
	
	// compute domain size
	int compute_size;

	// pids of compute nodes of the local patch, shape -> (Nh)
	int* compute_pids;


	// ====================== Halo Layer Description ======================== //
	
	// Number of neighbor patches
	int Nnbrs;

	// halo layer information for each neighboring patch, shape -> (Nnbrs)
	halo_struct* halos;

} patch_struct;

#endif

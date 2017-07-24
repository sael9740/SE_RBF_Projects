/******************************************* CONFIG.H **********************************************
 * This header is used for definitions that are common to or are not intrinsically unique to a 	   *
 * single header file within the greater RBF-FD Advection code.									   *
 **************************************************************************************************/

#ifndef ADV_CONFIG_H
#define ADV_CONFIG_H

#include <stdlib.h>

#define GHOST_LAYER_DEPTH 3
#define RBF_POLY_ORDER 5
#define RBF_NPOLY(ORDER)  (((ORDER + 1) * (ORDER + 2))/2)
#define RBF_PHS_ORDER 5

#define TRUE 1
#define FALSE 0

/******************************* CACHE ALIGNMENT AND VECTORIZATION ********************************/

// Cache Alignment (in number of FP elements)
#define CACHE_ALIGN 8

// Number of SIMD elements for vectorized/GPU FP operations (in number of FP elements)
#ifndef SIMD_LENGTH
  #define SIMD_LENGTH 8
#endif

/* Alignment Functions:
 * - Used to round numbers up or down to a specific alignment
 * - Examples: 
 *   	ALIGNUP(15,8) returns 16
 *   	ALIGNDOWN(15,8) returns 8
 */
#define ALIGNUP(N,M) (((((N)-1)/(M))+1)*(M))
#define ALIGNDOWN(N,M) (((N)/(M))*(M))


/************************************** PHYSICAL CONSTANTS ****************************************/

#define EARTH_RADIUS 6.37122e6
#define GRAVITATIONAL_CONSTANT 9.80616e0

typedef struct phys_constants_struct {

	double R;
	double g;

} phys_constants_struct;


/************************************ RUNTIME CONFIGURATION ***************************************/

// Maximum file path size
#define MAX_PATH_SIZE 200

/* DEFAULT PARAMS:
 * - These values are used when the associated variable is not explicitly defined in the rt 
 *   environment.
 */
#define DEFAULT_NODESET_FROM_FILE TRUE
#define DEFAULT_USE_METIS TRUE
#define DEFAULT_NUM_LAYERS 32
#define DEFAULT_STENCIL_SIZE 55
#define DEFAULT_MODEL_HEIGHT 12000.0
#define DEFAULT_TIMESTEP 600.0

/* RUNTIME CONFIG STRUCT:
 * - Holds all data associated with the rt environment configuration options such as input 
 *   files, rbf stencil size, number of levels, etc.
 */
typedef struct rt_config_struct {

	// Nodeset
	int nodeset_from_file;
	char nodeset_file[MAX_PATH_SIZE];

	// Vertical Layers
	int num_layers;
	double model_height;
	
	// RBF-FD Specifications
	int stencil_size;

	// Timestepping
	double timestep_length;

	// MPI
	int use_metis;

} rt_config_struct;


/********************************************* DOMAINS ********************************************/

/* NODESET STRUCT:
 * - Holds all data associated with a particular nodeset as well as the RBF-FD stencil 
 *   adjacency/distance matrices.
 */
typedef struct nodeset_struct {

	/***** Sizes/Dimensions *****/
	int Nnodes;		// Number of nodes in nodeset
	int n;			// RBF-FD stencil size


	/***** Coordinates *****/

	/* Cartesian Coordinates (on unit sphere):
	 * - Size -> Nnodes elements
	 */
	double* x;
	double* y;
	double* z;

	/* Spherical Coordinates:
	 * - Size -> Nnodes elements
	 * - meridonal angle -> lambda: [0,2*pi]
	 * - latitudinal angle -> phi: [-pi/2,pi/2]
	 */
	double* lambda;
	double* phi;


	/***** RBF-FD Stencils *****/

	/* Adjacency Matrix:
	 * - Size -> Nnodes rows by n columns (row major)
	 * - Rows of idx contain the node ids of the neighbor nodes in the stencil centered at the node 
	 *   with id that of the row number.
	 * - EXAMPLE: idx[(i*n)+j] returns the node id of the jth neighbor in the stencil centered at 
	 *   node i
	 */
	int* idx;

	/* Distance Matrix: 
	 * - Size -> Nnodes rows by n columns (row major)
	 * - Contains the euclidean distance between the two nodes of the associated stencil in the 
	 *   same position of the idx matrix described above.
	 * - EXAMPLE: D[(i*n)+j] returns the distance between node i and the jth neighbor node in the 
	 *   stencil cenered at node i.
	 */
	double* D;

} nodeset_struct;

/* STRUCT - LAYERS_STRUCT
 * - Holds all data describing the vertical layer structure of the domain
 */
typedef struct layers_struct {

	/***** Sizes/Dimensions *****/
	int Nv;			// Number of layers inside the domain
	int Nvg;		// Number of ghost layers at each boundary
	int Nvt;		// Number of layers in domain including ghost layers
	int pNvt;		// Number of layers in domain including ghost layers and padding layers

	double dh;		// layer height


	/***** Coordinates *****/

	/* Height and Spherical Radius
	 * - Shape -> Nvt elements
	 */
	double* h;
	double* r;


	/***** Layer Dependent Data *****/

	/* Pressure and Density
	 * - Shape: Nvt elements
	 */
	double* p;
	double* rho;

} layers_struct;

/* DOMAINS STRUCT:
 * - Holds the global nodeset as well as data describing the MPI partitioning of this nodeset
 */
typedef struct domains_struct {

	/***** Sizes/Dimensions *****/
	int Nnodes;		// Number of nodes in the global domain's nodeset
	int Nparts;		// Number of MPI partitions (same as mpi_size)


	/***** Nodeset and Stencils *****/

	// Nodeset of the global domain
	nodeset_struct global_nodeset[1];


	/***** MPI Patch Partitioning *****/

	/* Partition Ids:
	 * - Size -> Nnodes
	 * - Array holding the partition ids for each node in the nodeset (these are the compute nodes 
	 *   of the associated MPI patch)
	 * - EXAMPLE: part_ids[i] returns the partition id that is assigned to node i of the global 
	 *   nodeset
	 */
	int* part_ids;

	/* Partition Sizes:
	 * - Size -> Nparts
	 * - Array holding the number of nodes assigned to each partition
	 */
	int* part_sizes;
	
	/* Partition Start Ids:
	 * - Size -> Nparts
	 * - Array containing the initial node id of each partition in the global domain (note this is 
	 *   only assigned/relevant once the nodeset has been reordered to accommodate contiguous 
	 *   partitions)
	 */
	int* part_start_ids;

} domains_struct;

/* HALO STRUCT:
 * - Holds all data required for MPI halo layer communication with a single neighboring patch such 
 *   as the neghbors rank and mappings describing the halo layers in both domains
 */
typedef struct halos_struct {

	/***** Collective Sizes/Dimensions *****/
	
	int Nnbrs;						// Number of neighboring patches
	int halo_size_sum;				// total number of halo nodes
	int nbr_halo_size_sum;			// total number of neighboring patch halo nodes


	/***** Neighbor Specific Scalars *****/
	
	int* nbr_ranks;					// mpi ranks of neighbors
	int* halo_sizes;				// current rank's halo sizes in neighbor patches
	int* halo_offsets;				// offsets for each halo in coallesced halo buffer
	int* nbr_halo_sizes;			// neighbor's halo sizse in the local patch
	int* nbr_halo_offsets;			// offsets for each halo in coallesced neighbor halo buffer


	/***** Halo Layer Data/Mappings *****/

	// local patch ids of collective neighbor halo layer in the current rank's domain
	int* nbr_hid2pid;
	int* nbr_hid2gid;

	// local patch ids of the local patch's collective halo layer
	int* hid2pid;
	int* hid2gid;

} halos_struct;


typedef struct rbffds_DMs_struct {

	int part_Nnodes;
	int n;

	int* idx;
	
	double* D;

	double* H;

	double* Dx;
	double* Dy;
	double* Dz;

	double* L;

} rbffd_DMs_struct;

/* PATCH STRUCT:
 * - Holds ALL data associated with an MPI patch including the nodeset, state variables, and the 
 *   halo layer descriptions necessary for MPI communication (see halo_struct below)
 */
typedef struct patch_struct {

	/***** Sizes/Dimensions *****/
	int Nnodes;					// total number of nodes with ghost nodes in the patch
	int part_Nnodes;			// number of nodes in the patch's partition (number of compute nodes)


	/***** Nodeset Data *****/
	
	nodeset_struct nodeset[1];		// local patch nodeset

	rbffd_DMs_struct rbffd_DMs[1];

	/* Mappings to/from Global Domain
	 * - Note we generally refer to:
	 * 		- pid -> local patch node id
	 * 		- gid -> global domain node id
	 */
	int* pid2gid;		// mapping from pid to gid
	int* gid2pid;		// mapping from gid to pid


	/***** Partition and Halo Data *****/

	int* part_pids;				// node ids of the patch's partition

	halos_struct halos[1];		// halo layer data for all neighboring patches


} patch_struct;

#endif

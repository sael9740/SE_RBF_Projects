#include "include/layers.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

extern int mpi_size;
extern int mpi_rank;

extern phys_constants_struct phys_constants[1];

/* FUNCTION - GET_LAYERS
 * - Determines the vertical layer structure based on the given model height (htop) and number of 
 *   vertical layers (Nv)
 */
void get_layers(layers_struct* layers, double htop, int Nv) {

	// get number of ghost layers and total number of layers
	int Nvg = GHOST_LAYER_DEPTH;
	int Nvt = Nv + (2 * Nvg);

	// determine total number of layers with padding to ensure cache alignment
	int pNvt = ALIGNUP(Nvt, CACHE_ALIGN);

	// determine layer height
	double dh = htop / Nv;

	// allocate coordinate data space
	double* h = (double*) malloc(sizeof(double) * Nvt);
	double* r = (double*) malloc(sizeof(double) * Nvt);

	// get Earth radius
	double R = phys_constants->R;

	// determine layer heights
	for (int i = 0; i < Nvt; i++) {
		double height = dh * ((i - Nvg) + .5);
		h[i] = height;
		r[i] = R + height;
	}

	// assign layers struct data
	layers->Nv = Nv;
	layers->Nvg = Nvg;
	layers->Nvt = Nvt;
	layers->pNvt = pNvt;
	layers->dh = dh;

	layers->h = h;
	layers->r = r;

}


/* FUNCTION - PRINT_LAYERS
 * - Debugging/Validation 
 * - Simply prints the contents of the given layers struct
 */
void print_layers(layers_struct* layers) {

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\nRank %d layers:\n\tScalars ->  \tNv = %d, \tNvg = %d, \tNvt = %d, \tpNvt = %d, \tdt = %.1f\n\tLayer Data:\n",
					mpi_rank, layers->Nv, layers->Nvg, layers->Nvt, layers->pNvt, layers->dh); fflush(stdout);
			for (int i = 0; i < layers->Nvt; i++) {
				printf("\t\tlayer id = %3d -> \th = %7.1f m,\tr = %9.1f m\n", i, layers->h[i], layers->r[i]); fflush(stdout);
			}
		}
		usleep(1000);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}












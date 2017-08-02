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
	int Nvt = Nv + (2 * GHOST_SIZE);
	int Ng = GHOST_SIZE;

	// determine total number of layers with padding to ensure cache alignment
	int padded_Nv = ALIGNUP(Nv,CACHE_ALIGN);
	int padded_Nvt = ALIGNUP(Nvt, CACHE_ALIGN);

	// determine layer height
	double dh = htop / Nv;

	// allocate coordinate data space
	double* h = (double*) malloc(sizeof(double) * padded_Nv);
	double* r = (double*) malloc(sizeof(double) * padded_Nv);
	double* r_inv = (double*) malloc(sizeof(double) * padded_Nv);

	// get Earth radius
	double R = phys_constants->R;

	// determine layer heights
	for (int i = 0; i < padded_Nv; i++) {
		double height = dh * (i + .5);
		h[i] = height;
		r[i] = R + height;
		r_inv[i] = 1/(R + height);
	}

	// assign layers struct data
	layers->Nv = Nv;
	layers->Ng = Ng;
	layers->Nvt = Nvt;
	layers->padded_Nv = padded_Nv;
	layers->padded_Nvt = padded_Nvt;
	layers->dh = dh;
	layers->htop = htop;

	layers->h = h;
	layers->r = r;
	layers->r_inv = r_inv;

}


/* FUNCTION - PRINT_LAYERS
 * - Debugging/Validation 
 * - Simply prints the contents of the given layers struct
 */
void print_layers(layers_struct* layers) {

	for (int rank = 0; rank < mpi_size; rank++) {
		if (rank == mpi_rank) {
			printf("\nRank %d layers:\n\tScalars ->  \tNv = %d, \tNvt = %d, \tpadded_Nv = %d, \tpadded_Nvt = %d, \tdt = %.1f\n\tLayer Data:\n",
					mpi_rank, layers->Nv, layers->Nvt, layers->padded_Nv, layers->padded_Nvt, layers->dh); fflush(stdout);
			for (int i = 0; i < layers->Nvt; i++) {
				printf("\t\tlayer id = %3d -> \th = %7.1f m,\tr = %9.1f m\n", i, layers->h[i], layers->r[i]); fflush(stdout);
			}
		}
		usleep(1000);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}












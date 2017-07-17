#include "include/rbffd.h"
#include <stdlib.h>
#include <stdio.h>
#include <lapacke.h>
#include <math.h>

int* get_idx(double* D2, size_t Nh, size_t n);
double* get_D2(nodeset_struct ns1);
void print_idx(int* idx, size_t Nh, size_t n);


void get_rbffd_DMs(nodeset_struct ns1) {

	size_t n = 10;

	size_t Nh = ns1.Nh;

	double* D2 = get_D2(ns1);

	int* idx = get_idx(D2, Nh, n);
	print_idx(idx, Nh, n);

	//get_Dx(idx, D2, ns1, Nh, n);

}



void print_idx(int* idx, size_t Nh, size_t n) {

	printf("\n\nNeearest Neighbor Stencils:\n");
	for (int i = 0; i < Nh; i++) {
		printf("\nnode id = %d:",i);
		for (int k = 0; k < n; k++) {
			printf("\t%3d", idx[(i*n) + k]);
		}
	}

	printf("\n\n");
	/*printf("\n\nNeearest Neighbor Stencil Distances^2:\n");
	for (int i = 0; i < Nh; i++) {
		printf("\nnode id = %d:",i);
		for (int k = 0; k < n; k++) {
			printf("\t%5.4f", dist_idx[(i*n) + k]);
		}
	}*/

}

double* get_D2(nodeset_struct ns1) {

	size_t Nh = ns1.Nh;

	double* D2 = (double*) malloc(sizeof(double) * Nh * Nh);

	for (int i = 0; i < Nh; i++) {
		for (int j = 0; j < Nh; j++) {
			D2[(i*Nh) + j] = pow(ns1.x[i] - ns1.x[j], 2) + pow(ns1.y[i] - ns1.y[j], 2) + pow(ns1.z[i] - ns1.z[j], 2);
		}
	}

	//get_idx_matrix(dist_matrix, Nh, 10);

	return D2;
}

int* get_idx(double* D2, size_t Nh, size_t n) {
	
	int* idx = (int*) malloc(sizeof(int) * Nh * n);
	double* D2_idx = (double*) malloc(sizeof(double) * Nh * n);
	
	for (int i = 0; i < Nh * n; i++) {
		D2_idx[i] = 3.0;
		idx[i] = -1;
	}

	for (int i = 0; i < Nh; i++) {
		for (int j = 0; j < Nh; j++) {
			double d2 = D2[(i*Nh) + j];
			int idx_temp1 = j;
			for (int k = 0; k < n; k++) {
				if (D2_idx[(i*n)+k] > d2) {
					double d2_temp = d2;
					int idx_temp2 = idx_temp1;
					d2 = D2_idx[(i*n)+k];
					idx_temp1 = idx[(i*n)+k];
					D2_idx[(i*n)+k] = d2_temp;
					idx[(i*n)+k] = idx_temp2;

					if (idx_temp1 == -1) {
						break;
					}
				}
			}
		}
	}

	return (idx);

}

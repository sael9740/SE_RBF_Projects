#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <mkl_lapacke.h>
#include <mkl.h>

#include "include/rbffd_DMs.h"
#include "include/debug.h"

void get_part_rbffd_stencils(patch_struct* local_patch) {

	rbffd_DMs_struct* rbffd_DMs = local_patch->rbffd_DMs;
	nodeset_struct* patch_nodeset = local_patch->nodeset;
	int* partid2pid = local_patch->part_pids;
	int* patch_idx = patch_nodeset->idx;
	double* patch_D = patch_nodeset->D;

	int part_Nnodes = local_patch->part_Nnodes;
	int Nnodes = local_patch->Nnodes;
	int n = patch_nodeset->n;

	double* D = (double*) malloc(sizeof(double) * part_Nnodes * n);

	for (int partid = 0; partid < part_Nnodes; partid++) {
		int pid = partid2pid[partid];
		for (int sid = 0; sid < n; sid++) {
			D[(partid * n) + sid] = patch_D[(pid * n) + sid];
		}
	}

	int* idx = (int*) malloc(sizeof(int) * part_Nnodes * n);

	for (int partid = 0; partid < part_Nnodes; partid++) {
		int pid = partid2pid[partid];
		for (int sid = 0; sid < n; sid++) {
			idx[(partid * n) + sid] = patch_idx[(pid * n) + sid];
		}
	}

	rbffd_DMs->part_Nnodes = part_Nnodes;
	rbffd_DMs->n = n;
	rbffd_DMs->idx = idx;
	rbffd_DMs->D = D;

}

void init_rbffd_HH_rot_Ms(patch_struct* local_patch) {

	rbffd_DMs_struct* rbffd_DMs = local_patch->rbffd_DMs;
	nodeset_struct* nodeset = local_patch->nodeset;

	int* partid2pid = local_patch->part_pids;

	int Npart = rbffd_DMs->part_Nnodes;
	int n = rbffd_DMs->n;
	
	double* H = (double*) calloc(Npart * 9, sizeof(double));
	
	for (int partid = 0; partid < Npart; partid++) {
		
		int pid = partid2pid[partid];
		
		double x = nodeset->x[pid];
		double y = nodeset->y[pid];
		double z = nodeset->z[pid];

		double dot_lon = x*x + y*y;
		double v[3] = {x,y,1.0};

		double beta;
		if (dot_lon == 0.0) {
			beta = 0.0;
		}
		else {
			v[2] = z > 0.0 ? - dot_lon / (z + 1.0) : z - 1.0;
			beta = 2.0 * (v[2] * v[2]) / (dot_lon + (v[2] * v[2]));
			v[0] = v[0]/v[2]; v[1] = v[1]/v[2]; v[2] = v[2]/v[2];
		}

		for (int i = 0; i < 3; i++) {
			H[(partid*9) + (4*i)] = 1.0;
		}
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,3,3,1,-beta,v,1,v,3,1.0,&H[(partid*9)],3);
		//cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,3,1,3,1.0,&H[(partid*9)],3,xx,1,0.0,xxp,1);
	}

	rbffd_DMs->H = H;

}

void get_rbffd_DMs_Dxyz(patch_struct* local_patch) {

	rbffd_DMs_struct* rbffd_DMs = local_patch->rbffd_DMs;
	nodeset_struct* nodeset = local_patch->nodeset;

	int* partid2pid = local_patch->part_pids;

	int part_Nnodes = rbffd_DMs->part_Nnodes;
	int n = rbffd_DMs->n;

	int* idx = rbffd_DMs->idx;
	double* HH = rbffd_DMs->H;

	double* x = nodeset->x;
	double* y = nodeset->y;
	double* z = nodeset->z;

	double* X = (double*) malloc(sizeof(double) * n * 3);
	double* Xp = (double*) malloc(sizeof(double) * n * 3);
	
	int Npoly = RBF_NPOLY(RBF_POLY_ORDER);
	int size = Npoly+n;
	double* A = (double*) calloc(sizeof(double), size * size);
	double* B = (double*) calloc(sizeof(double), size * 3);
	double* W = (double*) calloc(sizeof(double), size * 3);
	lapack_int* ipiv = (lapack_int*) malloc(sizeof(lapack_int)*size);

	double* Dx = (double*) malloc(sizeof(double) * part_Nnodes * n);
	double* Dy = (double*) malloc(sizeof(double) * part_Nnodes * n);
	double* Dz = (double*) malloc(sizeof(double) * part_Nnodes * n);

	for (int partid = 0; partid < part_Nnodes; partid++) {

		// set up X and d
		for (int sid = 0; sid < n; sid++) {
			int pid = idx[(partid * n) + sid];
			X[(0*n) + sid] = x[pid];
			X[(1*n) + sid] = y[pid];
			X[(2*n) + sid] = z[pid];
		}
		
		// rotate all nodes in stencil to be centered on the z axis
		double* H = &HH[9*partid];
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,3,n,3,1.0,H,3,X,n,0.0,Xp,n);

		double* xp = &Xp[0*n];
		double* yp = &Xp[1*n];
		double* zp = &Xp[2*n];

		// assign A
		for (int i = 0; i < n; i++) {
			double xpi = xp[i];
			double ypi = yp[i];
			double zpi = zp[i];
			
			double dx,dy,dz,r;

			for (int j = 0; j < n; j++) {
				dx = xp[j] - xpi;
				dy = yp[j] - ypi;
				dz = zp[j] - zpi;
				r = L2_norm(dx,dy,dz);
				A[(i*size) + j] = phs(r, RBF_PHS_ORDER);
			}

			dx = xp[0] - xpi;
			dy = yp[0] - ypi;
			dz = zp[0] - zpi;
			r = L2_norm(dx,dy,dz);
			B[(3*i)] = ddri_phs(r, dx, RBF_PHS_ORDER);
			B[(3*i) + 1] = ddri_phs(r, dy, RBF_PHS_ORDER);
			B[(3*i) + 2] = 0.0;

		}

		int offset = n;
		for (int polyord = 0; polyord <= RBF_POLY_ORDER; polyord++) {
			for (int i = 0; i <= polyord; i++) {
				for (int j = 0; j < n; j++) {
					double a = pow(xp[j], polyord - i) * pow(yp[j],i);
					A[(i + offset) * size + j] = a;
					A[j*size + i + offset] = a;
				}
				B[3*(i+offset)] = polyord == 1 && i == 0 ? 1.0 : 0.0;
				B[3*(i+offset) + 1] = polyord == 1 && i == 1 ? 1.0 : 0.0;
				B[3*(i+offset) + 2] = 0.0;
			}
			offset += polyord+1;
		}

		for (int i = n; i < size; i++) {
			for (int j = n; j < size; j++) {
				A[(i*size) + j] = 0.0;
			}
		}
		
		LAPACKE_dgesv(LAPACK_ROW_MAJOR, size, 3, A, size, ipiv, B, 3);

		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,3,size,3,1.0,H,3,B,3,0.0,W,size);
	
		for (int sid = 0; sid < n; sid++) {
			Dx[(partid*n) + sid] = W[(0*size)+sid];
			Dy[(partid*n) + sid] = W[(1*size)+sid];
			Dz[(partid*n) + sid] = W[(2*size)+sid];
		}
	}
	rbffd_DMs->Dx = Dx;
	rbffd_DMs->Dy = Dy;
	rbffd_DMs->Dz = Dz;
}

double L2_norm(double x, double y, double z) {

	return sqrt(x*x + y*y + z*z);

}

double phs(double r, int k) {

	double val;
	double kk = (double) k;
	
	if (k%2 == 0)
		val = r == 0.0 ? r : log(r) * pow(r, kk);
	else
		val = pow(r, kk);

	return val;
}

double ddri_phs(double r, double ri, int k) {
	
	double val;
	double kk = (double) k;

	if (k%2 == 0)
		val = (r == 0.0 ? r : ri * pow(r, kk - 2.0) * (kk + log(r)));
	else
		val = kk * ri * pow(r, kk - 2.0);

	return val;

}














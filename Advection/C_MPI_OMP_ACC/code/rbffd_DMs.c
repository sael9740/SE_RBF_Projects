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

extern int mpi_size;
extern int mpi_rank;

extern phys_constants_struct phys_constants[1];

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
	}

	rbffd_DMs->H = H;

}

void get_rbffd_DMs(patch_struct* local_patch, int k_phs, int k_L, double epsilon) {


	rbffd_DMs_struct* rbffd_DMs = local_patch->rbffd_DMs;
	nodeset_struct* nodeset = local_patch->nodeset;
	layers_struct* layers = local_patch->layers;

	int* partid2pid = local_patch->part_pids;

	int part_Nnodes = rbffd_DMs->part_Nnodes;
	int n = rbffd_DMs->n;

	int* idx = rbffd_DMs->idx;
	double* HH = rbffd_DMs->H;

	double* x = nodeset->x;
	double* y = nodeset->y;
	double* z = nodeset->z;

	double R = phys_constants->R;

	double* X = (double*) malloc(sizeof(double) * n * 3);
	double* Xp = (double*) malloc(sizeof(double) * n * 3);
	
	int Npoly = RBF_NPOLY(RBF_POLY_ORDER);
	int size = n + Npoly;
	int size2 = n + 1;

	double* A = (double*) calloc(sizeof(double), size * size);
	double* A2 = (double*) calloc(sizeof(double), size2 * size2);
	double* B = (double*) calloc(sizeof(double), size * 3);
	double* B2 = (double*) calloc(sizeof(double), size2);
	double* W = (double*) calloc(sizeof(double), size * 3);

	lapack_int* ipiv = (lapack_int*) malloc(sizeof(lapack_int)*(size+size2));

	double* hDx = (double*) malloc(sizeof(double) * part_Nnodes * n);
	double* hDy = (double*) malloc(sizeof(double) * part_Nnodes * n);
	double* hDz = (double*) malloc(sizeof(double) * part_Nnodes * n);
	double* vDx = (double*) malloc(sizeof(double) * part_Nnodes * FD1_SIZE);
	double* vDy = (double*) malloc(sizeof(double) * part_Nnodes * FD1_SIZE);
	double* vDz = (double*) malloc(sizeof(double) * part_Nnodes * FD1_SIZE);
	double* L = (double*) malloc(sizeof(double) * part_Nnodes * n);

	double* hDxp = (double*) malloc(sizeof(double) * part_Nnodes * n);
	double* hDyp = (double*) malloc(sizeof(double) * part_Nnodes * n);

	//double* vDzp = rbffd_DMs->vDzp;
	double vDzp[FD1_SIZE] = UNIT_FD1_WEIGHTS;
	double* vD3p = (double*) calloc(3 * FD1_SIZE, sizeof(double));
	double* vD3 = (double*) calloc(3 * FD1_SIZE, sizeof(double));

	for (int i = 0; i < FD1_SIZE; i++) {
		double w = vDzp[i]/layers->dh;
		vDzp[i] = w;
		vD3p[(3*i) + 2] = w;
	}

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
				A[(i*size) + j] = phsrbf(r, RBF_PHS_ORDER);
				A2[(i*size2) + j] = garbf(r, epsilon);
			}

			dx = xp[0] - xpi;
			dy = yp[0] - ypi;
			dz = zp[0] - zpi;
			r = L2_norm(dx,dy,dz);
			B[(3*i)] = d1_phsrbf(r, dx, k_phs);
			B[(3*i) + 1] = d1_phsrbf(r, dy, k_phs);
			B[(3*i) + 2] = 0.0;
			B2[i] = L_garbf(r, epsilon, k_L);

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

		for (int i = 0; i < n; i++) {
			A2[(n*size2) + i] = 1.0;
			A2[(i*size2) + n] = 1.0;
		}
		A2[(size2*size2) - 1] = 0.0;
		B2[n] = 0.0;
		
		/*if (partid == 0 && mpi_rank == 0) {
			printf("\nNODE %d:\tx = %1.3e\ty = %1.3e\tz = %1.3e\n",partid,X[0],X[n],X[2*n]);
			printf("\n----> A_phs\n"); print_generic_fp_matrix(A,size,size,FALSE);
			printf("\n----> B_phs\n"); print_generic_fp_matrix(B,size,3,FALSE);
			printf("\n----> A_ga\n"); print_generic_fp_matrix(A2,size2,size2,FALSE);
			printf("\n----> B_Lga\n"); print_generic_fp_matrix(B2,size2,1,FALSE);
			printf("\n----> H\n"); print_generic_fp_matrix(H,3,3,FALSE);
			printf("\n----> Dzp\n"); print_generic_fp_matrix(vD3p,FD1_SIZE,3,FALSE);
		}*/

		int stat1 = LAPACKE_dgesv(LAPACK_ROW_MAJOR, size, 3, A, size, ipiv, B, 3);
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,3,size,3,1.0,H,3,B,3,0.0,W,size);

		int stat2 = LAPACKE_dgesv(LAPACK_ROW_MAJOR, size2, 1, A2, size2, &ipiv[size], B2, 1);

		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,3,FD1_SIZE,3,1.0,H,3,vD3p,3,0.0,vD3,FD1_SIZE);
	
		for (int sid = 0; sid < n; sid++) {
			hDx[(partid*n) + sid] = W[(0*size)+sid];
			hDy[(partid*n) + sid] = W[(1*size)+sid];
			hDz[(partid*n) + sid] = W[(2*size)+sid];

			hDxp[(partid*n) + sid] = B[(3*sid)+0];
			hDyp[(partid*n) + sid] = B[(3*sid)+1];

			L[(partid*n) + sid] = B2[sid]/R;
		}
		for (int vid = 0; vid < FD1_SIZE; vid++) {
			vDx[(partid*FD1_SIZE) + vid] = vD3[(0*FD1_SIZE) + vid];
			vDy[(partid*FD1_SIZE) + vid] = vD3[(1*FD1_SIZE) + vid];
			vDz[(partid*FD1_SIZE) + vid] = vD3[(2*FD1_SIZE) + vid];
		}
		/*if (partid == 0 && mpi_rank == 0) {
			printf("\nSTATUS: %d %d\nNODE %d:\tx = %1.3e\ty = %1.3e\tz = %1.3e\n",stat1,stat2,partid,X[0],X[n],X[2*n]);
			printf("\n----> Dxyp\n"); print_generic_fp_matrix(B,size,3,FALSE);
			printf("\n----> Dxyz\n"); print_generic_fp_matrix(W,3,size,FALSE);
			printf("\n----> L\n"); print_generic_fp_matrix(B2,size2,1,FALSE);
			printf("\n----> vDxyz\n"); print_generic_fp_matrix(vD3,3,FD1_SIZE,FALSE);
		}*/
	}
	
	rbffd_DMs->hDx = hDx;
	rbffd_DMs->hDy = hDy;
	rbffd_DMs->hDz = hDz;
	rbffd_DMs->vDx = vDx;
	rbffd_DMs->vDy = vDy;
	rbffd_DMs->vDz = vDz;
	rbffd_DMs->L = L;

	rbffd_DMs->hDxp = hDxp;
	rbffd_DMs->hDyp = hDyp;
	memcpy((void*) rbffd_DMs->vDzp, (void*) vDzp, sizeof(double) * FD1_SIZE);
}

double L2_norm(double x, double y, double z) {

	return sqrt(x*x + y*y + z*z);

}

double phsrbf(double r, int k) {

	if (k%2 == 0)
		return (r == 0.0 ? r : log(r) * pow(r,k));
	else
		return pow(r,k);

}

double d1_phsrbf(double r, double ri, int k) {
	
	if (k%2 == 0)
		return (r == 0.0 ? r : ri * pow(r, k-2) * (k + log(r)));
	else
		return (k * ri * pow(r, k-2));

}

double garbf(double r, double epsilon) {

	return exp(-pow(r*epsilon,2));

}

double L_garbf(double r, double epsilon, int k) {

	double lagpoly[MAX_HYPERVISCOSITY_ORDER];

	double epr2 = pow(epsilon*r, 2);

	lagpoly[0] = 1;
	lagpoly[1] = 4*epr2 - 4;

	for (int i = 1; i < k; i++) {
		lagpoly[i+1] = 4*(epr2 - (2*i + 1))*lagpoly[i] - (16*pow(i,2))*lagpoly[i-1];
	}

	return pow(epsilon, 2*k) * lagpoly[k] * garbf(r, epsilon);
}


void eval_partSVDotGradSV(patch_struct* local_patch, double alpha, part_SV_struct* SV_A, patch_SV_struct* SV_B, part_SV_struct* SV_C) {

	/***** Extract Local Patch Data *****/
	layers_struct* layers = local_patch->layers;
	rbffd_DMs_struct* rbffd_DMs = local_patch->rbffd_DMs;
	int* partid2pid = local_patch->part_pids;
	int Nnodes = local_patch->Nnodes;
	int part_Nnodes = local_patch->part_Nnodes;

	/***** Extract Layers Data *****/
	double* r_inv = layers->r_inv;
	int Nv = layers->Nv;
	int Ng = layers->Ng;
	int padded_Nv = layers->padded_Nv;
	int Nvt = layers->Nvt;
	int padded_Nvt = layers->padded_Nvt;

	/***** Extract RBFFD DM Data *****/
	int n = rbffd_DMs->n;
	int* idx = rbffd_DMs->idx;
	double* H = rbffd_DMs->H;
	double* hDxp = rbffd_DMs->hDxp;
	double* hDyp = rbffd_DMs->hDyp;
	double* vDzp = rbffd_DMs->vDzp;

	/***** Extract SV Data *****/
	double* A = SV_A->data;
	double* B = SV_B->data;
	double* C = SV_C->data;
	double* gradB = local_patch->SV_3D_temp->data;

	/***** Zero out gradB *****/
	for (int i = 0; i < 3*padded_Nv*part_Nnodes; i++) {
		gradB[i] = 0.0;
	}

	/***** Evaluate gradient(B) *****/

	// iterate through each partition node
	for (int partid = 0; partid < part_Nnodes; partid++) {
		
		// patch id of corresponding partition node
		int pid = partid2pid[partid];

		/***** Horizontal Differentiation *****/
		
		// Iterate through each stencil neighbor
		for (int sid = 0; sid < n; sid++) {

			// get pid of stencil neighbor
			int nbr_pid = idx[(partid * n) + sid];

			// get RBFFD d/dxp and d/dyp DM weights for stencil neighbor
			double hdxp = hDxp[(partid * n) + sid];
			double hdyp = hDyp[(partid * n) + sid];

			// iterate through vertical column
			for (int vid = 0; vid < Nv; vid++) {

				// value of SV at corresponding stencil neighbor/vertical level
				double b = B[(nbr_pid * padded_Nvt) + vid];

				// get corresponding DM weight/SV product and update differentiation sums
				gradB[(((3 * partid) + 0) * padded_Nv) + vid] += hdxp * b;
				gradB[(((3 * partid) + 1) * padded_Nv) + vid] += hdyp * b;
			}
		}

		// scale horizontal differentiation results for shell radius
		for (int vid = 0; vid < Nv; vid++) {
			
			// hDxp = hDxp/r, hDyp = hDyp/r
			gradB[(((3 * partid) + 0) * padded_Nv) + vid] *= r_inv[vid];
			gradB[(((3 * partid) + 1) * padded_Nv) + vid] *= r_inv[vid];

		}


		/***** Vertical Differentiation *****/
		
		// Iterate through each vertical FD stencil point
		for (int sid = 0; sid < FD1_SIZE; sid++) {

			// d/dzp weight for corresponding FD stencil id
			double vdzp = vDzp[sid];

			// iterate through each vertical level
			for (int vid = 0; vid < Nv; vid++) {

				// SV value at corresponding FD stencil point
				double b = B[(pid * padded_Nvt) + vid + (sid - Ng)];

				// determine FD weight/SV product and update differentiation sum
				gradB[(((3 * partid) + 2) * padded_Nv) + vid] += vdzp * b;

			}
		}
	}

	/***** C = alpha * A dot grad B *****/
	for (int partid = 0; partid < part_Nnodes; partid++) {

		for (int vid = 0; vid < Nv; vid++) {
			
			double udxp = A[(((3*partid) + 0) * padded_Nv) + vid] * gradB[(((3 * partid) + 0) * padded_Nv) + vid];
			double vdyp = A[(((3*partid) + 1) * padded_Nv) + vid] * gradB[(((3 * partid) + 1) * padded_Nv) + vid];
			double wdzp = A[(((3*partid) + 2) * padded_Nv) + vid] * gradB[(((3 * partid) + 2) * padded_Nv) + vid];

			C[(partid * padded_Nv) + vid] = alpha * (udxp + vdyp + wdzp);

		}
	}
}


void eval_sumpatchSVpartSV(patch_struct* local_patch, patch_SV_struct* SV_A, double alpha, part_SV_struct* SV_B, patch_SV_struct* SV_C) {

	/***** Extract Local Patch Data *****/
	layers_struct* layers = local_patch->layers;
	int* partid2pid = local_patch->part_pids;
	int part_Nnodes = local_patch->part_Nnodes;

	/***** Extract Layers Data *****/
	int Nv = layers->Nv;
	int padded_Nv = layers->padded_Nv;
	int Nvt = layers->Nvt;
	int padded_Nvt = layers->padded_Nvt;

	/***** Extract SV Data *****/
	double* A = SV_A->data;
	double* B = SV_B->data;
	double* C = SV_C->data;

	for (int partid = 0; partid < part_Nnodes; partid++) {
		int pid = partid2pid[partid];
		for (int vid = 0; vid < Nv; vid++) {
			C[(pid * padded_Nvt) + vid] = A[(pid * padded_Nvt) + vid] + alpha*B[(partid * padded_Nv) + vid];
		}
	}
}

void eval_sumpatchSVpatchSV(patch_struct* local_patch, patch_SV_struct* SV_A, double alpha, patch_SV_struct* SV_B, part_SV_struct* SV_C) {

	/***** Extract Local Patch Data *****/
	layers_struct* layers = local_patch->layers;
	int* partid2pid = local_patch->part_pids;
	int part_Nnodes = local_patch->part_Nnodes;
	int Nnodes = local_patch->Nnodes;

	/***** Extract Layers Data *****/
	int Nv = layers->Nv;
	int padded_Nv = layers->padded_Nv;
	int Nvt = layers->Nvt;
	int padded_Nvt = layers->padded_Nvt;

	/***** Extract SV Data *****/
	double* A = SV_A->data;
	double* B = SV_B->data;
	double* C = SV_C->data;

	for (int partid = 0; partid < part_Nnodes; partid++) {
		int pid = partid2pid[partid];
		for (int vid = 0; vid < Nv; vid++) {
			C[(partid * padded_Nv) + vid] = A[(pid * padded_Nvt) + vid] + alpha*B[(pid * padded_Nvt) + vid];
		}
	}
}


void eval_sumpartSVpartSV(patch_struct* local_patch, part_SV_struct* SV_A, double alpha, part_SV_struct* SV_B, part_SV_struct* SV_C) {

	/***** Extract Local Patch Data *****/
	layers_struct* layers = local_patch->layers;
	int part_Nnodes = local_patch->part_Nnodes;

	/***** Extract Layers Data *****/
	int Nv = layers->Nv;
	int padded_Nv = layers->padded_Nv;

	/***** Extract SV Data *****/
	double* A = SV_A->data;
	double* B = SV_B->data;
	double* C = SV_C->data;

	for (int partid = 0; partid < part_Nnodes; partid++) {
		for (int vid = 0; vid < Nv; vid++) {
			C[(partid * padded_Nv) + vid] = A[(partid * padded_Nv) + vid] + alpha*B[(partid * padded_Nv) + vid];
		}
	}
}


void update_Up(patch_struct* local_patch) {

	/***** Extract Local Patch Data *****/
	layers_struct* layers = local_patch->layers;
	rbffd_DMs_struct* rbffd_DMs = local_patch->rbffd_DMs;
	double* U = local_patch->SV_U->data;
	double* Up = local_patch->SV_Up->data;
	int part_Nnodes = local_patch->part_Nnodes;

	/***** Extract Layers Data *****/
	int Nv = layers->Nv;
	int Ng = layers->Ng;
	int padded_Nv = layers->padded_Nv;

	/***** Extract RBFFD DM Data *****/
	double* H = rbffd_DMs->H;

	/***** Perform Rotations *****/
	for (int partid = 0; partid < part_Nnodes; partid++) {
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, padded_Nv, 3, 1.0, &H[9*partid], 3, &U[3*padded_Nv*partid], padded_Nv, 0.0, &Up[3*padded_Nv*partid], padded_Nv);
	}
}



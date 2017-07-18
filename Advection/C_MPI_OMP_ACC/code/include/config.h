#ifndef ADV_CONFIG_H
#define ADV_CONFIG_H

#include <stdlib.h>

#define MAX_PATH_SIZE 200

typedef struct adv_params_struct {

	char nodesetFile[MAX_PATH_SIZE];
	size_t n;

} adv_params_struct;

typedef struct nodeset_struct {
	
	size_t Nh;
	size_t Nv;

	double* x;
	double* y;
	double* z;

} nodeset_struct;

#endif

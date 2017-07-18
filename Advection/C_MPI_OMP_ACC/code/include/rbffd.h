#ifndef ADV_RBFFD_H
#define ADV_RBFFD_H

#include "config.h"

// calculate quadrature weights (distances^2) for all node pairings
double* get_D(nodeset_struct ns1);

int* get_idx(double* D, size_t Nh, size_t n);

#endif

#ifndef ADV_IO_H
#define ADV_IO_H

#include "config.h"

// reads nodeset on unit sphere from netcdf file
void get_nodeset(nodeset_struct* nodeset);

void print_nodeset(nodeset_struct* nodeset);

#endif

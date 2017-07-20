#ifndef ADV_IO_H
#define ADV_IO_H

#include "config.h"

// reads nodeset on unit sphere from netcdf file
void get_ns1_xyz(char* nodesetFile, unit_nodeset_struct* ns1);

void print_ns1(unit_nodeset_struct* ns1);

#endif

#ifndef ADV_IO_H
#define ADV_IO_H

#include "config.h"

// reads nodeset on unit sphere from netcdf file
nodeset_struct get_ns1(char* nodesetFile);

void print_ns1(nodeset_struct ns1);

#endif

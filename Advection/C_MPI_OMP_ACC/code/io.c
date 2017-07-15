#include "include/io.h"

#include <netcdf.h>

// reads nodeset on unit sphere from netcdf file
nodeset_struct get_ns1(char* nodesetFile) {
	
	// init unit node set struct
	nodeset_struct ns1;

	// for unit nodeset Nv = 1
	ns1.Nv = 1;

	// init netcdf variable ids
	int ncid;
	int ns_gid;
	int Nh_did;
	int x_vid, y_vid, z_vid;

	// open file and get nodeset groupid
	nc_open(nodesetFile, NC_NOWRITE, &ncid);
	nc_inq_ncid(ncid, "nodeset", &ns_gid);
	
	// get number of horizontal nodes
	nc_inq_dimid(ncid, "Nh", &Nh_did);
	nc_inq_dimlen(ncid, Nh_did, &ns1.Nh);

	// allocate data space for x,y,z
	ns1.x = (double*) malloc(sizeof(double) * ns1.Nh);
	ns1.y = (double*) malloc(sizeof(double) * ns1.Nh);
	ns1.z = (double*) malloc(sizeof(double) * ns1.Nh);

	// read x,y,z
	nc_inq_varid(ns_gid, "x", &x_vid);
	nc_get_var_double(ns_gid, x_vid, ns1.x);
	nc_inq_varid(ns_gid, "y", &y_vid);
	nc_get_var_double(ns_gid, y_vid, ns1.y);
	nc_inq_varid(ns_gid, "z", &z_vid);
	nc_get_var_double(ns_gid, z_vid, ns1.z);

	// close file
	nc_close(ncid);

	// return the nodeset
	return ns1;
}

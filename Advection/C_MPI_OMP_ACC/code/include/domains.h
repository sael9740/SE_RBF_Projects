#ifndef ADV_DOMAINS_H
#define ADV_DOMAINS_H

#include "config.h"

void get_metis_partitioning(domains_struct* global_domains, int Nparts);

void reorder_domain_contiguous_parts(domains_struct* global_domains);

#endif

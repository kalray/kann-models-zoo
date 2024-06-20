/***
 * @copyright Copyright (C) 2024 Kalray SA. All rights reserved.
 * This code is Kalray proprietary and confidential.
 * Any use of the code for whatever purpose is subject
 * to specific written permission of Kalray SA.
 ***/
#include "custom_kernels_list.h"

#include <string.h>

const custom_kernel_fnct_t custom_kernel_table[KANN_NB_CUSTOM_KERNELS] = {
#define CUSTOM_KERNELS_FNCT(NAME)                                              \
    {(mppa_kann_clus_custom_kernel_fnct_pointer_t)                             \
         mppa_kann_clus_kernel_##NAME,                                         \
     "mppa_kann_clus_kernel_" #NAME},
    MPPA_CUSTOM_FOREACH_KERNELS(CUSTOM_KERNELS_FNCT)
#undef CUSTOM_KERNELS_FNCT
};

const int custom_kernel_table_len = KANN_NB_CUSTOM_KERNELS;

mppa_kann_clus_custom_kernel_fnct_pointer_t custom_kernel_get_fnct_ptr_byname(
    const char *name)
{
    for (int i = 0; i < custom_kernel_table_len; i++) {
        if (0 == strcmp(custom_kernel_table[i].name, name)) {
            return custom_kernel_table[i].fnct;
        }
    }
    return NULL;
}

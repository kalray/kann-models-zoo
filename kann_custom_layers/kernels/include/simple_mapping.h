/***
 * @copyright Copyright (C) 2021 Kalray SA. All rights reserved.
 * This code is Kalray proprietary and confidential.
 * Any use of the code for whatever purpose is subject
 * to specific written permission of Kalray SA.
 ***/
#ifndef SIMPLE_MAPPING_H
#define SIMPLE_MAPPING_H
#include "mppa_generic.h"

typedef struct __attribute__((__packed__)) tensor_offset_s {
    uint32_t buffer_id : 1;
    uint32_t offset : 31;
} tensor_offset_t;

static inline char *get_base_address_of_tensor(
  tensor_offset_t a, void *buffer0, void *buffer1)
    __attribute__((always_inline));

static inline char *get_base_address_of_tensor(
    tensor_offset_t a, void *buffer0, void *buffer1)
{
    return (char *)(a.buffer_id ? buffer1 : buffer0) + a.offset;
}

#define SIMPLE_MAPPING_SUBARGS(NB_SRC)                                         \
    const struct __attribute__((__packed__)) {                                 \
        tensor_offset_t dst;                                                   \
        tensor_offset_t src[NB_SRC];                                           \
        uint32_t cdim;                                                         \
        uint32_t nb_sdim;                                                      \
        struct {                                                               \
            uint32_t count;                                                    \
            uint32_t stride_dst;                                               \
            uint32_t stride_src[NB_SRC];                                       \
        } sdim[];                                                              \
    }

#define SIMPLE_MAPPING_KERNEL(NAME, NB_SRC, TYPE_IN, TYPE_OUT, FNCT, ARGS_T)   \
    void NAME(ARGS_T *restrict args, void *kann_data0, void *kann_data1)       \
    {                                                                          \
        uint32_t d, i;                                                         \
        uint32_t indexes[args->subargs.nb_sdim + 1];                           \
        /* compute the product of the dimensions counts                        \
         * also temporary store the partials products in indexes */            \
        uint32_t nb_values = args->subargs.cdim;                               \
        for (d = 0; d < args->subargs.nb_sdim; d++) {                          \
            indexes[d] = nb_values;                                            \
            nb_values *= args->subargs.sdim[d].count;                          \
        }                                                                      \
        /* get the range [beg, end[ of points that will be computed by this    \
         * PE, and get the corresponding indexes as a (nb_sdim + 1)            \
         * dimensional                                                         \
         * array */                                                            \
        uint32_t peid = KANN_GET_PE_ID();                                      \
        uint32_t beg = (nb_values * peid) / KANN_NB_CORES;                     \
        uint32_t end = (nb_values * (peid + 1)) / KANN_NB_CORES;               \
        uint32_t nb_iter = end - beg;                                          \
        /* use the partial dim products to compute the current indexes */      \
        div_t quot_rem = {.rem = beg};                                         \
        for (d = args->subargs.nb_sdim; d > 0; d--) {                          \
            quot_rem = div(quot_rem.rem, indexes[d - 1]);                      \
            indexes[d] = quot_rem.quot;                                        \
        }                                                                      \
        indexes[0] = quot_rem.rem;                                             \
        TYPE_OUT *dst_ptr = (TYPE_OUT *)get_base_address_of_tensor(            \
            args->subargs.dst, kann_data0, kann_data1);                        \
        dst_ptr += indexes[0];                                                 \
        const TYPE_IN *src_ptrs[NB_SRC];                                       \
        for (i = 0; i < NB_SRC; i++) {                                         \
            src_ptrs[i] = (const TYPE_IN *)get_base_address_of_tensor(         \
                args->subargs.src[i], kann_data0, kann_data1);                 \
            src_ptrs[i] += indexes[0];                                         \
        }                                                                      \
        for (d = 0; d < args->subargs.nb_sdim; d++) {                          \
            dst_ptr += indexes[d + 1] * args->subargs.sdim[d].stride_dst;      \
            for (i = 0; i < NB_SRC; i++) {                                     \
                src_ptrs[i] +=                                                 \
                    indexes[d + 1] * args->subargs.sdim[d].stride_src[i];      \
            }                                                                  \
        }                                                                      \
        while (nb_iter-- > 0) {                                                \
            *dst_ptr = FNCT(src_ptrs, args);                                   \
            indexes[0]++;                                                      \
            if (__builtin_expect(indexes[0] == args->subargs.cdim, 0)) {       \
                indexes[0] = 0;                                                \
                dst_ptr -= args->subargs.cdim;                                 \
                for (i = 0; i < NB_SRC; i++) {                                 \
                    src_ptrs[i] -= args->subargs.cdim;                         \
                }                                                              \
                for (d = 0; d < args->subargs.nb_sdim; d++) {                  \
                    indexes[d + 1]++;                                          \
                    dst_ptr += args->subargs.sdim[d].stride_dst;               \
                    for (i = 0; i < NB_SRC; i++) {                             \
                        src_ptrs[i] += args->subargs.sdim[d].stride_src[i];    \
                    }                                                          \
                    if (indexes[d + 1] < args->subargs.sdim[d].count) {        \
                        break;                                                 \
                    }                                                          \
                    indexes[d + 1] = 0;                                        \
                    dst_ptr -= args->subargs.sdim[d].stride_dst                \
                               * args->subargs.sdim[d].count;                  \
                    for (i = 0; i < NB_SRC; i++) {                             \
                        src_ptrs[i] -= args->subargs.sdim[d].stride_src[i]     \
                                       * args->subargs.sdim[d].count;          \
                    }                                                          \
                }                                                              \
            }                                                                  \
            dst_ptr++;                                                         \
            for (i = 0; i < NB_SRC; i++) {                                     \
                src_ptrs[i]++;                                                 \
            }                                                                  \
        }                                                                      \
        KANN_SMEM_FENCE();                                                     \
    }


#define VEC_SIMPLE_MAPPING_KERNEL(                                             \
  NAME, NB_SRC, TYPE_IN, TYPE_OUT, VEC_SIZE, VEC_TYPE, VEC_FNC, FNC, ARGS_T)   \
    void NAME(ARGS_T *restrict args, void *kann_data0, void *kann_data1)       \
    {                                                                          \
        uint32_t d, i;                                                         \
        uint32_t indexes[args->subargs.nb_sdim + 1];                           \
        /* compute the product of the dimensions counts                        \
         * also temporary store the partials products in indexes */            \
        uint32_t nb_values = args->subargs.cdim;                               \
        for (d = 0; d < args->subargs.nb_sdim; d++) {                          \
            indexes[d] = nb_values;                                            \
            nb_values *= args->subargs.sdim[d].count;                          \
        }                                                                      \
        /* get the range [beg, end[ of points that will be computed by this    \
         * PE, and get the corresponding indexes as a (nb_sdim + 1)            \
         * dimensional                                                         \
         * array */                                                            \
        uint32_t peid = KANN_GET_PE_ID();                                      \
        uint32_t beg = (nb_values * peid) / KANN_NB_CORES;                     \
        uint32_t end = (nb_values * (peid + 1)) / KANN_NB_CORES;               \
        uint32_t nb_iter = end - beg;                                          \
        /* use the partial dim products to compute the current indexes */      \
        div_t quot_rem = {.rem = beg};                                         \
        for (d = args->subargs.nb_sdim; d > 0; d--) {                          \
            quot_rem = div(quot_rem.rem, indexes[d - 1]);                      \
            indexes[d] = quot_rem.quot;                                        \
        }                                                                      \
        indexes[0] = quot_rem.rem;                                             \
        TYPE_OUT *dst_ptr = (TYPE_OUT *)get_base_address_of_tensor(            \
            args->subargs.dst, kann_data0, kann_data1);                        \
        dst_ptr += indexes[0];                                                 \
        const TYPE_IN *src_ptrs[NB_SRC];                                       \
        for (i = 0; i < NB_SRC; i++) {                                         \
            src_ptrs[i] = (const TYPE_IN *)get_base_address_of_tensor(         \
                args->subargs.src[i], kann_data0, kann_data1);                 \
            src_ptrs[i] += indexes[0];                                         \
        }                                                                      \
        for (d = 0; d < args->subargs.nb_sdim; d++) {                          \
            dst_ptr += indexes[d + 1] * args->subargs.sdim[d].stride_dst;      \
            for (i = 0; i < NB_SRC; i++) {                                     \
                src_ptrs[i] +=                                                 \
                    indexes[d + 1] * args->subargs.sdim[d].stride_src[i];      \
            }                                                                  \
        }                                                                      \
        while(nb_iter > 0) {                                                   \
            /* Compute SIMD_FUNC x VECTOR_SIZE increment steps */              \
            uint32_t index0_iter;                                              \
            index0_iter = MIN(                                                 \
              (nb_iter / VEC_SIZE),                                            \
              ((args->subargs.cdim - indexes[0]) / VEC_SIZE));                 \
            while (index0_iter-- > 0) {                                        \
                *(VEC_TYPE *)dst_ptr = VEC_FNC(                                \
                    *(const VEC_TYPE**)src_ptrs, args);                        \
                indexes[0]  += VEC_SIZE;                                       \
                dst_ptr     += VEC_SIZE;                                       \
                for (i = 0; i < NB_SRC; i++) {                                 \
                    src_ptrs[i] += VEC_SIZE;                                   \
                }                                                              \
                nb_iter -= VEC_SIZE;                                           \
            }                                                                  \
            /* Compute vector tail (step by step) */                           \
            index0_iter = MIN(nb_iter, (args->subargs.cdim - indexes[0]));     \
            while (index0_iter-- > 0) {                                        \
                *dst_ptr = FNC(src_ptrs, args);                                \
                indexes[0]++;                                                  \
                dst_ptr++;                                                     \
                for (i = 0; i < NB_SRC; i++) {                                 \
                    src_ptrs[i]++;                                             \
                }                                                              \
                nb_iter--;                                                     \
            }                                                                  \
            if (__builtin_expect(indexes[0] == args->subargs.cdim, 0)) {       \
                indexes[0] = 0;                                                \
                dst_ptr -= args->subargs.cdim;                                 \
                for (i = 0; i < NB_SRC; i++) {                                 \
                    src_ptrs[i] -= args->subargs.cdim;                         \
                }                                                              \
                for (d = 0; d < args->subargs.nb_sdim; d++) {                  \
                    indexes[d + 1]++;                                          \
                    dst_ptr += args->subargs.sdim[d].stride_dst;               \
                    for (i = 0; i < NB_SRC; i++)                               \
                        src_ptrs[i] += args->subargs.sdim[d].stride_src[i];    \
                    if (indexes[d + 1] < args->subargs.sdim[d].count) {        \
                        break;                                                 \
                    }                                                          \
                    indexes[d + 1] = 0;                                        \
                    dst_ptr -= args->subargs.sdim[d].stride_dst *              \
                                                  args->subargs.sdim[d].count; \
                    for (i = 0; i < NB_SRC; i++) {                             \
                        src_ptrs[i] -= args->subargs.sdim[d].stride_src[i]     \
                                                * args->subargs.sdim[d].count; \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        KANN_SMEM_FENCE();                                                     \
    }

#endif /*SIMPLE_MAPPING_H*/

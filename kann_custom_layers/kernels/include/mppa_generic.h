/***
 * @copyright Copyright (C) 2024 Kalray SA. All rights reserved.
 * This code is Kalray proprietary and confidential.
 * Any use of the code for whatever purpose is subject
 * to specific written permission of Kalray SA.
 ***/

#ifndef GENERIC_H
#define GENERIC_H

#include <mppa_cos.h>

typedef _Float16 float16_t;
typedef float float32_t;
typedef double float64_t;

typedef _Float16 float16x2_t  __attribute__((vector_size(2 *sizeof(_Float16))));
typedef _Float16 float16x4_t  __attribute__((vector_size(4 *sizeof(_Float16))));
typedef _Float16 float16x8_t  __attribute__((vector_size(8 *sizeof(_Float16))));
typedef _Float16 float16x16_t __attribute__((vector_size(16*sizeof(_Float16))));

typedef float32_t float32x2_t  __attribute__((vector_size(2 *sizeof(float32_t))));
typedef float32_t float32x4_t  __attribute__((vector_size(4 *sizeof(float32_t))));
typedef float32_t float32x8_t  __attribute__((vector_size(8 *sizeof(float32_t))));
typedef float32_t float32x16_t __attribute__((vector_size(16*sizeof(float32_t))));

#define KANN_NB_CORES         (16u)
#define KANN_GET_CLUSTER_ID() ((unsigned)__cos_get_cluster_id())
#define KANN_GET_PE_ID()      ((unsigned)__cos_get_cpu_id())
#define KANN_SMEM_FENCE()     __builtin_kvx_fence()

#endif /*GENERIC_H*/

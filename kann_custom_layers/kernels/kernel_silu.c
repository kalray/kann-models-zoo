/***
 * @copyright Copyright (C) 2024 Kalray SA. All rights reserved.
 * This code is Kalray proprietary and confidential.
 * Any use of the code for whatever purpose is subject
 * to specific written permission of Kalray SA.
 ***/

#include <fastmath.h>
#include "mppa_generic.h"
#include "simple_mapping.h"
#include "kernel_common.h"

#define KANN_PRESISTENT_DATA 1
#define VECTOR_SIZE  8
#define VECTOR_DTYPE float16x8_t
#define FUNC         silu_fast_tf16_tf16
#define VECTOR_FUNC  silu_fastx8_tf16_tf16
#define ARGS         args_silu_tf16_tf16_t


// SiLU FP16 KERNEL =========================================================//
// SiLU FP16 args struct
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_silu_tf16_tf16_args {
    float16_t alpha;
    SIMPLE_MAPPING_SUBARGS(1) subargs;     // single source
} ARGS;

//===========================================================================//
// SiLU Optimized FP16 callback function with taylor's series
//===========================================================================//
static inline float16_t silu_fast_tf16_tf16(
  const float16_t **src, ARGS *args)
{
    // Implementation of y = x * sigmoid(x)
    //   where sigmoid(x) => 1 / (1 + e(x))   for x <= 0
    //                       1 / (1 + 1/e(x)) for x > 0
    //   and e(x) => 1.0 + |x| + 0.555 * x^2 + 0.143 * x^4

    float a = args->alpha * fast_sigmoid(*src[0]);
    return *src[0] * a;
}

//============================================================================//
// SiLU Optimized FP16 callback function with taylor's series vectorized exec
//============================================================================//
static inline VECTOR_DTYPE silu_fastx8_tf16_tf16(
  const VECTOR_DTYPE *srcs, ARGS *args)
{
    // Implementation of y = x * sigmoid(x)
    //   where sigmoid(x) => 1 / (1 + e(x))   for x <= 0
    //                       1 / (1 + 1/e(x)) for x > 0
    //   and e(x) => 1.0 + |x| + 0.555 * x^2 + 0.143 * x^4
    // SIMD Vectors is used here using 8 elements

    // e(x) => 1.0 + |x| + 0.555 * x^2 + 0.143 * x^4
    VECTOR_DTYPE a = {
      1.000f,1.000f,1.000f,1.000f,1.000f,1.000f,1.000f,1.000f };
    VECTOR_DTYPE b = {
      0.555f,0.555f,0.555f,0.555f,0.555f,0.555f,0.555f,0.555f };
    VECTOR_DTYPE c = {
      0.143f,0.143f,0.143f,0.143f,0.143f,0.143f,0.143f,0.143f };
    VECTOR_DTYPE x = {
        ABS(srcs[0][0]),
        ABS(srcs[0][1]),
        ABS(srcs[0][2]),
        ABS(srcs[0][3]),
        ABS(srcs[0][4]),
        ABS(srcs[0][5]),
        ABS(srcs[0][6]),
        ABS(srcs[0][7]),
    };
    VECTOR_DTYPE x2 = x * x;
    VECTOR_DTYPE x4 = x2 * x2;
    VECTOR_DTYPE e = a + x + b * x2 + c * x4;
    VECTOR_DTYPE y = {
        1.0f / (1.0f + (srcs[0][0] > 0 ? 1.0f / e[0] : e[0])),
        1.0f / (1.0f + (srcs[0][1] > 0 ? 1.0f / e[1] : e[1])),
        1.0f / (1.0f + (srcs[0][2] > 0 ? 1.0f / e[2] : e[2])),
        1.0f / (1.0f + (srcs[0][3] > 0 ? 1.0f / e[3] : e[3])),
        1.0f / (1.0f + (srcs[0][4] > 0 ? 1.0f / e[4] : e[4])),
        1.0f / (1.0f + (srcs[0][5] > 0 ? 1.0f / e[5] : e[5])),
        1.0f / (1.0f + (srcs[0][6] > 0 ? 1.0f / e[6] : e[6])),
        1.0f / (1.0f + (srcs[0][7] > 0 ? 1.0f / e[7] : e[7]))
    };
    return srcs[0] * y;
}

//============================================================================//
// Vectorized SiLU FP16 simple mapping kernel call
// (define in simple_mapping.h)
VEC_SIMPLE_MAPPING_KERNEL(
    mppa_kann_clus_kernel_silu_x8_tf16_tf16,  // kernel_name
    1,                                        // nb sources
    float16_t,                                // src type
    float16_t,                                // dst type
    VECTOR_SIZE,                              // vector size
    VECTOR_DTYPE,                             // vector type
    VECTOR_FUNC,                              // vect compute fcnt
    FUNC,                                     // compute fcnt
    ARGS                                      // args struct
)
//============================================================================//

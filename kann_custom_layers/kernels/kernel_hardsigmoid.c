#include <fastmath.h>
#include "mppa_generic.h"
#include "simple_mapping.h"
#include "kernel_common.h"

#define VECTOR_SIZE  16
#define VECTOR_DTYPE float16x16_t
#define FUNC         hsigmoid_tf16_tf16
#define VECTOR_FUNC  hsigmoid_x16_tf16_tf16
#define ARGS_F32     args_mish_tf32_tf32_t
#define ARGS_F16     args_mish_tf16_tf16_t



// HardSigmoid FP32 args struct
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_hsigmoid_tf32_tf32_args {
    float32_t alpha;
    float32_t beta;
    SIMPLE_MAPPING_SUBARGS(1) subargs;  // single source
} ARGS_F32;


// HardSigmoid FP16 args struct
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_hsigmoid_tf16_tf16_args {
    float16_t alpha;
    float16_t beta;
    SIMPLE_MAPPING_SUBARGS(1) subargs;     // single source
} ARGS_F16;

// HardSigmoid FP32 KERNEL =========================================================//
// HardSigmoid FP32 callback function
static inline float32_t hsigmoid_tf32_tf32(
  const float32_t **src, ARGS_F32 *args)
{
    float32_t x = *src[0] * args->alpha + args->beta;
    return MAX(0, MIN(1, x));
}
// HardSigmoid FP32 simple mapping kernel call (define in simple_mapping.h)
SIMPLE_MAPPING_KERNEL(
    mppa_kann_clus_kernel_hsigmoid_tf32_tf32,  // kernel_name
    1,                                         // nb sources
    float32_t,                                 // src type
    float32_t,                                 // dst type
    hsigmoid_tf32_tf32,                        // compute fcnt
    ARGS_F32                                   // args struct
)

// HardSigmoid FP16 KERNEL =========================================================//
// HardSigmoid FP16 callback function
static inline float16_t FUNC(
  const float16_t **src, ARGS_F16 *args)
{
    float16_t x = *src[0] * args->alpha + args->beta;
    return MAX(0, MIN(1, x));
}

// HardSigmoid FP16 simple mapping kernel call (define in simple_mapping.h)
SIMPLE_MAPPING_KERNEL(
    mppa_kann_clus_kernel_hsigmoid_tf16_tf16,  // kernel_name
    1,                                         // nb sources
    float16_t,                                 // src type
    float16_t,                                 // dst type
    FUNC,                                      // compute fcnt
    ARGS_F16                                   // args struct
)
// Vectorized HardSigmoid FP16 callback function
static inline VECTOR_DTYPE VECTOR_FUNC(
  const VECTOR_DTYPE *srcs, ARGS_F16 *args)
{
    float16x16_t y = {
      MAX(0, MIN(1, srcs[0][0]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][1]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][2]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][3]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][4]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][5]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][6]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][7]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][8]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][9]  * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][10] * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][11] * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][12] * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][13] * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][14] * args->alpha + args->beta)),
      MAX(0, MIN(1, srcs[0][15] * args->alpha + args->beta)),
    };
    return y;
}

//============================================================================//
// SiLU Optimized FP16 callback function for SIMD execution
//============================================================================//
// Vectorized Mish FP16 simple mapping kernel call
// (define in simple_mapping.h)
VEC_SIMPLE_MAPPING_KERNEL(
    mppa_kann_clus_kernel_hsigmoid_x16_tf16_tf16,  // kernel_name
    1,                                             // nb sources
    float16_t,                                     // src type
    float16_t,                                     // dst type
    VECTOR_SIZE,                                   // vector size
    VECTOR_DTYPE,                                  // vector type
    VECTOR_FUNC,                                   // vect compute fcnt
    FUNC,                                          // compute fcnt
    ARGS_F16                                       // args struct
)
//============================================================================//

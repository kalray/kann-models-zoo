#include "kernel_common.h"


// SELU FP16 KERNEL ==========================================================//
// SeLU FP16 args struct
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_selu_tf16_tf16_args {
    float16_t alpha;
    float16_t scale;
    SIMPLE_MAPPING_SUBARGS(1) subargs;                       // single source
} args_selu_tf16_tf16_t;

// SeLU FP16 callback function
static inline float16_t selu_tf16_tf16(
    const float16_t **src,
    args_selu_tf16_tf16_t *args)
{
    return args->scale * (*src[0] > 0 ?
        *src[0] : args->alpha * expm1f(*src[0]) - 1);
}

SIMPLE_MAPPING_KERNEL(mppa_kann_clus_kernel_selu_tf16_tf16,  // kernel_name
                      1,                                     // nb sources
                      float16_t,                             // src type
                      float16_t,                             // dst type
                      selu_tf16_tf16,                        // compute fcnt
                      args_selu_tf16_tf16_t                  // args struct
                     )

// SELU FP32 KERNEL ==========================================================//
// SeLU FP32 args struct
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_selu_tf32_tf32_args {
    float32_t alpha;
    float32_t scale;
    SIMPLE_MAPPING_SUBARGS(1) subargs;  // single source
} args_selu_tf32_tf32_t;

// SeLU FP32 callback function
static inline float32_t selu_tf32_tf32(const float32_t **src,
                                       args_selu_tf32_tf32_t *args)
{
    return args->scale * (*src[0] > 0 ?
        *src[0] : args->alpha * expm1f(*src[0]));
}

SIMPLE_MAPPING_KERNEL(mppa_kann_clus_kernel_selu_tf32_tf32,  // kernel_name
                      1,                                     // nb sources
                      float32_t,                             // src type
                      float32_t,                             // dst type
                      selu_tf32_tf32,                        // compute fcnt
                      args_selu_tf32_tf32_t                  // args struct
                      )

//============================================================================//

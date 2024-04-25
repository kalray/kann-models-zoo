#include <fastmath.h>
#include "mppa_generic.h"
#include "simple_mapping.h"
#include "kernel_common.h"

#define KANN_PRESISTENT_DATA 1

#define VECTOR_SIZE 8
#define DTYPE_VEC   float16x8_t
#define FUNC        silu_fast_tf16_tf16
#define SIMD_FUNC   silu_fastx8_tf16_tf16


// SiLU FP32 KERNEL =========================================================//
// SiLU FP32 args struct
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_silu_tf32_tf32_args {
    float32_t alpha;
    SIMPLE_MAPPING_SUBARGS(1) subargs;  // single source
} args_silu_tf32_tf32_t;

// SiLU FP32 callback function
static inline float32_t silu_tf32_tf32(
  const float32_t **src, args_silu_tf32_tf32_t *args)
{
    return *src[0] / (1 + expf(-1.0 * args->alpha * *src[0]));
}

// SiLU FP32 simple mapping kernel call (define in simple_mapping.h)
SIMPLE_MAPPING_KERNEL(
    mppa_kann_clus_kernel_silu_tf32_tf32,  // kernel_name
    1,                                     // nb sources
    float32_t,                             // src type
    float32_t,                             // dst type
    silu_tf32_tf32,                        // compute fcnt
    args_silu_tf32_tf32_t                  // args struct
)

// SiLU FP16 KERNEL =========================================================//
// SiLU FP16 args struct
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_silu_tf16_tf16_args {
    float16_t alpha;
    SIMPLE_MAPPING_SUBARGS(1) subargs;     // single source
} args_silu_tf16_tf16_t;

// SiLU FP16 callback function
static inline float16_t silu_tf16_tf16(
  const float16_t **src, args_silu_tf16_tf16_t *args)
{
    return *src[0] / (1 + expf(-1.0 * args->alpha * *src[0]));
}

// SiLU FP16 simple mapping kernel call (define in simple_mapping.h)
SIMPLE_MAPPING_KERNEL(
    mppa_kann_clus_kernel_silu_tf16_tf16,  // kernel_name
    1,                                     // nb sources
    float16_t,                             // src type
    float16_t,                             // dst type
    silu_tf16_tf16,                        // compute fcnt
    args_silu_tf16_tf16_t                  // args struct
)


//===========================================================================//
// SiLU Optimized FP16 callback function with taylor's series
//===========================================================================//

static inline float16_t silu_fast_tf16_tf16(
  const float16_t **src, args_silu_tf16_tf16_t *args)
{
    // Implementation of y = x * sigmoid(x)
    //   where sigmoid(x) => 1 / (1 + e(x))   for x <= 0
    //                       1 / (1 + 1/e(x)) for x > 0
    //   and e(x) => 1.0 + |x| + 0.555 * x^2 + 0.143 * x^4

    float a = args->alpha * fast_sigmoid(*src[0]);
    return *src[0] * a;
}

// SiLU FP16 simple mapping kernel call (define in simple_mapping.h)
SIMPLE_MAPPING_KERNEL(
    mppa_kann_clus_kernel_silu_fast_tf16_tf16, // kernel_name
    1,                                     // nb sources
    float16_t,                             // src type
    float16_t,                             // dst type
    silu_fast_tf16_tf16,                   // compute fcnt
    args_silu_tf16_tf16_t                  // args struct
)


//============================================================================//
// SiLU Optimized FP16 callback function with taylor's series SIMD execution
//============================================================================//
static inline float16x8_t silu_fastx8_tf16_tf16(
  const float16x8_t src)
{
    // Implementation of y = x * sigmoid(x)
    //   where sigmoid(x) => 1 / (1 + e(x))   for x <= 0
    //                       1 / (1 + 1/e(x)) for x > 0
    //   and e(x) => 1.0 + |x| + 0.555 * x^2 + 0.143 * x^4
    // SIMD Vectors is used here using 8 elements

    // e(x) => 1.0 + |x| + 0.555 * x^2 + 0.143 * x^4
    float16x8_t a = { 1.000f,1.000f,1.000f,1.000f,1.000f,1.000f,1.000f,1.000f };
    float16x8_t b = { 0.555f,0.555f,0.555f,0.555f,0.555f,0.555f,0.555f,0.555f };
    float16x8_t c = { 0.143f,0.143f,0.143f,0.143f,0.143f,0.143f,0.143f,0.143f };
    float16x8_t x = {
        ABS(src[0]),
        ABS(src[1]),
        ABS(src[2]),
        ABS(src[3]),
        ABS(src[4]),
        ABS(src[5]),
        ABS(src[6]),
        ABS(src[7]),
    };
    float16x8_t x2 = x * x;
    float16x8_t x4 = x2 * x2;
    float16x8_t e = a + x + b * x2 + c * x4;
    float16x8_t y = {
        1.0f / (1.0f + (src[0] > 0 ? 1.0f / e[0] : e[0])),
        1.0f / (1.0f + (src[1] > 0 ? 1.0f / e[1] : e[1])),
        1.0f / (1.0f + (src[2] > 0 ? 1.0f / e[2] : e[2])),
        1.0f / (1.0f + (src[3] > 0 ? 1.0f / e[3] : e[3])),
        1.0f / (1.0f + (src[4] > 0 ? 1.0f / e[4] : e[4])),
        1.0f / (1.0f + (src[5] > 0 ? 1.0f / e[5] : e[5])),
        1.0f / (1.0f + (src[6] > 0 ? 1.0f / e[6] : e[6])),
        1.0f / (1.0f + (src[7] > 0 ? 1.0f / e[7] : e[7]))
    };
    return src * y;

    /* Alternatively another algorithm who's work well and faster */
    // float16x8_t a = { 1.000f,1.000f,1.000f,1.000f,1.000f,1.000f,1.000f,1.000f };
    // float16x8_t x = {
    //   fast_expf16(-src[0]),
    //   fast_expf16(-src[1]),
    //   fast_expf16(-src[2]),
    //   fast_expf16(-src[3]),
    //   fast_expf16(-src[4]),
    //   fast_expf16(-src[5]),
    //   fast_expf16(-src[6]),
    //   fast_expf16(-src[7]),
    // };
    // return src / (a + x);

}

//============================================================================//
// SiLU Optimized FP16 callback function for SIMD execution
//============================================================================//
void mppa_kann_clus_kernel_silu_x8_tf16_tf16 (
  args_silu_tf16_tf16_t *restrict args,
#if KANN_PRESISTENT_DATA > 0
  void *kann_data0, void *kann_data1
#else
  void *kann_data
#endif
  )
{
    uint32_t d;
#if KANN_PRESISTENT_DATA == 0
    char *data = (char *)kann_data;
#endif
    uint32_t indexes[args->subargs.nb_sdim + 1];
    /* compute the product of the dimensions counts
     * also temporary store the partials products in indexes */
    uint32_t nb_values = args->subargs.cdim;
    for (d = 0; d < args->subargs.nb_sdim; d++) {
        indexes[d] = nb_values;
        nb_values *= args->subargs.sdim[d].count;
    }
    /* get the range [beg, end[ of points that will be computed by this
     * PE, and get the corresponding indexes as a (nb_sdim + 1)
     * dimensional
     * array */
    uint32_t peid = KANN_GET_PE_ID();
    uint32_t beg = (nb_values * peid) / KANN_NB_CORES;
    uint32_t end = (nb_values * (peid + 1)) / KANN_NB_CORES;
    uint32_t nb_iter = end - beg;
    /* use the partial dim products to compute the current indexes */
    div_t quot_rem = {.rem = beg};
    for (d = args->subargs.nb_sdim; d > 0; d--) {
        quot_rem = div(quot_rem.rem, indexes[d - 1]);
        indexes[d] = quot_rem.quot;
    }
    indexes[0] = quot_rem.rem;

    // define destination pointer
#if KANN_PRESISTENT_DATA > 0
    // new feature : Weights in smem
     float16_t *dst_ptr = (float16_t *)get_base_address_of_tensor(
         args->subargs.dst, kann_data0, kann_data1);
     dst_ptr += indexes[0];
#else
    float16_t *dst_ptr = (float16_t *)&data[args->subargs.dst];
    dst_ptr += indexes[0];
#endif

    // define source pointer
    const float16_t *src_ptrs[1];
#if KANN_PRESISTENT_DATA > 0
    // new feature : Weights in smem
     src_ptrs[0] = (const float16_t *)get_base_address_of_tensor(
         args->subargs.src[0], kann_data0, kann_data1);
#else
    src_ptrs[0] = (const float16_t *)&data[args->subargs.src[0]]; //to remove
#endif
    src_ptrs[0] += indexes[0];
    for (d = 0; d < args->subargs.nb_sdim; d++) {
        dst_ptr += (int64_t)indexes[d + 1] * args->subargs.sdim[d].stride_dst;
        src_ptrs[0] += (int64_t)indexes[d + 1] * args->subargs.sdim[d].stride_src[0];
    }

    while (nb_iter > 0) {
        // define index local variable
        uint32_t index0_iter;
        // Compute SIMD_FUNC x VECTOR_SIZE increment steps
        index0_iter = MIN(
          (nb_iter / VECTOR_SIZE),
          ((args->subargs.cdim - indexes[0]) / VECTOR_SIZE)
        );
        while (index0_iter-- > 0) {
            DTYPE_VEC src = *(const DTYPE_VEC *)src_ptrs[0];
            *(DTYPE_VEC *)dst_ptr = SIMD_FUNC(src);
            indexes[0]  += VECTOR_SIZE;
            dst_ptr     += VECTOR_SIZE;
            src_ptrs[0] += VECTOR_SIZE;
            nb_iter     -= VECTOR_SIZE;
        }
        // Compute x1 steps
        index0_iter = MIN(nb_iter, (args->subargs.cdim - indexes[0]));
        while (index0_iter-- > 0) {
            *dst_ptr = FUNC(src_ptrs, args);
            indexes[0]  += 1;
            nb_iter     -= 1;
            dst_ptr     += 1;
            src_ptrs[0] += 1;
        }
        if (__builtin_expect(indexes[0] == args->subargs.cdim, 0)) {
            indexes[0] = 0;
            dst_ptr -= args->subargs.cdim;
            src_ptrs[0] -= args->subargs.cdim;
            for (d = 0; d < args->subargs.nb_sdim; d++) {
                indexes[d + 1]++;
                dst_ptr += args->subargs.sdim[d].stride_dst;
                src_ptrs[0] += args->subargs.sdim[d].stride_src[0];
                if (indexes[d + 1] < args->subargs.sdim[d].count) {
                    break;
                }
                indexes[d + 1] = 0;
                dst_ptr -= args->subargs.sdim[d].stride_dst * args->subargs.sdim[d].count;
                src_ptrs[0] -= args->subargs.sdim[d].stride_src[0] * args->subargs.sdim[d].count;
            }
        }
    }
    KANN_SMEM_FENCE();
}
#include <fastmath.h>
#include "mppa_generic.h"
#include "simple_mapping.h"
#include "kernel_common.h"


#define KANN_PRESISTENT_DATA 1
#define VECTOR_SIZE 16
#define DTYPE_VEC   float16x16_t
#define SIMD_FUNC   hsigmoid_x16_tf16_tf16
#define FUNC        hsigmoid_tf16_tf16


// HardSigmoid FP32 KERNEL =========================================================//
// HardSigmoid FP32 args struct
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_hsigmoid_tf32_tf32_args {
    float32_t alpha;
    float32_t beta;
    SIMPLE_MAPPING_SUBARGS(1) subargs;  // single source
} args_hsigmoid_tf32_tf32_t;

// HardSigmoid FP32 callback function
static inline float32_t hsigmoid_tf32_tf32(
  const float32_t **src, args_hsigmoid_tf32_tf32_t *args)
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
    args_hsigmoid_tf32_tf32_t                  // args struct
)

// HardSigmoid FP16 KERNEL =========================================================//
// HardSigmoid FP16 args struct
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_hsigmoid_tf16_tf16_args {
    float16_t alpha;
    float16_t beta;
    SIMPLE_MAPPING_SUBARGS(1) subargs;     // single source
} args_hsigmoid_tf16_tf16_t;

// HardSigmoid FP16 callback function
static inline float16_t hsigmoid_tf16_tf16(
  const float16_t **src, args_hsigmoid_tf16_tf16_t *args)
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
    hsigmoid_tf16_tf16,                        // compute fcnt
    args_hsigmoid_tf16_tf16_t                  // args struct
)

static inline float16x16_t hsigmoid_x16_tf16_tf16(
  const float16x16_t src, args_hsigmoid_tf16_tf16_t *args)
{
    float16x16_t y = {
      MAX(0, MIN(1, src[0]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[1]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[2]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[3]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[4]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[5]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[6]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[7]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[8]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[9]  * args->alpha + args->beta)),
      MAX(0, MIN(1, src[10] * args->alpha + args->beta)),
      MAX(0, MIN(1, src[11] * args->alpha + args->beta)),
      MAX(0, MIN(1, src[12] * args->alpha + args->beta)),
      MAX(0, MIN(1, src[13] * args->alpha + args->beta)),
      MAX(0, MIN(1, src[14] * args->alpha + args->beta)),
      MAX(0, MIN(1, src[15] * args->alpha + args->beta)),
    };
    return y;
}

//============================================================================//
// SiLU Optimized FP16 callback function for SIMD execution
//============================================================================//
void mppa_kann_clus_kernel_hsigmoid_x16_tf16_tf16 (
  args_hsigmoid_tf16_tf16_t *restrict args,
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
            *(DTYPE_VEC *)dst_ptr = SIMD_FUNC(src, args);
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
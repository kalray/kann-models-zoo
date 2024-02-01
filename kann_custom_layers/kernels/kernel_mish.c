#include <fastmath.h>
#include "mppa_generic.h"
#include "simple_mapping.h"
#include "kernel_common.h"

#define THRESHOLD_SOFTPLUS     4
#define THRESHOLD_TANHSOFTPLUS 5
#define THRESHOLD_TANH         4

#define APPROXIMATE 1
#define KANN_PRESISTENT_DATA 1

#define _FUNC mish_tf16_tf16
#define VECTOR_SIZE 8
#if VECTOR_SIZE == 8
  #define DTYPE_VEC float16x8_t
  #define _SIMD_FUNC mish_fastx8_tf16_tf16
#endif


// MISH FP32 args struct                =====================================//
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_mish_tf32_tf32 {
    float32_t beta;
    SIMPLE_MAPPING_SUBARGS(1) subargs;  // single source
} args_mish_tf32_tf32_t;

// MISH FP32 callback function           ====================================//
static inline float32_t mish_tf32_tf32(
    const float32_t **src,
    args_mish_tf32_tf32_t *args)
{
    // softplus(x) computation
    float32_t y_softplus_fp32 = logf(1 + expf(args->beta * *src[0]));
    y_softplus_fp32 = y_softplus_fp32 / args->beta;

    // x * tanh computation
    return *src[0] * tanhf(y_softplus_fp32);
}

SIMPLE_MAPPING_KERNEL(mppa_kann_clus_kernel_mish_tf32_tf32, // kernel_name
                      1,                                    // nb sources
                      float32_t,                            // src type
                      float32_t,                            // dst type
                      mish_tf32_tf32,                       // compute fcnt
                      args_mish_tf32_tf32_t                 // args struct
                     )


// CUSTOM LOGF, EXPF, TANHF             =====================================//

// EXPF16 --
inline double _expf(double a) {

    /* method and constants adapted for float16 (aka _Float16):
       in references:
       http://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c
       https://hackage.haskell.org/package/approximate-0.2.2.1/src/cbits/fast.c
       https://nic.schraudolph.org/pubs/Schraudolph99.pdf

       FLOAT16 : b |  |||||  |||||||||||
                  15 14  10  9         0

       sign : 1 bit
       exp  : 5 bits
       precision: 10 bits

       coef = 1 << 10 / log(2) = 1477,
       bias << 10 = (01111)b << 10 = 15360
    */

    /* Function for double precision */
    union { double d; long long x; } u;
    u.x = (long long)(6497320848556798LL * a + 0x3fef127e83d16f12LL);
    return u.d;

    /* Function for float precision */
    // union { float d; int x; } u;
    // u.x = (int) (12102203 * a + 1065353217);
    // return u.d;

    /* Function for float16 precision */
    // union { float16_t d; short x; } u;
    // u.x = (short)(1478 * a + 15360);
    // return u.d;
}

// LOGF32 --
inline float _logf(float a) {
    /* Function for double precision */
    // union { double d; long long x; } u = { a };
    // return (u.x - 4606921278410026770) * 1.539095918623324e-16;

    /* Function for float precision */
    union { float d; int x; } u = { a };
    return (u.x - 1064866805) * 8.262958405176314e-8f;

    /* Function for float16 precision */
    // union { float16_t d; short x; } u = { a };
    // return (u.x - 15360) * 0.0006765899864682003;
}

// TANHF16 --
float _tanhf(float16_t x) {
    // float16_t a = 1 + _expf(-2 * x);
    // float16_t b = 2 / a;
    // return b - 1;
    float16_t num = _expf(x) - _expf(-x);
    float16_t den = _expf(x) + _expf(-x);
    return (num * __builtin_kvx_frecw(den, ".s"));
}

//===========================================================================//
// MISH FP16                            =====================================//

// MISH FP16 args struct                =====================================//
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_mish_tf16_tf16_args {
    float16_t beta;
    SIMPLE_MAPPING_SUBARGS(1) subargs;
} args_mish_tf16_tf16_t;

// MISH FP16 callback function          =====================================//
static inline float16_t mish_tf16_tf16(
    const float16_t **src,
    args_mish_tf16_tf16_t *args)
{
    float16_t mish16 = 0;
#if APPROXIMATE > 0
    // Compute y = softplus(x)
    if (*src[0] < -THRESHOLD_SOFTPLUS) {
        mish16 = 0.0;
    }
    else if (*src[0] > THRESHOLD_SOFTPLUS) {
        mish16 = *src[0];
    }
    else {
        mish16 = _logf(_expf(*src[0]) + 1.0f);
        // Compute out = x * tanh(y)
        if (fabs(mish16) < THRESHOLD_TANH) {
            mish16 = fmaf(*src[0], _tanhf(mish16), 0);
        }
        else {
            mish16 = mish16<0 ? -*src[0]:*src[0];
        }
    }
#else
    mish16 = fmaf(*src[0], tanhf(logf(expf(*src[0]) + 1)), 0);
#endif
    return (args->beta * mish16);
}

SIMPLE_MAPPING_KERNEL(mppa_kann_clus_kernel_mish_tf16_tf16,    // kernel_name
                      1,                                       // nb sources
                      float16_t,                               // src type
                      float16_t,                               // dst type
                      mish_tf16_tf16,                          // compute fcnt
                      args_mish_tf16_tf16_t                    // args struct
                     )

//===========================================================================//
// MISH SIMD FP16                       =====================================//

// MISH SIMD FP16 callback function          ================================//
static inline float16x8_t mish_fastx8_tf16_tf16(
    const float16x8_t src)
{
#if APPROXIMATE > 0
    float16x8_t mish16x8 =
    {
      src[0] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (src[0] > THRESHOLD_SOFTPLUS ? src[0] :
          (src[0] * _tanhf(_logf(_expf(src[0])+1.0f)))),
      src[1] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (src[1] > THRESHOLD_SOFTPLUS ? src[1] :
          (src[1] * _tanhf(_logf(_expf(src[1])+1.0f)))),
      src[2] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (src[2] > THRESHOLD_SOFTPLUS ? src[2] :
          (src[2] * _tanhf(_logf(_expf(src[2])+1.0f)))),
      src[3] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (src[3] > THRESHOLD_SOFTPLUS ? src[3] :
          (src[3] * _tanhf(_logf(_expf(src[3])+1.0f)))),
      src[4] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (src[4] > THRESHOLD_SOFTPLUS ? src[4] :
          (src[4] * _tanhf(_logf(_expf(src[4])+1.0f)))),
      src[5] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (src[5] > THRESHOLD_SOFTPLUS ? src[5] :
          (src[5] * _tanhf(_logf(_expf(src[5])+1.0f)))),
      src[6] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (src[6] > THRESHOLD_SOFTPLUS ? src[6] :
          (src[6] * _tanhf(_logf(_expf(src[6])+1.0f)))),
      src[7] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (src[7] > THRESHOLD_SOFTPLUS ? src[7] :
          (src[7] * _tanhf(_logf(_expf(src[7])+1.0f)))),
   };
#else
    float16x8_t mish16x8 =
    {
      fmaf(src[0], tanhf(logf(expf(src[0])+1.0f)), 0.0),
      fmaf(src[1], tanhf(logf(expf(src[1])+1.0f)), 0.0),
      fmaf(src[2], tanhf(logf(expf(src[2])+1.0f)), 0.0),
      fmaf(src[3], tanhf(logf(expf(src[3])+1.0f)), 0.0),
      fmaf(src[4], tanhf(logf(expf(src[4])+1.0f)), 0.0),
      fmaf(src[5], tanhf(logf(expf(src[5])+1.0f)), 0.0),
      fmaf(src[6], tanhf(logf(expf(src[6])+1.0f)), 0.0),
      fmaf(src[7], tanhf(logf(expf(src[7])+1.0f)), 0.0),
   };
#endif
    return mish16x8;
}

// MISH SIMD FP16  function          =======================================//
void mppa_kann_clus_kernel_mish_x8_tf16_tf16 (
  args_mish_tf16_tf16_t *restrict args,
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
        // Compute _SIMD_FUNC x VECTOR_SIZE increment steps
        index0_iter = MIN(
          (nb_iter / VECTOR_SIZE),
          ((args->subargs.cdim - indexes[0]) / VECTOR_SIZE)
        );
        while (index0_iter-- > 0) {
            DTYPE_VEC src = *(const DTYPE_VEC *)src_ptrs[0];
            *(DTYPE_VEC *)dst_ptr = _SIMD_FUNC(src);
            indexes[0]  += VECTOR_SIZE;
            dst_ptr     += VECTOR_SIZE;
            src_ptrs[0] += VECTOR_SIZE;
            nb_iter     -= VECTOR_SIZE;
        }
        // Compute x1 steps
        index0_iter = MIN(nb_iter, (args->subargs.cdim - indexes[0]));
        while (index0_iter-- > 0) {
            *dst_ptr = _FUNC(src_ptrs, args);
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
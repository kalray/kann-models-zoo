#include <fastmath.h>
#include "mppa_generic.h"
#include "simple_mapping.h"
#include "kernel_common.h"

#define THRESHOLD_SOFTPLUS     4
#define THRESHOLD_TANHSOFTPLUS 5
#define THRESHOLD_TANH         4

#define APPROXIMATE 1
#define KANN_PRESISTENT_DATA 1

#define VECTOR_SIZE  8
#define VECTOR_DTYPE float16x8_t
#define FUNC         custom_mish_tf16_tf16
#define VECTOR_FUNC  mish_fastx8_tf16_tf16
#define ARGS         args_mish_tf16_tf16_t


// MISH FP16                            =====================================//
// MISH FP16 args struct                =====================================//
typedef struct __attribute__((__packed__))
mppa_kann_clus_kernel_mish_tf16_tf16_args {
    float16_t beta;
    SIMPLE_MAPPING_SUBARGS(1) subargs;
} ARGS;

//===========================================================================//
// MISH FP16 callback function          =====================================//
//===========================================================================//
static inline float16_t custom_mish_tf16_tf16(
    const float16_t **src, ARGS *args)
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


//===========================================================================//
// MISH Vectorized FP16                 =====================================//
//===========================================================================//
static inline VECTOR_DTYPE mish_fastx8_tf16_tf16(
    const VECTOR_DTYPE *srcs, ARGS *args)
{
#if APPROXIMATE > 0
    VECTOR_DTYPE mish16x8 =
    {
      srcs[0][0] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (srcs[0][0] > THRESHOLD_SOFTPLUS ? srcs[0][0] :
          (srcs[0][0] * _tanhf(_logf(_expf(srcs[0][0])+1.0f)))),
      srcs[0][1] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (srcs[0][1] > THRESHOLD_SOFTPLUS ? srcs[0][1] :
          (srcs[0][1] * _tanhf(_logf(_expf(srcs[0][1])+1.0f)))),
      srcs[0][2] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (srcs[0][2] > THRESHOLD_SOFTPLUS ? srcs[0][2] :
          (srcs[0][2] * _tanhf(_logf(_expf(srcs[0][2])+1.0f)))),
      srcs[0][3] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (srcs[0][3] > THRESHOLD_SOFTPLUS ? srcs[0][3] :
          (srcs[0][3] * _tanhf(_logf(_expf(srcs[0][3])+1.0f)))),
      srcs[0][4] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (srcs[0][4] > THRESHOLD_SOFTPLUS ? srcs[0][4] :
          (srcs[0][4] * _tanhf(_logf(_expf(srcs[0][4])+1.0f)))),
      srcs[0][5] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (srcs[0][5] > THRESHOLD_SOFTPLUS ? srcs[0][5] :
          (srcs[0][5] * _tanhf(_logf(_expf(srcs[0][5])+1.0f)))),
      srcs[0][6] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (srcs[0][6] > THRESHOLD_SOFTPLUS ? srcs[0][6] :
          (srcs[0][6] * _tanhf(_logf(_expf(srcs[0][6])+1.0f)))),
      srcs[0][7] < -THRESHOLD_SOFTPLUS ? 0.0 :
        (srcs[0][7] > THRESHOLD_SOFTPLUS ? srcs[0][7] :
          (srcs[0][7] * _tanhf(_logf(_expf(srcs[0][7])+1.0f)))),
   };
#else
    VECTOR_DTYPE mish16x8 =
    {
      fmaf(srcs[0][0], tanhf(logf(expf(srcs[0][0])+1.0f)), 0.0),
      fmaf(srcs[0][1], tanhf(logf(expf(srcs[0][1])+1.0f)), 0.0),
      fmaf(srcs[0][2], tanhf(logf(expf(srcs[0][2])+1.0f)), 0.0),
      fmaf(srcs[0][3], tanhf(logf(expf(srcs[0][3])+1.0f)), 0.0),
      fmaf(srcs[0][4], tanhf(logf(expf(srcs[0][4])+1.0f)), 0.0),
      fmaf(srcs[0][5], tanhf(logf(expf(srcs[0][5])+1.0f)), 0.0),
      fmaf(srcs[0][6], tanhf(logf(expf(srcs[0][6])+1.0f)), 0.0),
      fmaf(srcs[0][7], tanhf(logf(expf(srcs[0][7])+1.0f)), 0.0),
   };
#endif
    return mish16x8;
}

//============================================================================//
// Vectorized Mish FP16 simple mapping kernel call
// (define in simple_mapping.h)
VEC_SIMPLE_MAPPING_KERNEL(
    mppa_kann_clus_kernel_mish_x8_tf16_tf16,  // kernel_name
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
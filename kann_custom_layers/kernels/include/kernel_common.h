#ifndef COMMON_H
#define COMMON_H

#include <fastmath.h>
#include "mppa_generic.h"
#include "simple_mapping.h"

#define ABS(a) (a<0?-a:a)
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)<(b)?(b):(a))


// FAST SIGMOID IMPLEMENTATION FUNCTION =====================================//
static inline float16_t fast_sigmoid(float16_t value)
{
    // Implementation of y = sigmoid(x)
    //   where sigmoid(x) => 1 / (1 + e(x))   for x <= 0
    //                       1 / (1 + 1/e(x)) for x > 0
    //   and e(x) => 1.0 + |x| + 0.555 * x^2 + 0.143 * x^4

    float x = ABS(value);
    float x2 = x * x;
    float x4 = x2 * x2;
    float e = 1.0f + x + x2 * 0.555f + x4 * 0.143f;  // 0.2% error
    return 1.0f / (1.0f + (value > 0 ? 1.0f / e : e));
}

// FAST LOG IMPLEMENTATION FUNCTION ========================================//
inline double fast_logf16(double a) {
    union { double d; long long x; } u = { a };
    return (u.x - 4606921278410026770) * 1.539095918623324e-16;
}

// FAST EXP IMPLEMENTATION FUNCTION ========================================//
inline double fast_expf16(double a) {
    // Function for float32 precision
    union { double d; long long x; } u;
    u.x = (long long)(6497320848556798LL * a + 0x3fef127e83d16f12LL);
    return u.d;

    // Function for float32 precision
    // union { float d; int x; } u;
    // u.x = (int) (12102203 * a + 1065353217);
    // return u.d;

    // Function for float16 precision
    // union { float16_t d; short x; } u;
    // u.x = (short)(1478 * a + 15360);
    // return u.d;
}

// FAST TANH IMPLEMENTATION FUNCTION =======================================//
inline float16_t fast_tanhf16(float16_t a) {
    return (2 / (1 + fast_expf16(-2 * a)) - 1);
}

#endif /*COMMON_H*/
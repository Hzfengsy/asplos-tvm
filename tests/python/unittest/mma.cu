#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(128) main_kernel0(half* __restrict__ X, half* __restrict__ Y, half* __restrict__ C) {
  extern __shared__ uchar buf_dyn_shmem[];
  uint1 C_m16n8k8_matrixC[64];
  uint4 X_shared_dyn_local[4];
  uint4 Y_shared_dyn_local[4];
  half X_shared_dyn_m16n8k8_matrixA[32];
  half Y_shared_dyn_m16n8k8_matrixB[32];
  for (int i0_0_4_init = 0; i0_0_4_init < 4; ++i0_0_4_init) {
    for (int i1_0_4_init = 0; i1_0_4_init < 8; ++i1_0_4_init) {
      for (int i0_1_0 = 0; i0_1_0 < 2; ++i0_1_0) {
        C_m16n8k8_matrixC[(((i0_0_4_init * 16) + (i1_0_4_init * 2)) + i0_1_0)] = make_uint1(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
  }
  for (int ax0_ax1_fused_0_cache = 0; ax0_ax1_fused_0_cache < 4; ++ax0_ax1_fused_0_cache) {
    X_shared_dyn_local[ax0_ax1_fused_0_cache] = *(uint4*)(X + ((((((((int)blockIdx.x) >> 3) * 524288) + (ax0_ax1_fused_0_cache * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  for (int ax0_ax1_fused_0_cache1 = 0; ax0_ax1_fused_0_cache1 < 4; ++ax0_ax1_fused_0_cache1) {
    Y_shared_dyn_local[ax0_ax1_fused_0_cache1] = *(uint4*)(Y + ((((((ax0_ax1_fused_0_cache1 * 32768) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {
    *(uint4*)(((half*)buf_dyn_shmem) + ((((ax0_ax1_fused_0 * 1024) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 8192)) = X_shared_dyn_local[ax0_ax1_fused_0];
  }
  for (int ax0_ax1_fused_01 = 0; ax0_ax1_fused_01 < 4; ++ax0_ax1_fused_01) {
    *(uint4*)(((half*)buf_dyn_shmem) + (((ax0_ax1_fused_01 * 1024) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8))) = Y_shared_dyn_local[ax0_ax1_fused_01];
  }
  __syncthreads();
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[((((((int)threadIdx.y) & 1) * 2048) + (ax0_0 * 1024)) + 8192)])) + (((int)threadIdx.x) * 32)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_0 * 8)))[3])
      : "r"(addr)
    );
  }
  }
  for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 64) + (ax1_0 * 32))])) + (((((int)threadIdx.x) & 7) * 128) + ((((int)threadIdx.x) >> 3) * 8))))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[3])
      : "r"(addr)
    );
  }
  }
  for (int i2_0_0 = 0; i2_0_0 < 127; ++i2_0_0) {
    for (int ax0_ax1_fused_0_cache2 = 0; ax0_ax1_fused_0_cache2 < 4; ++ax0_ax1_fused_0_cache2) {
      X_shared_dyn_local[ax0_ax1_fused_0_cache2] = *(uint4*)(X + ((((((((((int)blockIdx.x) >> 3) * 524288) + (ax0_ax1_fused_0_cache2 * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (i2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
    }
    for (int ax0_ax1_fused_0_cache3 = 0; ax0_ax1_fused_0_cache3 < 4; ++ax0_ax1_fused_0_cache3) {
      Y_shared_dyn_local[ax0_ax1_fused_0_cache3] = *(uint4*)(Y + ((((((((i2_0_0 * 131072) + (ax0_ax1_fused_0_cache3 * 32768)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072));
    }
    for (int i2_0_1 = 0; i2_0_1 < 3; ++i2_0_1) {
      for (int ax0_01 = 0; ax0_01 < 2; ++ax0_01) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[((((((i2_0_0 & 1) * 4096) + ((((int)threadIdx.y) & 1) * 2048)) + (ax0_01 * 1024)) + (i2_0_1 * 8)) + 8200)])) + (((int)threadIdx.x) * 32)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_01 * 8)))[0]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_01 * 8)))[1]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_01 * 8)))[2]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_01 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax1_01 = 0; ax1_01 < 2; ++ax1_01) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[((((((i2_0_0 & 1) * 4096) + (i2_0_1 * 1024)) + ((((int)threadIdx.y) >> 1) * 64)) + (ax1_01 * 32)) + 1024)])) + (((((int)threadIdx.x) & 7) * 128) + ((((int)threadIdx.x) >> 3) * 8))))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_01 * 8)))[0]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_01 * 8)))[1]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_01 * 8)))[2]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_01 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int i0_0_4 = 0; i0_0_4 < 4; ++i0_0_4) {
        for (int i1_0_4 = 0; i1_0_4 < 8; ++i1_0_4) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_4 * 16) + (i1_0_4 * 2))))[0]), "=r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_4 * 16) + (i1_0_4 * 2))))[1])
      : "r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (i0_0_4 * 4)))[0]), "r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (i0_0_4 * 4)))[1]), "r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (i1_0_4 * 2)))[0]), "r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_4 * 16) + (i1_0_4 * 2))))[0]), "r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_4 * 16) + (i1_0_4 * 2))))[1]));
  }
        }
      }
    }
    for (int ax0_ax1_fused_02 = 0; ax0_ax1_fused_02 < 4; ++ax0_ax1_fused_02) {
      *(uint4*)(((half*)buf_dyn_shmem) + (((((((i2_0_0 + 1) & 1) * 4096) + (ax0_ax1_fused_02 * 1024)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 8192)) = X_shared_dyn_local[ax0_ax1_fused_02];
    }
    for (int ax0_ax1_fused_03 = 0; ax0_ax1_fused_03 < 4; ++ax0_ax1_fused_03) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((((i2_0_0 + 1) & 1) * 4096) + (ax0_ax1_fused_03 * 1024)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8))) = Y_shared_dyn_local[ax0_ax1_fused_03];
    }
    __syncthreads();
    for (int ax0_02 = 0; ax0_02 < 2; ++ax0_02) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[((((((i2_0_0 + 1) & 1) * 4096) + ((((int)threadIdx.y) & 1) * 2048)) + (ax0_02 * 1024)) + 8192)])) + (((int)threadIdx.x) * 32)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_02 * 8)))[0]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_02 * 8)))[1]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_02 * 8)))[2]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_02 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_02 = 0; ax1_02 < 2; ++ax1_02) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((i2_0_0 + 1) & 1) * 4096) + ((((int)threadIdx.y) >> 1) * 64)) + (ax1_02 * 32))])) + (((((int)threadIdx.x) & 7) * 128) + ((((int)threadIdx.x) >> 3) * 8))))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_02 * 8)))[0]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_02 * 8)))[1]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_02 * 8)))[2]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_02 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int i0_0_41 = 0; i0_0_41 < 4; ++i0_0_41) {
      for (int i1_0_41 = 0; i1_0_41 < 8; ++i1_0_41) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_41 * 16) + (i1_0_41 * 2))))[0]), "=r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_41 * 16) + (i1_0_41 * 2))))[1])
      : "r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (i0_0_41 * 4)))[0]), "r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (i0_0_41 * 4)))[1]), "r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (i1_0_41 * 2)))[0]), "r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_41 * 16) + (i1_0_41 * 2))))[0]), "r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_41 * 16) + (i1_0_41 * 2))))[1]));
  }
      }
    }
  }
  for (int i2_0_11 = 0; i2_0_11 < 3; ++i2_0_11) {
    for (int ax0_03 = 0; ax0_03 < 2; ++ax0_03) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.y) & 1) * 2048) + (ax0_03 * 1024)) + (i2_0_11 * 8)) + 12296)])) + (((int)threadIdx.x) * 32)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_03 * 8)))[0]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_03 * 8)))[1]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_03 * 8)))[2]), "=r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (ax0_03 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_03 = 0; ax1_03 < 2; ++ax1_03) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[((((i2_0_11 * 1024) + ((((int)threadIdx.y) >> 1) * 64)) + (ax1_03 * 32)) + 5120)])) + (((((int)threadIdx.x) & 7) * 128) + ((((int)threadIdx.x) >> 3) * 8))))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_03 * 8)))[0]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_03 * 8)))[1]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_03 * 8)))[2]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_03 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int i0_0_42 = 0; i0_0_42 < 4; ++i0_0_42) {
      for (int i1_0_42 = 0; i1_0_42 < 8; ++i1_0_42) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_42 * 16) + (i1_0_42 * 2))))[0]), "=r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_42 * 16) + (i1_0_42 * 2))))[1])
      : "r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (i0_0_42 * 4)))[0]), "r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (i0_0_42 * 4)))[1]), "r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (i1_0_42 * 2)))[0]), "r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_42 * 16) + (i1_0_42 * 2))))[0]), "r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_42 * 16) + (i1_0_42 * 2))))[1]));
  }
      }
    }
  }
  for (int i0_0_43 = 0; i0_0_43 < 4; ++i0_0_43) {
    for (int i1_0_43 = 0; i1_0_43 < 8; ++i1_0_43) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
      :  "=r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_43 * 16) + (i1_0_43 * 2))))[0]), "=r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_43 * 16) + (i1_0_43 * 2))))[1])
      : "r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (i0_0_43 * 4)))[0]), "r"(((unsigned *)(X_shared_dyn_m16n8k8_matrixA + (i0_0_43 * 4)))[1]), "r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (i1_0_43 * 2)))[0]), "r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_43 * 16) + (i1_0_43 * 2))))[0]), "r"(((unsigned *)(C_m16n8k8_matrixC + ((i0_0_43 * 16) + (i1_0_43 * 2))))[1]));
  }
    }
  }
  for (int ax0_04 = 0; ax0_04 < 8; ++ax0_04) {
    __syncthreads();
    for (int ax1_04 = 0; ax1_04 < 8; ++ax1_04) {
      *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 512) + (ax1_04 * 64)) + (((int)threadIdx.x) * 2)) + 8192)) = C_m16n8k8_matrixC[((((ax0_04 >> 1) * 16) + (ax1_04 * 2)) + (ax0_04 & 1))];
    }
    __syncthreads();
    for (int i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 = 0; i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 < 16; ++i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0) {
      C[((((((((((((((int)blockIdx.x) >> 3) * 524288) + (((i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 & 7) >> 2) * 262144)) + (ax0_04 * 32768)) + ((((int)threadIdx.y) & 1) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 >> 3) * 64)) + ((i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 & 3) * 16)) + ((((int)threadIdx.y) >> 1) * 8)) + (((int)threadIdx.x) & 7))] = ((half*)buf_dyn_shmem)[((((i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 8192)];
    }
  }
}



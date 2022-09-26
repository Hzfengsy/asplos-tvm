extern "C" __global__ void __launch_bounds__(128) main_kernel0(half* __restrict__ X, half* __restrict__ Y, half* __restrict__ C) {
  extern __shared__ uchar buf_dyn_shmem[];
  uint1 C_m16n8k8_matrixC[64];
  uint4 X_shared_dyn_local[4];
  uint4 Y_shared_dyn_local[4];
  half X_shared_dyn_m16n8k8_matrixA[16];
  half Y_shared_dyn_m16n8k8_matrixB[16];
  for (int i0_0_4_init = 0; i0_0_4_init < 4; ++i0_0_4_init) {
    for (int i1_0_4_init = 0; i1_0_4_init < 8; ++i1_0_4_init) {
      for (int i0_1_0 = 0; i0_1_0 < 2; ++i0_1_0) {
        C_m16n8k8_matrixC[(((i0_0_4_init * 16) + (i1_0_4_init * 2)) + i0_1_0)] = make_uint1(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
  }
  for (int i2_0_0 = 0; i2_0_0 < 128; ++i2_0_0) {
    for (int ax0_ax1_fused_0_cache = 0; ax0_ax1_fused_0_cache < 4; ++ax0_ax1_fused_0_cache) {
      X_shared_dyn_local[ax0_ax1_fused_0_cache] = *(uint4*)(X + (((((((((int)blockIdx.x) >> 3) * 524288) + (ax0_ax1_fused_0_cache * 131072)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (i2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {
      *(uint4*)(((half*)buf_dyn_shmem) + ((((ax0_ax1_fused_0 * 1024) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8)) + 4096)) = X_shared_dyn_local[ax0_ax1_fused_0];
    }
    for (int ax0_ax1_fused_0_cache1 = 0; ax0_ax1_fused_0_cache1 < 4; ++ax0_ax1_fused_0_cache1) {
      Y_shared_dyn_local[ax0_ax1_fused_0_cache1] = *(uint4*)(Y + (((((((i2_0_0 * 131072) + (ax0_ax1_fused_0_cache1 * 32768)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    for (int ax0_ax1_fused_01 = 0; ax0_ax1_fused_01 < 4; ++ax0_ax1_fused_01) {
      *(uint4*)(((half*)buf_dyn_shmem) + (((ax0_ax1_fused_01 * 1024) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8))) = Y_shared_dyn_local[ax0_ax1_fused_01];
    }
    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1) {
      for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.y) & 1) * 2048) + (ax0_0 * 1024)) + (i2_0_1 * 8)) + 4096)])) + (((int)threadIdx.x) * 32)))
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
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((i2_0_1 * 1024) + ((((int)threadIdx.y) >> 1) * 64)) + (ax1_0 * 32))])) + (((((int)threadIdx.x) & 7) * 128) + ((((int)threadIdx.x) >> 3) * 8))))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(Y_shared_dyn_m16n8k8_matrixB + (ax1_0 * 8)))[3])
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
  }
  for (int ax0_01 = 0; ax0_01 < 8; ++ax0_01) {
    __syncthreads();
    for (int ax1_01 = 0; ax1_01 < 8; ++ax1_01) {
      *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 512) + (ax1_01 * 64)) + (((int)threadIdx.x) * 2)) + 4096)) = C_m16n8k8_matrixC[((((ax0_01 >> 1) * 16) + (ax1_01 * 2)) + (ax0_01 & 1))];
    }
    __syncthreads();
    for (int ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 = 0; ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 < 16; ++ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0) {
      C[((((((((((((((int)blockIdx.x) >> 3) * 524288) + (((ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 & 7) >> 2) * 262144)) + (ax0_01 * 32768)) + ((((int)threadIdx.y) & 1) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 >> 3) * 64)) + ((ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 & 3) * 16)) + ((((int)threadIdx.y) >> 1) * 8)) + (((int)threadIdx.x) & 7))] = ((half*)buf_dyn_shmem)[((((ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 4096)];
    }
  }
}

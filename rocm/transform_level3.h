#pragma once

#include "util.h"
#include "mxm_level3.h"

/**
 * Transform wrapper for Level-3 (B in LDS + register accumulation).
 * Each K value gets its own kernel binary via template<T, int K>,
 * isolating register pressure to acc[K] for that specific K.
 */

template <typename T, int K>
__device__ void transform_level3_k(
    const T* t,
    const T* c,
    T*& result,
    T* workspace)
{
  constexpr int ndim   = 3;
  constexpr int K2NDIM = K * K * K;

  T *t0 = workspace, *t1 = result;
  auto tmp = t0; t0 = t1; t1 = tmp;

  /* B is already in LDS — mTxmq_level3_k loads it each call */
  mra::mTxmq_level3_k<T, K>(t0, t, c);
  for (int n = 1; n < ndim; ++n) {
    mra::mTxmq_level3_k<T, K>(t1, t0, c);
    auto tmp2 = t0; t0 = t1; t1 = tmp2;
  }
}

/* One kernel binary per K — register pressure is proportional to K, not max(K). */
template <typename T, int K>
LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK, 1)
__global__ void transform_kernel_level3_k(int nfuncs,
                                           const T* A, const T* B, T* C, T* workspace) {
  constexpr int K2NDIM = K * K * K;
  T* w = workspace + blockIdx.x * K2NDIM;
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    const T* a = A + i * K2NDIM;
    T* c       = C + i * K2NDIM;
    /* result pointer starts at c; workspace is w */
    T* result  = c;
    transform_level3_k<T, K>(a, B, result, w);
  }
}

template <typename T>
inline int transform_level3_shmem_size(int K) {
  return mra::mTxmq_level3_shmem_size<T>(K);
}

template <typename T>
inline void submit_transform_level3_bench(int nfuncs, int nblocks, int K,
                                          const T* A, const T* B, T* C, T* workspace,
                                          Stream stream)
{
  Dim3 thread_dims = mra::mTxmq_level3_blockdim<T>(K);
  int  smem_size   = mra::mTxmq_level3_shmem_size<T>(K);

#define DISPATCH_L3(Kval) \
  case Kval: \
    CONFIGURE_KERNEL((transform_kernel_level3_k<T, Kval>), smem_size); \
    CALL_KERNEL((transform_kernel_level3_k<T, Kval>), std::min(nfuncs, nblocks), \
                thread_dims, smem_size, stream, \
                (nfuncs, A, B, C, workspace)); \
    break;

  switch (K) {
    DISPATCH_L3( 6)
    DISPATCH_L3( 8)
    DISPATCH_L3(10)
    DISPATCH_L3(12)
    DISPATCH_L3(16)
    DISPATCH_L3(20)
    DISPATCH_L3(32)
    default:
      printf("submit_transform_level3_bench: unsupported K=%d\n", K);
  }
#undef DISPATCH_L3
}

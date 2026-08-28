#pragma once

#include "util.h"
#include "mxm_level4.h"

/**
 * Transform wrapper for Level-4 (MFMA on AMD GFX90A/GFX940, falls back to
 * Level-3 on other targets or for K values without native MFMA support).
 *
 * Each K value gets its own kernel binary via template<T, int K>,
 * isolating register pressure to the specific K being compiled.
 * Block dimension is 64 threads (one wavefront) for MFMA path.
 */

template <typename T, int K>
__device__ void transform_level4_k(
    const T* t,
    const T* c,
    T*& result,
    T* workspace)
{
  constexpr int ndim = 3;

  T *t0 = workspace, *t1 = result;
  auto tmp = t0; t0 = t1; t1 = tmp;

  mra::mTxmq_level4_k<T, K>(t0, t, c);
  for (int n = 1; n < ndim; ++n) {
    mra::mTxmq_level4_k<T, K>(t1, t0, c);
    auto tmp2 = t0; t0 = t1; t1 = tmp2;
  }
}

/* One kernel binary per K. */
template <typename T, int K>
LAUNCH_BOUNDS(64, 1)
__global__ void transform_kernel_level4_k(int nfuncs,
                                           const T* A, const T* B, T* C, T* workspace) {
  constexpr int K2NDIM = K * K * K;
  T* w = workspace + blockIdx.x * K2NDIM;
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    const T* a = A + i * K2NDIM;
    T* c       = C + i * K2NDIM;
    T* result  = c;
    transform_level4_k<T, K>(a, B, result, w);
  }
}

template <typename T>
inline int transform_level4_shmem_size(int K) {
  return mra::mTxmq_level4_shmem_size<T>(K);
}

template <typename T>
inline void submit_transform_level4_bench(int nfuncs, int nblocks, int K,
                                          const T* A, const T* B, T* C, T* workspace,
                                          Stream stream)
{
  Dim3 thread_dims = mra::mTxmq_level4_blockdim<T>(K);
  int  smem_size   = mra::mTxmq_level4_shmem_size<T>(K);

#define DISPATCH_L4(Kval) \
  case Kval: \
    CONFIGURE_KERNEL((transform_kernel_level4_k<T, Kval>), smem_size); \
    CALL_KERNEL((transform_kernel_level4_k<T, Kval>), std::min(nfuncs, nblocks), \
                thread_dims, smem_size, stream, \
                (nfuncs, A, B, C, workspace)); \
    break;

  switch (K) {
    DISPATCH_L4( 6)
    DISPATCH_L4( 8)
    DISPATCH_L4(10)
    DISPATCH_L4(12)
    DISPATCH_L4(16)
    DISPATCH_L4(20)
    DISPATCH_L4(32)
    default:
      printf("submit_transform_level4_bench: unsupported K=%d\n", K);
  }
#undef DISPATCH_L4
}

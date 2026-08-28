#pragma once

#include "util.h"
#include "mxm_wmma.h"

/**
 * Transform wrapper for Level-6 (nvcuda::wmma), the CUDA counterpart of the
 * rocWMMA level.
 *
 * The three GEMM passes ping-pong between the per-block workspace and C, the
 * same structure levels 2 and 3 use.  The rocWMMA source keeps the whole
 * tensor resident in LDS between passes; that is not reproduced here because
 * an 8x8x4 fragment covers K < 16 directly, so the LDS-resident small-K
 * special case it needs (transform_klt16) has no counterpart.
 */

template <typename T, int K>
__device__ void transform_wmma_k(
    const T* t,
    const T* c,
    T*& result,
    T* workspace)
{
  constexpr int ndim = 3;

  T *t0 = workspace, *t1 = result;
  auto tmp = t0; t0 = t1; t1 = tmp;

  mra::mTxmq_wmma_k<T, K>(t0, t, c);
  for (int n = 1; n < ndim; ++n) {
    mra::mTxmq_wmma_k<T, K>(t1, t0, c);
    auto tmp2 = t0; t0 = t1; t1 = tmp2;
  }
}

/* One kernel binary per K.  The launch bound is the block ceiling used by the
 * widest K (32 warps); narrower K launch fewer threads. */
template <typename T, int K>
LAUNCH_BOUNDS(mra::detail::WMMA_MAX_WARPS * MRA_WARP_SIZE, 1)
__global__ void transform_kernel_wmma_k(int nfuncs,
                                        const T* A, const T* B, T* C, T* workspace) {
  constexpr int K2NDIM = K * K * K;
  T* w = workspace + blockIdx.x * K2NDIM;
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    const T* a = A + i * K2NDIM;
    T* c       = C + i * K2NDIM;
    T* result  = c;
    transform_wmma_k<T, K>(a, B, result, w);
  }
}

template <typename T>
inline int transform_wmma_shmem_size(int K) {
  return mra::mTxmq_wmma_shmem_size<T>(K);
}

template <typename T>
inline Dim3 transform_wmma_blockdim(int K) {
  return mra::mTxmq_wmma_blockdim<T>(K);
}

template <typename T>
inline void submit_transform_wmma_bench(int nfuncs, int nblocks, int K,
                                        const T* A, const T* B, T* C, T* workspace,
                                        Stream stream)
{
  Dim3 thread_dims = mra::mTxmq_wmma_blockdim<T>(K);
  int  smem_size   = mra::mTxmq_wmma_shmem_size<T>(K);

#define DISPATCH_L6(Kval) \
  case Kval: \
    CONFIGURE_KERNEL((transform_kernel_wmma_k<T, Kval>), smem_size); \
    CALL_KERNEL((transform_kernel_wmma_k<T, Kval>), std::min(nfuncs, nblocks), \
                thread_dims, smem_size, stream, \
                (nfuncs, A, B, C, workspace)); \
    break;

  switch (K) {
    DISPATCH_L6( 6)
    DISPATCH_L6( 8)
    DISPATCH_L6(10)
    DISPATCH_L6(12)
    DISPATCH_L6(16)
    DISPATCH_L6(20)
    DISPATCH_L6(32)
    default:
      printf("submit_transform_wmma_bench: unsupported K=%d\n", K);
  }
#undef DISPATCH_L6
}

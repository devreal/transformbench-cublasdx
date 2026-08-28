#pragma once

#include "util.h"
#include "mxm_level2.h"

/**
 * Transform wrapper for Level-2 (B in LDS).
 * Follows the same structure as transform.h / transform_cublasdx.h.
 */

template <typename T>
__device__ void transform_level2(
    int K,
    const T* t,       /* input tensor  K^3 */
    const T* c,       /* coefficient matrix K^2 */
    T*& result,       /* output tensor K^3 (pointer updated on swap) */
    T* workspace)     /* per-block scratch K^3 */
{
  constexpr int ndim = 3;
  const T* pc = c;
  T *t0 = workspace, *t1 = result;
  /* swap so t0 points at the output buffer first */
  auto tmp = t0; t0 = t1; t1 = tmp;

  const int dimj = K;
  const int dimi = dimj * dimj;

  mra::mTxmq_level2(dimi, dimj, dimj, t0, t, pc);
  for (int n = 1; n < ndim; ++n) {
    mra::mTxmq_level2(dimi, dimj, dimj, t1, t0, pc);
    auto tmp2 = t0; t0 = t1; t1 = tmp2;
  }
  /* mTxmq_level2 ends with __syncthreads(); no extra sync needed */
}

template <typename T>
LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK, 4)
__global__ void transform_kernel_level2(int nfuncs, int K,
                                        const T* A, const T* B, T* C, T* workspace) {
  const int K2NDIM = K * K * K;
  T* w = workspace + blockIdx.x * K2NDIM;
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    const T* a = A + i * K2NDIM;
    T* c = C + i * K2NDIM;
    transform_level2(K, a, B, c, w);
  }
}

template <typename T>
inline int transform_level2_shmem_size(int K) {
  return mra::mTxmq_level2_shmem_size<T>(K);
}

template <typename T>
inline void submit_transform_level2_bench(int nfuncs, int nblocks, int K,
                                          const T* A, const T* B, T* C, T* workspace,
                                          Stream stream)
{
  Dim3 thread_dims = mra::mTxmq_level2_blockdim<T>(K);
  auto smem_size   = mra::mTxmq_level2_shmem_size<T>(K);
  CONFIGURE_KERNEL(transform_kernel_level2<T>, smem_size);
  CALL_KERNEL(transform_kernel_level2<T>, std::min(nfuncs, nblocks),
              thread_dims, smem_size, stream,
              (nfuncs, K, A, B, C, workspace));
}

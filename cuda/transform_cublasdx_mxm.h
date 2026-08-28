#ifndef HAVE_TRANSFORM_CUBLASDX_MXM_H
#define HAVE_TRANSFORM_CUBLASDX_MXM_H

#include <assert.h>
#include "util.h"
#include "mxm_cublasdx.h"

/**
 * Level 10 - cuBLASDx as a drop-in mTxmq.
 *
 * Where level 9 (transform_cublasdx.h) fuses all three passes and keeps the
 * intermediate in shared memory, this level calls a cuBLASDx block GEMM once
 * per pass, ping-ponging through the per-block workspace exactly like levels
 * 1-3.  The GEMM itself tiles A through shared memory with double buffering
 * (mxm_cublasdx.h), so the two levels bracket the cost of the fusion: level 9
 * saves the round trip to global memory between passes, level 10 does not.
 *
 * In the dual-target sources this path was unreachable: mxm_cublasdx.h declared
 * its entry point as `mTxmq(long, long, long, ...)`, and every call site passed
 * ints, which bind exactly to the `size_type` reference overload in mxm.h.  The
 * entry point is named mTxmq_cublasdx here so the choice is explicit.
 */

#if defined(MRA_HAVE_CUBLASDX) && MRA_HAVE_CUBLASDX

template <typename T>
__device__ void transform_cublasdx_mxm(
    int K,
    const T* t,
    const T* c,
    T*& result,
    T* workspace)
{
  constexpr const int ndim = 3;
  const T* pc = c;
  T *t0 = workspace, *t1 = result;
  { auto tmp = t0; t0 = t1; t1 = tmp; }

  const int dimj = K;
  const int dimi = dimj * dimj;
  mra::mTxmq_cublasdx(dimi, dimj, dimj, t0, t, pc);
  for (int n = 1; n < ndim; ++n) {
    mra::mTxmq_cublasdx(dimi, dimj, dimj, t1, t0, pc);
    auto tmp = t0; t0 = t1; t1 = tmp;
  }
  /* mTxmq_cublasdx ends with __syncthreads() */
}

template <typename T>
inline
LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK, 1)
__global__ void transform_kernel_cublasdx_mxm(int nfuncs, int K,
                                              const T* A, const T* B, T* C, T* workspace) {
  const int K2NDIM = K * K * K;
  T* w = workspace + blockIdx.x * K2NDIM;
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    const T* a = A + i * K2NDIM;
    T* c       = C + i * K2NDIM;
    transform_cublasdx_mxm(K, a, B, c, w);
  }
}

template <typename T>
inline int transform_cublasdx_mxm_shmem_size(int K) {
  return mra::mTxmq_cublasdx_shmem_size<T>(K);
}

template <typename T>
inline Dim3 transform_cublasdx_mxm_blockdim(int K) {
  return mra::mTxmq_cublasdx_blockdim<T>(K);
}

template <typename T>
inline void submit_transform_cublasdx_mxm_bench(int nfuncs, int nblocks, int K,
                                                const T* A, const T* B, T* C, T* workspace,
                                                Stream stream)
{
  Dim3 thread_dims = mra::mTxmq_cublasdx_blockdim<T>(K);
  int  smem_size   = mra::mTxmq_cublasdx_shmem_size<T>(K);
  CONFIGURE_KERNEL(transform_kernel_cublasdx_mxm<T>, smem_size);
  CALL_KERNEL(transform_kernel_cublasdx_mxm<T>, std::min(nfuncs, nblocks),
              thread_dims, smem_size, stream,
              (nfuncs, K, A, B, C, workspace));
}

#else // MRA_HAVE_CUBLASDX

template <typename T>
inline int transform_cublasdx_mxm_shmem_size(int /*K*/) { return 0; }

template <typename T>
inline Dim3 transform_cublasdx_mxm_blockdim(int /*K*/) { return Dim3(1, 1, 1); }

template <typename T>
inline void submit_transform_cublasdx_mxm_bench(int, int, int,
                                                const T*, const T*, T*, T*, Stream) {
  std::printf("cuBLASDx not available, cannot run benchmark\n");
}

#endif // MRA_HAVE_CUBLASDX

#endif // HAVE_TRANSFORM_CUBLASDX_MXM_H

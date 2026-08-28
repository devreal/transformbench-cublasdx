#ifndef HAVE_TRANSFORM_H
#define HAVE_TRANSFORM_H

#include <assert.h>
#include "util.h"
#include "mxm.h"

/*****************************************
 * Level 1 - reference transform via mTxmq
 *
 * Everything stays in global memory: B is re-read from HBM for every row of A
 * and A is re-read once per output column.  This is the correctness reference
 * that validate_levels compares every other level against, so it deliberately
 * uses the plain mxm.h kernel and nothing else.
 *
 * (The dual-target sources also pulled in mxm_cublasdx.h here, where its
 * `long`-typed mTxmq overload was silently shadowed by the exact-match
 * `size_type` overload from mxm.h.  In this tree the cuBLASDx block-GEMM path
 * is reachable on a level of its own - see transform_cublasdx_mxm.h.)
 *****************************************/

template <typename T>
__device__ void transform(
    int K,
    const T* t,
    const T* c,
    T*& result,
    T* workspace)
{
  constexpr const int ndim = 3; // fixed for benchmark
  const T* pc = c;
  T *t0=workspace, *t1=result;
  {
    auto tmp = t0;
    t0 = t1;
    t1 = tmp;
  }
  const int dimj = K;
  int dimi = dimj*dimj;
  mra::mTxmq(dimi, dimj, dimj, t0, t, pc);
  for (int n=1; n<ndim; ++n) {
    mra::mTxmq(dimi, dimj, dimj, t1, t0, pc);
    auto tmp = t0;
    t0 = t1;
    t1 = tmp;
  }
  /* no need to synchronize here, mTxmq synchronizes */
}

template<typename T>
inline
LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK, 4)
__global__ void transform_kernel(int nfuncs, int K, const T* A, const T* B, T* C, T* workspace) {

  const T *a, *b;
  T *c, *w;
  int K2NDIM = K*K*K;
  /* workspace is allocated for each thread-block */
  w = workspace + blockIdx.x * K2NDIM;
  /* iterate over all tensors */
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    a = A + i * K2NDIM;
    b = B;
    c = C + i * K2NDIM;
    transform(K, a, b, c, w);
  }
}

template<typename T>
inline int transform_shmem_size(int K) {
  /* use whatever mTxm says we need */
  return mra::mTxmq_shmem_size<T>(K);
}

template<typename T>
inline void submit_transform_bench(int nfuncs, int nblocks, int K,
                                  const T* A, const T* B, T* C, T* workspace,
                                  Stream stream)
{
  Dim3 thread_dims = mra::mTxmq_blockdim<T>(K);
  assert(block_size(thread_dims) <= MAX_THREADS_PER_BLOCK);
  size_type smem_size = mra::mTxmq_shmem_size<T>(K);
  size_type K2 = K*K;
  if (smem_size < K2*(size_type)sizeof(T)) {
    smem_size = K2*sizeof(T);
  }
  CONFIGURE_KERNEL(transform_kernel<T>, smem_size);
  CALL_KERNEL(transform_kernel<T>, std::min(nfuncs, nblocks), thread_dims, smem_size, stream, (nfuncs, K, A, B, C, workspace));
}

#endif // HAVE_TRANSFORM_H

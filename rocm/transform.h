#ifndef HAVE_TRANSFORM_H
#define HAVE_TRANSFORM_H

#include <assert.h>
#include "util.h"
#include "mxm_cublasdx.h"
#include "mxm.h"

/*****************************************
 * Regular transform function using mTxmq
 *****************************************/

#if defined(__HIP_DEVICE_COMPILE__)


/**
 * T'is the version for HIP! Hop!
 */

template <typename T>
__device__ void transform(
    int K,
    const T* a,
    const T* b,
    T*& c,
    T* workspace)
{
  constexpr const int ndim = 3; // fixed for benchmark

  extern __shared__ T b_shm[];

  for (int i = thread_id(); i < K*K; i += block_size()) b_shm[i] = b[i];
  const T* pc = b_shm;
  T *t0=workspace, *t1=c;
  //std::swap(t0,t1);
    auto tmp = t0;
    t0 = t1;
    t1 = tmp;
  const int dimj = K;
  int dimi = dimj*dimj;
  mra::mTxmq(dimi, dimj, dimj, t0, a, pc);
  for (int n=1; n<ndim; ++n) {
    mra::mTxmq(dimi, dimj, dimj, t1, t0, pc);
    auto tmp = t0;
    t0 = t1;
    t1 = tmp;
    //std::swap(t0,t1);
  }
  /* no need to synchronize here, mTxmq synchronizes */
}


#else // __HIP_DEVICE_COMPILE__

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
  //std::swap(t0,t1);
    auto tmp = t0;
    t0 = t1;
    t1 = tmp;
  const int dimj = K;
  int dimi = dimj*dimj;
  mra::mTxmq(dimi, dimj, dimj, t0, t, pc);
  for (int n=1; n<ndim; ++n) {
    mra::mTxmq(dimi, dimj, dimj, t1, t0, pc);
    auto tmp = t0;
    t0 = t1;
    t1 = tmp;
    //std::swap(t0,t1);
  }
  /* no need to synchronize here, mTxmq synchronizes */
}

#endif // __HIP_DEVICE_COMPILE__

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
  auto smem_size = mra::mTxmq_shmem_size<T>(K);
  size_type K2 = K*K;
  if (smem_size < K2*sizeof(T)) {
    smem_size = K2*sizeof(T);
  }
  CONFIGURE_KERNEL(transform_kernel<T>, smem_size);
  CALL_KERNEL(transform_kernel<T>, std::min(nfuncs, nblocks), thread_dims, smem_size, stream, (nfuncs, K, A, B, C, workspace));
}



#endif // HAVE_TRANSFORM_H

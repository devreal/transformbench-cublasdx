#ifndef HAVE_TRANSFORM_ROCWMMA_H
#define HAVE_TRANSFORM_ROCWMMA_H

#include "util.h"
#include "mxm.h"

#define ROCWMMA_NUM_THREADS 1024

#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>


template <size_type K, typename T>
__device__ void transform_klt16(
    const T* a,
    const T* b,
    T*& c)
{
  /* hold everything in shared memory */
  extern __shared__ char smem[];
  T* shmem = reinterpret_cast<T*>(smem);
  T* b_shmem = shmem;
  T* a_shmem = b_shmem + K * K;
  T* c_shmem = a_shmem + K * K * K;

  const size_type tid = thread_id();
  const size_type num_threads = block_size();

  /* load A and B into shared memory */
  for (int idx = tid; idx < K * K; idx += num_threads) {
    b_shmem[idx] = b[idx];
  }
  for (int idx = tid; idx < K * K * K; idx += num_threads) {
    a_shmem[idx] = a[idx];
  }
  __syncthreads();

  for (int d = 0; d < 3; ++d) {
    /* compute c = a * b, with c also in shared memory */
    for (int i = tid/K; i < K * K; i += num_threads/K) {
      T* ci = c_shmem + i * K;
      int j = tid % K;
      T sum = 0;
      for (long k = 0; k < K; ++k) { /* not parallelized */
        sum += a_shmem[k * K * K + i] * b_shmem[k * K + j];
      }
      if (d == 0) {
        ci[j] = sum;
      } else {
        ci[j] += sum;
      }
    }
    __syncthreads();

    /* swap A and C for the next iteration, so we always read from A and write to C */
    std::swap(a_shmem, c_shmem);
   }

   // write back result to global memory
   for (int idx = tid; idx < K * K * K; idx += num_threads) {
     c[idx] = a_shmem[idx]; // a_shmem is the final result after 3 iterations
   }

}

/**
 * This implementation only works on K=16. For other K values, we fall back to the Level-3 implementation.
 * The fragment size is 16x16x16.
 * The block dimension is 256 threads (one wavefront) to match the MFMA requirements.
 * We load B into a fragment and keep it there.
 * We load A into fragments. Each wave-front stores 4 input fragments and 4 output fragments.
 *
 */
template <size_type K, typename T>
__device__ void transform_rocwmma_k(
    const T* a,
    const T* b,
    T*& c,
    T* workspace)
{
  constexpr uint32_t WM = 16, WN = 16, WK = 16;
  constexpr uint32_t WAVE = 64;   // CDNA wavefront size
  constexpr const int ndim = 3; // fixed for benchmark

  using FragmentA = rocwmma::fragment<rocwmma::matrix_a, K, K, K, T, rocwmma::col_major>;
  using FragmentB = rocwmma::fragment<rocwmma::matrix_b, K, K, K, T, rocwmma::row_major>;
  using FragmentAcc = rocwmma::fragment<rocwmma::accumulator, K, K, K, T, rocwmma::row_major>;

  if constexpr (K < 16) {
    // Fallback to non mma implementation
    transform_klt16<K, T>(a, b, c);
    return;
  } else if constexpr (K > 16) {
    // Not supported, fallback to Level-3
    transform_level3_k<T, K>(a, b, c, workspace);
    return;
  } else {

    /* single shared memory region, holds A and C */
    extern __shared__ char smem[];
    T* shmem = reinterpret_cast<T*>(smem);

    int wave_id = thread_id() / WAVE;
    constexpr int num_waves = (ROCWMMA_NUM_THREADS / WAVE);
    constexpr int frags_per_wave = (K / num_waves);

    // load b into a fragment
    FragmentB b_frag;
    rocwmma::load_matrix_sync(b_frag, b, K);

    /* load A into shared memory */
    const double4* a2 = reinterpret_cast<const double4*>(a);
    double4* s2 = reinterpret_cast<double4*>(shmem);
    for (int idx = thread_id(); idx < (K * K * K) / 4; idx += block_size()) {
      s2[idx] = a2[idx];
    }
    __syncthreads();

    /* every wavefront handles 4 fragments */
    FragmentA a_frags[frags_per_wave];
    FragmentAcc acc_frags[frags_per_wave];


    for (int d = 0; d < ndim; ++d) {
      /* load all wavefront fragments */
      for (int i = 0; i < frags_per_wave; ++i)
      {
        /* load the current fragment */
        if (i < frags_per_wave - 1 || frags_per_wave == 1) {
          rocwmma::load_matrix_sync(a_frags[i], shmem + (i + wave_id * frags_per_wave) * K, K*K);
          // TODO: is it worth prefetching the next fragment?
          if constexpr (frags_per_wave > 1) {
            rocwmma::load_matrix_sync(a_frags[i+1], shmem + (i+1 + wave_id * frags_per_wave) * K, K*K);
          }
        }
        rocwmma::fill_fragment(acc_frags[i], static_cast<T>(0));
        rocwmma::mma_sync(acc_frags[i], a_frags[i], b_frag, acc_frags[i]);
      }

      /* write back all fragments */
      T* c_ptr = (d == ndim - 1) ? c : shmem;
      
      if (d < ndim -1) rocwmma::synchronize_workgroup();
      
      for (int i = 0; i < frags_per_wave; ++i)
      {
        rocwmma::store_matrix_sync(c_ptr + (i + wave_id * frags_per_wave) * K * K,
                                  acc_frags[i], K);
      }

      rocwmma::synchronize_workgroup();
    }
  }
}

#endif // not __HIP_DEVICE_COMPILE__

// fwd-decl for kernel
template <size_type K, typename T>
__device__ void transform_rocwmma_k(
    const T* a,
    const T* b,
    T*& c,
    T* workspace);

/* One kernel binary per K — register pressure is proportional to K, not max(K). */
template <typename T, int K>
LAUNCH_BOUNDS(ROCWMMA_NUM_THREADS, 1)
__global__ void transform_rocwmma(int nfuncs,
                                  const T* A, const T* B, T* C, T* workspace) {
  constexpr int K2NDIM = K * K * K;
  T* w = workspace + blockIdx.x * K2NDIM;
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    const T* a = A + i * K2NDIM;
    T* c       = C + i * K2NDIM;
    /* result pointer starts at c; workspace is w */
    T* result  = c;
    transform_rocwmma_k<K>(a, B, result, w);
  }
}

template <typename T>
inline int transform_rocwmma_shmem_size(int K) {
  if (K < 16) {
    // For K<=16, we load A and B into shared memory. We need space for A (K^3), B (K^2), and C (K^3).
    return (K*K*K + K*K) * sizeof(T);
  } else if (K == 16) {
    // For K==16, we hold one copy of A/C in LDS
    return K*K*K * sizeof(T);
  } else {
    return transform_level3_shmem_size<T>(K);
  }
}

template <typename T>
inline Dim3 transform_rocwmma_blockdim(int K) {
  return {ROCWMMA_NUM_THREADS, 1, 1};
}

template <typename T>
inline void submit_transform_rocwmma_bench(int nfuncs, int nblocks, int K,
                                           const T* A, const T* B, T* C, T* workspace,
                                           Stream stream)
{
  Dim3 thread_dims = transform_rocwmma_blockdim<T>(K);
  int  smem_size   = transform_rocwmma_shmem_size<T>(K);

#define DISPATCH_L3(Kval) \
  case Kval: \
    CONFIGURE_KERNEL((transform_rocwmma<T, Kval>), smem_size); \
    CALL_KERNEL((transform_rocwmma<T, Kval>), std::min(nfuncs, nblocks), \
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


#endif // HAVE_TRANSFORM_ROCWMMA_H

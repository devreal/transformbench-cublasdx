#pragma once

#include "util.h"

/**
 * Level 3: B in LDS + register accumulation.
 *          Each thread owns a full row of the output tile held in a compile-time
 *          register array T acc[K].  The k-loop loads a[k,i] once and FMAs it
 *          against all K columns of B (from LDS), eliminating redundant global
 *          loads and keeping the hot loop inside the register file.
 *
 * Use mTxmq_level3_k<T, K> (K known at compile time) so that each K value
 * gets its own kernel binary with isolated register pressure.
 *
 * c(i,j) = sum_k a(k,i)*b(k,j)
 *   A: K^2 x K  col-major   a[k,i] = a[k*dimi + i]
 *   B: K   x K  row-major   b[k,j] = b[k*dimj + j]
 *   C: K^2 x K  row-major   c[i,j] = c[i*dimj + j]
 */

namespace mra {

namespace detail {

/**
 * Inner kernel: B is already in b_shmem, register array acc[K] accumulates
 * the dot product.  Compile-time K keeps acc[] in VGPRs.
 */
template <typename T, int K, bool Q = false>
__device__ void mTxmq_level3_impl(T* __restrict__ c, const T* a, const T* b_shmem) {
  constexpr int DIMI = K * K;

  for (int i = (int)threadIdx.x; i < DIMI; i += (int)blockDim.x) {
    T acc[K];

    if constexpr (Q) {
      for (int j = 0; j < K; ++j) acc[j] = T(0);
    } else {
      for (int j = 0; j < K; ++j) acc[j] = c[i * K + j];
    }

    /* k-loop: load a[k,i] once, FMA with all K entries of row k of B */
    const T* aik = a + i;   /* a[0,i] in col-major */
    for (int k = 0; k < K; ++k, aik += DIMI) {
      T aki = *aik;
      for (int j = 0; j < K; ++j) {
        acc[j] += aki * b_shmem[k * K + j];
      }
    }

    for (int j = 0; j < K; ++j) c[i * K + j] = acc[j];
  }
}

} // namespace detail


/**
 * K-templated entry point — one binary per K value.
 * Each instantiation sees only acc[K] for its specific K,
 * keeping register pressure proportional to K rather than max(K).
 */
template <typename T, int K>
__device__ void mTxmq_level3_k(T* __restrict__ c, const T* a, const T* b) {
  // extern __shared__ char smem_level3[];
  // T* b_shmem = reinterpret_cast<T*>(smem_level3);

  // for (int idx = (int)threadIdx.x; idx < K * K; idx += (int)blockDim.x)
  //   b_shmem[idx] = b[idx];
  // __syncthreads();

  detail::mTxmq_level3_impl<T, K, true>(c, a, b);
  __syncthreads();
}

template <typename T>
constexpr size_type mTxmq_level3_shmem_size(size_type K) {
  return K * K * sizeof(T);
}

template <typename T>
constexpr Dim3 mTxmq_level3_blockdim(int /*K*/) {
  return Dim3(MAX_THREADS_PER_BLOCK, 1, 1);
}

} // namespace mra

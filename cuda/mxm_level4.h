#pragma once

#include "util.h"
#include "dmma.h"
#include "mxm_level3.h"   /* for the Level-3 fallback */

/**
 * Level 4: NVIDIA FP64 tensor cores (DMMA), one warp per thread block.
 *
 * This is the CUDA counterpart of the CDNA level-4 kernel.  There, one
 * 64-lane wavefront issues v_mfma_f64_16x16x4f64 to produce a 16x16 output
 * tile with a 4-deep contraction.  Here, one 32-lane warp issues
 * mma.sync.m8n8k4.f64 to produce an 8x8 output tile with the same 4-deep
 * contraction, so the tile grid is finer but the shape of the kernel - a single
 * warp walking every output tile in sequence, with B parked in shared memory
 * and A streamed straight from global - is unchanged.
 *
 * Block dimension is 32 threads (one warp), matching the AMD kernel's
 * one-wavefront block.  Level 5 is the multi-warp, shared-memory-staged variant.
 *
 * K must be a multiple of 8 for the DMMA path (K = 8, 16, 32).  Every other K
 * (6, 10, 12, 20) transparently falls back to the level-3 register-blocking
 * kernel, as does any GPU older than sm_80.
 *
 * c(i,j) = sum_k a(k,i)*b(k,j)
 *   A: K^2 x K  col-major   a[k,i] = a[k*dimi + i]
 *   B: K   x K  row-major   b[k,j] = b[k*dimj + j]
 *   C: K^2 x K  row-major   c[i,j] = c[i*dimj + j]
 */

namespace mra {

namespace detail {

#if MRA_HAVE_DMMA

/**
 * DMMA kernel for compile-time K.  Requires blockDim.x == 32 (one warp).
 * B must already be resident in b_smem.
 *
 * Leading dimensions: A uses ldm = K^2 and B/C use ldm = K.  Both are even for
 * every supported K, satisfying the WMMA 16-byte stride requirement for double.
 */
template <typename T, int K>
__device__ void mTxmq_level4_dmma(T* __restrict__ c, const T* a, const T* b_smem) {
  static_assert(std::is_same_v<T, double>,
                "mTxmq_level4_dmma: FP64 tensor cores operate on double only");
  static_assert(K % DMMA_N == 0,
                "mTxmq_level4_dmma: K must be a multiple of 8 for the 8x8x4 tile");

  constexpr int DIMI      = K * K;
  constexpr int ROW_TILES = DIMI / DMMA_M;   /* 8-row tiles of the output   */
  constexpr int COL_TILES = K    / DMMA_N;   /* 8-column tiles of the output */

  /* One warp walks the whole (ROW_TILES x COL_TILES) grid. */
  for (int r = 0; r < ROW_TILES; ++r) {
    for (int ct = 0; ct < COL_TILES; ++ct) {
      FragC acc;
      nvcuda::wmma::fill_fragment(acc, 0.0);

      /* K/4 steps of 4-deep contraction */
      for (int k = 0; k < K; k += DMMA_K) {
        FragA a_frag;
        FragB b_frag;
        /* A^T[r*8 .. r*8+8, k .. k+4] read from the col-major K^2 x K source */
        dmma_load_a(a_frag, a,      k, r  * DMMA_M, DIMI);
        /* B[k .. k+4, ct*8 .. ct*8+8] read from row-major shared memory */
        dmma_load_b(b_frag, b_smem, k, ct * DMMA_N, K);
        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
      }

      dmma_store_c(c, acc, r * DMMA_M, ct * DMMA_N, K);
    }
  }
}

#endif /* MRA_HAVE_DMMA */

} // namespace detail


/* Public entry-point: always clears C (mTxmq semantics, Q=true). */
template <typename aT, typename bT, typename cT>
__device__ void mTxmq_level4(size_type dimi, size_type dimj, size_type dimk,
                              cT* __restrict__ c, const aT* a, const bT* b) {
  extern __shared__ char smem_level4[];
  bT* b_smem = reinterpret_cast<bT*>(smem_level4);

  /* Load B into shared memory */
  for (int idx = (int)threadIdx.x; idx < dimk * dimj; idx += (int)blockDim.x) {
    b_smem[idx] = b[idx];
  }
  __syncthreads();

#if MRA_HAVE_DMMA
  /* DMMA path: only for K divisible by 8 */
  if constexpr (std::is_same_v<cT, double>) {
    if (dimi == dimj * dimj) {
      if (dimj == 8) {
        detail::mTxmq_level4_dmma<cT, 8>(c, a, b_smem);
        __syncthreads();
        return;
      } else if (dimj == 16) {
        detail::mTxmq_level4_dmma<cT, 16>(c, a, b_smem);
        __syncthreads();
        return;
      } else if (dimj == 32) {
        detail::mTxmq_level4_dmma<cT, 32>(c, a, b_smem);
        __syncthreads();
        return;
      }
    }
  }
  /* Fall through to Level-3 register blocking for other K values */
#endif

  /* Level-3 fallback (also the path on pre-sm_80 hardware) */
  if (dimi == dimj * dimj) {
    if      (dimj ==  6) detail::mTxmq_level3_impl<cT,  6, true>(c, a, b_smem);
    else if (dimj ==  8) detail::mTxmq_level3_impl<cT,  8, true>(c, a, b_smem);
    else if (dimj == 10) detail::mTxmq_level3_impl<cT, 10, true>(c, a, b_smem);
    else if (dimj == 12) detail::mTxmq_level3_impl<cT, 12, true>(c, a, b_smem);
    else if (dimj == 16) detail::mTxmq_level3_impl<cT, 16, true>(c, a, b_smem);
    else if (dimj == 20) detail::mTxmq_level3_impl<cT, 20, true>(c, a, b_smem);
    else if (dimj == 32) detail::mTxmq_level3_impl<cT, 32, true>(c, a, b_smem);
    else {
      if (is_team_lead()) printf("mTxmq_level4: unsupported K=%d\n", (int)dimj);
    }
  }
  __syncthreads();
}

/**
 * K-templated entry point - one binary per K value.
 * Loads B into shared memory, then dispatches to DMMA (if available) or the
 * Level-3 fallback.  Requires blockDim.x == 32 (one warp) on the DMMA path.
 */
template <typename T, int K>
__device__ void mTxmq_level4_k(T* __restrict__ c, const T* a, const T* b) {
  extern __shared__ char smem_level4[];
  T* b_smem = reinterpret_cast<T*>(smem_level4);

  for (int idx = (int)threadIdx.x; idx < K * K; idx += (int)blockDim.x)
    b_smem[idx] = b[idx];
  __syncthreads();

#if MRA_HAVE_DMMA
  if constexpr (detail::dmma_supports_k(K) && std::is_same_v<T, double>) {
    detail::mTxmq_level4_dmma<T, K>(c, a, b_smem);
    __syncthreads();
    return;
  }
#endif
  /* Level-3 register-blocking fallback */
  detail::mTxmq_level3_impl<T, K, true>(c, a, b_smem);
  __syncthreads();
}

template <typename T>
constexpr size_type mTxmq_level4_shmem_size(size_type K) {
  return K * K * sizeof(T);
}

template <typename T>
constexpr Dim3 mTxmq_level4_blockdim(int /*K*/) {
  return Dim3(MRA_WARP_SIZE, 1, 1);   /* one warp */
}

} // namespace mra

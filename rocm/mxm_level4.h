#pragma once

#include "util.h"
#include "mxm_level3.h"   /* for the Level-3 fallback */

/**
 * Level 4: AMD MFMA (Matrix Fused Multiply-Accumulate) for FP64.
 *
 * On GFX90A / GFX940 the v_mfma_f64_16x16x4f64 instruction computes a
 * 16x16 output tile with a 4-deep contraction in one wavefront step.
 * Thread layout (64-thread wavefront, cbsz=0, abid=0, blgp=0):
 *
 *   A input  (16 rows x 4 cols  = 64 elements): thread t -> A[t/4][t%4]
 *   B input  (4  rows x 16 cols = 64 elements): thread t -> B[t/16][t%16]
 *   C/D out  (16 rows x 16 cols = 256 elements, 4 per thread):
 *             thread t -> D[{(t/16)*4 + 0..3}][t%16]
 *
 * The blockdim for Level 4 is {64,1,1} (one wavefront).
 *
 * For K values not handled by MFMA (or on non-AMD targets), the implementation
 * transparently falls back to the Level-3 register-blocking kernel.
 *
 * c(i,j) = sum_k a(k,i)*b(k,j)
 *   A: K^2 x K  col-major   a[k,i] = a[k*dimi + i]
 *   B: K   x K  row-major   b[k,j] = b[k*dimj + j]
 *   C: K^2 x K  row-major   c[i,j] = c[i*dimj + j]
 */

namespace mra {

namespace detail {

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__))

/* AMD vector type for 4 doubles — the return type of the MFMA builtin */
typedef double mfma_d4 __attribute__((ext_vector_type(4)));

/**
 * MFMA kernel for compile-time K.
 * Requires blockDim.x == 64 (one wavefront).
 * B must already be loaded into b_shmem.
 */
template <typename T, int K>
__device__ void mTxmq_level4_mfma(T* __restrict__ c, const T* a, const T* b_shmem) {
  static_assert(K % 16 == 0,
                "mTxmq_level4_mfma: K must be a multiple of 16 for 16x16 MFMA");
  static_assert(K * K % 16 == 0,
                "mTxmq_level4_mfma: K^2 must be a multiple of 16");

  constexpr int DIMI     = K * K;
  constexpr int ROW_TILES = DIMI / 16;   /* number of 16-row tiles */
  constexpr int COL_TILES = K   / 16;   /* number of 16-col tiles */

  const int tid = (int)threadIdx.x;     /* 0..63 */

  /* Thread's contribution indices within one MFMA tile */
  const int a_row_in_tile = tid / 4;    /* 0..15 */
  const int a_col_in_tile = tid % 4;    /* 0..3  */
  const int b_row_in_tile = tid / 16;   /* 0..3  */
  const int b_col_in_tile = tid % 16;   /* 0..15 */
  const int d_col_in_tile = tid % 16;
  const int d_row_grp     = (tid / 16) * 4; /* first of 4 output rows this thread owns */

  for (int r = 0; r < ROW_TILES; ++r) {
    for (int ct = 0; ct < COL_TILES; ++ct) {
      mfma_d4 acc = {0.0, 0.0, 0.0, 0.0};

      /* loop over 4-deep contraction blocks */
      for (int k_block = 0; k_block < K; k_block += 4) {
        /* A element: A^T[r*16 + a_row_in_tile, k_block + a_col_in_tile]
         * A is col-major K^2 x K: a[k,i] = a[k*DIMI + i]
         * so A^T[i, k] = a[k*DIMI + i]                                  */
        int a_i = r * 16 + a_row_in_tile;
        int a_k = k_block + a_col_in_tile;
        double a_elem = (double)a[a_k * DIMI + a_i];

        /* B element: B[k_block + b_row_in_tile, ct*16 + b_col_in_tile]
         * B is row-major K x K: b[k,j] = b[k*K + j]                     */
        int b_k = k_block + b_row_in_tile;
        int b_j = ct * 16 + b_col_in_tile;
        double b_elem = (double)b_shmem[b_k * K + b_j];

        acc = (mfma_d4)__builtin_amdgcn_mfma_f64_16x16x4f64(
                a_elem, b_elem, (mfma_d4)acc, 0, 0, 0);
      }

      /* Store 4 output elements owned by this thread:
       * rows r*16 + d_row_grp + 0..3,  col ct*16 + d_col_in_tile        */
      int c_col     = ct * 16 + d_col_in_tile;
      int c_row_base = r * 16 + d_row_grp;
      c[(c_row_base + 0) * K + c_col] = (T)acc[0];
      c[(c_row_base + 1) * K + c_col] = (T)acc[1];
      c[(c_row_base + 2) * K + c_col] = (T)acc[2];
      c[(c_row_base + 3) * K + c_col] = (T)acc[3];
    }
  }
}

#endif /* AMD MFMA guard */

} // namespace detail


/* Public entry-point: always clears C (mTxmq semantics, Q=true). */
template <typename aT, typename bT, typename cT>
__device__ void mTxmq_level4(size_type dimi, size_type dimj, size_type dimk,
                              cT* __restrict__ c, const aT* a, const bT* b) {
  extern __shared__ char smem_level4[];
  bT* b_shmem = reinterpret_cast<bT*>(smem_level4);

  /* Load B into LDS */
  for (int idx = (int)threadIdx.x; idx < dimk * dimj; idx += (int)blockDim.x) {
    b_shmem[idx] = b[idx];
  }
  __syncthreads();

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__))
  /* MFMA path: only for K divisible by 16 */
  if (dimi == dimj * dimj) {
    if (dimj == 16) {
      detail::mTxmq_level4_mfma<cT, 16>(c, a, b_shmem);
      __syncthreads();
      return;
    } else if (dimj == 32) {
      detail::mTxmq_level4_mfma<cT, 32>(c, a, b_shmem);
      __syncthreads();
      return;
    }
  }
  /* Fall through to Level-3 register blocking for other K values */
#endif

  /* Level-3 fallback (also the path on CUDA / non-GFX90A AMD) */
  if (dimi == dimj * dimj) {
    if      (dimj ==  6) detail::mTxmq_level3_impl<cT,  6, true>(c, a, b_shmem);
    else if (dimj ==  8) detail::mTxmq_level3_impl<cT,  8, true>(c, a, b_shmem);
    else if (dimj == 10) detail::mTxmq_level3_impl<cT, 10, true>(c, a, b_shmem);
    else if (dimj == 12) detail::mTxmq_level3_impl<cT, 12, true>(c, a, b_shmem);
    else if (dimj == 16) detail::mTxmq_level3_impl<cT, 16, true>(c, a, b_shmem);
    else if (dimj == 20) detail::mTxmq_level3_impl<cT, 20, true>(c, a, b_shmem);
    else if (dimj == 32) detail::mTxmq_level3_impl<cT, 32, true>(c, a, b_shmem);
    else {
      if (is_team_lead()) printf("mTxmq_level4: unsupported K=%d\n", (int)dimj);
    }
  }
  __syncthreads();
}

/**
 * K-templated entry point — one binary per K value.
 * Loads B into LDS, then dispatches to MFMA (if available) or Level-3 fallback.
 * Requires blockDim.x == 64 (one wavefront) on MFMA path.
 */
template <typename T, int K>
__device__ void mTxmq_level4_k(T* __restrict__ c, const T* a, const T* b) {
  extern __shared__ char smem_level4[];
  T* b_shmem = reinterpret_cast<T*>(smem_level4);

  for (int idx = (int)threadIdx.x; idx < K * K; idx += (int)blockDim.x)
    b_shmem[idx] = b[idx];
  __syncthreads();

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__))
  if constexpr (K % 16 == 0) {
    detail::mTxmq_level4_mfma<T, K>(c, a, b_shmem);
    __syncthreads();
    return;
  }
#endif
  /* Level-3 register-blocking fallback */
  detail::mTxmq_level3_impl<T, K, true>(c, a, b_shmem);
  __syncthreads();
}

template <typename T>
constexpr size_type mTxmq_level4_shmem_size(size_type K) {
  return K * K * sizeof(T);
}

template <typename T>
constexpr Dim3 mTxmq_level4_blockdim(int /*K*/) {
  return Dim3(64, 1, 1);   /* one wavefront */
}

} // namespace mra

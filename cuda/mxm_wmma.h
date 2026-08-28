#pragma once

#include "util.h"
#include "dmma.h"
#include "mxm_level3.h"

/**
 * Level 6: WMMA implementation of mTxmq, c(i,j) = sum_k a(k,i) * b(k,j).
 *
 * CUDA counterpart of the rocWMMA level.  Both go through the vendor's
 * warp-cooperative matrix API rather than raw intrinsics, but the fragment
 * geometry differs and that changes the tiling arithmetic throughout:
 *
 *                        rocWMMA (CDNA)      nvcuda::wmma (sm_80+)
 *   FP64 fragment        16 x 16 x 4         8 x 8 x 4
 *   lanes per fragment   64                  32
 *   column padding       N -> multiple of 16 N -> multiple of 8
 *
 * Matrices:
 *   A : [K x K^2]   col-major in the multiply (leading dimension = K^2)
 *   B : [K x K  ]   row-major (leading dimension = K)
 *   C : [K^2 x K]   row-major (leading dimension = K)
 *
 * Supported K: any multiple of 4 (the FP64 contraction tile).  That covers
 * K = 4, 8, 12, 16, 20, 32; K = 6 and 10 fall back to level 3.
 *
 * Because the fragment is 8 wide rather than 16, K = 4 and 12 are the only
 * supported values that still need column padding, and the smaller tile means
 * the K < 16 special case that rocWMMA needs (its fragment cannot express a
 * K < 16 GEMM at all) has no counterpart here - the WMMA path covers those K
 * values directly.
 *
 * Work assignment: one warp per output tile, striding when there are more
 * tiles than warps.  Block size is capped at 32 warps (1024 threads).
 *
 * Shared memory:
 *   smem_b        [K x N_PAD]                 zero-padded B
 *   smem_stage    [NWARPS x 8 x 8]            per-warp output tile, only when
 *                                             N is not a multiple of 8
 *
 * When N is a multiple of 8 every column of a tile is valid and each warp
 * stores straight to global C.  Otherwise B's padding makes the surplus
 * columns zero but they must not reach C, so each warp lands its tile in an
 * 8x8 shared scratch and copies out only the N valid columns.  (rocWMMA stages
 * the entire M x N_PAD output instead; per-warp staging keeps shared memory
 * bounded - the full-output variant would need 75 KB at K = 20.)
 */

namespace mra {

namespace detail {

/* Zero-padded column count and tile counts for a given K. */
constexpr int wmma_n_pad(int K)   { return dmma_pad_n(K); }
constexpr int wmma_m_tiles(int K) { return (K * K) / DMMA_M; }
constexpr int wmma_n_tiles(int K) { return wmma_n_pad(K) / DMMA_N; }
constexpr int wmma_tiles(int K)   { return wmma_m_tiles(K) * wmma_n_tiles(K); }

constexpr int WMMA_MAX_WARPS = 32;   /* 1024 threads, the block ceiling */

constexpr int wmma_warps(int K) {
  return wmma_tiles(K) < WMMA_MAX_WARPS ? wmma_tiles(K) : WMMA_MAX_WARPS;
}

/** K values the WMMA path handles: the contraction must tile by 4. */
constexpr bool wmma_supports_k(int K) {
  return K > 0 && (K % DMMA_K) == 0;
}

/** True when B/C need column zero-padding. */
constexpr bool wmma_needs_pad(int K) { return K != wmma_n_pad(K); }

/** Shared memory in bytes: padded B, plus per-warp staging when padding. */
template <typename T>
constexpr size_type wmma_shmem_bytes(int K) {
  return static_cast<size_type>(
      (K * wmma_n_pad(K)
       + (wmma_needs_pad(K) ? wmma_warps(K) * DMMA_M * DMMA_N : 0))
      * (int)sizeof(T));
}

#if MRA_HAVE_DMMA

/**
 * Core device function: C[M x N] = A^T[M x K] x B[K x N]
 *   where M = K^2, N = K.
 */
template <typename T, int K>
__device__ void mTxmq_wmma_core(T* __restrict__ c, const T* a, const T* b, T* smem) {
  static_assert(std::is_same_v<T, double>,
                "mTxmq_wmma_core: FP64 tensor cores operate on double only");
  static_assert(K % DMMA_K == 0,
                "K must be divisible by the FP64 contraction tile (4)");

  constexpr int M      = K * K;
  constexpr int N      = K;
  constexpr int N_PAD  = wmma_n_pad(K);
  constexpr int M_TILES = wmma_m_tiles(K);
  constexpr int TOTAL_TILES = wmma_tiles(K);
  constexpr int NWARPS = wmma_warps(K);
  constexpr bool NEEDS_PAD = wmma_needs_pad(K);

  static_assert(M % DMMA_M == 0,
                "K^2 must be 8-aligned; guaranteed whenever K % 4 == 0");

  T* smem_b     = smem;                    /* [K x N_PAD]            */
  T* smem_stage = smem_b + K * N_PAD;      /* [NWARPS x 8 x 8], padded case */

  /* --- Phase 1: load B into smem_b with zero padding -------------------- */
  for (int idx = (int)threadIdx.x; idx < K * N_PAD; idx += (int)blockDim.x) {
    const int ki = idx / N_PAD;
    const int ni = idx % N_PAD;
    smem_b[idx] = (ni < N) ? b[ki * N + ni] : T(0);
  }
  __syncthreads();

  /* --- Phase 2: one warp per output tile, striding over tiles ----------- */
  const int warp_id = (int)threadIdx.x / MRA_WARP_SIZE;
  const int lane    = (int)threadIdx.x % MRA_WARP_SIZE;

  for (int tile = warp_id; tile < TOTAL_TILES; tile += NWARPS) {
    const int tile_m  = tile % M_TILES;
    const int tile_n  = tile / M_TILES;
    const int m_start = tile_m * DMMA_M;
    const int n_start = tile_n * DMMA_N;

    FragC c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0);

    for (int k = 0; k < K; k += DMMA_K) {
      /* A^T[m_start .. m_start+8, k .. k+4].  A is stored row-major as
       * [K rows x M cols], i.e. a[k*M + i] = A[k][i] = A^T[i][k].  Declaring
       * the fragment col_major with ldm = M makes element [m_i][k_j] resolve
       * to ptr[k_j*M + m_i] = a[(k+k_j)*M + m_start+m_i] = A^T[m_start+m_i][k+k_j]. */
      FragA a_frag;
      dmma_load_a(a_frag, a, k, m_start, M);

      /* B tile: smem_b[k .. k+4, n_start .. n_start+8], row-major, ldm = N_PAD */
      FragB b_frag;
      dmma_load_b(b_frag, smem_b, k, n_start, N_PAD);

      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if constexpr (!NEEDS_PAD) {
      /* Every column of the tile is valid: store straight to global C.
       * Warps own disjoint row ranges, so there is no write conflict. */
      dmma_store_c(c, c_frag, m_start, n_start, N);
    } else {
      /* Surplus columns are zero but out of bounds for C.  Land the tile in
       * this warp's 8x8 scratch and copy out only the N valid columns. */
      T* stage = smem_stage + warp_id * DMMA_M * DMMA_N;
      nvcuda::wmma::store_matrix_sync(stage, c_frag, DMMA_N,
                                      nvcuda::wmma::mem_row_major);
      __syncwarp();
      for (int idx = lane; idx < DMMA_M * DMMA_N; idx += MRA_WARP_SIZE) {
        const int mi = idx / DMMA_N;
        const int ni = idx % DMMA_N;
        if (n_start + ni < N) {
          c[(size_t)(m_start + mi) * N + n_start + ni] = stage[idx];
        }
      }
      __syncwarp();   /* scratch is reused on the next tile of this warp */
    }
  }
  __syncthreads();
}

#endif // MRA_HAVE_DMMA

} // namespace detail


/**
 * K-templated entry point - one binary per K value.
 * Falls back to level-3 register blocking when K is not a multiple of 4 or the
 * build targets a GPU without FP64 tensor cores.
 */
template <typename T, int K>
__device__ void mTxmq_wmma_k(T* __restrict__ c, const T* a, const T* b) {
  extern __shared__ char smem_wmma[];
  T* smem = reinterpret_cast<T*>(smem_wmma);

#if MRA_HAVE_DMMA
  if constexpr (detail::wmma_supports_k(K) && std::is_same_v<T, double>) {
    detail::mTxmq_wmma_core<T, K>(c, a, b, smem);
    return;
  }
#endif
  for (int idx = (int)threadIdx.x; idx < K * K; idx += (int)blockDim.x)
    smem[idx] = b[idx];
  __syncthreads();
  detail::mTxmq_level3_impl<T, K, true>(c, a, smem);
  __syncthreads();
}

template <typename T>
inline size_type mTxmq_wmma_shmem_size(int K) {
  if (MRA_DMMA_SUPPORTED && detail::wmma_supports_k(K)) {
    return detail::wmma_shmem_bytes<T>(K);
  }
  return static_cast<size_type>(K * K * (int)sizeof(T));   /* B only, fallback */
}

/**
 * Block dimension: one warp per output tile, capped at 32 warps.
 *   K= 4 :   2 warps =   64 threads
 *   K= 8 :   8 warps =  256 threads
 *   K=12 :  32 warps = 1024 threads (36 tiles, warps stride)
 *   K=16 :  32 warps = 1024 threads (64 tiles, warps stride)
 */
template <typename T>
inline Dim3 mTxmq_wmma_blockdim(int K) {
  if (MRA_DMMA_SUPPORTED && detail::wmma_supports_k(K)) {
    return Dim3(detail::wmma_warps(K) * MRA_WARP_SIZE, 1, 1);
  }
  return Dim3(MAX_THREADS_PER_BLOCK, 1, 1);   /* level-3 fallback */
}

} // namespace mra

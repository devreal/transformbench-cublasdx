#ifndef MRA_OPS_MXM_ROCWMMA_H
#define MRA_OPS_MXM_ROCWMMA_H

/**
 * rocWMMA implementation of mTxmq: c(i,j) = sum_k a(k,i) * b(k,j)
 *
 * Matrices (all row-major):
 *   A : [K_ord × K_ord²]  (used transposed;  leading dimension = K_ord²)
 *   B : [K_ord × K_ord ]  (leading dimension = K_ord)
 *   C : [K_ord² × K_ord]  (leading dimension = K_ord)
 *
 * Supported K_ord values: multiples of 4 with K_ord ≤ 16
 *   K=4, 8, 12, 16 (fp32/fp64).  Falls back for K=6, 10, 20+.
 *
 *   Rationale: MFMA K-tile is 4 (MFMA_F*_16x16x4), so K%4=0 is required.
 *   K_ord%4=0 implies K_ord²%16=0, so A/C have no partial M-tiles.
 *   K_ord≤16 keeps total tiles = K_ord²/16 ≤ 16, fitting in a 1024-thread block.
 *
 * Required block configuration:
 *   blockDim.x = mTxmq_rocwmma_nthreads(K_ord)
 *             = (K_ord² / 16) * warpSize         (warpSize = 64 on AMD)
 *
 * Required shared memory:
 *   mTxmq_shmem_size<T>(K_ord) * sizeof(T) bytes
 *   (= mTxmq_rocwmma_shmem_bytes<T, K_ord>())
 *
 * Integration: include this header before "mra/ops/mxm.h"; the macro
 * MRA_HAVE_MTXMQ defined here suppresses the fallback definition in mxm.h.
 */

#include <rocwmma/rocwmma.hpp>
#include "util.h"

#ifdef __HIP_DEVICE_COMPILE__


namespace mra {

namespace detail {

  // WMMA fragment tile size (M × N × K_TILE) on AMD hardware.
  // M and N are always 16; K_TILE depends on precision.
  constexpr int ROCWMMA_TILE  = 16;   // fragment M and N dimension
  // MFMA K-tile: 4 for both MFMA_F32_16x16x4F32 and MFMA_F64_16x16x4F64
  template<typename T> constexpr int rocwmma_k_tile = 4;

  /**
   * Core device function: C[M×N] = A^T[M×K] × B[K×N]
   *   where M = K_ord², N = K = K_ord.
   *
   * Shared memory layout (smem must hold mTxmq_rocwmma_shmem_bytes<T,K_ord> bytes):
   *   smem_b[K_ord × N_PAD]    zero-padded B; N_PAD = ceil(K_ord/16)*16 = 16
   *   smem_c[M × N_PAD]        staged output  (only allocated when N < N_PAD)
   *
   * When N == N_PAD (K_ord == 16): smem_c is omitted and each WMMA tile stores
   * directly to global C, keeping shared memory usage minimal (~2 KB for fp64).
   *
   * When N < N_PAD (K_ord < 16): B is padded with zeros so the WMMA tiles
   * produce correct results in the valid columns; the full N_PAD-wide output is
   * staged in smem_c, and only the N valid columns are written back to global C.
   */
  template<int K_ord, typename T>
  __device__ void mTxmq_rocwmma_core(T* __restrict__ c, const T* a, const T* b, T* smem)
  {
    static_assert(K_ord % rocwmma_k_tile<T> == 0,
                  "K_ord must be divisible by the MFMA K tile size (4 for fp32/fp64)");

    constexpr int M      = K_ord * K_ord;
    constexpr int N      = K_ord;
    constexpr int K      = K_ord;
    constexpr int K_TILE = rocwmma_k_tile<T>;

    // N_PAD: round K_ord up to the next multiple of 16.
    // For K_ord ≤ 16 this is always 16.
    constexpr int N_PAD = ((N + ROCWMMA_TILE - 1) / ROCWMMA_TILE) * ROCWMMA_TILE;

    // M is already a multiple of 16 when K_ord % 4 == 0 (K_ord² = (4n)² = 16n²).
    static_assert(M % ROCWMMA_TILE == 0,
                  "K_ord² must be 16-aligned; this is guaranteed when K_ord % 4 == 0");

    constexpr int M_TILES     = M / ROCWMMA_TILE;      // output row tiles
    constexpr int N_TILES     = N_PAD / ROCWMMA_TILE;  // = 1 for K_ord ≤ 16
    constexpr int TOTAL_TILES = M_TILES * N_TILES;

    // True when B/C need column zero-padding (K_ord not a multiple of 16).
    constexpr bool NEEDS_PAD  = (N != N_PAD);

    // ── Shared memory layout ───────────────────────────────────────────────
    T* smem_b = smem;                     // [K × N_PAD]
    T* smem_c = smem_b + K * N_PAD;      // [M × N_PAD]  (used only when NEEDS_PAD)

    // ── Phase 1: load B into smem_b with zero-padding ─────────────────────
    // smem_b[ki][ni] = B[ki][ni]  if ni < N,  else 0.
    for (int idx = threadIdx.x; idx < K * N_PAD; idx += blockDim.x) {
      const int ki = idx / N_PAD;
      const int ni = idx % N_PAD;
      smem_b[idx] = (ni < N) ? b[ki * N + ni] : T(0);
    }
    __syncthreads();

    // ── Phase 2: WMMA computation (one wavefront per output tile) ─────────
    const int warp_id = threadIdx.x / warpSize;

    rocwmma::fragment<rocwmma::accumulator,
                      ROCWMMA_TILE, ROCWMMA_TILE, K_TILE, T> c_frag;
    rocwmma::fill_fragment(c_frag, T(0));

    if (warp_id < TOTAL_TILES) {
      // tile_m : row tile index [0, M_TILES)
      // tile_n : col tile index [0, N_TILES) — always 0 for K_ord ≤ 16
      const int tile_m  = warp_id % M_TILES;
      const int tile_n  = warp_id / M_TILES;
      const int m_start = tile_m * ROCWMMA_TILE;
      const int n_start = tile_n * ROCWMMA_TILE;

      for (int k = 0; k < K; k += K_TILE) {
        // ── A fragment ────────────────────────────────────────────────────
        // We want A^T[m_start : m_start+16, k : k+K_TILE].
        // A is stored row-major as [K rows × M cols].
        // Using col_major layout for matrix_a: element [m_i][k_j] is read from
        //   ptr[k_j * ld + m_i]  where ptr = a + k*M + m_start,  ld = M.
        // This gives a[k*M + m_start + k_j*M + m_i]
        //          = A[k + k_j][m_start + m_i]
        //          = A^T[m_start + m_i][k + k_j]. ✓
        rocwmma::fragment<rocwmma::matrix_a,
                          ROCWMMA_TILE, ROCWMMA_TILE, K_TILE,
                          T, rocwmma::col_major> a_frag;

        rocwmma::load_matrix_sync(a_frag,
                                  a + k * M + m_start,
                                  static_cast<uint32_t>(M));

        // ── B fragment ────────────────────────────────────────────────────
        // B tile: smem_b[k : k+K_TILE, n_start : n_start+16]
        // Stored row-major in smem_b with stride N_PAD.
        rocwmma::fragment<rocwmma::matrix_b,
                          ROCWMMA_TILE, ROCWMMA_TILE, K_TILE,
                          T, rocwmma::row_major> b_frag;

        rocwmma::load_matrix_sync(b_frag,
                                  smem_b + k * N_PAD + n_start,
                                  static_cast<uint32_t>(N_PAD));

        rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      if constexpr (!NEEDS_PAD) {
        // ── K_ord == 16: N == N_PAD, no column padding ───────────────────
        // Each tile's 16 output columns are all valid.  Store directly to C
        // (row-major, leading dimension N).  Warps write to non-overlapping
        // row ranges so there are no global-memory conflicts.
        rocwmma::store_matrix_sync(c + m_start * N + n_start,
                                   c_frag,
                                   static_cast<uint32_t>(N),
                                   rocwmma::mem_row_major);
      } else {
        // ── K_ord < 16: N < N_PAD, stage through smem_c ──────────────────
        // Each WMMA tile covers N_PAD columns, but only the first N are valid.
        // Store the full N_PAD-wide tile to smem_c so that the copy phase can
        // extract just the N valid columns without corrupting adjacent data.
        rocwmma::store_matrix_sync(
            smem_c + tile_m * ROCWMMA_TILE * N_PAD + tile_n * ROCWMMA_TILE,
            c_frag,
            static_cast<uint32_t>(N_PAD),
            rocwmma::mem_row_major);
      }
    }

    // ── Phase 3: copy valid columns from smem_c to global C ───────────────
    // Only needed when N < N_PAD.
    if constexpr (NEEDS_PAD) {
      __syncthreads();  // wait for all tiles to finish storing to smem_c
      // smem_c is [M × N_PAD] row-major; copy the valid [M × N] sub-block.
      for (int idx = threadIdx.x; idx < M * N; idx += blockDim.x) {
        const int mi = idx / N;
        const int ni = idx % N;
        c[idx] = smem_c[mi * N_PAD + ni];
      }
    }
  }

  // ── Shared-memory size helpers ─────────────────────────────────────────────

  template<typename T, int K_ord>
  constexpr size_t mTxmq_rocwmma_shmem_bytes() {
    constexpr int N_PAD      = ((K_ord + ROCWMMA_TILE - 1) / ROCWMMA_TILE) * ROCWMMA_TILE;
    constexpr bool NEEDS_PAD = (K_ord != N_PAD);
    constexpr size_t smem_b  = static_cast<size_t>(K_ord) * N_PAD;
    constexpr size_t smem_c  = NEEDS_PAD
                                  ? static_cast<size_t>(K_ord) * K_ord * N_PAD
                                  : 0;
    return (smem_b + smem_c) * sizeof(T);
  }

} // namespace detail


// ── Public interface ───────────────────────────────────────────────────────────

/**
 * rocWMMA-accelerated mTxmq: c(i,j) = sum_k a(k,i) * b(k,j)
 *
 *   A : [dimk × dimi] row-major  (transposed in the multiply)
 *   B : [dimk × dimj] row-major
 *   C : [dimi × dimj] row-major
 *
 * Requires dimi == dimk² and dimj == dimk.
 * Dispatches to rocWMMA for dimk ∈ {4, 8, 12, 16}; prints a diagnostic for
 * other values (callers should fall back to a reference implementation for those).
 */
template<typename aT, typename bT, typename cT>
__device__ void mTxmq(size_type dimi, size_type dimj, size_type dimk,
                      cT* __restrict__ c, const aT* a, const bT* b)
{
  static_assert(std::is_same_v<aT, bT> && std::is_same_v<bT, cT>,
                "rocWMMA mTxmq requires identical input and output types");

  extern __shared__ char smem_raw[];
  cT* smem = reinterpret_cast<cT*>(smem_raw);

  if (dimi == dimk * dimk && dimj == dimk) {
    switch (dimk) {
      case  4: detail::mTxmq_rocwmma_core< 4, cT>(c, a, b, smem); break;
      case  8: detail::mTxmq_rocwmma_core< 8, cT>(c, a, b, smem); break;
      case 12: detail::mTxmq_rocwmma_core<12, cT>(c, a, b, smem); break;
      case 16: detail::mTxmq_rocwmma_core<16, cT>(c, a, b, smem); break;
      default:
        if (threadIdx.x == 0)
          printf("mTxmq_rocwmma: unsupported dimk=%u "
                 "(supported: 4, 8, 12, 16 when dimi=dimk² and dimj=dimk)\n", dimk);
    }
  } else {
    if (threadIdx.x == 0)
      printf("mTxmq_rocwmma: unsupported dimi=%u dimj=%u dimk=%u\n", dimi, dimj, dimk);
  }
}

/**
 * Required shared memory, in units of T elements.
 * Pass mTxmq_shmem_size<T>(K) * sizeof(T) to the kernel's shared memory
 * allocation, e.g. via hipLaunchKernelGGL's sharedMemBytes argument.
 *
 * Shared memory requirements (fp64):
 *   K= 4 :    80 elements =   640 B
 *   K= 8 :  1152 elements =  9.0 KB
 *   K=12 :  2496 elements = 19.5 KB
 *   K=16 :   256 elements =  2.0 KB   (direct store path; no smem_c needed)
 */
template<typename T>
constexpr size_type mTxmq_shmem_size(size_type K) {
  switch (K) {
    case  4: return static_cast<size_type>(
                 detail::mTxmq_rocwmma_shmem_bytes<T,  4>() / sizeof(T));
    case  8: return static_cast<size_type>(
                 detail::mTxmq_rocwmma_shmem_bytes<T,  8>() / sizeof(T));
    case 12: return static_cast<size_type>(
                 detail::mTxmq_rocwmma_shmem_bytes<T, 12>() / sizeof(T));
    case 16: return static_cast<size_type>(
                 detail::mTxmq_rocwmma_shmem_bytes<T, 16>() / sizeof(T));
    default: return 0;
  }
}

/**
 * Required block thread count for K_ord (AMD wavefront = 64 threads).
 *   K= 4 :   64 threads  (  1 wavefront)
 *   K= 8 :  256 threads  (  4 wavefronts)
 *   K=12 :  576 threads  (  9 wavefronts)
 *   K=16 : 1024 threads  ( 16 wavefronts)
 */
template<typename T>
constexpr size_type mTxmq_rocwmma_nthreads(size_type K) {
  // (K² / 16) wavefronts × 64 threads/wavefront
  return static_cast<size_type>((K * K / 16) * 64);
}

} // namespace mra

#define MRA_HAVE_MTXMQ 1

#endif // __HIP_DEVICE_COMPILE__
#endif // MRA_OPS_MXM_ROCWMMA_H

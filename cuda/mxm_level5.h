#pragma once

#include "util.h"
#include "dmma.h"
#include "mxm_level3.h"

/**
 * Level 5: shared-memory-staged A with FP64 tensor cores, 8-warp block.
 *
 * CUDA counterpart of the CDNA level-5 kernel.  The AMD version runs 256
 * threads as 4 wavefronts of 64 and hands each wavefront a 16-row MFMA tile;
 * this version runs the same 256 threads as 8 warps of 32 and hands each warp
 * an 8-row DMMA tile.  The staging strategy - the point of the level - is
 * identical: A is pulled through shared memory in strips so that each element
 * crosses the memory bus once instead of once per output column tile.
 *
 * For K=16, A is 256x16 (col-major).  A is loaded in NCHUNKS strips of
 * CHUNK_ROWS rows each.  All 256 threads cooperate to load one strip into
 * shared memory, then each of the 8 warps takes a disjoint set of 8x8 subtiles
 * of that strip and runs mma.sync.m8n8k4.f64 against B (always resident in
 * shared memory).  Once all warps finish, the next strip is loaded.
 *
 * For K=16:
 *   CHUNK_ROWS = 64   (K^2/4)  - loads all of A in 4 strips
 *   NCHUNKS    = 4
 *   Each warp: 8 rows per chunk = 1 tile of 8x8 per column tile
 *   Total C coverage: 4 chunks x (8 warps x 8 rows) = 256 rows = K^2
 *
 * CHUNK_ROWS is chosen at compile time as the largest fraction of DIMI
 * (DIMI/4, DIMI/8, DIMI/16) whose A strip fits alongside B in 64 KB of shared
 * memory and that is a multiple of NWARPS*8 = 64, so tiles divide evenly.
 * For K=32: CHUNK_ROWS=128, NCHUNKS=8, TILES_PER_WARP=2 per chunk.
 * For K=8 the whole of A is one 64-row strip.
 *
 * Shared memory layout:
 *   [0   ]: B  K x K row-major          (K*K doubles, loaded once)
 *   [K*K ]: A_strip  K x A_STRIDE col-major, A_STRIDE = CHUNK_ROWS + 2
 *             a_smem[k*A_STRIDE + row_local] = A^T[row_base+row_local][k]
 *
 * The +2 padding shifts each k-column off the 32-bank alignment that an
 * unpadded CHUNK_ROWS stride would land on, and keeps the stride even, which
 * the WMMA API requires of a double leading dimension (16 bytes / 8 bytes = 2).
 * The AMD source pads by +1 for the same reason; +1 would be rejected here.
 *
 * Global memory load pattern (K=16, CHUNK_ROWS=64):
 *   256 threads load 1024 elements in 4 passes; each pass reads 64 consecutive
 *   doubles from one k-column of A - a fully coalesced 512-byte burst.
 *
 * K must be a multiple of 8 (K = 8, 16, 32); other K values and pre-sm_80
 * hardware fall back to the level-3 register-blocking kernel.
 *
 * c(i,j) = sum_k a(k,i)*b(k,j)
 *   A: K^2 x K  col-major  a[k,i] = a[k*K^2+i]
 *   B: K   x K  row-major  b[k,j] = b[k*K +j]
 *   C: K^2 x K  row-major  c[i,j] = c[i*K +j]
 */

namespace mra {

namespace detail {

constexpr int LEVEL5_NWARPS  = 8;                              /* 256 threads */
constexpr int LEVEL5_NTHREAD = LEVEL5_NWARPS * MRA_WARP_SIZE;
constexpr int LEVEL5_BUDGET  = 64 * 1024;                      /* shared memory ceiling */

/* Padded column stride of the staged A strip: even, as WMMA requires. */
constexpr int level5_a_stride(int chunk_rows) { return chunk_rows + 2; }

/* A candidate chunk size is usable when it splits DIMI evenly, splits evenly
 * across the 8 warps' 8-row tiles, and its strip fits beside B. */
constexpr bool level5_chunk_ok(int K, int elem_bytes, int cr) {
  return cr >= LEVEL5_NWARPS * DMMA_M
      && (cr % (LEVEL5_NWARPS * DMMA_M)) == 0
      && ((K * K) % cr) == 0
      && K * level5_a_stride(cr) * elem_bytes
             <= LEVEL5_BUDGET - K * K * elem_bytes;
}

constexpr int level5_chunk_rows(int K, int elem_bytes) {
  return level5_chunk_ok(K, elem_bytes, (K * K) / 4)  ? (K * K) / 4
       : level5_chunk_ok(K, elem_bytes, (K * K) / 8)  ? (K * K) / 8
       : level5_chunk_ok(K, elem_bytes, (K * K) / 16) ? (K * K) / 16
       : LEVEL5_NWARPS * DMMA_M;   /* K=8: DIMI is itself one 64-row strip */
}

#if MRA_HAVE_DMMA

template <typename T, int K>
__device__ void mTxmq_level5_dmma(T* __restrict__ c, const T* a, T* b_smem) {
  static_assert(std::is_same_v<T, double>,
                "mTxmq_level5_dmma: FP64 tensor cores operate on double only");
  static_assert(K % DMMA_N == 0, "mTxmq_level5_dmma: K must be a multiple of 8");

  constexpr int DIMI = K * K;

  constexpr int CHUNK_ROWS    = level5_chunk_rows(K, (int)sizeof(T));
  constexpr int NCHUNKS       = DIMI / CHUNK_ROWS;
  constexpr int A_STRIDE      = level5_a_stride(CHUNK_ROWS);
  constexpr int ROWS_PER_WARP = CHUNK_ROWS / LEVEL5_NWARPS;
  constexpr int TILES_PER_WARP = ROWS_PER_WARP / DMMA_M;
  constexpr int COL_TILES     = K / DMMA_N;

  static_assert(TILES_PER_WARP >= 1, "level 5: chunk too small for 8 warps");

  const int tid_block = (int)threadIdx.x;                /* 0..255 */
  const int warp_id   = tid_block / MRA_WARP_SIZE;       /* 0..7   */

  /* A strip buffer sits directly after B in shared memory */
  T* a_smem = b_smem + DIMI;

  for (int chunk = 0; chunk < NCHUNKS; ++chunk) {
    const int row_base = chunk * CHUNK_ROWS;   /* first global A^T row in strip */

    /* --- Cooperative load of the A strip (all 256 threads) --------------- */
    for (int idx = tid_block; idx < K * CHUNK_ROWS; idx += LEVEL5_NTHREAD) {
      const int row_local = idx % CHUNK_ROWS;  /* row within strip */
      const int k         = idx / CHUNK_ROWS;  /* k-column of A    */
      a_smem[k * A_STRIDE + row_local] = a[(size_t)k * DIMI + row_base + row_local];
    }
    __syncthreads();   /* strip fully staged before any DMMA begins */

    /* --- DMMA: each warp owns TILES_PER_WARP consecutive 8-row tiles ----- */
    const int warp_row_start = warp_id * ROWS_PER_WARP;

    for (int t = 0; t < TILES_PER_WARP; ++t) {
      const int local_row = warp_row_start + t * DMMA_M;   /* tile start in strip */

      for (int ct = 0; ct < COL_TILES; ++ct) {
        FragC acc;
        nvcuda::wmma::fill_fragment(acc, 0.0);

        /* K/4 steps of 4-deep contraction */
        for (int kb = 0; kb < K; kb += DMMA_K) {
          FragA a_frag;
          FragB b_frag;
          dmma_load_a(a_frag, a_smem, kb, local_row,    A_STRIDE);
          dmma_load_b(b_frag, b_smem, kb, ct * DMMA_N,  K);
          nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        dmma_store_c(c, acc, row_base + local_row, ct * DMMA_N, K);
      }
    }

    __syncthreads();   /* all warps done before the strip is overwritten */
  }
}

#endif /* MRA_HAVE_DMMA */

} // namespace detail


template <typename T, int K>
__device__ void mTxmq_level5_k(T* __restrict__ c, const T* a, const T* b) {
  extern __shared__ char smem_level5[];
  T* b_smem = reinterpret_cast<T*>(smem_level5);

  /* All threads cooperate to load B once - it stays resident throughout */
  for (int idx = (int)threadIdx.x; idx < K * K; idx += (int)blockDim.x)
    b_smem[idx] = b[idx];
  __syncthreads();

#if MRA_HAVE_DMMA
  if constexpr (detail::dmma_supports_k(K) && std::is_same_v<T, double>) {
    detail::mTxmq_level5_dmma<T, K>(c, a, b_smem);
    __syncthreads();
    return;
  }
#endif
  detail::mTxmq_level3_impl<T, K, true>(c, a, b_smem);
  __syncthreads();
}

/* Host-side sizing - must mirror the constants used inside the kernel. */
template <typename T>
inline size_type mTxmq_level5_shmem_size(int K) {
  const int DIMI = K * K;
  if ((K % mra::detail::DMMA_N) != 0) {
    return static_cast<size_type>(DIMI * (int)sizeof(T));   /* level-3 fallback */
  }
  const int chunk_rows = mra::detail::level5_chunk_rows(K, (int)sizeof(T));
  const int a_stride   = mra::detail::level5_a_stride(chunk_rows);
  return static_cast<size_type>((DIMI + K * a_stride) * (int)sizeof(T));
}

template <typename T>
constexpr Dim3 mTxmq_level5_blockdim(int /*K*/) {
  return Dim3(detail::LEVEL5_NTHREAD, 1, 1);   /* 8 warps */
}

} // namespace mra

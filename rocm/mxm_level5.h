#pragma once

#include "util.h"
#include "mxm_level3.h"

/**
 * Level 5: LDS-staged A with v_mfma_f64_16x16x4f64, 4-wavefront block.
 *
 * For K=16, A is 256×16 (col-major).  A is loaded in NCHUNKS strips of
 * CHUNK_ROWS rows each.  All 256 threads cooperate to load one strip into
 * LDS, then each of the 4 wavefronts takes a disjoint 16×16 subtile of that
 * strip and runs v_mfma_f64_16x16x4f64 against B (always resident in LDS).
 * Once all 4 wavefronts finish, the next strip is loaded.
 *
 * For K=16 (primary target):
 *   CHUNK_ROWS = 64   (K²/4)   — loads all of A in 4 strips
 *   NCHUNKS    = 4
 *   Each wavefront: 1 tile of 16×16 per chunk → 4 tiles total = 64 rows each
 *   Total C coverage: 4 chunks × (4 WFs × 16 rows) = 256 rows = K² ✓
 *
 * CHUNK_ROWS is chosen at compile time as the largest fraction of DIMI
 * (DIMI/4, DIMI/8, DIMI/16) whose A strip fits alongside B in 64 KB LDS.
 * CHUNK_ROWS is always a multiple of NWARPS×16=64 so tiles divide evenly.
 * For K=32: CHUNK_ROWS=128, NCHUNKS=8, TILES_PER_WF=2 per chunk.
 *
 * LDS layout:
 *   [0   ]: B  K×K row-major         (K*K doubles, loaded once)
 *   [K*K ]: A_strip  K×(CHUNK_ROWS+1) col-major with +1 padding per k-column
 *             a_lds[k*(CHUNK_ROWS+1) + row_local] = A^T[row_base+row_local][k]
 *             Padding shifts LDS banks between k-groups, avoiding 4-way conflicts.
 *
 * Global memory load pattern (K=16, CHUNK_ROWS=64):
 *   256 threads load 1024 elements in 4 passes; each pass loads 64 consecutive
 *   doubles from one k-column of A → full 512-byte coalesced transaction.
 *
 * v_mfma_f64_16x16x4f64 thread layout (tid = lane in wavefront, 0..63):
 *   a_row     = tid/4   (0..15): row in the 16×4 A operand
 *   a_k       = tid%4   (0..3 ): k-depth offset within the 4-wide block
 *   b_k       = tid/16  (0..3 ): k-depth offset in the 4×16 B operand
 *   b_col     = tid%16  (0..15): column
 *   d_row_grp = (tid/16)*4      : first of the 4 output rows owned by this thread
 *   d_col     = tid%16
 *
 * c(i,j) = sum_k a(k,i)*b(k,j)
 *   A: K²×K  col-major  a[k,i] = a[k*K²+i]
 *   B: K×K   row-major  b[k,j] = b[k*K +j]
 *   C: K²×K  row-major  c[i,j] = c[i*K +j]
 */

namespace mra {

namespace detail {

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__))

typedef double mfma_d4 __attribute__((ext_vector_type(4)));

template <typename T, int K>
__device__ void mTxmq_level5_mfma(T* __restrict__ c, const T* a, T* b_shmem) {
  static_assert(K % 16 == 0, "mTxmq_level5_mfma: K must be a multiple of 16");

  constexpr int DIMI    = K * K;
  constexpr int NWARPS  = 4;

  /* --- Compile-time chunk sizing ------------------------------------------ */
  constexpr int LDS_BUDGET     = 64 * 1024;
  constexpr int B_BYTES        = DIMI * (int)sizeof(T);
  /* A strip with +1 padding: K*(CHUNK_ROWS+1)*sizeof(T) bytes               */
  /* CHUNK_ROWS+1 <= (LDS_BUDGET - B_BYTES) / (K*sizeof(T))                  */
  constexpr int MAX_CHUNK_ROWS = (LDS_BUDGET - B_BYTES) / (K * (int)sizeof(T)) - 1;
  /* Pick largest DIMI/N that fits AND is a multiple of NWARPS*16=64           */
  constexpr int CHUNK_ROWS = (DIMI / 4  <= MAX_CHUNK_ROWS) ? DIMI / 4  :
                             (DIMI / 8  <= MAX_CHUNK_ROWS) ? DIMI / 8  : DIMI / 16;
  constexpr int NCHUNKS       = DIMI / CHUNK_ROWS;
  constexpr int A_STRIDE      = CHUNK_ROWS + 1;    /* padded LDS column stride */
  constexpr int ROWS_PER_WF   = CHUNK_ROWS / NWARPS;     /* rows per WF per chunk   */
  constexpr int TILES_PER_WF  = ROWS_PER_WF / 16;        /* 16-row tiles per WF     */

  /* --- Thread indices ----------------------------------------------------- */
  const int tid_block = (int)threadIdx.x;   /* 0..255                         */
  const int warp_id   = tid_block / 64;     /* 0..3  — selects SIMD unit      */
  const int tid       = tid_block % 64;     /* 0..63 — lane within wavefront  */

  /* v_mfma_f64_16x16x4f64 lane mapping                                       */
  const int a_row     = tid / 4;            /* 0..15: row in A operand        */
  const int a_k       = tid % 4;            /* 0..3:  k-depth                 */
  const int b_k       = tid / 16;           /* 0..3:  k-depth in B operand    */
  const int b_col     = tid % 16;           /* 0..15: column                  */
  const int d_row_grp = (tid / 16) * 4;    /* first of 4 output rows         */
  const int d_col     = tid % 16;

  /* A strip buffer sits directly after B in LDS                              */
  T* a_lds = b_shmem + DIMI;

  /* =========================================================================
   * Outer loop: one A strip per iteration.
   * ========================================================================= */
  for (int chunk = 0; chunk < NCHUNKS; ++chunk) {
    const int row_base = chunk * CHUNK_ROWS;   /* first global A^T row in strip */

    /* --- Cooperative load of A strip (all 256 threads) ------------------- */
    /* a_lds[k * A_STRIDE + row_local] = a[k*DIMI + row_base + row_local]   */
    /*                                                                        */
    /* Load order: idx steps over [0, K*CHUNK_ROWS) in strides of 256.       */
    /* For K=16, CHUNK_ROWS=64: each pass covers one complete k-column       */
    /* (64 consecutive doubles), giving a fully-coalesced 512-byte burst.    */
    for (int idx = tid_block; idx < K * CHUNK_ROWS; idx += 256) {
      const int row_local = idx % CHUNK_ROWS;  /* row within strip           */
      const int k         = idx / CHUNK_ROWS;  /* k-column of A              */
      a_lds[k * A_STRIDE + row_local] = a[k * DIMI + row_base + row_local];
    }
    __syncthreads();   /* strip fully in LDS before any MFMA begins          */

    /* --- MFMA: each wavefront owns TILES_PER_WF consecutive 16-row tiles - */
    const int wf_local_row_start = warp_id * ROWS_PER_WF;

    for (int t = 0; t < TILES_PER_WF; ++t) {
      const int local_row = wf_local_row_start + t * 16;  /* tile start in strip */

      mfma_d4 acc = {0.0, 0.0, 0.0, 0.0};

      /* K/4 steps of 4-deep contraction                                      */
      for (int kb = 0; kb < K; kb += 4) {
        /* A^T[row_base + local_row + a_row][kb + a_k] from LDS               */
        const double a_elem =
            (double)a_lds[(kb + a_k) * A_STRIDE + local_row + a_row];

        /* B[kb + b_k][b_col] from LDS (row-major)                            */
        const double b_elem =
            (double)b_shmem[(kb + b_k) * K + b_col];

        acc = (mfma_d4)__builtin_amdgcn_mfma_f64_16x16x4f64(
            a_elem, b_elem, (mfma_d4)acc, 0, 0, 0);
      }

      /* Write 4 output elements to C (row-major)                             */
      const int c_row = row_base + local_row + d_row_grp;
      c[(c_row + 0) * K + d_col] = (T)acc[0];
      c[(c_row + 1) * K + d_col] = (T)acc[1];
      c[(c_row + 2) * K + d_col] = (T)acc[2];
      c[(c_row + 3) * K + d_col] = (T)acc[3];
    }

    __syncthreads();   /* all WFs done before strip is overwritten            */
  }
}

#endif /* AMD MFMA guard */

} // namespace detail


template <typename T, int K>
__device__ void mTxmq_level5_k(T* __restrict__ c, const T* a, const T* b) {
  extern __shared__ char smem_level5[];
  T* b_shmem = reinterpret_cast<T*>(smem_level5);

  /* All threads cooperate to load B once — stays resident throughout          */
  for (int idx = (int)threadIdx.x; idx < K * K; idx += (int)blockDim.x)
    b_shmem[idx] = b[idx];
  __syncthreads();

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__))
  if constexpr (K % 16 == 0) {
    detail::mTxmq_level5_mfma<T, K>(c, a, b_shmem);
    __syncthreads();
    return;
  }
#endif
  detail::mTxmq_level3_impl<T, K, true>(c, a, b_shmem);
  __syncthreads();
}

template <typename T>
inline size_type mTxmq_level5_shmem_size(int K) {
  const int DIMI         = K * K;
  const int b_bytes      = DIMI * (int)sizeof(T);
  const int max_chunk    = (64 * 1024 - b_bytes) / (K * (int)sizeof(T)) - 1;
  const int chunk_rows   = (DIMI/4  <= max_chunk) ? DIMI/4  :
                           (DIMI/8  <= max_chunk) ? DIMI/8  : DIMI/16;
  const int a_stride     = chunk_rows + 1;         /* padded                  */
  return static_cast<size_type>((DIMI + K * a_stride) * (int)sizeof(T));
}

template <typename T>
constexpr Dim3 mTxmq_level5_blockdim(int /*K*/) {
  return Dim3(256, 1, 1);   /* 4 wavefronts, one per SIMD unit                */
}

} // namespace mra


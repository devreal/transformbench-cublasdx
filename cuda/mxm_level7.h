#pragma once

#include "util.h"
#include "dmma.h"
#include "mxm_level3.h"

/**
 * Level 7: B resident in tensor-core registers across all three GEMMs.
 *
 * CUDA counterpart of the CDNA level-7 kernel.  Both versions chase the same
 * idea: the K x K matrix B is the one operand shared by all three passes of the
 * transform, so load it into registers once and never touch memory for it
 * again.  The AMD source does this by hand, packing B into VGPRs according to
 * the v_mfma_f64_16x16x4f64 lane mapping.  Here a `wmma::fragment<matrix_b,...>`
 * *is* the register-resident operand, so the same trick falls out of holding
 * the fragments live across the three passes.
 *
 * Block = 256 threads (8 warps).  Warp w owns rows
 * [w*K^2/8, (w+1)*K^2/8) of every output, as 8-row DMMA tiles.
 *
 * --- Pointer trick ---
 * After each GEMM, C [K^2 x K] written row-major to shared memory is
 * reinterpreted as A [K x K^2] col-major for the next GEMM via identical flat
 * indices:
 *   Write: buf[i*K + j]    (C row-major, i in [0,K^2), j in [0,K))
 *   Read:  buf[k*K^2 + i]  (A col-major, k in [0,K), i in [0,K^2))
 * Since K*K^2 = K^3 = K^2*K, both index the same flat buffer - just different
 * shapes.  This is the standard MADNESS mTxmq pointer trick that makes the 3D
 * separable transform work in place.
 *
 * --- Single-buffer reuse ---
 * Within gemm7_pass every A fragment is pulled into registers before any write
 * to dst, and a __syncthreads() separates the two phases, so the same shared
 * buffer is safe to overwrite:
 *   GEMM 1: global A -> buf (shared, row-major)
 *   GEMM 2: buf      -> buf (shared, in place)
 *   GEMM 3: buf      -> C   (global, row-major)
 *
 * Shared memory = K^3 * sizeof(T).  For K=16: 16^3 * 8 = 32,768 bytes.
 *
 * --- A note on bank conflicts ---
 * The AMD source applies an XOR swizzle to its LDS addresses, because a
 * K^2-element stride aliases onto the same banks for every k.  The same
 * aliasing exists here (a 256-double stride is a multiple of the 32-bank
 * cycle, so the four k-groups of a fragment contend 4 ways), but the swizzle
 * cannot be carried over: `load_matrix_sync`/`store_matrix_sync` compute their
 * own lane addresses from a base pointer and a stride, leaving no place to
 * inject an address permutation.  Padding is not an option either - it would
 * break the pointer trick, which depends on the two views aliasing exactly.
 * The conflicts are therefore accepted here; correctness is unaffected.
 *
 * Supported: K in {8, 16} on sm_80+.  Other K values are dispatched to the
 * level-3 kernel by the host-side submit function, and the device-side
 * fallback below keeps the three-pass chain intact for pre-sm_80 builds.
 */

namespace mra {

namespace detail {

constexpr int LEVEL7_NWARPS  = 8;
constexpr int LEVEL7_NTHREAD = LEVEL7_NWARPS * MRA_WARP_SIZE;   /* 256 */

/**
 * K values that get the register-resident-B tensor-core path.
 *
 * K must be a multiple of 8 (tile geometry) and small enough that the whole A
 * partition of a warp fits in registers: a warp holds
 * (K^2/8/8) x (K/4) A fragments plus (K/4) x (K/8) B fragments.  At K=16 that
 * is 16 + 8 doubles per lane, plus 8 accumulator tiles; K=32 would need 128 A
 * fragments per lane and spill.
 */
constexpr bool level7_supports_k(int K) {
  return (K % DMMA_N) == 0 && K >= DMMA_N && K <= 16;
}

#if MRA_HAVE_DMMA

/**
 * One GEMM pass: dst = src^T x B, with B already in b_frags.
 *
 * src is always addressed as A [K x K^2] col-major (ldm = K^2) and dst as
 * C [K^2 x K] row-major (ldm = K) - the two halves of the pointer trick.  That
 * holds whether src/dst live in global or shared memory, so no layout template
 * parameters are needed.
 *
 * The pass is split into load / sync / compute+store so that GEMM 2 can read
 * and write the same buffer: no lane writes until every lane has read.
 */
template <typename T, int K>
__device__ __forceinline__ void gemm7_pass(
    const T*  src,
    T*        dst,
    FragB     b_frags[K / DMMA_K][K / DMMA_N],
    int       warp_row_offset)
{
  /* src and dst deliberately carry no __restrict__: GEMM 2 passes the same
   * buffer for both, and promising the compiler they cannot alias would make
   * that call ill-formed. */
  constexpr int K2            = K * K;
  constexpr int ROWS_PER_WARP = K2 / LEVEL7_NWARPS;
  constexpr int TILES_PER_WARP = ROWS_PER_WARP / DMMA_M;
  constexpr int NSTEPS        = K / DMMA_K;
  constexpr int COL_TILES     = K / DMMA_N;

  /* --- Pre-load this warp's whole A partition into registers -------------- */
  FragA a_frags[TILES_PER_WARP][NSTEPS];
  #pragma unroll
  for (int t = 0; t < TILES_PER_WARP; ++t) {
    #pragma unroll
    for (int s = 0; s < NSTEPS; ++s) {
      dmma_load_a(a_frags[t][s], src,
                  s * DMMA_K,                            /* contraction offset */
                  warp_row_offset + t * DMMA_M,          /* row in A^T         */
                  K2);                                   /* col-major ldm      */
    }
  }

  /* Every lane has read; writes may now proceed even if dst aliases src. */
  __syncthreads();

  /* --- Accumulate and store ---------------------------------------------- */
  #pragma unroll
  for (int t = 0; t < TILES_PER_WARP; ++t) {
    #pragma unroll
    for (int ct = 0; ct < COL_TILES; ++ct) {
      FragC acc;
      nvcuda::wmma::fill_fragment(acc, 0.0);
      #pragma unroll
      for (int s = 0; s < NSTEPS; ++s) {
        nvcuda::wmma::mma_sync(acc, a_frags[t][s], b_frags[s][ct], acc);
      }
      dmma_store_c(dst, acc,
                   warp_row_offset + t * DMMA_M,   /* row in C   */
                   ct * DMMA_N,                    /* col in C   */
                   K);                             /* row-major ldm */
    }
  }
}

/**
 * Three-GEMM chain for level 7.
 *
 * B is loaded into fragments once and stays in registers for all three passes.
 * A single shared buffer (K^3 doubles) is reused in place.
 */
template <typename T, int K>
__device__ void mTxmq_level7_dmma(
    T* __restrict__       c,     /* output [K^2 x K] row-major, global */
    const T* __restrict__ a,     /* input  [K x K^2] col-major, global */
    const T* __restrict__ b,     /* B      [K x K]   row-major, global */
    T*                    buf)   /* shared scratch: K^3 doubles        */
{
  static_assert(std::is_same_v<T, double>,
                "mTxmq_level7_dmma: FP64 tensor cores operate on double only");
  static_assert(level7_supports_k(K), "mTxmq_level7_dmma: unsupported K");

  constexpr int K2        = K * K;
  constexpr int NSTEPS    = K / DMMA_K;
  constexpr int COL_TILES = K / DMMA_N;

  const int warp_id         = (int)threadIdx.x / MRA_WARP_SIZE;
  const int warp_row_offset = warp_id * (K2 / LEVEL7_NWARPS);

  /* Load B into registers once - resident for all three passes. */
  FragB b_frags[NSTEPS][COL_TILES];
  #pragma unroll
  for (int s = 0; s < NSTEPS; ++s) {
    #pragma unroll
    for (int ct = 0; ct < COL_TILES; ++ct) {
      dmma_load_b(b_frags[s][ct], b, s * DMMA_K, ct * DMMA_N, K);
    }
  }

  /* GEMM 1: A (global) -> buf (shared, row-major) */
  gemm7_pass<T, K>(a, buf, b_frags, warp_row_offset);
  __syncthreads();

  /* GEMM 2: buf reread col-major via the pointer trick -> buf, in place */
  gemm7_pass<T, K>(buf, buf, b_frags, warp_row_offset);
  __syncthreads();

  /* GEMM 3: buf reread col-major -> c (global, row-major) */
  gemm7_pass<T, K>(buf, c, b_frags, warp_row_offset);
}

#endif /* MRA_HAVE_DMMA */

} // namespace detail


/**
 * Public interface: executes the full three-GEMM transform chain.
 *
 * `workspace` is used only by the non-tensor-core fallback, which ping-pongs
 * between it and `c` exactly as the level-3 transform does.  (The AMD source's
 * fallback runs a single pass here rather than three, which silently produces a
 * one-pass result; this version keeps the chain intact.)
 */
template <typename T, int K>
__device__ void mTxmq_level7_k(
    T* __restrict__       c,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T*                    workspace)
{
  extern __shared__ char smem_level7[];
  T* buf = reinterpret_cast<T*>(smem_level7);

#if MRA_HAVE_DMMA
  if constexpr (detail::level7_supports_k(K) && std::is_same_v<T, double>) {
    (void)workspace;
    detail::mTxmq_level7_dmma<T, K>(c, a, b, buf);
    return;
  }
#endif
  /* Fallback: level-3 register blocking, full three-pass chain. */
  for (int idx = (int)threadIdx.x; idx < K * K; idx += (int)blockDim.x)
    buf[idx] = b[idx];
  __syncthreads();

  T *t0 = workspace, *t1 = c;
  { auto tmp = t0; t0 = t1; t1 = tmp; }

  detail::mTxmq_level3_impl<T, K, true>(t0, a, buf);
  __syncthreads();
  for (int n = 1; n < 3; ++n) {
    detail::mTxmq_level3_impl<T, K, true>(t1, t0, buf);
    __syncthreads();
    auto tmp = t0; t0 = t1; t1 = tmp;
  }
}

template <typename T>
inline size_type mTxmq_level7_shmem_size(int K) {
  if (MRA_DMMA_SUPPORTED && detail::level7_supports_k(K)) {
    /* Flat K^3 buffer: C [K^2 x K] row-major reinterpreted as A [K x K^2]
     * col-major via the pointer trick.  No padding - it would break aliasing. */
    return static_cast<size_type>(K * K * K * (int)sizeof(T));
  }
  return static_cast<size_type>(K * K * (int)sizeof(T));   /* B only, fallback */
}

template <typename T>
constexpr Dim3 mTxmq_level7_blockdim(int /*K*/) {
  return Dim3(detail::LEVEL7_NTHREAD, 1, 1);
}

} // namespace mra

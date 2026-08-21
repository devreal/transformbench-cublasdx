#pragma once

#include "util.h"
#include "mxm_level3.h"

/**
 * Level 7: B resident in VGPRs, A loaded directly from HBM/LDS into VGPRs (transposed),
 *          no LDS used for A or B.
 *
 * Computes C[K²×K] = A^T[K²×K] × B[K×K] using v_mfma_f64_16x16x4f64.
 *
 * Block = 256 threads (4 wavefronts, one per SIMD/Matrix core).
 *
 * --- B register layout ---
 * B [K×K] row-major.  For the MFMA lane mapping: b_k = tid/16 (0..3), b_col = tid%16 (0..15).
 * Each lane pre-loads its K/4 relevant elements into b_reg[NSTEPS] and holds them in VGPRs
 * across all three GEMMs.
 *
 * --- A register layout ---
 * A [K×K²] col-major in source memory (global for GEMM 1, LDS for GEMMs 2&3).
 * Wave w handles rows [w*ROWS_PER_WF .. w*ROWS_PER_WF+ROWS_PER_WF) of A^T.
 * MFMA A-operand lane mapping: a_row = tid/4 (0..15), a_k = tid%4 (0..3).
 * The transposed read is achieved by the MFMA lane mapping — no explicit LDS transpose.
 *
 * --- Pointer trick ---
 * After each GEMM, C [K²×K] written row-major to LDS is reinterpreted as A [K×K²]
 * col-major for the next GEMM via identical flat indices:
 *   Write: buf[i*K + j]   (C row-major, i ∈ [0,K²), j ∈ [0,K))
 *   Read:  buf[k*K² + i]  (A col-major, k ∈ [0,K), i ∈ [0,K²))
 * Since K*K² = K³ = K²*K, both index the same flat buffer — just different shapes.
 * This is the standard MADNESS mTxmq pointer trick enabling 3D separable transforms.
 *
 * --- XOR swizzle for LDS bank conflicts ---
 * gfx90a: 32 banks × 4 bytes.  bank_of_double = (addr_in_doubles × 2) % 32.
 * K²=256 doubles → stride (256×2)%32 = 0 → all k-values alias → 16-way conflict.
 *
 * Fix: apply a consistent XOR swizzle to all LDS addresses (both write and read):
 *   lds_swizzle(flat) = flat ^ (((flat >> 8) & 3) << 3)
 *
 * Write side (flat = i*K + j):
 *   flat >> 8 = i*K/256 = i/16   →  for K=16, i/16 is exactly the k-group index
 *   (rows i ∈ [0,16) belong to k=0, rows i ∈ [16,32) to k=1, etc.)
 *
 * Read side (flat = k*K² + i):
 *   flat >> 8 = k*K²/256 = k     →  same swizzle key as write side
 *
 * XOR value = (k & 3) * 8 ∈ {0, 8, 16, 24}: scatters k=0..3 to banks {0,8,16,24},
 * reducing 16-way conflicts to 2-way (theoretical minimum for 64 lanes / 32 banks).
 *
 * --- Single-buffer reuse ---
 * Within gemm7_pass the full A pre-load into a_reg completes before any write to dst.
 * Combined with __syncthreads() between GEMMs, the same LDS buffer is safely reused:
 *   GEMM 1: global A → buf (LDS, row-major + swizzle)
 *   GEMM 2: buf (LDS) → buf (LDS, in-place, row-major + swizzle)
 *   GEMM 3: buf (LDS) → C (global, plain row-major, no swizzle)
 *
 * LDS size = K³ * sizeof(T).  For K=16: 16³ × 8 = 32,768 bytes (32 KB).
 *
 * --- GEMM chain: global memory traffic ---
 * GEMM 1: reads A from global.
 * GEMMs 2&3: no global memory traffic for A.
 * GEMM 3: writes final C to global.
 * B remains in VGPRs throughout.
 *
 * Supported: K multiple of 16, gfx90a / gfx940.  Falls back to L3 otherwise.
 */

namespace mra {

namespace detail {

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__))

typedef double mfma_d4 __attribute__((ext_vector_type(4)));

/**
 * XOR swizzle for LDS bank conflict reduction.
 *
 * Applied consistently on both LDS write (flat = i*K + j) and LDS read
 * (flat = k*K² + i): in both cases flat>>8 == k (for K=16), giving the same
 * per-k-group XOR offset of (k & 3) * 8.
 *
 * Not applied to global memory accesses (GEMM 1 read, GEMM 3 write).
 */
__device__ __forceinline__ int lds_swizzle(int flat) {
    return flat ^ (((flat >> 8) & 3) << 3);
}

/**
 * One GEMM pass: dst = src^T × B (B already in b_reg[]).
 *
 * SWIZZLE_SRC — true  → apply lds_swizzle to src read address (LDS source)
 *               false → use flat col-major address as-is  (global source)
 * SWIZZLE_DST — true  → apply lds_swizzle to dst write address (LDS dest)
 *               false → use flat row-major address as-is  (global dest)
 *
 * Read path  (A [K×K²] col-major, standard pointer-trick flat layout):
 *   flat_src = (s*4 + a_k) * K² + wave_row_offset + t*16 + a_row
 *   src[SWIZZLE_SRC ? lds_swizzle(flat_src) : flat_src]
 *
 * Write path (C [K²×K] row-major):
 *   flat_dst = (base_row + r) * K + d_col
 *   dst[SWIZZLE_DST ? lds_swizzle(flat_dst) : flat_dst]
 *
 * GEMM 1: <SWIZZLE_SRC=false, SWIZZLE_DST=true>  — global A  → LDS (swizzled)
 * GEMM 2: <SWIZZLE_SRC=true,  SWIZZLE_DST=true>  — LDS → LDS (in-place, swizzled)
 * GEMM 3: <SWIZZLE_SRC=true,  SWIZZLE_DST=false> — LDS → global C (plain row-major)
 */
template <typename T, int K, bool SWIZZLE_SRC, bool SWIZZLE_DST>
__device__ __forceinline__ void gemm7_pass(
    const T* __restrict__ src,
    T* __restrict__       dst,
    const double          b_reg[K / 4],
    const int             wave_row_offset,
    const int             a_row,
    const int             a_k,
    const int             d_row_grp,
    const int             d_col)
{
    constexpr int K2           = K * K;
    constexpr int ROWS_PER_WF  = K2 / 4;
    constexpr int TILES_PER_WF = ROWS_PER_WF / 16;
    constexpr int NSTEPS       = K / 4;

    /* --- Pre-load A partition into registers (transposed in-flight) -----------
     * A [K×K²] col-major: src[k * K² + i] where k = s*4+a_k, i = wave_row_offset+t*16+a_row.
     * MFMA lane mapping (a_row=tid/4, a_k=tid%4) achieves the in-flight transpose:
     * no extra LDS step needed.  XOR swizzle applied when reading from LDS so that
     * swizzled-write addresses are matched exactly by swizzled-read addresses. */
    double a_reg[TILES_PER_WF][NSTEPS];
    #pragma unroll
    for (int t = 0; t < TILES_PER_WF; ++t)
        #pragma unroll
        for (int s = 0; s < NSTEPS; ++s) {
            const int flat_src = (s * 4 + a_k) * K2
                                 + wave_row_offset + t * 16 + a_row;
            a_reg[t][s] = (double)src[SWIZZLE_SRC ? lds_swizzle(flat_src) : flat_src];
        }

    /* --- Issue all MFMAs for all tiles before draining any accumulator --------
     * Separate AGPR sets per tile allow the matrix core to pipeline tiles
     * while the VALU/LDS units write completed tile results. */
    mfma_d4 acc[TILES_PER_WF];
    #pragma unroll
    for (int t = 0; t < TILES_PER_WF; ++t)
        acc[t] = {0.0, 0.0, 0.0, 0.0};

    #pragma unroll
    for (int t = 0; t < TILES_PER_WF; ++t)
        #pragma unroll
        for (int s = 0; s < NSTEPS; ++s)
            acc[t] = (mfma_d4)__builtin_amdgcn_mfma_f64_16x16x4f64(
                         a_reg[t][s], b_reg[s], (mfma_d4)acc[t], 0, 0, 0);

    /* --- Write results to dst -------------------------------------------------
     * Row-major C [K²×K]: dst[i*K + j].
     * Pointer trick: the same flat buffer is later read as col-major A [K×K²]:
     *   A[k][i] = buf[k*K² + i]   (flat index unchanged, shape reinterpreted).
     * XOR swizzle applied on LDS writes matches the swizzle on subsequent reads.
     * No swizzle on the final global write (GEMM 3). */
    #pragma unroll
    for (int t = 0; t < TILES_PER_WF; ++t) {
        const int base_row = wave_row_offset + t * 16 + d_row_grp;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int flat_dst = (base_row + r) * K + d_col;
            dst[SWIZZLE_DST ? lds_swizzle(flat_dst) : flat_dst] = (T)acc[t][r];
        }
    }
}

/**
 * Three-GEMM chain for level 7.
 *
 * B loaded once into VGPRs, resident through all three GEMMs.
 * Single LDS buffer (K³ doubles) reused in-place; XOR swizzle on all LDS I/O.
 *
 *   GEMM 1: gemm7_pass<SWIZZLE_SRC=false, SWIZZLE_DST=true>  — global A → LDS
 *   GEMM 2: gemm7_pass<SWIZZLE_SRC=true,  SWIZZLE_DST=true>  — LDS → LDS (in-place)
 *   GEMM 3: gemm7_pass<SWIZZLE_SRC=true,  SWIZZLE_DST=false> — LDS → global C
 */
template <typename T, int K>
__device__ void mTxmq_level7_mfma(
    T* __restrict__       c,         /* output [K²×K] row-major, global          */
    const T* __restrict__ a,         /* input  [K×K²] col-major, global          */
    const T* __restrict__ b,         /* B      [K×K]  row-major, global          */
    T*                    buf)       /* LDS scratch: K³ doubles                  */
{
    constexpr int K2     = K * K;
    constexpr int NSTEPS = K / 4;

    const int tid_block       = (int)threadIdx.x;
    const int warp_id         = tid_block / 64;
    const int tid             = tid_block % 64;

    /* MFMA lane indices — fixed for all three GEMMs. */
    const int a_row           = tid / 4;
    const int a_k             = tid % 4;
    const int b_k             = tid / 16;
    const int b_col           = tid % 16;
    const int d_row_grp       = (tid / 16) * 4;
    const int d_col           = tid % 16;
    const int wave_row_offset = warp_id * (K2 / 4);

    /* -----------------------------------------------------------------------
     * Load B into VGPRs once.
     * Lane tid holds b_reg[s] = B[(s*4 + b_k)*K + b_col] for s=0..NSTEPS-1.
     * ----------------------------------------------------------------------- */
    double b_reg[NSTEPS];
    #pragma unroll
    for (int s = 0; s < NSTEPS; ++s)
        b_reg[s] = (double)b[(s * 4 + b_k) * K + b_col];

    /* -----------------------------------------------------------------------
     * GEMM 1: A (global, unpadded K² stride) → buf (LDS, row-major + swizzle)
     * ----------------------------------------------------------------------- */
    gemm7_pass<T, K, /*SWIZZLE_SRC=*/false, /*SWIZZLE_DST=*/true>(
        a, buf, b_reg, wave_row_offset, a_row, a_k, d_row_grp, d_col);
    __syncthreads();

    /* -----------------------------------------------------------------------
     * GEMM 2: buf (LDS, swizzled) reread as col-major via pointer trick
     *         → buf (LDS, in-place, row-major + swizzle)
     * Full A pre-load into a_reg completes before any write, so same buffer
     * is safe to overwrite.
     * ----------------------------------------------------------------------- */
    gemm7_pass<T, K, /*SWIZZLE_SRC=*/true, /*SWIZZLE_DST=*/true>(
        buf, buf, b_reg, wave_row_offset, a_row, a_k, d_row_grp, d_col);
    __syncthreads();

    /* -----------------------------------------------------------------------
     * GEMM 3: buf (LDS, swizzled) reread as col-major → c (global, row-major)
     * ----------------------------------------------------------------------- */
    gemm7_pass<T, K, /*SWIZZLE_SRC=*/true, /*SWIZZLE_DST=*/false>(
        buf, c, b_reg, wave_row_offset, a_row, a_k, d_row_grp, d_col);
}

#endif /* AMD MFMA guard */

} // namespace detail


/**
 * Public interface: executes the full three-GEMM transform chain.
 * Dispatches to mTxmq_level7_mfma on gfx90a/gfx940; falls back to L3 elsewhere.
 */
template <typename T, int K>
__device__ void mTxmq_level7_k(
    T* __restrict__       c,
    const T* __restrict__ a,
    const T* __restrict__ b)
{
    extern __shared__ char smem_level7[];
    T* buf = reinterpret_cast<T*>(smem_level7);

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__))
    if constexpr (K % 16 == 0) {
        detail::mTxmq_level7_mfma<T, K>(c, a, b, buf);
        return;
    }
#endif
    /* Fallback: L3. Load B into the LDS buffer and run one GEMM at a time. */
    for (int idx = (int)threadIdx.x; idx < K * K; idx += (int)blockDim.x)
        buf[idx] = b[idx];
    __syncthreads();
    detail::mTxmq_level3_impl<T, K, true>(c, a, buf);
    __syncthreads();
}

template <typename T>
inline size_type mTxmq_level7_shmem_size(int K) {
    /* Flat K³ buffer: C [K²×K] row-major reinterpreted as A [K×K²] col-major
     * via the pointer trick.  No padding needed — swizzle handles bank conflicts. */
    return static_cast<size_type>(K * K * K * (int)sizeof(T));
}

template <typename T>
constexpr Dim3 mTxmq_level7_blockdim(int /*K*/) {
    return Dim3(256, 1, 1);
}

} // namespace mra

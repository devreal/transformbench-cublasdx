#pragma once

#include "util.h"

/**
 * FP64 tensor-core (DMMA) support layer.
 *
 * This header is the CUDA stand-in for the `__builtin_amdgcn_mfma_f64_16x16x4f64`
 * intrinsic that levels 4, 5 and 7 use on CDNA.  The two hardware units are not
 * interchangeable, and the differences drive every shape constant downstream:
 *
 *                                 CDNA (gfx90a/gfx940)   NVIDIA (sm_80+)
 *   instruction                   v_mfma_f64_16x16x4f64  mma.sync.m8n8k4.f64
 *   tile (M x N x K)              16 x 16 x 4            8 x 8 x 4
 *   lanes cooperating             64 (wavefront)         32 (warp)
 *   accumulators per lane         4                      2
 *
 * The AMD sources index MFMA operands by hand (thread t supplies A[t/4][t%4]
 * and so on).  That lane mapping is specific to CDNA and does not carry over,
 * so the ports below drive the tensor cores through `nvcuda::wmma` instead:
 * `load_matrix_sync` owns the lane->element mapping, which keeps the port
 * correct without hard-coding an undocumented register layout.  The structural
 * choices that distinguish the levels from one another - who stages A, where B
 * lives, how many warps cooperate - are preserved exactly.
 *
 * Constraints inherited from the WMMA API for `double`:
 *   - shape must be 8 x 8 x 4; no other FP64 fragment geometry exists
 *   - requires __CUDA_ARCH__ >= 800 (A100 / H100); sm_70 has no FP64 tensor core
 *   - `ldm` must be a multiple of 16 bytes / sizeof(double) = 2 elements, so all
 *     padded shared-memory strides below are kept even
 *   - every lane of the warp must reach the mma_sync call
 *
 * Levels that cannot use DMMA - either because the GPU predates sm_80 or
 * because K is not a multiple of the tile size - fall back to the level-3
 * register-blocking kernel, exactly as the AMD sources fall back for K values
 * without a native MFMA shape.
 */

/* Host-visible availability: MRA_CUDA_ARCH is supplied by CMake. */
#if defined(MRA_CUDA_ARCH) && (MRA_CUDA_ARCH >= 80)
#  define MRA_DMMA_SUPPORTED 1
#else
#  define MRA_DMMA_SUPPORTED 0
#endif

/* Device-side availability: only true while compiling for sm_80 or newer. */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#  define MRA_HAVE_DMMA 1
#else
#  define MRA_HAVE_DMMA 0
#endif

#if MRA_HAVE_DMMA
#include <mma.h>
#endif

namespace mra {
namespace detail {

/* FP64 tensor-core tile geometry (the only shape NVIDIA offers). */
constexpr int DMMA_M = 8;
constexpr int DMMA_N = 8;
constexpr int DMMA_K = 4;

/* Round n up to the next multiple of DMMA_N. */
constexpr int dmma_pad_n(int n) {
  return ((n + DMMA_N - 1) / DMMA_N) * DMMA_N;
}

/**
 * True when K admits the plain (unpadded) DMMA tiling used by levels 4, 5 and 7:
 *   K % DMMA_K == 0  so the contraction divides into whole 4-deep steps
 *   K % DMMA_N == 0  so the output columns divide into whole 8-wide tiles
 * K % 8 == 0 implies both, and also K^2 % 8 == 0 for the row tiles.
 */
constexpr bool dmma_supports_k(int K) {
  return (K % DMMA_N) == 0 && K > 0;
}

#if MRA_HAVE_DMMA

/**
 * Fragment aliases for  C[M x N] = A^T[M x K] * B[K x N].
 *
 * A is stored col-major as a[k * M + i] (that is, A^T[i][k]), so the matrix_a
 * fragment is declared col_major with ldm = M: element [i][k] is then read from
 * ptr[k * ldm + i], which is precisely the source layout.  B is row-major with
 * ldm = N.  This mirrors the operand handling in the rocWMMA source one-to-one.
 */
using FragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                                     DMMA_M, DMMA_N, DMMA_K,
                                     double, nvcuda::wmma::col_major>;
using FragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                     DMMA_M, DMMA_N, DMMA_K,
                                     double, nvcuda::wmma::row_major>;
using FragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                                     DMMA_M, DMMA_N, DMMA_K,
                                     double>;

/** Load one 8x4 tile of A^T starting at row `row`, contraction offset `k`. */
__device__ __forceinline__
void dmma_load_a(FragA& frag, const double* a, int k, int row, int ldm) {
  nvcuda::wmma::load_matrix_sync(frag, a + (size_t)k * ldm + row, ldm);
}

/** Load one 4x8 tile of B starting at contraction offset `k`, column `col`. */
__device__ __forceinline__
void dmma_load_b(FragB& frag, const double* b, int k, int col, int ldm) {
  nvcuda::wmma::load_matrix_sync(frag, b + (size_t)k * ldm + col, ldm);
}

/** Store one 8x8 accumulator tile to row-major C at [row][col]. */
__device__ __forceinline__
void dmma_store_c(double* c, const FragC& frag, int row, int col, int ldm) {
  nvcuda::wmma::store_matrix_sync(c + (size_t)row * ldm + col, frag, ldm,
                                  nvcuda::wmma::mem_row_major);
}

#endif // MRA_HAVE_DMMA

} // namespace detail
} // namespace mra

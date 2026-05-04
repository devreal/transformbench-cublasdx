#pragma once
#include <cassert>
#include "util.h"

// HIP/AMD-only — uses v_mfma_f64_4x4x4f64 and v_mfma_f64_16x16x4f64 intrinsics.
#if defined(__HIP__)

// transform_blocked_k20.h — corner-turn 3D transform at K=20.
//
// Two variants live here:
//   * scalar  — `transform_kernel_blocked_k20_scalar`        (correctness baseline)
//   * MFMA    — `transform_kernel_blocked_k20_mfma`          (v_mfma_f64_4x4x4f64)
//
// Structural notes (shared):
//
//  * K=20 needs 20 slabs but one wave (64 lanes) per slab would require
//    20 × 64 = 1280 threads, past the 1024/block limit.  We use 10 waves
//    and give each wave two k' slabs (w, w+10).
//
//  * LDS budget is tight.  K³ = 8000 doubles = 64 KB.  No room for a
//    B_lds, and bank-conflict padding (K → K+1) would push us to 8400
//    doubles.  B lives in HBM and goes through the L1 cache; at
//    K²=400 it's 3.2 KB and fits L1 comfortably.
//
// Logical flow (matches CPU reference cpu_transform3d_blocked):
//
//   distribute   : buf[i, j, k] ← A[i, j, k]           canonical layout
//   Pass 1       : buf[a, j, k] ← Σ_i B[i, a] · buf[i, j, k]
//   Pass 2       : buf[a, b, k] ← Σ_j buf[a, j, k] · B[j, b]
//   "corner turn": implicit -- buf is already (a, b, k); wave w
//                  simply reads rows a ∈ {w, w+10} in pass 3 instead of
//                  the k' it wrote.  No LDS shuffle needed.
//   Pass 3       : C[a, b, c]   ← Σ_k buf[a, b, k] · B[k, c]
//                  (write directly to HBM)

// ============================================================================
// Scalar variant (correctness baseline)
// ============================================================================

template<typename T>
__global__
__launch_bounds__(640, 1)
void transform_kernel_blocked_k20_scalar(int nfuncs,
                                         const T* __restrict__ A,
                                         const T* __restrict__ B,
                                         T* __restrict__ C,
                                         T* __restrict__ /*ws unused*/)
{
    static_assert(std::is_same<T, double>::value, "K=20 blocked: FP64 only");

    constexpr int K         = 20;
    constexpr int K2        = K * K;        // 400
    constexpr int K3        = K * K * K;    // 8000
    constexpr int NWAVES    = 10;
    constexpr int SPW       = 2;            // slabs per wave
    constexpr int LANES     = 64;
    constexpr int MAX_SLOTS = 7;            // ceil(K²/64)

    extern __shared__ unsigned char _smem_k20[];
    T* buf = reinterpret_cast<T*>(_smem_k20);

    const int w    = threadIdx.y;
    const int t    = threadIdx.x;
    const int tid  = w * LANES + t;
    const int nthr = blockDim.x * blockDim.y;

    const int cube = blockIdx.x;
    if (cube >= nfuncs) return;
    const T* a_ptr = A + (size_t)cube * K3;
    T*       c_ptr = C + (size_t)cube * K3;

    for (int idx = tid; idx < K3; idx += nthr) buf[idx] = a_ptr[idx];
    __syncthreads();

    // Pass 1
    {
        double acc[SPW][MAX_SLOTS];
        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;
            #pragma unroll
            for (int slot = 0; slot < MAX_SLOTS; ++slot) {
                const int cell_idx = slot * LANES + t;
                if (cell_idx >= K2) { acc[ss][slot] = 0.0; continue; }
                const int a = cell_idx / K, j = cell_idx % K;
                double s = 0.0;
                #pragma unroll
                for (int i = 0; i < K; ++i) s += B[i * K + a] * buf[i * K2 + j * K + kp];
                acc[ss][slot] = s;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;
            #pragma unroll
            for (int slot = 0; slot < MAX_SLOTS; ++slot) {
                const int cell_idx = slot * LANES + t;
                if (cell_idx >= K2) continue;
                const int a = cell_idx / K, j = cell_idx % K;
                buf[a * K2 + j * K + kp] = acc[ss][slot];
            }
        }
    }
    __syncthreads();

    // Pass 2
    {
        double acc[SPW][MAX_SLOTS];
        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;
            #pragma unroll
            for (int slot = 0; slot < MAX_SLOTS; ++slot) {
                const int cell_idx = slot * LANES + t;
                if (cell_idx >= K2) { acc[ss][slot] = 0.0; continue; }
                const int a = cell_idx / K, b = cell_idx % K;
                double s = 0.0;
                #pragma unroll
                for (int j = 0; j < K; ++j) s += buf[a * K2 + j * K + kp] * B[j * K + b];
                acc[ss][slot] = s;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;
            #pragma unroll
            for (int slot = 0; slot < MAX_SLOTS; ++slot) {
                const int cell_idx = slot * LANES + t;
                if (cell_idx >= K2) continue;
                const int a = cell_idx / K, b = cell_idx % K;
                buf[a * K2 + b * K + kp] = acc[ss][slot];
            }
        }
    }
    __syncthreads();

    // Pass 3 (direct to HBM)
    #pragma unroll
    for (int ss = 0; ss < SPW; ++ss) {
        const int a = w + ss * NWAVES;
        #pragma unroll
        for (int slot = 0; slot < MAX_SLOTS; ++slot) {
            const int cell_idx = slot * LANES + t;
            if (cell_idx >= K2) continue;
            const int b = cell_idx / K, c = cell_idx % K;
            double s = 0.0;
            #pragma unroll
            for (int kp = 0; kp < K; ++kp) s += buf[a * K2 + b * K + kp] * B[kp * K + c];
            c_ptr[a * K2 + b * K + c] = s;
        }
    }
}

// ============================================================================
// MFMA variant — v_mfma_f64_4x4x4f64
//
// Layout (empirically confirmed in test_mfma_4x4x4_layout*.hip):
//   The instruction computes FOUR INDEPENDENT 4×4×4 GEMMs per call.
//   With lane S = 16·α + 4·g + β  (g = group, α = S/16, β = (S/4)%4),
//   each of the four groups g ∈ {0,1,2,3} operates on its own 4×4 A,
//   4×4 B and 4×4 D:
//     A_g[m = β ][k = α]  at lane S
//     B_g[k = α ][n = β]  at lane S
//     D_g[m = α ][n = β]  at lane S
//
// GEMM plan per pass, per slab:
//   Output is 20×20 = 5×5 tiles of 4×4.  Inner contraction K_inner=20=5×4.
//   Per output tile: 5 MFMA calls to accumulate the K-slices.
//   Per MFMA call: 4 tiles computed simultaneously (one per g).
//   25 tiles ⇒ 7 MFMA rounds (25/4 rounds = 6 full + 1 partial, last
//   round leaves groups 1,2,3 idle -- fed zeros so their output is
//   discarded cleanly).
// ============================================================================

__device__ inline double mfma_4x4x4_f64(double a, double b, double c) {
    return __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, 0, 0, 0);
}

template<typename T>
__global__
__launch_bounds__(640, 1)
void transform_kernel_blocked_k20_mfma(int nfuncs,
                                       const T* __restrict__ A,
                                       const T* __restrict__ B,
                                       T* __restrict__ C,
                                       T* __restrict__ /*ws unused*/)
{
    static_assert(std::is_same<T, double>::value, "K=20 blocked: FP64 only");

    constexpr int K             = 20;
    constexpr int K2            = K * K;
    constexpr int K3            = K * K * K;
    constexpr int NWAVES        = 10;
    constexpr int SPW           = 2;
    constexpr int LANES         = 64;
    constexpr int TILES_DIM     = 5;          // K / 4
    constexpr int K_SLICES      = 5;          // K / 4
    constexpr int TOTAL_TILES   = TILES_DIM * TILES_DIM;   // 25
    constexpr int MFMA_ROUNDS   = 7;          // ceil(25/4)

    extern __shared__ unsigned char _smem_k20m[];
    T* buf = reinterpret_cast<T*>(_smem_k20m);

    const int w    = threadIdx.y;
    const int t    = threadIdx.x;
    const int tid  = w * LANES + t;
    const int nthr = blockDim.x * blockDim.y;

    // Lane decomposition: S = 16α + 4g + β   with β = S%4, g = (S/4)%4, α = S/16.
    const int beta  =  t        & 3;    // β: low 2 bits
    const int g     = (t >> 2)  & 3;    // g: middle 2 bits -- MFMA group id
    const int alpha = (t >> 4)  & 3;    // α: high 2 bits

    const int cube = blockIdx.x;
    if (cube >= nfuncs) return;
    const T* a_ptr = A + (size_t)cube * K3;
    T*       c_ptr = C + (size_t)cube * K3;

    // ----- Distribute -----
    for (int idx = tid; idx < K3; idx += nthr) buf[idx] = a_ptr[idx];
    __syncthreads();

    // ----- Pass 1: buf[a, j, kp] = Σ_i B[i, a] · buf[i, j, kp] -----
    // MFMA A_g[m=β][k=α] ← B[i=4·k_tile+α, a=4·a_tile_g+β]
    // MFMA B_g[k=α][n=β] ← buf[i=4·k_tile+α, j=4·j_tile_g+β, kp]
    // D_g[m=α][n=β]       ↦ buf[a=4·a_tile_g+α, j=4·j_tile_g+β, kp]
    {
        double tile_acc[SPW][MFMA_ROUNDS];

        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;

            #pragma unroll
            for (int round = 0; round < MFMA_ROUNDS; ++round) {
                const int tile_idx = round * 4 + g;
                const bool valid   = tile_idx < TOTAL_TILES;
                const int a_tile   = valid ? tile_idx / TILES_DIM : 0;
                const int j_tile   = valid ? tile_idx % TILES_DIM : 0;

                double acc = 0.0;
                #pragma unroll
                for (int k_tile = 0; k_tile < K_SLICES; ++k_tile) {
                    const int gi = 4 * k_tile + alpha;
                    const int ga = 4 * a_tile + beta;
                    const int gj = 4 * j_tile + beta;
                    const double aval = valid ? B[gi * K + ga] : 0.0;
                    const double bval = valid ? buf[gi * K2 + gj * K + kp] : 0.0;
                    acc = mfma_4x4x4_f64(aval, bval, acc);
                }
                tile_acc[ss][round] = acc;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;
            #pragma unroll
            for (int round = 0; round < MFMA_ROUNDS; ++round) {
                const int tile_idx = round * 4 + g;
                if (tile_idx >= TOTAL_TILES) continue;
                const int a_tile = tile_idx / TILES_DIM;
                const int j_tile = tile_idx % TILES_DIM;
                const int ga     = 4 * a_tile + alpha;
                const int gj     = 4 * j_tile + beta;
                buf[ga * K2 + gj * K + kp] = tile_acc[ss][round];
            }
        }
    }
    __syncthreads();

    // ----- Pass 2: buf[a, b, kp] = Σ_j buf[a, j, kp] · B[j, b] -----
    // MFMA A_g[m=β][k=α] ← buf[a=4·a_tile_g+β, j=4·j_tile+α, kp]
    //   (here "m" of the mfma = "a" of our GEMM; "k" of the mfma = "j")
    // MFMA B_g[k=α][n=β] ← B[j=4·j_tile+α, b=4·b_tile_g+β]
    // D_g[m=α][n=β]       ↦ buf[a=4·a_tile_g+α, b=4·b_tile_g+β, kp]
    {
        double tile_acc[SPW][MFMA_ROUNDS];

        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;

            #pragma unroll
            for (int round = 0; round < MFMA_ROUNDS; ++round) {
                const int tile_idx = round * 4 + g;
                const bool valid   = tile_idx < TOTAL_TILES;
                const int a_tile   = valid ? tile_idx / TILES_DIM : 0;
                const int b_tile   = valid ? tile_idx % TILES_DIM : 0;

                double acc = 0.0;
                #pragma unroll
                for (int j_tile = 0; j_tile < K_SLICES; ++j_tile) {
                    const int ga = 4 * a_tile + beta;
                    const int gj = 4 * j_tile + alpha;
                    const int gb = 4 * b_tile + beta;
                    const double aval = valid ? buf[ga * K2 + gj * K + kp] : 0.0;
                    const double bval = valid ? B[gj * K + gb] : 0.0;
                    acc = mfma_4x4x4_f64(aval, bval, acc);
                }
                tile_acc[ss][round] = acc;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;
            #pragma unroll
            for (int round = 0; round < MFMA_ROUNDS; ++round) {
                const int tile_idx = round * 4 + g;
                if (tile_idx >= TOTAL_TILES) continue;
                const int a_tile = tile_idx / TILES_DIM;
                const int b_tile = tile_idx % TILES_DIM;
                const int ga     = 4 * a_tile + alpha;
                const int gb     = 4 * b_tile + beta;
                buf[ga * K2 + gb * K + kp] = tile_acc[ss][round];
            }
        }
    }
    __syncthreads();

    // ----- Pass 3: C[a, b, c] = Σ_k buf[a, b, k] · B[k, c] -----
    // After pass 2 the buf layout is (a, b, k').  Wave w now "owns"
    // rows a ∈ {w, w+10} -- it reads from cross-wave contributions.
    //
    // For pass 3, fix a = the wave's owned row.  The per-slab GEMM is
    // over (b, c) with inner k'.
    //
    // MFMA A_g[m=β][k=α] ← buf[a=fixed, b=4·b_tile_g+β, k'=4·k_tile+α]
    //   ("m" = "b" of the GEMM, "k" = "k'")
    // MFMA B_g[k=α][n=β] ← B[k'=4·k_tile+α, c=4·c_tile_g+β]
    // D_g[m=α][n=β]       ↦ C[a=fixed, b=4·b_tile_g+α, c=4·c_tile_g+β]
    #pragma unroll
    for (int ss = 0; ss < SPW; ++ss) {
        const int a = w + ss * NWAVES;

        double tile_acc[MFMA_ROUNDS];

        #pragma unroll
        for (int round = 0; round < MFMA_ROUNDS; ++round) {
            const int tile_idx = round * 4 + g;
            const bool valid   = tile_idx < TOTAL_TILES;
            const int b_tile   = valid ? tile_idx / TILES_DIM : 0;
            const int c_tile   = valid ? tile_idx % TILES_DIM : 0;

            double acc = 0.0;
            #pragma unroll
            for (int k_tile = 0; k_tile < K_SLICES; ++k_tile) {
                const int gb = 4 * b_tile + beta;
                const int gk = 4 * k_tile + alpha;
                const int gc = 4 * c_tile + beta;
                const double aval = valid ? buf[a * K2 + gb * K + gk] : 0.0;
                const double bval = valid ? B[gk * K + gc] : 0.0;
                acc = mfma_4x4x4_f64(aval, bval, acc);
            }
            tile_acc[round] = acc;
        }

        // No __syncthreads needed: writes go to HBM, not LDS.
        #pragma unroll
        for (int round = 0; round < MFMA_ROUNDS; ++round) {
            const int tile_idx = round * 4 + g;
            if (tile_idx >= TOTAL_TILES) continue;
            const int b_tile = tile_idx / TILES_DIM;
            const int c_tile = tile_idx % TILES_DIM;
            const int gb     = 4 * b_tile + alpha;
            const int gc     = 4 * c_tile + beta;
            c_ptr[a * K2 + gb * K + gc] = tile_acc[round];
        }
    }
}

// ============================================================================
// Hybrid MFMA variant — 16×16×4 main + 4×4×4 strips + 4×4 corner
//
// Profile of the pure-4×4×4 path showed MFMA units at 12 % busy (starved by
// per-instruction VALU/LDS overhead), not HBM-bound.  The fix is to push
// more FMAs per instruction by using v_mfma_f64_16x16x4f64 for the bulk
// 16×16 sub-tile; the remaining 16×4, 4×16 and 4×4 pieces are handled by
// v_mfma_f64_4x4x4f64 with 4, 4 and 1 groups respectively.
//
// Per slab per pass: 5 k-slices × 4 MFMA kinds = 20 MFMAs (vs. 35 for the
// pure-4×4×4 path -- a 43 % reduction in instruction issues).
//
// Tile partition of a 20×20 output slab:
//     ┌────────────────┬──────┐
//     │  16×16 main    │16×4  │  right strip  (4 tiles, 4 MFMA groups)
//     │                │right │
//     ├────────────────┼──────┤
//     │  4×16 bottom   │ 4×4  │  corner  (1 tile, 1 MFMA group, 3 wasted)
//     │                │corner│
//     └────────────────┴──────┘
//
// Lane roles (inherit both layouts):
//   16×16×4  A (16×4):  lane t → A[t%16][t/16]      col-major
//            B ( 4×16): lane t → B[t/16][t%16]      row-major
//            D (16×16): lane t acc[e] → D[(t/16)+4e][t%16]
//   4×4×4    (per group g):
//            A_g[m=β][k=α]    at S = 16α + 4g + β
//            B_g[k=α][n=β]
//            D_g[m=α][n=β]
//
// Optimisations on top of the vanilla hybrid:
//   (a) Wide double4 distribute from HBM.
//   (b) Pass 1 → Pass 2 register fusion for the main + right strip tiles.
//       The K=16 kernel already uses this trick; we swap pass 1's main/right
//       operands so the accumulator layout matches pass 2's A-frag directly:
//         feed A_frag = buf, B_frag = B   ⇒  D[m][n] = temp1[n][m]
//         at lane t acc_main[e] = temp1[t%16][(t/16) + 4e]      (main)
//         at lane S acc_right   = temp1[4g+β][16+α]             (right strip)
//       Pass 2 main slices 0..3 (j∈0..15) read acc_main[0..3] directly; slice 4
//       (j∈16..19) reads acc_right.  Pass 2 right strip does the same — its
//       A-frag layout at lane S is temp1[4g+β][4p+α] which matches acc_main
//       for p=0..3 and acc_right for p=4.  Bottom strip and corner can't be
//       fused cleanly (α/β transpose mismatch) and still use LDS.
// ============================================================================

template<typename T>
__global__
__launch_bounds__(640, 1)
void transform_kernel_blocked_k20_hybrid(int nfuncs,
                                         const T* __restrict__ A,
                                         const T* __restrict__ B,
                                         T* __restrict__ C,
                                         T* __restrict__ /*ws unused*/)
{
    static_assert(std::is_same<T, double>::value, "K=20 blocked: FP64 only");

    constexpr int K        = 20;
    constexpr int K2       = K * K;
    constexpr int K3       = K * K * K;
    constexpr int NWAVES   = 10;
    constexpr int SPW      = 2;
    constexpr int K_SLICES = K / 4;      // 5

    extern __shared__ unsigned char _smem_hyb[];
    T* buf = reinterpret_cast<T*>(_smem_hyb);

    const int w    = threadIdx.y;
    const int t    = threadIdx.x;
    const int tid  = w * 64 + t;
    const int nthr = blockDim.x * blockDim.y;

    // 4×4×4 lane decomposition: S = 16α + 4g + β
    const int beta   =  t       & 3;
    const int g      = (t >> 2) & 3;
    const int alpha  = (t >> 4) & 3;
    // 16×16×4 lane decomposition
    const int tmod16 =  t       & 15;
    const int tdiv16 =  t >> 4;

    const int cube = blockIdx.x;
    if (cube >= nfuncs) return;
    const T* a_ptr = A + (size_t)cube * K3;
    T*       c_ptr = C + (size_t)cube * K3;

    // --- Wide (double4) distribute.  One load per 4 doubles = 2000 loads total.
    // a_ptr[] is hipMalloc'd (16-byte aligned) and K³ = 8000 is divisible by 4.
    // Each thread issues at most ceil(2000/640) = 4 double4 loads.
    {
        const double4* a_ptr_vec = reinterpret_cast<const double4*>(a_ptr);
        constexpr int NVEC = K3 / 4;          // 2000
        for (int idx = tid; idx < NVEC; idx += nthr) {
            double4 v = a_ptr_vec[idx];
            const int base = idx * 4;
            buf[base + 0] = v.x;
            buf[base + 1] = v.y;
            buf[base + 2] = v.z;
            buf[base + 3] = v.w;
        }
    }
    __syncthreads();

    // --------------------------------------------------------------------
    // PASS 1 (swapped operands) : keep acc_main and acc_right in registers
    // across the sync so pass 2 can read them directly instead of going
    // through LDS.  See header comment for the layout derivation.
    //
    // With A_frag = buf and B_frag = B, the MFMA produces D = temp1^T.
    // Concretely: at lane t  acc_main[e] = temp1[t%16][(t/16) + 4e].
    //
    // Bottom strip and corner are NOT swapped (their α/β transpose doesn't
    // align with pass 2's bottom/corner A-frag layout) and still go to LDS.
    // --------------------------------------------------------------------
    v4f64  p1_main[SPW];   // pass-1 main   accumulator, survives to pass 2
    double p1_right[SPW];  // pass-1 right  accumulator, survives to pass 2

    {
        double acc_bottom[SPW], acc_corner[SPW];

        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            p1_main[ss]    = v4f64{0.0, 0.0, 0.0, 0.0};
            p1_right[ss]   = 0.0;
            acc_bottom[ss] = 0.0;
            acc_corner[ss] = 0.0;
        }

        #pragma unroll
        for (int ks = 0; ks < K_SLICES; ++ks) {
            #pragma unroll
            for (int ss = 0; ss < SPW; ++ss) {
                const int kp = w + ss * NWAVES;

                // main 16×16 (SWAPPED: A_frag=buf, B_frag=B)
                //   feeds D[m=(t/16)+4e][n=t%16] = temp1[t%16][(t/16)+4e]
                {
                    const int gi = 4*ks + tdiv16;      // global i (0..19)
                    const int gj = tmod16;              // pass-1 "m" from buf = j
                    const double av = buf[gi * K2 + gj * K + kp];
                    const double bv = B[gi * K + gj];   // pass-1 "n" from B = a
                    p1_main[ss] = mfma_16x16x4_f64_shared(av, bv, p1_main[ss]);
                }
                // right strip 16×4 (SWAPPED)
                //   at lane S acc_right = temp1[4g+β][16+α]
                {
                    const int gi = 4*ks + alpha;
                    const int gj = 16 + beta;           // j from buf = 16..19
                    const int ga = 4*g + beta;          // a from B   = 0..15
                    const double av = buf[gi * K2 + gj * K + kp];
                    const double bv = B[gi * K + ga];
                    p1_right[ss] = mfma_4x4x4_f64(av, bv, p1_right[ss]);
                }
                // bottom strip 4×16 (NOT swapped — goes to LDS)
                {
                    const int gi = 4*ks + alpha;
                    const int ga = 16 + beta;
                    const int gj = 4*g + beta;
                    const double av = B[gi * K + ga];
                    const double bv = buf[gi * K2 + gj * K + kp];
                    acc_bottom[ss] = mfma_4x4x4_f64(av, bv, acc_bottom[ss]);
                }
                // corner 4×4 (NOT swapped — goes to LDS, group 0 only)
                {
                    const int gi = 4*ks + alpha;
                    const int ga = 16 + beta;
                    const int gj = 16 + beta;
                    const bool vg = (g == 0);
                    const double av = vg ? B[gi * K + ga] : 0.0;
                    const double bv = vg ? buf[gi * K2 + gj * K + kp] : 0.0;
                    acc_corner[ss] = mfma_4x4x4_f64(av, bv, acc_corner[ss]);
                }
            }
        }

        __syncthreads();  // all pass-1 LDS reads done; about to overwrite buf

        // Write only bottom + corner to LDS.  Main and right stay in
        // (p1_main, p1_right) registers for the pass-2 fused reads.
        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;
            {
                const int ga = 16 + alpha;
                const int gj = 4*g + beta;
                buf[ga * K2 + gj * K + kp] = acc_bottom[ss];
            }
            if (g == 0) {
                const int ga = 16 + alpha;
                const int gj = 16 + beta;
                buf[ga * K2 + gj * K + kp] = acc_corner[ss];
            }
        }
    }
    __syncthreads();

    // --------------------------------------------------------------------
    // PASS 2 : D[a, b] = Σ_j temp1[a, j, kp] · B_filter[j, b]
    //   Main & right strip read A-frag from (p1_main, p1_right) registers:
    //     ks=0..3 → A-frag = p1_main[ss][ks]          (j ∈ 0..15)
    //     ks=4    → A-frag = p1_right[ss]             (j ∈ 16..19)
    //   Bottom strip and corner still read from LDS.
    // --------------------------------------------------------------------
    {
        v4f64  acc_main[SPW];
        double acc_right[SPW], acc_bottom[SPW], acc_corner[SPW];

        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            acc_main[ss]   = v4f64{0.0, 0.0, 0.0, 0.0};
            acc_right[ss]  = 0.0;
            acc_bottom[ss] = 0.0;
            acc_corner[ss] = 0.0;
        }

        #pragma unroll
        for (int ks = 0; ks < K_SLICES; ++ks) {
            #pragma unroll
            for (int ss = 0; ss < SPW; ++ss) {
                const int kp = w + ss * NWAVES;

                // A-frag for main & right: from registers (fused).
                // ks=0..3 → p1_main[ss][ks] at this lane (4g+β, α encoding).
                // ks=4   → p1_right[ss]    at this lane.
                const double a_main_av  = (ks < 4) ? p1_main[ss][ks] : p1_right[ss];
                // For the 4x4x4 right strip, the A-frag value at lane S is the
                // same register value (the encoding matches — see header).
                const double a_right_av = (ks < 4) ? p1_main[ss][ks] : p1_right[ss];

                // main 16×16  (a∈0..15, b∈0..15)
                {
                    const int gj = 4*ks + tdiv16;
                    const int gb = tmod16;
                    const double bv = B[gj * K + gb];
                    acc_main[ss] = mfma_16x16x4_f64_shared(a_main_av, bv, acc_main[ss]);
                }
                // right strip 16×4  (a∈0..15, b∈16..19)
                {
                    const int gj = 4*ks + alpha;
                    const int gb = 16 + beta;
                    const double bv = B[gj * K + gb];
                    acc_right[ss] = mfma_4x4x4_f64(a_right_av, bv, acc_right[ss]);
                }
                // bottom strip 4×16  (a∈16..19, b∈0..15)  — LDS read
                {
                    const int ga = 16 + beta;
                    const int gj = 4*ks + alpha;
                    const int gb = 4*g + beta;
                    const double av = buf[ga * K2 + gj * K + kp];
                    const double bv = B[gj * K + gb];
                    acc_bottom[ss] = mfma_4x4x4_f64(av, bv, acc_bottom[ss]);
                }
                // corner 4×4  (a∈16..19, b∈16..19)  — LDS read, group 0 only
                {
                    const int ga = 16 + beta;
                    const int gj = 4*ks + alpha;
                    const int gb = 16 + beta;
                    const bool vg = (g == 0);
                    const double av = vg ? buf[ga * K2 + gj * K + kp] : 0.0;
                    const double bv = vg ? B[gj * K + gb] : 0.0;
                    acc_corner[ss] = mfma_4x4x4_f64(av, bv, acc_corner[ss]);
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int ss = 0; ss < SPW; ++ss) {
            const int kp = w + ss * NWAVES;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int ga = tdiv16 + 4*e;
                const int gb = tmod16;
                buf[ga * K2 + gb * K + kp] = acc_main[ss][e];
            }
            {
                const int ga = 4*g + alpha;
                const int gb = 16 + beta;
                buf[ga * K2 + gb * K + kp] = acc_right[ss];
            }
            {
                const int ga = 16 + alpha;
                const int gb = 4*g + beta;
                buf[ga * K2 + gb * K + kp] = acc_bottom[ss];
            }
            if (g == 0) {
                const int ga = 16 + alpha;
                const int gb = 16 + beta;
                buf[ga * K2 + gb * K + kp] = acc_corner[ss];
            }
        }
    }
    __syncthreads();

    // --------------------------------------------------------------------
    // PASS 3 : C[a, b, c] = Σ_k temp2[a, b, k] · B_filter[k, c]
    //   For each wave-owned a, do the hybrid GEMM over (b, c); inner k=k'.
    //   A_mfma[b, k'] = buf[a, b, k'];  B_mfma[k', c] = B_filter[k', c]
    //   Output written directly to HBM.
    // --------------------------------------------------------------------
    #pragma unroll
    for (int ss = 0; ss < SPW; ++ss) {
        const int a = w + ss * NWAVES;    // wave's owned row

        v4f64  acc_main   = v4f64{0.0, 0.0, 0.0, 0.0};
        double acc_right  = 0.0;
        double acc_bottom = 0.0;
        double acc_corner = 0.0;

        #pragma unroll
        for (int ks = 0; ks < K_SLICES; ++ks) {
            // main 16×16  (b∈0..15, c∈0..15)
            {
                const int gb  = tmod16;
                const int gkp = 4*ks + tdiv16;
                const int gc  = tmod16;
                const double av = buf[a * K2 + gb * K + gkp];
                const double bv = B[gkp * K + gc];
                acc_main = mfma_16x16x4_f64_shared(av, bv, acc_main);
            }
            // right strip 16×4  (b∈0..15, c∈16..19)
            {
                const int gb  = 4*g + beta;
                const int gkp = 4*ks + alpha;
                const int gc  = 16 + beta;
                const double av = buf[a * K2 + gb * K + gkp];
                const double bv = B[gkp * K + gc];
                acc_right = mfma_4x4x4_f64(av, bv, acc_right);
            }
            // bottom strip 4×16  (b∈16..19, c∈0..15)
            {
                const int gb  = 16 + beta;
                const int gkp = 4*ks + alpha;
                const int gc  = 4*g + beta;
                const double av = buf[a * K2 + gb * K + gkp];
                const double bv = B[gkp * K + gc];
                acc_bottom = mfma_4x4x4_f64(av, bv, acc_bottom);
            }
            // corner 4×4  (b∈16..19, c∈16..19).  Only group 0.
            {
                const int gb  = 16 + beta;
                const int gkp = 4*ks + alpha;
                const int gc  = 16 + beta;
                const bool vg = (g == 0);
                const double av = vg ? buf[a * K2 + gb * K + gkp] : 0.0;
                const double bv = vg ? B[gkp * K + gc] : 0.0;
                acc_corner = mfma_4x4x4_f64(av, bv, acc_corner);
            }
        }

        // Writeback directly to HBM C.
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int gb = tdiv16 + 4*e;
            const int gc = tmod16;
            c_ptr[a * K2 + gb * K + gc] = acc_main[e];
        }
        {
            const int gb = 4*g + alpha;
            const int gc = 16 + beta;
            c_ptr[a * K2 + gb * K + gc] = acc_right;
        }
        {
            const int gb = 16 + alpha;
            const int gc = 4*g + beta;
            c_ptr[a * K2 + gb * K + gc] = acc_bottom;
        }
        if (g == 0) {
            const int gb = 16 + alpha;
            const int gc = 16 + beta;
            c_ptr[a * K2 + gb * K + gc] = acc_corner;
        }
    }
}

// ============================================================================
// 2-block-per-tensor SPLIT variant — padded LDS + B_lds
//
// Profile of the 1-block hybrid revealed 23-way LDS bank conflicts from the
// stride-K=20 access pattern (GCD(20, 32)=4).  Adding a K+1 pad fixes the
// conflict but overflows 64 KB in a single block.  Splitting the k'-axis
// across 2 blocks (each handles K/2=10 slabs) buys the padded layout *and*
// a B_lds cache:
//
//   buf_padded  : 10 · K · BK doubles = 10 · 20 · 21 = 4200   (33.6 KB)
//   B_lds       : K · K doubles       = 400                    ( 3.2 KB)
//   total                                                    ( 36.8 KB)
//
// LDS layout: buf[kp_local · K · BK + i · BK + j], BK = 21.
//   stride over i = BK  (GCD(21,32)=1) -> conflict-free
//   stride over j = 1                   -> conflict-free
//   stride over kp_local = K·BK=420    -> 8-way conflict (pass 3 contraction)
//
// Grid: (nfuncs, 2).  Block 0 handles kp ∈ [0, 10); block 1 handles [10, 20).
// Each block computes a partial-sum over its k' range and atomic-adds to C.
// Caller must zero C before launching (submit helper does hipMemsetAsync).
//
// Pass 3 inner contraction has K_inner = 10 = 2 full 4-wide MFMA k-slices + 1
// partial (α∈{0,1} valid, α∈{2,3} fed zero).  Handled by a per-lane guard.
// ============================================================================

template<typename T>
__global__
__launch_bounds__(640, 2)
void transform_kernel_blocked_k20_split(int nfuncs,
                                        const T* __restrict__ A,
                                        const T* __restrict__ B,
                                        T* __restrict__ C,
                                        T* __restrict__ /*ws unused*/)
{
    static_assert(std::is_same<T, double>::value, "K=20 split: FP64 only");

    constexpr int K        = 20;
    constexpr int K_HALF   = K / 2;            // 10 slabs per block
    constexpr int BK       = K + 1;             // 21 - bank-conflict pad on j
    constexpr int K2       = K * K;
    constexpr int K3       = K * K * K;
    constexpr int K_SLICES_FULL = K / 4;        // 5 (pass 1 & 2: full K)
    // Pass 3 inner is only K_HALF=10 per block → 3 slices, last one partial.
    constexpr int K_SLICES_P3   = (K_HALF + 3) / 4;  // 3
    constexpr int WAVE_STRIDE   = K * BK;       // 20 * 21 = 420

    extern __shared__ unsigned char _smem_split[];
    T* buf   = reinterpret_cast<T*>(_smem_split);
    T* B_lds = buf + K_HALF * WAVE_STRIDE;      // after padded buf

    const int w     = threadIdx.y;              // 0..9   kp_local this wave owns
    const int t     = threadIdx.x;              // 0..63
    const int tid   = w * 64 + t;
    const int nthr  = blockDim.x * blockDim.y;  // 640

    // 4×4×4 lane decomposition
    const int beta   =  t       & 3;
    const int g      = (t >> 2) & 3;
    const int alpha  = (t >> 4) & 3;
    // 16×16×4 lane decomposition
    const int tmod16 =  t       & 15;
    const int tdiv16 =  t >> 4;

    const int cube       = blockIdx.x;
    const int block_half = blockIdx.y;          // 0 or 1
    if (cube >= nfuncs) return;
    const int kp_start   = block_half * K_HALF; // 0 or 10

    const T* a_ptr = A + (size_t)cube * K3;
    T*       c_ptr = C + (size_t)cube * K3;

    // --- Cache B into B_lds ---
    for (int idx = tid; idx < K2; idx += nthr) B_lds[idx] = B[idx];

    // --- Distribute: load A[:, :, kp ∈ block's range] into buf ---
    // HBM layout: A[i*K² + j*K + kp].  LDS: buf[kp_local * WAVE_STRIDE + i * BK + j].
    for (int idx = tid; idx < K_HALF * K2; idx += nthr) {
        int kp_local = idx / K2;
        int i        = (idx / K) % K;
        int j        = idx %  K;
        int kp_g     = kp_start + kp_local;
        buf[kp_local * WAVE_STRIDE + i * BK + j] = a_ptr[i * K2 + j * K + kp_g];
    }
    __syncthreads();

    // ============ PASS 1 : D[a, j] = Σ_i B[i, a] · buf[i, j, kp] ============
    {
        v4f64  acc_main[1];
        double acc_right = 0.0, acc_bottom = 0.0, acc_corner = 0.0;
        acc_main[0] = v4f64{0.0, 0.0, 0.0, 0.0};

        #pragma unroll
        for (int ks = 0; ks < K_SLICES_FULL; ++ks) {
            // main 16×16  (a∈0..15, j∈0..15)
            {
                const int gi = 4*ks + tdiv16;
                const int ga = tmod16;
                const double av = B_lds[gi * K + ga];
                const double bv = buf[w * WAVE_STRIDE + gi * BK + ga];
                acc_main[0] = mfma_16x16x4_f64_shared(av, bv, acc_main[0]);
            }
            // right strip 16×4  (a∈0..15, j∈16..19)
            {
                const int gi = 4*ks + alpha;
                const int ga = 4*g + beta;
                const int gj = 16 + beta;
                const double av = B_lds[gi * K + ga];
                const double bv = buf[w * WAVE_STRIDE + gi * BK + gj];
                acc_right = mfma_4x4x4_f64(av, bv, acc_right);
            }
            // bottom strip 4×16  (a∈16..19, j∈0..15)
            {
                const int gi = 4*ks + alpha;
                const int ga = 16 + beta;
                const int gj = 4*g + beta;
                const double av = B_lds[gi * K + ga];
                const double bv = buf[w * WAVE_STRIDE + gi * BK + gj];
                acc_bottom = mfma_4x4x4_f64(av, bv, acc_bottom);
            }
            // corner 4×4  (a∈16..19, j∈16..19)  group 0 only
            {
                const int gi = 4*ks + alpha;
                const int ga = 16 + beta;
                const int gj = 16 + beta;
                const bool vg = (g == 0);
                const double av = vg ? B_lds[gi * K + ga] : 0.0;
                const double bv = vg ? buf[w * WAVE_STRIDE + gi * BK + gj] : 0.0;
                acc_corner = mfma_4x4x4_f64(av, bv, acc_corner);
            }
        }

        __syncthreads();

        // Writeback pass 1 → buf (same slab, reinterpret as (a, j, kp_local))
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int ga = tdiv16 + 4*e;
            const int gj = tmod16;
            buf[w * WAVE_STRIDE + ga * BK + gj] = acc_main[0][e];
        }
        {
            const int ga = 4*g + alpha;
            const int gj = 16 + beta;
            buf[w * WAVE_STRIDE + ga * BK + gj] = acc_right;
        }
        {
            const int ga = 16 + alpha;
            const int gj = 4*g + beta;
            buf[w * WAVE_STRIDE + ga * BK + gj] = acc_bottom;
        }
        if (g == 0) {
            const int ga = 16 + alpha;
            const int gj = 16 + beta;
            buf[w * WAVE_STRIDE + ga * BK + gj] = acc_corner;
        }
    }
    __syncthreads();

    // ============ PASS 2 : D[a, b] = Σ_j temp1[a, j, kp] · B[j, b] ============
    {
        v4f64  acc_main[1];
        double acc_right = 0.0, acc_bottom = 0.0, acc_corner = 0.0;
        acc_main[0] = v4f64{0.0, 0.0, 0.0, 0.0};

        #pragma unroll
        for (int ks = 0; ks < K_SLICES_FULL; ++ks) {
            // main 16×16  (a∈0..15, b∈0..15)
            {
                const int ga = tmod16;
                const int gj = 4*ks + tdiv16;
                const int gb = tmod16;
                const double av = buf[w * WAVE_STRIDE + ga * BK + gj];
                const double bv = B_lds[gj * K + gb];
                acc_main[0] = mfma_16x16x4_f64_shared(av, bv, acc_main[0]);
            }
            // right strip 16×4  (a∈0..15, b∈16..19)
            {
                const int ga = 4*g + beta;
                const int gj = 4*ks + alpha;
                const int gb = 16 + beta;
                const double av = buf[w * WAVE_STRIDE + ga * BK + gj];
                const double bv = B_lds[gj * K + gb];
                acc_right = mfma_4x4x4_f64(av, bv, acc_right);
            }
            // bottom strip 4×16  (a∈16..19, b∈0..15)
            {
                const int ga = 16 + beta;
                const int gj = 4*ks + alpha;
                const int gb = 4*g + beta;
                const double av = buf[w * WAVE_STRIDE + ga * BK + gj];
                const double bv = B_lds[gj * K + gb];
                acc_bottom = mfma_4x4x4_f64(av, bv, acc_bottom);
            }
            // corner 4×4  (a∈16..19, b∈16..19)  group 0 only
            {
                const int ga = 16 + beta;
                const int gj = 4*ks + alpha;
                const int gb = 16 + beta;
                const bool vg = (g == 0);
                const double av = vg ? buf[w * WAVE_STRIDE + ga * BK + gj] : 0.0;
                const double bv = vg ? B_lds[gj * K + gb] : 0.0;
                acc_corner = mfma_4x4x4_f64(av, bv, acc_corner);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int ga = tdiv16 + 4*e;
            const int gb = tmod16;
            buf[w * WAVE_STRIDE + ga * BK + gb] = acc_main[0][e];
        }
        {
            const int ga = 4*g + alpha;
            const int gb = 16 + beta;
            buf[w * WAVE_STRIDE + ga * BK + gb] = acc_right;
        }
        {
            const int ga = 16 + alpha;
            const int gb = 4*g + beta;
            buf[w * WAVE_STRIDE + ga * BK + gb] = acc_bottom;
        }
        if (g == 0) {
            const int ga = 16 + alpha;
            const int gb = 16 + beta;
            buf[w * WAVE_STRIDE + ga * BK + gb] = acc_corner;
        }
    }
    __syncthreads();

    // ============ PASS 3 : partial[a, b, c] = Σ_{kp_local ∈ [0, K_HALF)} ============
    //                        buf[kp_local, a, b] · B[kp_start+kp_local, c]
    // Output atomic-added to HBM C (caller pre-zeros C).
    //
    // Wave w owns 2 a-values: a ∈ {w, w + K_HALF} = {w, w+10}.
    // K_inner = K_HALF = 10 → 3 MFMA k-slices (last partial, 2 of 4 valid).
    #pragma unroll
    for (int ss = 0; ss < 2; ++ss) {
        const int a = w + ss * K_HALF;     // wave's owned a-row

        v4f64  acc_main  = v4f64{0.0, 0.0, 0.0, 0.0};
        double acc_right = 0.0, acc_bottom = 0.0, acc_corner = 0.0;

        #pragma unroll
        for (int ks = 0; ks < K_SLICES_P3; ++ks) {
            // For 16×16×4 MFMA, the K=4 inner lanes are the ones with tdiv16 ∈ [0, 4).
            //   gkp = 4*ks + tdiv16.  Valid iff gkp < K_HALF.
            const int gkp_main = 4*ks + tdiv16;
            const bool v_main  = (gkp_main < K_HALF);

            const int gkp_strip = 4*ks + alpha;
            const bool v_strip  = (gkp_strip < K_HALF);

            // main 16×16  (b∈0..15, c∈0..15)
            //   A_mfma[m=b][k=kp_local] = temp2[a_fixed, b, kp_local]
            //                           = buf[kp_local, a_fixed, b]
            //   Lane t's (m, k) = (t%16, t/16)  →  b = t%16, kp_local offset = t/16
            {
                const int gb  = tmod16;
                const int gc  = tmod16;
                const int kpg = kp_start + gkp_main;
                const double av = v_main ? buf[gkp_main * WAVE_STRIDE + a * BK + gb] : 0.0;
                const double bv = v_main ? B_lds[kpg * K + gc] : 0.0;
                acc_main = mfma_16x16x4_f64_shared(av, bv, acc_main);
            }
            // right strip 16×4  (b∈0..15, c∈16..19)
            {
                const int gb = 4*g + beta;
                const int gc = 16 + beta;
                const int kpg = kp_start + gkp_strip;
                const double av = v_strip ? buf[gkp_strip * WAVE_STRIDE + a * BK + gb] : 0.0;
                const double bv = v_strip ? B_lds[kpg * K + gc] : 0.0;
                acc_right = mfma_4x4x4_f64(av, bv, acc_right);
            }
            // bottom strip 4×16  (b∈16..19, c∈0..15)
            {
                const int gb = 16 + beta;
                const int gc = 4*g + beta;
                const int kpg = kp_start + gkp_strip;
                const double av = v_strip ? buf[gkp_strip * WAVE_STRIDE + a * BK + gb] : 0.0;
                const double bv = v_strip ? B_lds[kpg * K + gc] : 0.0;
                acc_bottom = mfma_4x4x4_f64(av, bv, acc_bottom);
            }
            // corner 4×4  (b∈16..19, c∈16..19)
            {
                const int gb = 16 + beta;
                const int gc = 16 + beta;
                const int kpg = kp_start + gkp_strip;
                const bool vg = (g == 0) && v_strip;
                const double av = vg ? buf[gkp_strip * WAVE_STRIDE + a * BK + gb] : 0.0;
                const double bv = vg ? B_lds[kpg * K + gc] : 0.0;
                acc_corner = mfma_4x4x4_f64(av, bv, acc_corner);
            }
        }

        // --- atomic add partial to HBM C ---
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int gb = tdiv16 + 4*e;
            const int gc = tmod16;
            atomicAdd(&c_ptr[a * K2 + gb * K + gc], (double)acc_main[e]);
        }
        {
            const int gb = 4*g + alpha;
            const int gc = 16 + beta;
            atomicAdd(&c_ptr[a * K2 + gb * K + gc], acc_right);
        }
        {
            const int gb = 16 + alpha;
            const int gc = 4*g + beta;
            atomicAdd(&c_ptr[a * K2 + gb * K + gc], acc_bottom);
        }
        if (g == 0) {
            const int gb = 16 + alpha;
            const int gc = 16 + beta;
            atomicAdd(&c_ptr[a * K2 + gb * K + gc], acc_corner);
        }
    }
}

// ============================================================================
// Dispatch helpers
// ============================================================================

template<typename T>
inline size_type blocked_k20_shmem_size() {
    return (size_type)(20 * 20 * 20 * sizeof(T));  // 64 KB at FP64 (1-block variants)
}

template<typename T>
inline size_type blocked_k20_split_shmem_size() {
    // buf[K_HALF · K · BK] + B_lds[K · K] = 10·20·21 + 20·20 = 4600 doubles
    return (size_type)((10 * 20 * 21 + 20 * 20) * sizeof(T));  // 36.8 KB
}

template<typename T>
inline Dim3 blocked_k20_blockdim() {
    return Dim3(64, 10, 1);
}

// Default path for K=20: hybrid 16×16×4 + 4×4×4 MFMA (1 block per tensor).
//
// The 2-block _split variant (with padded LDS + B_lds) was implemented and
// measured to be ~1.9× SLOWER at N=2048 despite eliminating bank conflicts
// (23.9 → 0.2-way).  The overheads that killed it:
//   * atomicAdd read-modify-write on C  (HBM traffic 258 → 752 MB)
//   * pre-zero hipMemsetAsync on C      (+128 MB)
//   * extra VALU for 420-stride indexing + partial-k-slice guards (+2.6×)
// Net: MFMA busy fell 11 → 6 % because the extra VALU starved them further.
// Conclusion: bank conflicts were a visible counter but not the critical path.
//
// Alternative kernels kept for reference / future experimentation:
//   transform_kernel_blocked_k20_split  (2 blocks + padded LDS + atomics)
//   transform_kernel_blocked_k20_mfma   (pure 4×4×4)
//   transform_kernel_blocked_k20_scalar (correctness baseline)
template<typename T>
inline void submit_transform_bench_blocked_k20(int nfuncs,
                                               const T* A, const T* B, T* C, T* workspace,
                                               Stream stream)
{
    Dim3      td   = blocked_k20_blockdim<T>();
    size_type smem = blocked_k20_shmem_size<T>();
    CONFIGURE_KERNEL((transform_kernel_blocked_k20_hybrid<T>), smem);
    CALL_KERNEL((transform_kernel_blocked_k20_hybrid<T>), nfuncs, td, smem, stream,
                (nfuncs, A, B, C, workspace));
}

#endif // __HIP__

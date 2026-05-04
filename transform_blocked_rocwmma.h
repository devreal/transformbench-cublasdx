#pragma once
#include <cassert>
#include "util.h"

// HIP/AMD-only — uses rocWMMA fragments / mma_sync.
#if defined(__HIP__)

#if defined(__HIP_DEVICE_COMPILE__)
#include <rocwmma/rocwmma.hpp>
#endif

// transform_blocked_rocwmma.h — block-distributed 3D transform, rocWMMA variant.
//
// Mirrors the algorithm in transform_blocked.h (L7) but replaces the manual
// MFMA calls with rocwmma::mma_sync.  rocWMMA hides the MFMA fragment layouts
// and emits the same underlying v_mfma_f64_16x16x4f64 on GFX90A.
//
// Layout differences from L7:
//   * buf uses the per-wave layout  buf[s*WB + i*BK + j]  (not L7's canonical
//     (i, j, k) form) so rocWMMA's standard row-major loads work directly
//     with ldm = BK.
//   * No register fusion between pass 1 and pass 2 (fragment contents are
//     opaque to us), so pass 1's output goes through LDS.
//
// Thread block: K waves × 64 threads = 1024 at K=16.  One block per tensor.
// LDS budget: buf (K × K × (K+1) padded) + B cache (K × K)  = 36 KB at K=16.

#if defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__)
namespace rw = rocwmma;

// Fragment types for f64 16×16×4 MFMA on GFX90A.
using FragA    = rw::fragment<rw::matrix_a,    16, 16, 4, double, rw::row_major>;
using FragAT   = rw::fragment<rw::matrix_a,    16, 16, 4, double, rw::col_major>; // used to load B as B^T
using FragB    = rw::fragment<rw::matrix_b,    16, 16, 4, double, rw::row_major>;
using FragAcc  = rw::fragment<rw::accumulator, 16, 16, 4, double>;
#endif

template<typename T, int K>
__global__
__launch_bounds__(K * 64, 1)
void transform_kernel_blocked_rocwmma(int nfuncs,
                                      const T* __restrict__ A,
                                      const T* __restrict__ B,
                                      T* __restrict__ C,
                                      T* __restrict__ /*workspace unused*/)
{
#if defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__)
    static_assert(std::is_same<T, double>::value, "rocWMMA path is FP64 only");
    static_assert(K == 16, "rocWMMA path: K=16 only for now");

    constexpr int K2 = K * K;
    constexpr int K3 = K * K * K;
    constexpr int ELEMS_PER_LANE = K2 / 64;
    constexpr int NMFMA = K / 4;

    // Bank-conflict pad identical to L7.
    constexpr int BK = K + 1;
    constexpr int WB = K * BK;

    extern __shared__ unsigned char _smem_rwmma[];
    T* buf   = reinterpret_cast<T*>(_smem_rwmma);
    T* B_lds = buf + K * WB;

    const int s    = threadIdx.y;    // wave index (0..K-1)
    const int t    = threadIdx.x;    // lane 0..63
    const int tid  = s * 64 + t;

    // --- Cache B into LDS (wide double4 loads, like L7) ---
    static_assert(K2 % 4 == 0);
    if (tid < K2 / 4) {
        const double4* B_vec     = reinterpret_cast<const double4*>(B);
        double4*       B_lds_vec = reinterpret_cast<double4*>(B_lds);
        B_lds_vec[tid] = B_vec[tid];
    }
    __syncthreads();

    // One block = one tensor.
    const int cube = blockIdx.x;
    if (cube >= nfuncs) return;
    const T* a_ptr = A + (size_t)cube * K3;
    T*       c_ptr = C + (size_t)cube * K3;

    // --- Distribute: coalesced HBM load into per-wave layout ---
    // Each thread loads 4 contiguous HBM doubles as a double4, placing them
    // across 4 WAVES (since consecutive HBM elements have consecutive k, and
    // wave s owns k=s).  Per-wave layout: buf[s*WB + i*BK + j] = A[i, j, s].
    {
        static_assert(K == 16);
        const double4* a_ptr_vec = reinterpret_cast<const double4*>(a_ptr);
        const int base_idx = 4 * tid;
        const int i        =  base_idx >> 8;
        const int j        = (base_idx >> 4) & 15;
        const int k_start  =  base_idx &  15;  // 0, 4, 8, or 12

        double4 v = a_ptr_vec[tid];
        // 4 cross-wave stores (destination wave = k_start + e for e=0..3).
        buf[(k_start + 0) * WB + i * BK + j] = v.x;
        buf[(k_start + 1) * WB + i * BK + j] = v.y;
        buf[(k_start + 2) * WB + i * BK + j] = v.z;
        buf[(k_start + 3) * WB + i * BK + j] = v.w;
    }
    __syncthreads();

    // --- Pass 1 (SWAPPED operands): compute blk^T · B ---
    // A_frag = blk^T: col_major load of the per-wave blk region.
    // B_frag = B: row_major load.
    // Output D[j][a] = temp1[a][j].  In the fragment, thread t's acc1.x[e] =
    // temp1[t%16][(t/16) + 4e] = temp1[t%16][p*4 + t/16]  (commutative).
    // That is EXACTLY the value pass 2's matrix_a fragment wants at iter p=e,
    // so we can feed acc1 straight into pass 2 without any LDS round-trip.
    FragAcc acc1;
    rw::fill_fragment(acc1, 0.0);
    #pragma unroll
    for (int p = 0; p < NMFMA; ++p) {
        FragAT a_frag;                   // col_major → blk^T
        FragB  b_frag;                   // B
        rw::load_matrix_sync(a_frag, &buf[s * WB + p * 4 * BK],   BK);
        rw::load_matrix_sync(b_frag, &B_lds[p * 4 * K],           K);
        rw::mma_sync(acc1, a_frag, b_frag, acc1);
    }
    // No LDS writeback.  acc1.x[p] is pass 2's a_frag value at iter p.

    // --- Pass 2: temp2 = temp1 · B  (consumes acc1 directly) ---
    FragAcc acc2;
    rw::fill_fragment(acc2, 0.0);
    #pragma unroll
    for (int p = 0; p < NMFMA; ++p) {
        FragA a_frag_p;
        // Populate matrix_a's per-lane element from acc1's p-th acc value.
        a_frag_p.x[0] = acc1.x[p];
        FragB b_frag;
        rw::load_matrix_sync(b_frag, &B_lds[p * 4 * K], K);
        rw::mma_sync(acc2, a_frag_p, b_frag, acc2);
    }
    // acc2 holds pass 2 output (temp2) in the same per-lane layout as L7's
    // acc2 — so acc2.x[e] IS the corner-turn stash for iter e.

    // --- Corner turn: direct cross-wave write from acc2 (no stash read) ---
    __syncthreads();  // all waves done with pass-1 reads from own region
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; ++e) {
        int idx = t + e * 64;
        int a_ix = idx / K;
        int b_ix = idx % K;
        buf[a_ix * WB + b_ix * BK + s] = acc2.x[e];
    }
    __syncthreads();

    // --- Pass 3: result = temp2_reshuffled · B ---
    // Wave s now represents a=s.  buf[s*WB + b*BK + k'] = temp2[a=s, b, k'].
    FragAcc acc3;
    rw::fill_fragment(acc3, 0.0);
    #pragma unroll
    for (int p = 0; p < NMFMA; ++p) {
        FragA a_frag;
        FragB b_frag;
        rw::load_matrix_sync(a_frag, &buf[s * WB + p * 4],        BK);
        rw::load_matrix_sync(b_frag, &B_lds[p * 4 * K],           K);
        rw::mma_sync(acc3, a_frag, b_frag, acc3);
    }

    // --- Store to HBM ---
    // Canonical C[a*K² + b*K + c] with a=s.
    rw::store_matrix_sync(&c_ptr[(size_t)s * K2], acc3, K, rw::mem_row_major);
#endif // __HIP__ && __HIP_DEVICE_COMPILE__
}

template<typename T>
inline size_type blocked_rocwmma_shmem_size(int K) {
    return (size_type)((K * K * (K + 1) + K * K) * sizeof(T));
}

template<typename T>
inline Dim3 blocked_rocwmma_blockdim(int K) {
    return Dim3(64, K, 1);
}

template<typename T>
inline void submit_transform_bench_blocked_rocwmma(int nfuncs, int /*nblocks*/, int K,
                                                   const T* A, const T* B, T* C, T* workspace,
                                                   Stream stream)
{
    if (K == 16) {
        constexpr int Kv = 16;
        Dim3      td   = blocked_rocwmma_blockdim<T>(Kv);
        size_type smem = blocked_rocwmma_shmem_size<T>(Kv);
        CONFIGURE_KERNEL((transform_kernel_blocked_rocwmma<T, Kv>), smem);
        CALL_KERNEL((transform_kernel_blocked_rocwmma<T, Kv>), nfuncs, td, smem, stream,
                    (nfuncs, A, B, C, workspace));
    } else {
        fprintf(stderr, "blocked rocwmma transform: K=%d not supported (K=16 only)\n", K);
        assert(false);
    }
}

#endif // __HIP__

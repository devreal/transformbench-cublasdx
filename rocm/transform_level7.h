#pragma once

#include "util.h"
#include "mxm_level7.h"

/**
 * Transform wrapper for Level 7.
 *
 * Unlike levels 1-6, the three-GEMM chain is executed inside a single call to
 * mTxmq_level7_k because B must remain resident in VGPRs across all three GEMMs.
 * A single LDS buffer of K*(K²+1) doubles is reused across all three GEMMs; only
 * the final output is written to global memory C.
 *
 * For K=16: LDS = 16*257*8 = 32,864 bytes (~32 KB), well within the 64 KB limit.
 * occupancy=1 is retained to maximise VGPR headroom for B and A register arrays.
 */

template <typename T, int K>
LAUNCH_BOUNDS(256, 1)
__global__ void transform_kernel_level7_k(int nfuncs,
                                           const T* A, const T* B, T* C)
{
    constexpr int K3 = K * K * K;
    for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
        const T* a = A + i * K3;
        T*       c = C + i * K3;
        mra::mTxmq_level7_k<T, K>(c, a, B);
    }
}

template <typename T>
inline int transform_level7_shmem_size(int K) {
    return (int)mra::mTxmq_level7_shmem_size<T>(K);
}

template <typename T>
inline Dim3 transform_level7_blockdim(int K) {
    return mra::mTxmq_level7_blockdim<T>(K);
}

template <typename T>
inline void submit_transform_level7_bench(int nfuncs, int nblocks, int K,
                                          const T* A, const T* B, T* C, T* /*workspace*/,
                                          Stream stream)
{
    Dim3 thread_dims = mra::mTxmq_level7_blockdim<T>(K);
    int  smem_size   = transform_level7_shmem_size<T>(K);

#define DISPATCH_L7(Kval) \
    case Kval: \
        CONFIGURE_KERNEL((transform_kernel_level7_k<T, Kval>), smem_size); \
        CALL_KERNEL((transform_kernel_level7_k<T, Kval>), std::min(nfuncs, nblocks), \
                    thread_dims, smem_size, stream, \
                    (nfuncs, A, B, C)); \
        break;

    switch (K) {
        DISPATCH_L7( 8)
        DISPATCH_L7(12)
        DISPATCH_L7(16)
        DISPATCH_L7(20)
        DISPATCH_L7(32)
        default:
            printf("submit_transform_level7_bench: unsupported K=%d\n", K);
    }
#undef DISPATCH_L7
}

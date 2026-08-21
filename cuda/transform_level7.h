#pragma once

#include "util.h"
#include "mxm_level7.h"
#include "transform_level3.h"

/**
 * Transform wrapper for Level 7.
 *
 * Unlike levels 1-6 the three-GEMM chain runs inside a single call to
 * mTxmq_level7_k, because B has to stay in tensor-core registers across all
 * three GEMMs.  One shared buffer of K^3 doubles is reused in place via the
 * pointer trick; only the final output reaches global memory C.
 *
 * For K=16: shared memory = 16^3 * 8 = 32,768 bytes.  occupancy=1 is retained
 * to leave register headroom for the B fragments and the per-warp A partition.
 *
 * K values outside {8, 16} are dispatched to the level-3 kernel by the submit
 * function below rather than being handled in-kernel, so the fallback keeps
 * level-3's own block and shared-memory configuration.
 */

template <typename T, int K>
LAUNCH_BOUNDS(mra::detail::LEVEL7_NTHREAD, 1)
__global__ void transform_kernel_level7_k(int nfuncs,
                                          const T* A, const T* B, T* C, T* workspace)
{
    constexpr int K3 = K * K * K;
    T* w = workspace + blockIdx.x * K3;
    for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
        const T* a = A + i * K3;
        T*       c = C + i * K3;
        mra::mTxmq_level7_k<T, K>(c, a, B, w);
    }
}

template <typename T>
inline int transform_level7_shmem_size(int K) {
    return (int)mra::mTxmq_level7_shmem_size<T>(K);
}

template <typename T>
inline Dim3 transform_level7_blockdim(int K) {
    if (MRA_DMMA_SUPPORTED && mra::detail::level7_supports_k(K)) {
        return mra::mTxmq_level7_blockdim<T>(K);
    }
    return mra::mTxmq_level3_blockdim<T>(K);
}

template <typename T>
inline void submit_transform_level7_bench(int nfuncs, int nblocks, int K,
                                          const T* A, const T* B, T* C, T* workspace,
                                          Stream stream)
{
    /* K outside the register-resident-B range runs the level-3 kernel, which
     * brings its own block dim and shared-memory size. */
    if (!MRA_DMMA_SUPPORTED || !mra::detail::level7_supports_k(K)) {
        submit_transform_level3_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, stream);
        return;
    }

    Dim3 thread_dims = mra::mTxmq_level7_blockdim<T>(K);
    int  smem_size   = transform_level7_shmem_size<T>(K);

#define DISPATCH_L7(Kval) \
    case Kval: \
        CONFIGURE_KERNEL((transform_kernel_level7_k<T, Kval>), smem_size); \
        CALL_KERNEL((transform_kernel_level7_k<T, Kval>), std::min(nfuncs, nblocks), \
                    thread_dims, smem_size, stream, \
                    (nfuncs, A, B, C, workspace)); \
        break;

    switch (K) {
        DISPATCH_L7( 8)
        DISPATCH_L7(16)
        default:
            printf("submit_transform_level7_bench: unsupported K=%d\n", K);
    }
#undef DISPATCH_L7
}

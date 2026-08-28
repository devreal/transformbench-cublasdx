#pragma once

/**
 * Level 8 - Kronecker product GEMM (cuBLAS).
 *
 * MATHEMATICAL BACKGROUND
 * -----------------------
 * The standard 3-pass transform applies B^T along each mode of a K x K x K tensor:
 *
 *   Pass 1: T1[j0,i1,i2] = SUM_{i0} A[i0,i1,i2] * B[i0,j0]   (contract mode 0)
 *   Pass 2: T2[j0,j1,i2] = SUM_{i1} T1[j0,i1,i2] * B[i1,j1]  (contract mode 1)
 *   Pass 3: C [j0,j1,j2] = SUM_{i2} T2[j0,j1,i2] * B[i2,j2]  (contract mode 2)
 *
 * Vectorising the tensor (flattening to K^3 elements) turns this into a single
 * matrix-vector product:
 *
 *   vec(C) = KronMat * vec(A)
 *
 * where KronMat = B^T (x) B^T (x) B^T is the three-fold Kronecker product
 * (a K^3 x K^3 matrix).  Each entry is:
 *
 *   KronMat[b, a] = B[a%K][b%K] * B[(a/K)%K][(b/K)%K] * B[a/K^2][b/K^2]
 *
 * with a = input linear index  (i0 + K*i1 + K^2*i2)
 *      b = output linear index (j0 + K*j1 + K^2*j2)
 *
 * IMPLEMENTATION
 * --------------
 * 1. build_kron_kernel - one GPU thread per (b, a) entry; called ONCE before
 *                        the timing loop and cached for all subsequent batches.
 *
 * 2. submit_transform_kron_bench - calls cublasDgemm:
 *
 *      C [K^3 x nfuncs] = KronMat [K^3 x K^3] x A [K^3 x nfuncs]
 *
 *    Tensors are stored contiguously (tensor f occupies A[f*K^3 .. (f+1)*K^3-1]),
 *    so the batch dimension maps naturally to GEMM columns.
 *
 * TRADE-OFFS
 * ----------
 *   Pros
 *     - Single API call into a fully tuned library GEMM; one large DGEMM keeps
 *       the SMs busy where the per-tensor kernels of L1-L7 cannot.
 *
 *   Cons
 *     - KronMat memory = K^6 x 8 bytes: 6 MB at K=10, 128 MB at K=16,
 *       512 MB at K=20 - impractical beyond K ~ 16.
 *     - FLOPs reported are 2*K^6*N (actual GEMM work), not the 3*2*K^4*N
 *       mathematical minimum, so raw GFlop/s are not directly comparable
 *       to L1-L7 or L9/L10.
 */

#include "util.h"
#include <cublas_v2.h>

using blasHandle_t = cublasHandle_t;
#define BLAS_OP_N     CUBLAS_OP_N
#define blasCreate    cublasCreate
#define blasDestroy   cublasDestroy
#define blasSetStream cublasSetStream
#define blasDgemm     cublasDgemm

// ---------------------------------------------------------------------------
// Kernel: build the K^3 x K^3 Kronecker product matrix (column-major).
//
//   KronMat[I, J] = B^T[i0,j0] * B^T[i1,j1] * B^T[i2,j2]
//
// Index decomposition (first index fastest = column-major vector):
//   I = i0 + K*i1 + K^2*i2
//   J = j0 + K*j1 + K^2*j2
//
// B is row-major K x K, so B^T[i,j] = B[j*K + i].
// ---------------------------------------------------------------------------
template <typename T>
__global__ void build_kron_kernel(int K, const T* __restrict__ B,
                                   T* __restrict__ KronMat)
{
    const int K3 = K * K * K;
    const int I  = blockIdx.x * blockDim.x + threadIdx.x;
    const int J  = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= K3 || J >= K3) return;

    const int i0 = I % K,        j0 = J % K;
    const int i1 = (I / K) % K,  j1 = (J / K) % K;
    const int i2 = I / (K * K),  j2 = J / (K * K);

    // B^T[i,j] = B[j*K + i]  (B is row-major)
    KronMat[(size_t)I + (size_t)J * K3] = B[j0*K + i0] * B[j1*K + i1] * B[j2*K + i2];
}

// ---------------------------------------------------------------------------
// Build the Kronecker matrix on the device (call once before timing).
// KronMat must already be allocated with K^3 x K^3 elements.
// ---------------------------------------------------------------------------
template <typename T>
inline void build_kron_matrix(int K, const T* B_dev, T* KronMat_dev,
                               Stream stream)
{
    const int K3 = K * K * K;
    dim3 block(16, 16);
    dim3 grid((K3 + 15) / 16, (K3 + 15) / 16);
    CALL_KERNEL(build_kron_kernel<T>, grid, block, 0, stream,
                (K, B_dev, KronMat_dev));
}

// ---------------------------------------------------------------------------
// Submit one round of the Kronecker GEMM (called inside the timing loop).
//
//   C[K^3 x nfuncs] = KronMat[K^3 x K^3] x A[K^3 x nfuncs]
//
// A and C are treated as column-major (each contiguous K^3-block = one tensor).
// ---------------------------------------------------------------------------
template <typename T>
inline void submit_transform_kron_bench(int nfuncs, int K,
                                         const T* A, const T* KronMat, T* C,
                                         blasHandle_t blas_handle,
                                         Stream stream)
{
    const int K3 = K * K * K;
    const double alpha = 1.0, beta = 0.0;
    blasSetStream(blas_handle, stream);
    blasDgemm(blas_handle,
              BLAS_OP_N, BLAS_OP_N,
              K3, nfuncs, K3,
              &alpha,
              KronMat, K3,
              A,       K3,
              &beta,
              C,       K3);
}

// Required by the benchmark dispatch (values are unused for the Kronecker level).
template <typename T>
inline int kron_shmem_size(int /*K*/) { return 0; }

inline Dim3 kron_blockdim(int /*K*/) { return {1, 1, 1}; }

#pragma once


#include "util.h"
#include "transform.h"
/*****************************************
 * Level-5 transform function using ROCWMMA.
 * Follows the same structure as transform.h / transform_cublasdx.h.
 *****************************************/

static constexpr int WM = 16, WN = 16, WK = 4;
static constexpr int WAVE     = 64;   // CDNA wavefront size
static constexpr int K_CHUNK  = 16;   // k-output tile size for Phase 3

template<int N>
struct C {
    static constexpr int NP     = ((N + WM - 1) / WM) * WM;
    static constexpr int TILES  = NP / WM;               // tiles per dim
    static constexpr int KT     = NP / WK;               // K-tiles per GEMM
    static constexpr int BSIZE  = TILES * TILES * WAVE;  // threads per block
    static constexpr int KCHUNKS = (N + K_CHUNK - 1) / K_CHUNK;
    static constexpr int PAIRS  = (WM * WN) / WAVE;      // = 4, always
};



#if defined(__HIP_DEVICE_COMPILE__)

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

template<int N>
__device__ void
transform3d_rocwmma(double* __restrict__ result,
                    const double* __restrict__ t,
                    const double* __restrict__ c)
{
    constexpr int NP     = C<N>::NP;
    constexpr int TILES  = C<N>::TILES;
    constexpr int KT     = C<N>::KT;
    constexpr int KCHUNKS = C<N>::KCHUNKS;
    constexpr int PAIRS  = C<N>::PAIRS;
    extern __shared__ double shared_mem[];  // dynamically allocated shared memory
    double *s_c   = shared_mem;  // transform matrix, permanent
    double *s_work = shared_mem + NP * NP;  // reused: t-slice → u → w

    // Phase 1: A = c^T via col_major load of c (transposes it)
    using FragA1 = rocwmma::fragment<rocwmma::matrix_a,   WM, WN, WK, double, rocwmma::col_major>;
    using FragB1 = rocwmma::fragment<rocwmma::matrix_b,   WM, WN, WK, double, rocwmma::row_major>;
    // Phase 2: A = u (row_major), B = c (row_major, no transpose)
    using FragA2 = rocwmma::fragment<rocwmma::matrix_a,   WM, WN, WK, double, rocwmma::row_major>;
    using FragB2 = rocwmma::fragment<rocwmma::matrix_b,   WM, WN, WK, double, rocwmma::row_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator, WM, WN, WK, double>;

    const int wid = threadIdx.x / WAVE;
    const int wm  = wid / TILES;
    const int wn  = wid % TILES;

    // Load c into shared memory, zero-pad beyond N
    for (int idx = threadIdx.x; idx < NP * NP; idx += blockDim.x) {
        int ip = idx / NP, i = idx % NP;
        s_c[idx] = (ip < N && i < N) ? c[ip * N + i] : 0.0;
    }
    __syncthreads();

    // Outer k-chunk loop: controls register usage for Phase 3
    for (int kchunk = 0; kchunk < KCHUNKS; ++kchunk) {

        const int k0 = kchunk * K_CHUNK;
        const int k1 = min(k0 + K_CHUNK, N);

        double acc[PAIRS][K_CHUNK] = {};   // 4×16 = 64 fp64 per thread

        for (int kp = 0; kp < N; ++kp) {

            // Load t[:, :, kp] into s_work (zero-pad beyond N)
            // Note: t[ip, jp, kp] has stride N in the jp direction;
            // cache in shared memory so WMMA loads are coalesced.
            for (int idx = threadIdx.x; idx < NP * NP; idx += blockDim.x) {
                int ip = idx / NP, jp = idx % NP;
                s_work[idx] = (ip < N && jp < N) ? t[ip*N*N + jp*N + kp] : 0.0;
            }
            __syncthreads();

            // --- Phase 1: u = c^T @ t_slice ---
            // col_major load of s_c gives c^T as matrix_a
            FragAcc u_frag;
            rocwmma::fill_fragment(u_frag, 0.0);
            for (int kt = 0; kt < KT; ++kt) {
                FragA1 a_frag;  FragB1 b_frag;
                rocwmma::load_matrix_sync(a_frag,
                    s_c    + wm*WM    + kt*WK*NP, NP);  // c^T tile (wm, kt)
                rocwmma::load_matrix_sync(b_frag,
                    s_work + kt*WK*NP + wn*WN,    NP);  // t tile (kt, wn)
                rocwmma::mma_sync(u_frag, a_frag, b_frag, u_frag);
            }
            rocwmma::store_matrix_sync(s_work + wm*WM*NP + wn*WN,
                                        u_frag, NP, rocwmma::mem_row_major);
            __syncthreads();

            // --- Phase 2: w = u @ c ---
            FragAcc w_frag;
            rocwmma::fill_fragment(w_frag, 0.0);
            for (int kt = 0; kt < KT; ++kt) {
                FragA2 a_frag;  FragB2 b_frag;
                rocwmma::load_matrix_sync(a_frag,
                    s_work + wm*WM*NP + kt*WK, NP);   // u tile (wm, kt)
                rocwmma::load_matrix_sync(b_frag,
                    s_c    + kt*WK*NP + wn*WN, NP);   // c tile (kt, wn)
                rocwmma::mma_sync(w_frag, a_frag, b_frag, w_frag);
            }
            rocwmma::store_matrix_sync(s_work + wm*WM*NP + wn*WN,
                                        w_frag, NP, rocwmma::mem_row_major);
            __syncthreads();

            // --- Phase 3: acc[p][k-k0] += w[i,j] * c[kp, k] for k in [k0, k1) ---
            for (int p = 0; p < PAIRS; ++p) {
                int ij = threadIdx.x + p * (int)blockDim.x;
                if (ij < NP * NP) {
                    int i = ij / NP, j = ij % NP;
                    if (i < N && j < N) {
                        double w_ij = s_work[ij];
                        for (int k = k0; k < k1; ++k)
                            acc[p][k - k0] += w_ij * s_c[kp * NP + k];
                    }
                }
            }
            __syncthreads();

        } // end kp loop

        // Write this k-chunk of result to global memory
        for (int p = 0; p < PAIRS; ++p) {
            int ij = threadIdx.x + p * (int)blockDim.x;
            if (ij < NP * NP) {
                int i = ij / NP, j = ij % NP;
                if (i < N && j < N)
                    for (int k = k0; k < k1; ++k)
                        result[i*N*N + j*N + k] = acc[p][k - k0];
            }
        }

    } // end kchunk loop
}

// Dispatch

__device__
void transform3d(int n, const double* t, const double* c, double* result, double* workspace)
{
  // Launches one block; grid = 1 since this is a single-transform kernel.
  // For batched transforms, grid > 1 and pass a stride/batch index.
  #define LAUNCH(NN) \
      transform3d_rocwmma<NN>(result, t, c)

  switch (n) {
    case  6: LAUNCH( 6); break;   case  8: LAUNCH( 8); break;
    case 10: LAUNCH(10); break;   case 12: LAUNCH(12); break;
    case 16: LAUNCH(16); break;   case 20: LAUNCH(20); break;
    case 24: LAUNCH(24); break;   case 32: LAUNCH(32); break;
    default: /* fallback to scalar kernel */ break;
  }
  #undef LAUNCH
}

#endif // __HIP_DEVICE_COMPILE__

__device__
void transform3d(int n, const double* t, const double* c, double* result, double* workspace);

template <typename T>
LAUNCH_BOUNDS(1024, 4)
__global__ void transform_kernel_level5(int nfuncs, int K,
                                        const T* A, const T* B, T* C, T* workspace) {
  const int K2NDIM = K * K * K;
  T* w = workspace + blockIdx.x * K2NDIM;
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    const T* a = A + i * K2NDIM;
    T* c = C + i * K2NDIM;
    transform3d(K, a, B, c, w);
  }
}

/**
  ┌─────────┬────────────┬────────────┐
  │ n (= N) │ blockDim.x │ Wavefronts │
  ├─────────┼────────────┼────────────┤
  │ 6       │ 384        │ 6          │
  ├─────────┼────────────┼────────────┤
  │ 8       │ 512        │ 8          │
  ├─────────┼────────────┼────────────┤
  │ 10      │ 640        │ 10         │
  ├─────────┼────────────┼────────────┤
  │ 12      │ 768        │ 12         │
  ├─────────┼────────────┼────────────┤
  │ 16      │ 1024       │ 16         │
  └─────────┴────────────┴────────────┘
 */

template<typename T>
inline Dim3 transform_level5_thread_num(size_type K) {
  switch (K) {
    case  6: return {384, 1, 1};   case  8: return {512, 1, 1};
    case 10: return {640, 1, 1};   case 12: return {768, 1, 1};
    case 16: return {1024, 1, 1};  default: return {1024, 1, 1};
  }
}

template <typename T>
inline int transform_level5_shmem_size(int K) {
  const int NP  = ((K + WM - 1) / WM) * WM;
  auto smem_size   = 2*NP*NP * sizeof(T);  // for s_c and s_work
  return smem_size;
}

template <typename T>
inline void submit_transform_level5_bench(int nfuncs, int nblocks, int K,
                                          const T* a, const T* b, T* c, T* workspace,
                                          Stream stream)
  {
  Dim3 thread_dims = transform_level5_thread_num<T>(K);
  int smem_size = transform_level5_shmem_size<T>(K);
  CONFIGURE_KERNEL(transform_kernel_level5<T>, smem_size);
  CALL_KERNEL(transform_kernel_level5<T>, std::min(nfuncs, nblocks),
              thread_dims, smem_size, stream,
              (nfuncs, K, a, b, c, workspace));
}



#include <iostream>
#include <chrono>

#include "transform.h"
#include "transform_cublasdx.h"
#include "transform_level2.h"
#include "transform_level3.h"
#include "transform_level4.h"
#include "transform_level5.h"
#include "transform_rocwmma.h"
#include "transform_level7.h"
#include "transform_kron.h"
#include "mxm_cublasdx.h"
#include "util.h"

/**
 * Optimization levels:
 *   1 - L1: thread-parallel over j, serial k-loop, all global memory (mxm.h fallback)
 *   2 - L2: B in LDS, threads distributed over rows
 *   3 - L3: B in LDS + register accumulation (acc[K] in VGPRs)
 *   4 - L4: AMD MFMA (GFX90A/GFX940) for K=16,32; falls back to L3 elsewhere
 *   5 - L5: cuBLASDx (NVIDIA only, double-buffered block GEMM with Tensor Cores)
 *   6 - L6: Single GEMM via K³×K³ Kronecker product (B^T ⊗ B^T ⊗ B^T)
 */

template<typename T>
void transform_bench(int nreps, int ntasks, int nfuncs, int nblocks, int K, int level, int num_streams) {

  std::vector<Stream> streams(num_streams); // PaRSEC uses 4 streams by default
  T* A, *B, *C, *workspace;
  MALLOC(&A, nfuncs * K * K * K * sizeof(T)); // N x KxKxK tensors
  MALLOC(&B, K * K * sizeof(T)); // KxK matrix
  MALLOC(&C, nfuncs * K * K * K * sizeof(T)); // N x KxKxK tensors
  MALLOC(&workspace, nblocks * K * K * K * sizeof(T)); // per-block scratch

  for (int i = 0; i < num_streams; ++i) {
    CREATE_STREAM(&streams[i]);
  }

  /* Warn early if a level is unavailable */
  if (level == 5 && !MRA_HAVE_CUBLASDX) {
    std::cerr << "Warning: level 5 (cuBLASDx) requested but not available; "
                 "falling back to level 1\n";
    level = 1;
  }

  /* Resolve default level */
  if (level <= 0) {
    level = (MRA_HAVE_CUBLASDX) ? 5 : 3;
  }

  const char* level_names[] = {
    "",           /* unused [0] */
    "L1-global",  /* 1 */
    "L2-lds_b",   /* 2 */
    "L3-regblk",  /* 3 */
    "L4-mfma",    /* 4 */
    "L5",/* 5 */
    "L6-rocwmma",/* 6 */
    "L7-builtins",/* 7 */
    "L8-kron"     /* 8 */
  };

  /* Print shmem and thread dims for this level */
  int smem_size = 0;
  Dim3 thread_dims = {1, 1, 1};
  switch (level) {
    case 1:
      smem_size   = mra::mTxmq_shmem_size<T>(K);
      thread_dims = mra::mTxmq_blockdim<T>(K);
      break;
    case 2:
      smem_size   = transform_level2_shmem_size<T>(K);
      thread_dims = mra::mTxmq_level2_blockdim<T>(K);
      break;
    case 3:
      smem_size   = transform_level3_shmem_size<T>(K);
      thread_dims = mra::mTxmq_level3_blockdim<T>(K);
      break;
    case 4:
      smem_size   = transform_level4_shmem_size<T>(K);
      thread_dims = mra::mTxmq_level4_blockdim<T>(K);
      break;
    case 5:
      smem_size   = transform_level5_shmem_size<T>(K);
      thread_dims = mra::mTxmq_blockdim<T>(K);
      break;
    case 6:
      smem_size   = transform_rocwmma_shmem_size<T>(K);
      thread_dims = transform_rocwmma_blockdim<T>(K);
      break;
    case 7:
      smem_size   = transform_level7_shmem_size<T>(K);
      thread_dims = transform_level7_blockdim<T>(K);
      break;
    case 8:
      smem_size   = kron_shmem_size<T>(K);
      thread_dims = kron_blockdim(K);
      break;
  }

  /* Level 8: build Kronecker matrix once, before the timing loop */
  T* KronMat = nullptr;
  blasHandle_t blas_handle{};
  if (level == 8) {
    const int K3 = K * K * K;
    const size_t kron_bytes = (size_t)K3 * K3 * sizeof(T);
    std::cout << "L8-kron: allocating " << kron_bytes / (1024*1024.0)
              << " MB for " << K3 << "x" << K3 << " Kronecker matrix\n";
    MALLOC(&KronMat, kron_bytes);
    blasCreate(&blas_handle);
    build_kron_matrix<T>(K, B, KronMat, streams[0]);
    SYNC_STREAM(streams[0]);
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  for (int i = 0; i < nreps+1; ++i) {
    beg = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < ntasks; ++t) {
      switch (level) {
        case 1:
          submit_transform_bench(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 2:
          submit_transform_level2_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 3:
          submit_transform_level3_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 4:
          submit_transform_level4_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 5:
          submit_transform_level5_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 6:
          submit_transform_rocwmma_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 7:
          submit_transform_level7_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 8:
          submit_transform_kron_bench<T>(nfuncs, K, A, KronMat, C, blas_handle, streams[t%num_streams]);
          break;
      }
    }
    for (int t = 0; t < num_streams; ++t) {
      SYNC_STREAM(streams[t]);
    }
    end = std::chrono::high_resolution_clock::now();

    /* skip warm-up */
    if (i > 0) {
      auto us = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
      /* L8 does one K³×K³ GEMM per task (2·K⁶ FLOPs); others do 3 passes (2·3·K⁴ FLOPs) */
      uint64_t flops = (level == 8)
          ? (uint64_t)ntasks * 2 * (uint64_t)K*K*K * (uint64_t)K*K*K * nfuncs
          : (uint64_t)ntasks * K * K * K * K * 3 * 2 * nfuncs;
      std::cout << "Transform"
                << ";level=" << level_names[level]
                << ";nfuncs=" << nfuncs
                << ";nblocks=" << nblocks
                << ";K=" << K
                << ";tasks=" << ntasks
                << ";threads={" << thread_dims.x << "," << thread_dims.y << "," << thread_dims.z << "}"
                << ";smem=" << smem_size
                << ";Time(us)=" << us
                << ";GFlop=" << flops*1e-9
                << ";Gflop/s=" << (1e-3 * flops) / us
                << std::endl;
    }
  }

  if (level == 8) {
    blasDestroy(blas_handle);
    FREE(KronMat);
  }

  FREE(A);
  FREE(B);
  FREE(C);
  FREE(workspace);
}

int main(int argc, char **argv) {

  auto opt = OptionParser(argc, argv);

  int nreps  = opt.parse("-r", 5);
  int ntasks = opt.parse("-n", 500);
  int N      = opt.parse("-N", 2048);  /* number of functions */
  int K      = opt.parse("-K", 16);   /* number of coefficients */
  int M      = opt.parse("-M", 512);  /* max number of blocks */
  int level  = opt.parse("-l", 0);    /* 0 = auto, 1-5 = explicit */
  int num_streams = opt.parse("-s", 4); /* number of concurrent streams to use */

  /* Legacy -m flag: force level 1 */
  if (opt.exists("-m")) level = 1;

  std::cout << "Running benchmark"
            << " nreps=" << nreps
            << " ntasks=" << ntasks
            << " N=" << N
            << " K=" << K
            << " M=" << M
            << " level=" << (level <= 0 ? (MRA_HAVE_CUBLASDX ? 5 : 3) : level)
            << std::endl;

  transform_bench<double>(nreps, ntasks, N, M, K, level, num_streams);
}

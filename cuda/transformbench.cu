#include <iostream>
#include <chrono>
#include <vector>

#include "util.h"
#include "transform.h"                 // L1
#include "transform_level2.h"          // L2
#include "transform_level3.h"          // L3
#include "transform_level4.h"          // L4
#include "transform_level5.h"          // L5
#include "transform_wmma.h"            // L6
#include "transform_level7.h"          // L7
#include "transform_kron.h"            // L8
#include "transform_cublasdx.h"        // L9
#include "transform_cublasdx_mxm.h"    // L10

/**
 * Optimization levels (CUDA):
 *   1  - L1: thread-parallel over j, serial k-loop, all global memory (reference)
 *   2  - L2: B in shared memory, threads distributed over rows
 *   3  - L3: B in shared memory + register accumulation (acc[K] in registers)
 *   4  - L4: FP64 tensor cores (mma.sync.m8n8k4.f64), one warp; L3 fallback
 *   5  - L5: FP64 tensor cores, A staged through shared memory, 8 warps
 *   6  - L6: nvcuda::wmma, one warp per output tile
 *   7  - L7: FP64 tensor cores with B resident in registers across all 3 GEMMs
 *   8  - L8: single K^3 x K^3 DGEMM via the Kronecker product (cuBLAS)
 *   9  - L9: cuBLASDx, all three GEMMs fused in shared memory
 *  10  - L10: cuBLASDx as a per-pass block GEMM
 *
 * Levels 4-7 need sm_80 or newer; on older targets they fall back to L3.
 * Levels 9 and 10 need cuBLASDx headers at build time.
 */

template<typename T>
void transform_bench(int nreps, int ntasks, int nfuncs, int nblocks, int K, int level, int num_streams) {

  std::vector<Stream> streams(num_streams); // PaRSEC uses 4 streams by default
  T* A, *B, *C, *workspace;
  MALLOC(&A, (size_t)nfuncs * K * K * K * sizeof(T)); // N x KxKxK tensors
  MALLOC(&B, (size_t)K * K * sizeof(T));              // KxK matrix
  MALLOC(&C, (size_t)nfuncs * K * K * K * sizeof(T)); // N x KxKxK tensors
  MALLOC(&workspace, (size_t)nblocks * K * K * K * sizeof(T)); // per-block scratch

  for (int i = 0; i < num_streams; ++i) {
    CREATE_STREAM(&streams[i]);
  }

  /* Warn early if a level is unavailable */
  if ((level == 9 || level == 10) && !MRA_HAVE_CUBLASDX) {
    std::cerr << "Warning: level " << level << " (cuBLASDx) requested but cuBLASDx "
                 "was not found at build time; falling back to level 3\n";
    level = 3;
  }
  if (level >= 4 && level <= 7 && !MRA_DMMA_SUPPORTED) {
    std::cerr << "Warning: level " << level << " needs FP64 tensor cores (sm_80+); "
                 "this build targets sm_" << MRA_CUDA_ARCH
              << ", so the kernels will run their level-3 fallback\n";
  }

  /* Resolve default level */
  if (level <= 0) {
    level = (MRA_HAVE_CUBLASDX) ? 9 : 3;
  }

  const char* level_names[] = {
    "",                  /* unused [0]  */
    "L1-global",         /* 1  */
    "L2-smem_b",         /* 2  */
    "L3-regblk",         /* 3  */
    "L4-dmma",           /* 4  */
    "L5-dmma-staged",    /* 5  */
    "L6-wmma",           /* 6  */
    "L7-dmma-breg",      /* 7  */
    "L8-kron",           /* 8  */
    "L9-cublasdx",       /* 9  */
    "L10-cublasdx-mxm"   /* 10 */
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
      thread_dims = transform_level4_blockdim<T>(K);
      break;
    case 5:
      smem_size   = transform_level5_shmem_size<T>(K);
      thread_dims = transform_level5_blockdim<T>(K);
      break;
    case 6:
      smem_size   = transform_wmma_shmem_size<T>(K);
      thread_dims = transform_wmma_blockdim<T>(K);
      break;
    case 7:
      smem_size   = transform_level7_shmem_size<T>(K);
      thread_dims = transform_level7_blockdim<T>(K);
      break;
    case 8:
      smem_size   = kron_shmem_size<T>(K);
      thread_dims = kron_blockdim(K);
      break;
    case 9:
      smem_size   = transform_cublasdx_shmem_size<T>(K);
      thread_dims = transform_cublasdx_blockdim<T>(K);
      break;
    case 10:
      smem_size   = transform_cublasdx_mxm_shmem_size<T>(K);
      thread_dims = transform_cublasdx_mxm_blockdim<T>(K);
      break;
  }

  /* Level 8: build the Kronecker matrix once, before the timing loop */
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
          submit_transform_wmma_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 7:
          submit_transform_level7_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 8:
          submit_transform_kron_bench<T>(nfuncs, K, A, KronMat, C, blas_handle, streams[t%num_streams]);
          break;
        case 9:
          submit_transform_cublasdx_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
        case 10:
          submit_transform_cublasdx_mxm_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
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
      /* L8 does one K^3 x K^3 GEMM per task (2*K^6 FLOPs); the rest do 3 passes (3*2*K^4 FLOPs) */
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

  for (int i = 0; i < num_streams; ++i) {
    DESTROY_STREAM(streams[i]);
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
  int K      = opt.parse("-K", 16);    /* number of coefficients */
  int M      = opt.parse("-M", 512);   /* max number of blocks */
  int level  = opt.parse("-l", 0);     /* 0 = auto, 1-10 = explicit */
  int num_streams = opt.parse("-s", 4);/* number of concurrent streams to use */

  /* Legacy -m flag: force level 1 */
  if (opt.exists("-m")) level = 1;

  std::cout << "Running benchmark"
            << " nreps=" << nreps
            << " ntasks=" << ntasks
            << " N=" << N
            << " K=" << K
            << " M=" << M
            << " level=" << (level <= 0 ? (MRA_HAVE_CUBLASDX ? 9 : 3) : level)
            << std::endl;

  transform_bench<double>(nreps, ntasks, N, M, K, level, num_streams);
}


#include <iostream>
#include <chrono>

#include "transform.h"
#include "transform_cublasdx.h"
#include "transform_level2.h"
#include "transform_level3.h"
#include "transform_level4.h"
#include "transform_level5.h"
#include "mxm_cublasdx.h"
#include "util.h"

/**
 * Optimization levels:
 *   0 - Default
 *   1 - L1: thread-parallel over j, serial k-loop, all global memory (mxm.h fallback)
 *   2 - L2: B in LDS, threads distributed over rows
 *   3 - L3: B in LDS + register accumulation (acc[K] in VGPRs)
 *   4 - L4: AMD MFMA (GFX90A/GFX940) for K=16,32; falls back to L3 elsewhere
 *   5 - L5: transform3d
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

  const char* level_names[] = {
    "Default",           /* unused [0] */
    "L1-global",  /* 1 */
    "L2-lds_b",   /* 2 */
    "L3-regblk",  /* 3 */
    "L4-mfma",    /* 4 */
    "L5-transform3-rocwmma" /* 5 */
  };

  /* Print shmem and thread dims for this level */
  int smem_size = 0;
  Dim3 thread_dims = {1, 1, 1};
  switch (level) {
    case 0:
      smem_size   = transform_cublasdx_shmem_size<T>(K);
      thread_dims = mra::mTxmq_blockdim<T>(K);
      break;
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
      thread_dims = transform_level5_thread_num<T>(K);
      break;
  }

  // check correctness by comparing against base case (level 1)
  if (level > 1) {
    submit_transform_bench(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  for (int i = 0; i < nreps+1; ++i) {
    beg = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < ntasks; ++t) {
      switch (level) {
        case 0:
          submit_transform_cublasdx_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%num_streams]);
          break;
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
      }
    }
    for (int t = 0; t < num_streams; ++t) {
      SYNC_STREAM(streams[t]);
    }
    end = std::chrono::high_resolution_clock::now();

    /* skip warm-up */
    if (i > 0) {
      auto us = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
      uint64_t flops = (uint64_t)ntasks * K * K * K * K * 3 * 2 /* multiply-add */ * nfuncs;
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
            << " level=" << level
            << std::endl;

  transform_bench<double>(nreps, ntasks, N, M, K, level, num_streams);
}

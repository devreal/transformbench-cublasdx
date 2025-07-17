
#include <iostream>
#include <chrono>

#include "transform.h"
#include "transform_cublasdx.h"
#include "mxm_cublasdx.h"
#include "util.h"

template<typename T>
void transform_bench(int nreps, int ntasks, int nfuncs, int nblocks, int K, bool use_mTxm) {

  cudaStream_t streams[4]; // PaRSEC uses 4 streams by default
  T* A, *B, *C, *workspace;
  cudaMalloc(&A, nfuncs * K * K * K * sizeof(T)); // N x KxKxK tensors
  cudaMalloc(&B, K * K * sizeof(T)); // KxK matrix
  cudaMalloc(&C, nfuncs * K * K * K * sizeof(T)); // N x KxKxK tensors
  cudaMalloc(&workspace, nblocks * K * K * K * sizeof(T)); // N x KxKxK tensors

  for (int i = 0; i < 4; ++i) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  for (int i = 0; i < nreps+1; ++i) {
    beg = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < ntasks; ++t) {
      if (!use_mTxm) {
        submit_transform_cublasdx_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, streams[t%4]);
      } else {
        submit_transform_bench(nfuncs, nblocks, K, A, B, C, workspace, streams[t%4]);
      }
    }
    for (int t = 0; t < 4; ++t) {
      cudaStreamSynchronize(streams[t]);
    }
    end = std::chrono::high_resolution_clock::now();

    /* skip warm-up */
    if (i > 0) {
      auto us = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
      uint64_t flops = (uint64_t)ntasks * K * K * K * K * 3 * 2 /* multiply-add */ * nfuncs;
      Dim3 thread_dims = mra::mTxmq_blockdim<T>(K);
      std::cout << "Transform nfuncs = " << nfuncs << ";nblocks = " << nblocks << ";K = " << K << ";tasks = " << ntasks
                << ";threads = {" << thread_dims.x << ", " << thread_dims.y << ", " << thread_dims.z << "}"
                << ";smem = " << ((!use_mTxm) ? transform_cublasdx_shmem_size<T>(K) : mra::mTxmq_shmem_size<T>(K))
                << ";Time (microseconds) = "
                << us
                << ";GFlop = " << flops*1e-9
                << ";Gflop/s = " << (1e-3 * flops) / us
                << std::endl;
    }
  }

  // cleanup
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(workspace);
}

int main(int argc, char **argv) {

  auto opt = OptionParser(argc, argv);

  int nreps = opt.parse("-r", 5);
  int ntasks = opt.parse("-n", 500);
  int N = opt.parse("-N", 2048); // number of functions
  int K = opt.parse("-K", 16); // number of coefficients
  int M = opt.parse("-M", 512); // max number of blocks
  bool use_mTxm = opt.exists("-m"); // 0 for mTxmq, 1 for cublasdx
  std::cout << "Running benchmark with " << nreps << " repetitions, " << ntasks << " tasks, "
            << N << " functions, " << K << " coefficients, " << M << " blocks"
            << (use_mTxm ? " (mTxmq)" : " (cublasdx)")
            << std::endl;

  transform_bench<double>(nreps, ntasks, N, M, K, use_mTxm);
}

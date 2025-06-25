
#include <iostream>
#include <chrono>
#include "mxm_cublasdx.h"
#include "util.h"

template <typename T>
__device__ void transform(
    int K,
    const T* t,
    const T* c,
    T*& result,
    T* workspace)
{
  constexpr const int ndim = 3; // fixed for benchmark
  const T* pc = c;
  T *t0=workspace, *t1=result;
  std::swap(t0,t1);
  const int dimj = K;
  int dimi = dimj*dimj;
  mra::mTxmq(dimi, dimj, dimj, t0, t, pc);
  for (int n=1; n<ndim; ++n) {
    mra::mTxmq(dimi, dimj, dimj, t1, t0, pc);
    std::swap(t0,t1);
  }
  /* no need to synchronize here, mTxmq synchronizes */
}

template<typename T>
static
LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK, 4)
__global__ void transform_kernel(int nfuncs, int K, const T* A, const T* B, T* C, T* workspace) {

  const T *a, *b;
  T *c, *w;
  int K2NDIM = K*K*K;
  /* workspace is allocated for each thread-block */
  w = workspace + blockIdx.x * K2NDIM;
  /* iterate over all tensors */
  for (int i = blockIdx.x; i < nfuncs; i += gridDim.x) {
    a = A + i * K2NDIM;
    b = B;
    c = C + i * K2NDIM;
    transform(K, a, b, c, w);
  }
}

template<typename T>
static void submit_transform_bench(int nfuncs, int nblocks, int K,
                                   const T* A, const T* B, T* C, T* workspace,
                                   cudaStream_t stream)
{
  Dim3 thread_dims = mra::mTxmq_blockdim<T>(K);
  assert(block_size(thread_dims) <= MAX_THREADS_PER_BLOCK);
  auto smem_size = mra::mTxmq_shmem_size<T>(K);
  CONFIGURE_KERNEL(transform_kernel<T>, smem_size);
  CALL_KERNEL(transform_kernel<T>, std::min(nfuncs, nblocks), thread_dims, smem_size, stream, (nfuncs, K, A, B, C, workspace));
}

template<typename T>
void transform_bench(int nreps, int ntasks, int nfuncs, int nblocks, int K) {

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
      submit_transform_bench(nfuncs, nblocks, K, A, B, C, workspace, streams[t%4]);
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
                << ";smem = " << mra::mTxmq_shmem_size<T>(K)
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
  int M = opt.parse("-M", 128); // max number of blocks
  std::cout << "Running benchmark with " << nreps << " repetitions, " << ntasks << " tasks, "
            << N << " functions, " << K << " coefficients, " << M << " blocks"
            << std::endl;

  transform_bench<double>(nreps, ntasks, N, M, K);
}

#pragma once

/**
 * Cross-cutting helpers for the CUDA build: launch macros, error checking,
 * thread-index helpers and the command-line option parser.
 *
 * This is the CUDA-only counterpart of the dual-target util.h in the parent
 * directory; every HIP branch has been removed.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 128

#define LAUNCH_BOUNDS(__NT, __NB) __launch_bounds__(__NT, __NB)

typedef int32_t size_type;

using Dim3 = dim3;

/* NVIDIA warp size.  The AMD sources assume a 64-lane wavefront; every place
 * that constant appears has been re-derived for 32-lane warps. */
#define MRA_WARP_SIZE 32

typedef cudaStream_t Stream;

#define SYNC_STREAM(stream)   (void)cudaStreamSynchronize(stream)
#define CREATE_STREAM(stream) (void)cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking)
#define DESTROY_STREAM(stream) (void)cudaStreamDestroy(stream)

#define MALLOC(ptr, size)        (void)cudaMalloc(ptr, size)
#define FREE(ptr)                (void)cudaFree(ptr)
#define MEMCPY_H2D(dst, src, size) (void)cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define MEMCPY_D2H(dst, src, size) (void)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)

#define CALL_KERNEL(name, block, thread, shared, stream, args)                          \
  do {                                                                                  \
    name<<<block, thread, shared, stream>>> args ;                                       \
    { auto _err = cudaGetLastError();                                                   \
      if (_err != cudaSuccess) {                                                        \
        std::cout << "kernel submission failed with " << shared << "B smem at "         \
                  << __FILE__ << ":" << __LINE__ << ": "                                \
                  << cudaGetErrorString(_err) << std::endl;                             \
        throw std::runtime_error("kernel submission failed");                           \
      }                                                                                 \
    }                                                                                   \
  } while (0)

/**
 * Opt in to more than the 48 KB of dynamic shared memory that a kernel gets by
 * default.  Tracked per instantiation so the (comparatively expensive) driver
 * call happens only when the requirement grows.
 */
#define CONFIGURE_KERNEL(name, shared)                                                  \
  do {                                                                                  \
    static int smem_size_config = 0;                                                    \
    if (smem_size_config < (int)(shared)) {                                             \
      cudaFuncSetAttribute(name, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);  \
      { auto _err = cudaGetLastError();                                                 \
        if (_err != cudaSuccess) {                                                      \
          std::cout << "kernel configuration failed with " << shared << "B smem at "    \
                    << __FILE__ << ":" << __LINE__ << ": "                              \
                    << cudaGetErrorString(_err) << std::endl;                           \
          throw std::runtime_error("kernel configuration failed");                      \
        }                                                                               \
        smem_size_config = (int)(shared);                                               \
      }                                                                                 \
    }                                                                                   \
  } while (0)


#if defined(__CUDA_ARCH__)
#define HAVE_DEVICE_ARCH 1
#define SCOPE __host__ __device__ inline
#define SYNCTHREADS() __syncthreads()
#define SHARED __shared__
#define THROW(s) do { std::printf(s); __trap(); } while (0)
#else
#define SCOPE inline
#define SYNCTHREADS()
#define SHARED
#define THROW(s) do { throw std::runtime_error(s); } while (0)
#endif // __CUDA_ARCH__


constexpr inline Dim3 max_thread_dims(int K) {
  int x = K;
  int y = std::min(K, MAX_THREADS_PER_BLOCK / x);
  int z = 1;
  return Dim3(x, y, z);
}

constexpr inline int max_threads(int K) {
  Dim3 thread_dims = max_thread_dims(K);
  return thread_dims.x * thread_dims.y * thread_dims.z;
}

__device__ inline int thread_id() {
  return blockDim.x * ((blockDim.y * threadIdx.z) + threadIdx.y) + threadIdx.x;
}

/* Two overloads rather than a `blockDim` default argument: nvcc rejects the
 * builtin as a default argument of a __host__ __device__ function. */
__host__ __device__ inline int block_size(Dim3 block) {
  return block.x * block.y * block.z;
}

__device__ inline int block_size() {
  return blockDim.x * blockDim.y * blockDim.z;
}

__device__ inline bool is_team_lead() {
  return (0 == (threadIdx.x + threadIdx.y + threadIdx.z));
}


struct OptionParser {

  private:
    char **m_begin;
    char **m_end;

    static inline const char *empty = "";

  public:
    OptionParser(int argc, char **argv)
    : m_begin(argv), m_end(argv+argc)
    { }

    std::string_view get(const std::string &option) {
      char **itr = std::find(m_begin, m_end, option);
      if (itr != m_end && ++itr != m_end) return std::string_view(*itr);
      return std::string_view(empty);
    }

    bool exists(const std::string &option) {
      return std::find(m_begin, m_end, option) != m_end;
    }

    int index(const std::string &option) {
      char **itr = std::find(m_begin, m_end, option);
      if (itr != m_end) return (int)(itr - m_end);
      return -1;
    }

    int parse(std::string_view option, int default_value) {
      int N = default_value;
      char **itr = std::find(m_begin, m_end, option);
      if (++itr < m_end) {
        N = std::stoi(*itr);
      }
      return N;
    }

    long parse(std::string_view option, long default_value) {
      long N = default_value;
      char **itr = std::find(m_begin, m_end, option);
      if (++itr < m_end) {
        N = std::stol(*itr);
      }
      return N;
    }

    double parse(std::string_view option, double default_value = 0.25) {
      double N = default_value;
      char **itr = std::find(m_begin, m_end, option);
      if (++itr < m_end) {
        N = std::stod(*itr);
      }
      return N;
    }

  }; // struct OptionParser

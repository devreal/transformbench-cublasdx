#pragma once

#include <string>
#include <cuda_runtime.h>

/**
 * A bunch of helper macros, options parsign, etc.
 */

#define MAX_THREADS_PER_BLOCK 128

#define LAUNCH_BOUNDS(__NT, __NB) __launch_bounds__(__NT, __NB)

using Dim3 = dim3;

#define CALL_KERNEL(name, block, thread, shared, stream, args)                          \
  do {                                                                                  \
    name<<<block, thread, shared, stream>>> args ;                                      \
    if (cudaPeekAtLastError() != cudaSuccess) {                                         \
      std::cout << "kernel submission failed with " << shared << "B smem at "           \
                << __FILE__ << ":" << __LINE__ << ": "                                  \
                << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;              \
      throw std::runtime_error("kernel configuration failed");                          \
    }                                                                                   \
  } while (0)

#define CONFIGURE_KERNEL(name, shared)                                                  \
  do {                                                                                  \
    static int smem_size_config = 0;                                                    \
    if (smem_size_config < shared) {                                                    \
      cudaFuncSetAttribute(name, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);  \
      if (cudaPeekAtLastError() != cudaSuccess) {                                       \
        std::cout << "kernel configuration failed with " << shared << "B smem at "      \
                  << __FILE__ << ":" << __LINE__ << ": "                                \
                  << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;            \
        throw std::runtime_error("kernel configuration failed");                        \
      }                                                                                 \
    }                                                                                   \
  } while (0)


#if defined(__CUDA_ARCH__)
#define THROW(s) do { std::printf(s); __trap(); } while(0)
#else  // __CUDA_ARCH__
#define THROW(s) do { throw std::runtime_error(s); } while(0)
#endif // __CUDA_ARCH__

constexpr inline Dim3 max_thread_dims(int K) {
//  int x = 32;
//  int y = MAX_THREADS_PER_BLOCK / x;
  int x = MAX_THREADS_PER_BLOCK;
  int y = 1;
  int z = 1;
  return Dim3(x, y, z);
}

constexpr inline int max_threads(int K) {
  Dim3 thread_dims = max_thread_dims(K);
  return thread_dims.x*thread_dims.y*thread_dims.z;
}

__device__ int thread_id() {
  return blockDim.x * ((blockDim.y * threadIdx.z) + threadIdx.y) + threadIdx.x;
}

__host__ __device__ int block_size(Dim3 block = blockDim) {
  return block.x * block.y * block.z;
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
      std::string token;
      int N = default_value;
      char **itr = std::find(m_begin, m_end, option);
      if (++itr < m_end) {
        N = std::stoi(*itr);
      }
      return N;
    }

    long parse(std::string_view option, long default_value) {
      std::string token;
      long N = default_value;
      char **itr = std::find(m_begin, m_end, option);
      if (++itr < m_end) {
        N = std::stol(*itr);
      }
      return N;
    }

    double parse(std::string_view option, double default_value = 0.25) {
      std::string token;
      double N = default_value;
      char **itr = std::find(m_begin, m_end, option);
      if (++itr < m_end) {
        N = std::stod(*itr);
      }
      return N;
    }

  }; // struct Options

#pragma once

#include <string>

/**
 * A bunch of helper macros, options parsign, etc.
 */

#define MAX_THREADS_PER_BLOCK 128

#define LAUNCH_BOUNDS(__NT, __NB) __launch_bounds__(__NT, __NB)

typedef int32_t size_type;


#ifdef MRA_HAVE_CUDA
#include <cuda_runtime.h>

using Dim3 = dim3;

typedef cudaStream_t Stream;
#define SYNC_STREAM(cudaStream) cudaStreamSynchronize(cudaStream)
#define CREATE_STREAM(cudaStream) cudaStreamCreateWithFlags(cudaStream, cudaStreamNonBlocking)

#define MALLOC(ptr, size) cudaMalloc(ptr, size)
#define FREE(ptr) cudaFree(ptr)
#define MEMCPY_H2D(dst, src, size) cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define MEMCPY_D2H(dst, src, size) cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)

#define CALL_KERNEL(name, block, thread, shared, stream, args)                          \
  do {                                                                                  \
    name<<<block, thread, shared, stream>>> args ;                                      \
    { auto _err = cudaGetLastError();                                                   \
      if (_err != cudaSuccess) {                                                        \
        std::cout << "kernel submission failed with " << shared << "B smem at "         \
                  << __FILE__ << ":" << __LINE__ << ": "                               \
                  << cudaGetErrorString(_err) << std::endl;                            \
        throw std::runtime_error("kernel configuration failed");                        \
      }                                                                                 \
    }                                                                                   \
  } while (0)

#define CONFIGURE_KERNEL(name, shared)                                                  \
  do {                                                                                  \
    static int smem_size_config = 0;                                                    \
    if (smem_size_config < shared) {                                                    \
      cudaFuncSetAttribute(name, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);  \
      { auto _err = cudaGetLastError();                                                 \
        if (_err != cudaSuccess) {                                                      \
          std::cout << "kernel configuration failed with " << shared << "B smem at "    \
                    << __FILE__ << ":" << __LINE__ << ": "                             \
                    << cudaGetErrorString(_err) << std::endl;                          \
          throw std::runtime_error("kernel configuration failed");                      \
        }                                                                               \
        smem_size_config = shared;                                                      \
      }                                                                                 \
    }                                                                                   \
  } while (0)

#elif defined(MRA_HAVE_HIP)

#include <hip/hip_runtime.h>

using Dim3 = dim3;

typedef hipStream_t Stream;
#define SYNC_STREAM(hipStream) (void)hipStreamSynchronize(hipStream)
#define CREATE_STREAM(hipStream) (void)hipStreamCreateWithFlags(hipStream, hipStreamNonBlocking)

#define MALLOC(ptr, size) (void)hipMalloc(ptr, size)
#define FREE(ptr) (void)hipFree(ptr)
#define MEMCPY_H2D(dst, src, size) (void)hipMemcpy(dst, src, size, hipMemcpyHostToDevice)
#define MEMCPY_D2H(dst, src, size) (void)hipMemcpy(dst, src, size, hipMemcpyDeviceToHost)

#define CALL_KERNEL(name, block, thread, shared, stream, args)                          \
  do {                                                                                  \
    name<<<block, thread, shared, stream>>> args ;                                      \
    { auto _err = hipGetLastError();                                                    \
      if (_err != hipSuccess) {                                                         \
        std::cout << "kernel submission failed with " << shared << "B smem at "         \
                  << __FILE__ << ":" << __LINE__ << ": "                               \
                  << hipGetErrorString(_err) << std::endl;                             \
        throw std::runtime_error("kernel configuration failed");                        \
      }                                                                                 \
    }                                                                                   \
  } while (0)

#define CONFIGURE_KERNEL(name, shared)                                                  \
  do {                                                                                  \
    static int smem_size_config = 0;                                                    \
    if (smem_size_config < shared) {                                                    \
      const void* func_ptr = reinterpret_cast<void*>(name);                             \
      (void)hipFuncSetAttribute(func_ptr, hipFuncAttributeMaxDynamicSharedMemorySize, shared);  \
      { auto _err = hipGetLastError();                                                  \
        if (_err != hipSuccess) {                                                       \
          std::cout << "kernel configuration failed with " << shared << "B smem at "    \
                    << __FILE__ << ":" << __LINE__ << ": "                             \
                    << hipGetErrorString(_err) << std::endl;                           \
          throw std::runtime_error("kernel configuration failed");                      \
        }                                                                               \
        smem_size_config = shared;                                                      \
      }                                                                                 \
    }                                                                                   \
  } while (0)

#endif // MRA_HAVE_CUDA


#if defined(__CUDA_ARCH__)
#define THROW(s) do { std::printf(s); __trap(); } while(0)
#define HAVE_DEVICE_ARCH 1
#define SCOPE __device__ __host__
#define SYNCTHREADS() __syncthreads()
#elif defined(__HIP__)
#define SCOPE __device__ __host__ inline
// TODO: how to abort a kernel on AMD?
#define THROW(s) do { std::printf(s); } while(0)
#if defined(__HIP_DEVICE_COMPILE__)
  #define SYNCTHREADS() __syncthreads()
  #define SHARED __shared__
  #define HAVE_DEVICE_ARCH 1
#else
  #define SYNCTHREADS()
  #define SHARED
#endif // __HIP_DEVICE_COMPILE__
#else  // __CUDA_ARCH__
#define THROW(s) do { throw std::runtime_error(s); } while(0)
#define SCOPE inline
#define SYNCTHREADS()
#endif // __CUDA_ARCH__


constexpr inline Dim3 max_thread_dims(int K) {
  int x = K;
  int y = std::min(K, MAX_THREADS_PER_BLOCK / x);
  //int x = MAX_THREADS_PER_BLOCK;
  //int y = 1;
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

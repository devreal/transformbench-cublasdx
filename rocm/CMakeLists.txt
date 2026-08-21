cmake_minimum_required(VERSION 3.10)

project(transformbench LANGUAGES CXX)

set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(TTG_ENABLE_CUDA ON)
else(CMAKE_CUDA_COMPILER)
  message(WARNING "CUDA compiler not found")
endif(CMAKE_CUDA_COMPILER)
set(HAVE_CUDA ${CMAKE_CUDA_COMPILER} CACHE BOOL "True if we can compile .cu files")

check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
  set(TTG_ENABLE_HIP ON)
else(CMAKE_HIP_COMPILER)
  message(WARNING "HIP compiler not found")
endif(CMAKE_HIP_COMPILER)
set(HAVE_HIP ${CMAKE_HIP_COMPILER} CACHE BOOL "True if we can compile .hip files")

option(USE_SUGGEST_LAYOUT "Use suggested layout instead of get_layout" ON)
option(DEBUG_TENSOR_TYPE "Compile-time print cute tensor types (breaks build)" OFF)
set(USE_CUBLASDX_VERSION "25.06" CACHE STRING "Version of cublasDx to use")


# fetch cublasDx
if (CMAKE_CUDA_COMPILER)
  include(FetchContent)
  FetchContent_Declare(
    cublasdx
    URL https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx/nvidia-mathdx-${USE_CUBLASDX_VERSION}.0.tar.gz
  )
  FetchContent_MakeAvailable(cublasdx)
  FetchContent_GetProperties(cublasdx
    SOURCE_DIR CUBLASDX_SOURCE_DIR
    BINARY_DIR CUBLASDX_BINARY_DIR
  )

  # look for cublasDx
  find_package(mathdx REQUIRED COMPONENTS cublasdx HINTS ${CUBLASDX_SOURCE_DIR}/nvidia/mathdx/25.06/)
  if (TARGET mathdx::cublasdx)
    message(STATUS "Found cublasDx at ${mathdx_CUBLASDX_DIR}")
  else()
    message(FATAL_ERROR "cublasDx not found")
  endif()

endif(CMAKE_CUDA_COMPILER)

# Simple interface that holds cublasDx and CUDA settings
add_library(libmra INTERFACE)
if (CMAKE_CUDA_COMPILER)
  # Link against cublasDx and CUDA
  target_link_libraries(libmra INTERFACE mathdx::cublasdx)
  # Set the CUDA architecture
  target_compile_definitions(libmra INTERFACE MRA_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES} MRA_HAVE_CUDA=1)
  # Enable support for constexpr and extended lambdas
  target_compile_options(libmra INTERFACE --expt-relaxed-constexpr --extended-lambda)

  # Add the transformbench executable
  add_executable(transformbench_cuda transformbench.cu)

  # Link against the MRA interface
  target_link_libraries(transformbench_cuda PUBLIC libmra)


  if (USE_SUGGEST_LAYOUT)
      # Enable using suggested layout instead of get_layout
      target_compile_definitions(transformbench_cuda PUBLIC USE_SUGGEST_LAYOUT)
  endif (USE_SUGGEST_LAYOUT)

  if (DEBUG_TENSOR_TYPE)
      # Enable compile-time printing of cute tensor types (breaks build)
      target_compile_definitions(transformbench_cuda PUBLIC DEBUG_TENSOR_TYPE)
  endif (DEBUG_TENSOR_TYPE)
endif()

if (CMAKE_HIP_COMPILER)
  # Ensure ROCm cmake configs (hip, hipblas, etc.) are findable
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm-6.4.3 /opt/rocm)
  find_package(hipblas REQUIRED)

  #target_link_libraries(libmra INTERFACE mathdx::cublasdx)
  # Set the CUDA architecture
  target_compile_definitions(libmra INTERFACE MRA_HAVE_HIP=1)
  # Enable support for constexpr and extended lambdas
  #target_compile_options(libmra INTERFACE --expt-relaxed-constexpr --extended-lambda)

  # Add the transformbench executable
  add_executable(transformbench_hip transformbench.hip)

  # Link against the MRA interface and hipBLAS (for level 6 Kronecker GEMM)
  target_link_libraries(transformbench_hip PUBLIC libmra roc::hipblas)

  # Correctness test: validate any optimization level against the L1 reference
  add_executable(validate_levels validate_levels.hip)
  target_link_libraries(validate_levels PUBLIC libmra roc::hipblas)

  if (USE_SUGGEST_LAYOUT)
      # Enable using suggested layout instead of get_layout
      target_compile_definitions(transformbench_hip PUBLIC USE_SUGGEST_LAYOUT)
  endif (USE_SUGGEST_LAYOUT)

  if (DEBUG_TENSOR_TYPE)
      # Enable compile-time printing of cute tensor types (breaks build)
      target_compile_definitions(transformbench_hip PUBLIC DEBUG_TENSOR_TYPE)
  endif (DEBUG_TENSOR_TYPE)
endif ()
cmake_minimum_required(VERSION 3.18)

project(transformbench_cuda LANGUAGES CXX CUDA)

# sm_80 (A100) is the minimum for FP64 tensor cores, which levels 4-7 use.
# Override with -DCMAKE_CUDA_ARCHITECTURES=90 for Hopper.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(USE_CUBLASDX        "Fetch and build against cuBLASDx (levels 9 and 10)" ON)
option(USE_SUGGEST_LAYOUT  "Use suggested layout instead of get_layout" ON)
option(DEBUG_TENSOR_TYPE   "Compile-time print cute tensor types (breaks build)" OFF)
set(USE_CUBLASDX_VERSION "25.06" CACHE STRING "Version of cuBLASDx to use")

# MRA_CUDA_ARCH gates the FP64 tensor-core paths at compile time; take the
# first entry when several architectures are requested.
list(GET CMAKE_CUDA_ARCHITECTURES 0 MRA_PRIMARY_ARCH)
string(REGEX REPLACE "[^0-9]" "" MRA_PRIMARY_ARCH "${MRA_PRIMARY_ARCH}")
message(STATUS "Building for CUDA architecture sm_${MRA_PRIMARY_ARCH}")
if (MRA_PRIMARY_ARCH LESS 80)
  message(WARNING "sm_${MRA_PRIMARY_ARCH} has no FP64 tensor cores; "
                  "levels 4-7 will run their level-3 fallback")
endif()

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# cuBLASDx (levels 9 and 10)
# ---------------------------------------------------------------------------
set(MRA_CUBLASDX_FOUND OFF)
if (USE_CUBLASDX)
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

  find_package(mathdx REQUIRED COMPONENTS cublasdx
               HINTS ${CUBLASDX_SOURCE_DIR}/nvidia/mathdx/${USE_CUBLASDX_VERSION}/)
  if (TARGET mathdx::cublasdx)
    message(STATUS "Found cuBLASDx at ${mathdx_CUBLASDX_DIR}")
    set(MRA_CUBLASDX_FOUND ON)
  else()
    message(FATAL_ERROR "cuBLASDx not found")
  endif()
endif()

# ---------------------------------------------------------------------------
# Common settings
# ---------------------------------------------------------------------------
add_library(libmra INTERFACE)
target_compile_definitions(libmra INTERFACE
  MRA_HAVE_CUDA=1
  MRA_CUDA_ARCH=${MRA_PRIMARY_ARCH})
# constexpr host functions in device code, and extended lambdas
target_compile_options(libmra INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
# cuBLAS backs the Kronecker level
target_link_libraries(libmra INTERFACE CUDA::cublas)

if (MRA_CUBLASDX_FOUND)
  target_link_libraries(libmra INTERFACE mathdx::cublasdx)
endif()

function(mra_configure_target tgt)
  target_link_libraries(${tgt} PUBLIC libmra)
  set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  if (USE_SUGGEST_LAYOUT)
    target_compile_definitions(${tgt} PUBLIC USE_SUGGEST_LAYOUT)
  endif()
  if (DEBUG_TENSOR_TYPE)
    target_compile_definitions(${tgt} PUBLIC DEBUG_TENSOR_TYPE)
  endif()
endfunction()

# The benchmark driver
add_executable(transformbench_cuda transformbench.cu)
mra_configure_target(transformbench_cuda)

# Correctness test: validate any optimization level against the L1 reference
add_executable(validate_levels validate_levels.cu)
mra_configure_target(validate_levels)

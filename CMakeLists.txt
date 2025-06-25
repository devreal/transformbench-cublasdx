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
  message(FATAL_ERROR "CUDA compiler not found")
endif(CMAKE_CUDA_COMPILER)
set(HAVE_CUDA ${CMAKE_CUDA_COMPILER} CACHE BOOL "True if we can compile .cu files")


# look for cublasDx
find_package(mathdx REQUIRED COMPONENTS cublasdx)
if (TARGET mathdx::cublasdx)
  message(STATUS "Found cublasDx at ${mathdx_CUBLASDX_DIR}")
endif()

# Add the executable
add_executable(transformbench transformbench.cu)

# Link against cuBLASLt and CUDA
target_link_libraries(transformbench mathdx::cublasdx)

# Set the CUDA architecture
target_compile_definitions(transformbench PUBLIC MRA_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES})

# Enable support for constexpr and extended lambdas
target_compile_options(transformbench PUBLIC --expt-relaxed-constexpr --extended-lambda)

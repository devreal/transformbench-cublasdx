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

# Simple interface that holds cublasDx and CUDA settings
add_library(libmra INTERFACE)
# Link against cublasDx and CUDA
target_link_libraries(libmra mathdx::cublasdx)
# Set the CUDA architecture
target_compile_definitions(libmra PUBLIC MRA_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES})
# Enable support for constexpr and extended lambdas
target_compile_options(libmra PUBLIC --expt-relaxed-constexpr --extended-lambda)

# Add the transformbench executable
add_executable(transformbench transformbench.cu)

# Link against the MRA interface
target_link_libraries(transformbench libmra)


# Add the transformbench executable with suggested layout
add_executable(transformbench_suggested transformbench.cu)

# Link against the MRA interface
target_link_libraries(transformbench_suggested libmra)

# Enable using suggested layout instead of get_layout
target_compile_definitions(transformbench_suggested PUBLIC USE_SUGGEST_LAYOUT)
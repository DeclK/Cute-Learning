cmake_minimum_required(VERSION 3.23.1)
project(cute_learning LANGUAGES CUDA CXX)

# CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set include directories
set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../flashdecoding/src/cutlass/include)

# Create executable
add_executable(tensor_practical tensor_practical.cu)

# Set include directories
target_include_directories(tensor_practical PRIVATE ${CUTLASS_INCLUDE_DIR})

# Set CUDA compilation flags
target_compile_options(tensor_practical PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -arch=sm_80 
        -std=c++17 
        --expt-relaxed-constexpr
        -w
    >
)
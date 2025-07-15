# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cute-Learning is a CUDA/C++ repository showcasing high-performance implementations using Cutlass CuTe library. The project focuses on optimized GPU kernels for deep learning primitives including GEMM (General Matrix Multiply), GEMV (General Matrix-Vector), Flash Attention, and other tensor operations.

## Architecture & Key Patterns

### Core Components
- **GEMM**: Multiple implementations (v1-v4, hopper, cublas, stream-k) with progressive optimizations
- **GEMV**: Matrix-vector multiplication variants including fast implementations and PyTorch bindings
- **FlashDecoding**: Flash attention mechanism for transformer models
- **Data Operations**: LDSM, tile copy, dequantization utilities

### Technology Stack
- **CUDA**: Primary language for GPU kernels (C++17, CUDA 17)
- **Cutlass CuTe**: Tensor operation library used throughout
- **PyTorch**: Integration for testing (flashdecoding components)
- **cuBLAS**: Baseline comparisons and fallbacks

### Build Systems
- **Make**: Used for individual components (gemm/, gemv/, others/)
- **CMake**: Used for flashdecoding/ and sm90_decode/ (minimum 3.23.1)
- **Architecture**: sm_80 (default), sm_90a for newer components

## Build Commands

### GEMM Variants
```bash
# Navigate to specific variant and build
cd gemm/gemm_v1 && make

# Common build patterns:
cd gemm/gemm_v[1-4] && make                    # Basic variants
cd gemm/hopper/gemm_wgmma_rs && make          # Hopper architecture
cd gemm/cublas && make                        # cuBLAS baseline
```

### FlashDecoding
```bash
cd flashdecoding
mkdir build && cd build
cmake ..
make -j
```

### Testing
```bash
# FlashDecoding testing
cd flashdecoding/build
./test_single_decode    # Unit tests
./bench_single_decode   # Performance benchmarks

# Individual component testing
cd gemm/gemm_v1 && make && ./gemm  # Run with default parameters
```

### GEMV Variants
```bash
cd gemv/fast_gemv && make
./gemv  # Matrix-vector multiplication

cd gemv/gemv_torch && python gemv.py  # PyTorch integration
```

## Key File Patterns

### GEMM Implementation Structure
- `gemm.cu`: Main kernel implementation
- `utils.h`: Helper functions and utilities
- `Makefile`: Build configuration with nvcc flags

### FlashDecoding Structure
- `src/flash_fwd_*.cu`: Forward pass implementations
- `src/include/`: Header files with kernel traits and utilities
- `src/flash_api.h`: Public API definitions
- `test.ipynb`: Jupyter notebook for testing

### Common Build Flags
- `-arch=sm_80`: Target architecture
- `-std=c++17`: C++ standard
- `-I../../flashdecoding/src/cutlass/include`: Cutlass include path
- `--expt-relaxed-constexpr`: CUDA compilation flag

## Development Workflow

1. **Select Component**: Choose specific directory (e.g., gemm/gemm_v3)
2. **Build**: Run `make` in component directory
3. **Test**: Execute generated binary (usually `./gemm` or `./test_*`)
4. **Iterate**: Modify .cu files and rebuild

## Performance Analysis
- **NCU Integration**: Some components include `ncu.sh` scripts for profiling
- **Benchmarking**: Built-in timing using `std::chrono`
- **Validation**: CPU reference implementations for correctness checking
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CUDA/C++ learning repository focused on Cutlass CuTe (CUDA Templates for Tensor operations) library. The project demonstrates various GPU-accelerated tensor operations including basic tensor creation, vector arithmetic, and PyTorch integration.

## Build System

### CMake Configuration
- **Build System**: CMake 3.23.1+
- **Language**: CUDA C++17
- **Target Architecture**: sm_80 (Ampere)
- **Compiler Flags**: `--expt-relaxed-constexpr`, `-fPIC`

### Build Commands
```bash
# Build all executables
mkdir build && cd build
cmake ..
make -j

# Build specific targets
make tensor_basics    # Basic tensor operations
make vector_add       # Vector addition kernel
make torch_sample     # PyTorch integration example
```

## Key Components

### 1. Tensor Operations (`tensor_basics.cu`)
- **Purpose**: Demonstrates Cutlass CuTe tensor creation and manipulation
- **Key Features**:
  - Tensor creation from existing data
  - Tensor slicing and reshaping
  - Layout manipulation (flatten, squeeze)
  - Multi-dimensional tensor indexing

### 2. Vector Addition (`vector_add.cu`)
- **Purpose**: GPU-accelerated vector arithmetic using CuTe tensors
- **Kernel**: `vector_add<kNumElemPerThread>` - optimized vectorized operation
- **Operation**: z = ax + by + c (where a,b,c are scalars)
- **Data Type**: half precision (FP16)
- **Optimization**: 8 elements per thread processing

### 3. PyTorch Integration (`torch_sample.cu`)
- **Purpose**: CUDA kernel integration with PyTorch tensors
- **Function**: `add_one_to_tensor` - adds 1 to each element
- **Requirements**: 
  - Input must be float32 CUDA tensor
  - PyTorch development headers required
- **Extension**: Includes pybind11 for Python module creation

### 4. Helper Utilities (`helper.h`)
- **Core Functions**:
  - Data generation: `cpu_rand_data`, `cpu_const_data`, `cpu_arange_data`
  - Validation: `cpu_compare`, `gpu_compare` (with tolerance)
  - Tensor inspection: `print_tensor_info`, `print_tensor_values`
  - Color-coded output: `printf_fail`, `printf_ok`

## Development Workflow

### Prerequisites
- CUDA toolkit (11.8+)
- PyTorch with CUDA support
- Cutlass library (included via relative path: `../cutlass/include`)

### Testing
```bash
# After building
./tensor_basics    # Test tensor operations
./vector_add       # Test vector addition
./torch_sample     # Test PyTorch integration
```

### File Structure
- `tensor_basics.cu`: Tensor creation and manipulation examples
- `vector_add.cu`: Vector arithmetic with CuTe tensors
- `torch_sample.cu`: PyTorch tensor processing with CUDA kernels
- `helper.h`: Common utilities for data generation and validation
- `CMakeLists.txt`: Build configuration with PyTorch integration

## Architecture Patterns

### CuTe Tensor Usage
- **Creation**: `make_tensor(ptr, shape)` - creates tensor from device/host memory
- **Layout**: `make_shape()`, `make_layout()` - define tensor dimensions and strides
- **Operations**: `local_tile()`, `recast()`, `copy()` - optimized tensor operations
- **Memory**: Uses both host (CPU) and device (GPU) memory pointers

### CUDA Kernel Patterns
- **Thread mapping**: 1D grid/block configuration for vector operations
- **Vectorization**: Processing multiple elements per thread (8 in vector_add)
- **Memory coalescing**: Using `LDG.128`/`STG.128` for efficient memory access
- **Type handling**: Half precision (FP16) with half2 vector types
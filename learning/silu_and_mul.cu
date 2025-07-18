#include <cute/tensor.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include "helper.h"

using namespace cute;

// create a silu matmul kernel
template <int kNumElemPerThread = 8>
__global__ void silu_and_mul(half* output, 
                            const half *input,
                            const int dim) {

}

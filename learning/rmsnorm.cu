#include <cute/tensor.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <helper.h>

using namespace cute;
__global__ void rmsnorm(half* output,
                         const half* input,
                         int b,
                         int n,
                         int c) {
    const int bid = blockIdx.x;
    const int tx = threadIdx.x; // tid in warp
    const int ty = threadIdx.y; // warp id
    const int num_threads = blockDim.x * blockDim.y;
    const int n WARPS_PER_BLOCK = blockDim.y;

    Tensor in_t = make_tensor(input, make_shape(b*n, c), make_stride(c, _1{}));
    Tensor out_t = make_tensor(output, make_shape(b*n, c), make_stride(c, _1{}));

    Tensor in_block_t = in_t(bid, _);
    Tensor out_block_t = out_t(bid, _);
}

int main() {
    // make data
    int b = 1;
    int n = 1024;
    int c = 2048;

    half* input, *output;
    half* d_input, *d_output;

    input = create_host_data<half>({b, n ,c});
    output = create_host_data<half>({b, n, c}, 0);

    cudaMalloc(&d_input, b * n * c * sizeof(half));
    cudaMalloc(&d_output, b * n * c * sizeof(half));
}
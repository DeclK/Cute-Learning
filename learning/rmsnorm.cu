#include <cute/tensor.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include "helper.h"

using namespace cute;
__global__ void rmsnorm(half* output,
                         const half* input,
                         const half* weight,
                         const float eps,
                         int b,
                         int n,
                         int c) {
    const int bid = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = threadIdx.x + ty * blockDim.x;
    const int num_threads = blockDim.x * blockDim.y;
    const int n_warps = blockDim.y;

    extern __shared__ float smem[];
    Tensor smem_t = make_tensor(make_smem_ptr(smem), make_shape(n_warps));

    Tensor in_t = make_tensor(input, make_shape(b*n, c), make_stride(c, _1{}));
    Tensor out_t = make_tensor(output, make_shape(b*n, c), make_stride(c, _1{}));
    Tensor in_block_t = in_t(bid, _);
    Tensor out_block_t = out_t(bid, _);

    // step-1: reduce 8 elements per thread
    float square_sum = 0;
    Tensor in_thread_t = make_tensor(in_block_t.data(), make_shape(c/8, _8{}), make_stride(_8{}, _1{}));
    Tensor in_thread_t_r = make_tensor_like(in_thread_t(0, _));
    for(int i=tid; i<c/8; i+=num_threads){
        copy(in_thread_t(i, _), in_thread_t_r);
        for (int j=0; j < 8; j++){
            square_sum += float(in_thread_t_r[j]) * float(in_thread_t_r[j]);
        }
    }

    // step-2: warp reduction
    for(int offset=16; offset>1; offset /= 2) {
        square_sum += __shfl_xor_sync(0xffffffff, square_sum, offset);
    }

    // step-3: block reduction
    smem[ty] = square_sum;
    __syncthreads();
    if (ty==0) {
        square_sum = (tx < n_warps) ? smem_t(tx) : 0.f;
        for (int offset=16; offset > 1; offset /= 2) {
            square_sum += __shfl_xor_sync(0xffffffff, square_sum, offset);
        }
        smem_t(0) = square_sum;
    }
    __syncthreads();
    square_sum = smem_t(0);

    float rms_scale = rsqrt(square_sum / c + eps);

    Tensor out_thread_t = make_tensor(out_block_t.data(), make_shape(c/8, _8{}), make_stride(_8{}, _1{}));
    Tensor out_thread_t_r = make_tensor_like(out_thread_t(0, _));
    Tensor weight_t = make_tensor(weight, make_shape(c/8, _8{}), make_stride(_8{}, _1{}));
    Tensor weight_t_r = make_tensor_like(weight_t(0, _));
    
    for (int i=tid; i < c/8; i += num_threads) {
        copy(in_thread_t(i, _), in_thread_t_r);
        copy(weight_t(i, _), weight_t_r);
        for (int j=0; j < 8; j++){ 
            // FIX: Use local weight copy
            out_thread_t_r[j] = float(in_thread_t_r[j]) * rms_scale * float(weight_t_r[j]);
        } 
        copy(out_thread_t_r, out_thread_t(i, _));
    }
}

void run_kernel(half* output, half* input, half* weight, float eps, int b, int n, int c) {
    int num_threads = std::min(c / 8, 1024);
    int num_warps = num_threads / 32;
    int smem_size = num_warps * sizeof(float);
    dim3 block(32, num_warps);
    dim3 grid(b * n);

    cudaFuncSetAttribute(rmsnorm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    rmsnorm<<<grid, block, smem_size>>>(output, input, weight, eps, b, n, c);
}

int main() {
    // make data
    int b = 1;
    int n = 2048 * 2;
    int c = 2048 * 2;
    float eps = 1e-5f;

    half* input, *output, *weight;
    half* d_input, *d_output, *d_weight;

    input = create_host_data<half>({b, n ,c});
    output = create_host_data<half>({b, n, c}, 0);
    weight = create_host_data<half>({c}, 1.0f);  // Initialize weights to 1.0

    cudaMalloc(&d_input, b * n * c * sizeof(half));
    cudaMalloc(&d_output, b * n * c * sizeof(half));
    cudaMalloc(&d_weight, c * sizeof(half));

    // Copy data to device
    cudaMemcpy(d_input, input, b * n * c * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, c * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    run_kernel(d_output, d_input, d_weight, eps, b, n, c);
    GPU_Clock timer;
    timer.start();
    for (int i=0; i < 100; i++){
        run_kernel(d_output, d_input, d_weight, eps, b, n, c);
    }
    auto elapsed = timer.milliseconds();
    printf("Kernel execution time: %.4f ms\n", elapsed / 100.0f);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Wait for kernel completion
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(output, d_output, b * n * c * sizeof(half), cudaMemcpyDeviceToHost);

}
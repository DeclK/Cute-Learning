#include <cute/tensor.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include "vec_dtypes.cuh"
#include "helper.h"

#define endl print("\n")
using namespace cute;

__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

// create a silu matmul kernel
template <int vec_size = 8>
__global__ void silu_and_mul(half* output, 
                            const half *input,
                            const int b,
                            const int n,
                            const int c) {
    Tensor t_input = make_tensor(input, make_shape(b * n, 2 * c), make_stride(2 * c, _1{}));
    Tensor t_output = make_tensor(output, make_shape(b * n, c), make_stride(c, _1{}));

    // get thread data, no 2d grid is needed
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    Tensor input_block = t_input(bid, _);// this does the job of local tile
    Tensor output_block = t_output(bid, _);

    // assuming the dimension c is divisible by vec_size
    // reshape the data into (2, c) -> (2, c // 8, 8)
    Tensor input_block_g = make_tensor(input_block.data(), 
                                make_shape(2, c / vec_size, Int<vec_size>{}),
                                make_stride(c, Int<vec_size>{}, _1{}));
    Tensor output_block_g = make_tensor(output_block.data(), 
                                 make_shape(c / vec_size, Int<vec_size>{}),
                                 make_stride(Int<vec_size>{}, _1{}));


    Tensor input_a_thread_r = make_tensor_like(input_block_g(0, 0, _));
    Tensor input_x_thread_r = make_tensor_like(input_block_g(0, 0, _));
    Tensor output_thread_r = make_tensor_like(output_block_g(0, _));

    for(int i = tid; i < c / vec_size; i += num_threads) {
      // copy data from g to r
      copy(input_block_g(0, i, _), input_a_thread_r);
      copy(input_block_g(1, i, _), input_x_thread_r);


      // calculate output
      for(int j = 0; j < vec_size; j++) {
        output_thread_r(j) = __float2half(silu(__half2float(input_a_thread_r(j))) * __half2float(input_x_thread_r(j)));
      }

      // copy data from r to g
      copy(output_thread_r, output_block_g(i, _));
    }
}


__global__ void silu_and_mul_flashinfer(half* output, const half *input, const int c) {
  constexpr int vec_size = 8;
  const int token_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;
  const int stride = blockDim.x;
  const int offset = token_idx * 2 * c;

  for (int i = thread_idx; i < c / vec_size; i += stride) {
    flashinfer::vec_t<float, vec_size> x_vec, y_vec, out_vec;
    x_vec.cast_load(input + offset + i * vec_size);
    y_vec.cast_load(input + offset + c + i * vec_size);

    for (int j = 0; j < vec_size; j++) {
      out_vec[j] = silu(x_vec[j]) * y_vec[j];
    }
    out_vec.cast_store(output + token_idx * c + i * vec_size);
  }
}

__global__ void silu_and_mul_reference(half* output, const half *input, const int c) {
  const int token_idx = blockIdx.x;
  for (int i = threadIdx.x; i < c; i += blockDim.x) {
    // calculate silu and mul
    float a = __half2float(input[token_idx * 2 * c + i]);
    float x = __half2float(input[token_idx * 2 * c + i + c]);
    output[token_idx * c + i] = __float2half(silu(a) * x);
  }
}

void run_kernel(half* output, 
                const half *input,
                const int b,
                const int n,
                const int c,
                const int check = false) {
  dim3 block(std::min(1024, c / 8));
  // dim3 block(256);
  dim3 grid(b * n);
  
  silu_and_mul<8><<<grid, block>>>(output, input, b, n, c);
  // silu_and_mul_reference<<<grid, block>>>(output, input, c);
  // silu_and_mul_flashinfer<<<grid, block>>>(output, input, c);
  
  if (check) {CUDA_CHECK(cudaDeviceSynchronize());}
}


int main() {
  // create input and output tensors
  const int b = 1;
  const int n = 2048 * 2;
  const int c = 2048 * 2;

  half *input, *output;
  input = create_host_data<half>({b, n, 2 * c});
  output = create_host_data<half>({b, n, c});

  // create device memory
  half *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, b * n * 2 * c * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_output, b * n * c * sizeof(half)));

  // copy input to device
  CUDA_CHECK(cudaMemcpy(d_input, input, b * n * 2 * c * sizeof(half), cudaMemcpyHostToDevice));

  run_kernel(d_output, d_input, b, n, c);
  GPU_Clock timer;
  timer.start();
  for(int i = 0; i < 100; i++) {
    run_kernel(d_output, d_input, b, n, c);
  }
  auto elapsed = timer.milliseconds();
  printf("Kernel execution time: %.4f ms\n", elapsed / 100.0f);

  // copy output back to host
  CUDA_CHECK(cudaMemcpy(output, d_output, b * n * c * sizeof(half), cudaMemcpyDeviceToHost));
  Tensor output_t = make_tensor(output, make_shape(b * n, c), make_stride(c, _1{}));
  Tensor input_t = make_tensor(input, make_shape(b * n, 2 * c), make_stride(2 * c, _1{}));
  print((float)input_t(0, 0));endl;
  print((float)input_t(0, c));endl;  
  print((float)output_t(0, 0));endl;
  
  // free device memory
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  
  // free host memory
  free(input);
  free(output);
  return 0;
}
#include <cute/tensor.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include "helper.h"
using namespace cute;
#define endl print("\n")

// z = ax + by + c
template <int kNumElemPerThread = 8>
__global__ void vector_add(
    half *z, int num, const half *x, const half *y, const half a, const half b, const half c) {
  using namespace cute;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / kNumElemPerThread) { // 未处理非对齐问题
    return;
  }

  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // LDG.128
  copy(txr, txR);
  copy(tyr, tyR);

  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};

  auto tzR2 = recast<half2>(tzR);
  auto txR2 = recast<half2>(txR);
  auto tyR2 = recast<half2>(tyR);

#pragma unroll
  for (int i = 0; i < size(tzR2); ++i) {
    // two hfma2 instruction
    tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
  }

  auto tzRx = recast<half>(tzR2);

  // STG.128
  copy(tzRx, tzr);
}

// z = ax + by + c
template <int kNumElemPerThread = 8>
__global__ void vector_add_half(
    half *z, int num, const half *x, const half *y, const half a, const half b, const half c) {
  using namespace cute;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / kNumElemPerThread) { // 未处理非对齐问题
    return;
  }

  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // LDG.128
  copy(txr, txR);
  copy(tyr, tyR);

#pragma unroll
  for (int i = 0; i < size(tzR); ++i) {
    // two hfma2 instruction
    tzR(i) = txR(i) * a + (tyR(i) * b + c);
  }

  // STG.128
  copy(tzR, tzr);
}

void run_kernel(half *z, int num, const half *x, const half *y, const half a, const half b, const half c) {
  int block_size = 256;
  int grid_size = (num + block_size * 8 - 1) / (block_size * 8);
  
  vector_add<8><<<grid_size, block_size>>>(z, num, x, y, a, b, c);
  
  cudaDeviceSynchronize();
}

int main(){
    // create CUDA streams, allocate memory, etc.
    const int num = 1024 * 1024; // 1M elements
    half *z, *x, *y;

    half *host_z = new half[num];
    half *host_x = new half[num];
    half *host_y = new half[num];
    half *host_z_ref = new half[num];

    cudaMalloc(&z, num * sizeof(half));
    cudaMalloc(&x, num * sizeof(half));
    cudaMalloc(&y, num * sizeof(half));

    // Initialize host data
    Tensor t_z = make_tensor(host_z, make_shape(num));
    Tensor t_x = make_tensor(host_x, make_shape(num));
    Tensor t_y = make_tensor(host_y, make_shape(num));
    Tensor t_z_ref = make_tensor(host_z_ref, make_shape(num));
    cpu_const_data(&t_x, 1.0f);
    cpu_const_data(&t_y, 2.0f);
    cpu_const_data(&t_z, 8.0f);

    // Copy data to device
    cudaMemcpy(x, host_x, num * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(y, host_y, num * sizeof(half), cudaMemcpyHostToDevice);

    // run kernel
    GPU_Clock timer;
    // warm up
    run_kernel(z, num, x, y, 1.0f, 2.0f, 3.0f);
    timer.start();
    for(int i=0; i<100; i++) {
        run_kernel(z, num, x, y, 1.0f, 2.0f, 3.0f);
    }
    auto used_time = timer.milliseconds() / 100;
    print("run time: ");print(used_time);endl;

    // Copy result back to host
    cudaMemcpy(host_z_ref, z, num * sizeof(half), cudaMemcpyDeviceToHost);

    // CPU reference computation, need to convert to float, cpu can't do half
    // cpu_vector_add(host_z, num, host_x, host_y, 1.0f, 2.0f, 3.0f);
    // auto t_z_float = recast<float>(t_z); // this won't work.
    cpu_compare(t_z_ref, t_z, 0.00001f);

    // cast the 1st element to float and print
    float first_element = (float)host_z_ref[0];
    print(first_element);
}
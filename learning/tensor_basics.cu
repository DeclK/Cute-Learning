#include <cuda.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "helper.h"

using namespace cute;

int main() {
    std::cout << "=== Cutlass CuTe Tensor Creation with Real Data ===" << std::endl;
    std::cout << std::endl;

    // 1. Float tensor with actual data
    std::cout << "1. Float tensor with actual data:" << std::endl;
    
    float* float_data = new float[12]; // 3x4 = 12 elements
    auto tensor_float = make_tensor(make_gmem_ptr(float_data), make_shape(3, 4));
    auto tensor_float_1 = make_tensor(float_data, make_shape(3, 4));
    
    cpu_const_data(tensor_float, 1.0);
    print_tensor_info(tensor_float, "Float Tensor (3x4)");
    print_tensor_values(tensor_float, "Float values");

    // 2. TODO: Int8 tensor with actual data
    std::cout << "\n2. Int8 tensor with actual data:" << std::endl;
    
    int8_t* int8_data = new int8_t[8]; // 2x4 = 8 elements
    auto tensor_int8 = make_tensor(make_gmem_ptr(int8_data), make_shape(2, 4));
    return 0;
}
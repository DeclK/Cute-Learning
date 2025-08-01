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
    auto tensor_float = make_tensor(float_data, make_shape(3, 4));
    auto tensor_float_1 = make_tensor(float_data, make_shape(3, 4));
    auto tensor_float_2 = make_tensor(float_data, make_shape(3, make_shape(2, 2)));
    cpu_arange_data(&tensor_float_2);
    print(tensor_float_2(0, 0)); print("\n");

    // slice a tensor
    Tensor new_tensor = tensor_float_2(0, _);
    print_tensor(new_tensor); print("\n");

    // flatten
    print(flatten(tensor_float_2.layout())); print("\n");

    // take
    print_tensor(take<1,2>(tensor_float_2)); print("\n");

    // squeeze
    Tensor squeezed_tensor = make_tensor(float_data, append(tensor_float.layout(), make_layout(1, 0)));

    // different with torch
    auto tensor3 = make_tensor(float_data, make_shape(3, 2, 2));
    // print(tensor3(0,0)); // this would cause error
    print(tensor3(0,0,0));

    Layout x = make_layout(make_shape(128, 256), LayoutRight{});
    Layout y = make_layout(make_shape(8));
    auto identity_layout = make_identity_layout(make_shape(2, 1, 2));
    auto out = logical_divide(x, y);
    print(flatten(out));endl;
    print(group<0, 1>(out));endl;
    print(select<0, 1>(x));endl;
    print(take<0, 1>(x));endl;

    // tiler
    auto tiler1 = make_tile(_, make_shape(_8{}));
    print(tiler1);endl;
    print(logical_divide(x, tiler1));endl;
    // print(zipped_divide(x, tiler1));endl;

    auto permuted = select<1, 0>(x);
    print(permuted);endl;
    print(select<0>(x.shape()));endl;
    print(get<0>(x));
}
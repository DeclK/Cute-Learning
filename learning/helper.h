/* Helper contains the following:
1. data creation
2. results checking
3. print functions
4. cuda check
*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <stdarg.h>

#define CUDA_CHECK(err) \
  if (err != cudaSuccess) { \
    printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  }

void printf_fail(const char *fmt, ...) {
  int red = 31;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

void printf_ok(const char *fmt, ...) {
  int red = 32;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

// // Pythonic print function - prints a line and appends newline
// inline void print(const char* str) {
//     printf("%s\n", str);
// }

// // Pythonic print with printf-style formatting
// inline void print(const char* fmt, ...) {
//     va_list args;
//     va_start(args, fmt);
//     vprintf(fmt, args);
//     va_end(args);
//     printf("\n");
// }

// // Pythonic print for C++ strings
// inline void print(const std::string& str) {
//     printf("%s\n", str.c_str());
// }

template <typename T>
void cpu_rand_data(T *c) {
  auto t = *c;

  using ValueType = typename T::value_type;

  int n = size(t);
  for (int i = 0; i < n; ++i) {
    float v = ((rand() % 200) - 100.f) * 0.01f;
    // printf("v = %f\n", v);
    t(i) = ValueType(v);
  }
}

template <typename T>
void cpu_const_data(T *c, float k) {
  auto t = *c;

  int n = size(t);
  for (int i = 0; i < n; ++i) {
    t(i) = k;
  }
}

template <typename T>
void cpu_arange_data(T *c) {
  auto t = *c;
  int n = size(t);
  for (int i = 0; i < n; ++i) {
    t(i) = i;
  }
}

template <typename T>
void cpu_arange_data_T(T *c, int M, int K) {
  auto t = *c;
  int n = size(t);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      t(m, k) = m * K + k;
    }
  }
}

template <typename T>
void cpu_gemm(T *c, const T &a, const T &b) {
  using namespace cute;

  using ValueType = typename T::value_type;

  int m = size<0>(a);
  int n = size<0>(b);
  int k = size<1>(a);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float s = 0.f;

      for (int kk = 0; kk < k; ++kk) {
        float v1 = a(i, kk);
        float v2 = b(j, kk);
        s += v1 * v2;
      }

      (*c)(i, j) = ValueType(s);
    }
  }
}

template <typename T>
void cpu_compare(const T &x, const T &y, float threshold) {
  using namespace cute;

  if (size(x) != size(y)) {
    fprintf(stderr, "lenght not equal x = %d, y = %d\n", size(x), size(y));
    exit(9);
  }

  int n = size(x);
  float diff_max = 0;
  int diff_count = 0;
  for (int i = 0; i < n; ++i) {
    float v0 = x(i);
    float v1 = y(i);

    diff_max = max(diff_max, fabs(v0 - v1));

    if (fabs(v0 - v1) > threshold) {
      ++diff_count;
    }
  }
  if (diff_count > 0) {
    printf("check fail: max_diff = %f, diff_count = %d\n", diff_max,
           diff_count);
  } else {
    printf("cpu check ok\n");
  }
}

template <typename T>
__global__ void gpu_compare_kernel(const T *x, const T *y, int n,
                                   float threshold, int *count,
                                   float *max_error) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  float v0 = x[idx];
  float v1 = y[idx];

  float diff = fabs(v0 - v1);
  if (diff > threshold) {
    atomicAdd(count, 1);

    // for positive floating point, there int representation is in the same
    // order.
    int int_diff = *((int *)(&diff));
    atomicMax((int *)max_error, int_diff);
  }
}

template <typename T>
void gpu_compare(const T *x, const T *y, int n, float threshold) {
  int *num_count;
  float *max_error;
  cudaMalloc(&num_count, sizeof(int));
  cudaMalloc(&max_error, sizeof(float));
  cudaMemset(num_count, 0, sizeof(int));
  cudaMemset(max_error, 0, sizeof(float));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  gpu_compare_kernel<<<grid, block>>>(x, y, n, threshold, num_count, max_error);
  int num = 0;
  float error = 0;
  cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (num == 0) {
    printf_ok("check ok, max_error = %f\n", error);
  } else {
    float p = (100.f * num) / n;
    printf_fail("===============================\n");
    printf_fail("check fail: diff %.1f%% = %d/%d max_error = %f\n", p, num, n,
                error);
    printf_fail("===============================\n");
  }
}

template <typename TensorType>
void print_tensor_info(const TensorType& tensor, const std::string& name) {
    std::cout << "=== " << name << " ===" << std::endl;
    std::cout << "Shape: " << shape(tensor) << std::endl;
    std::cout << "Stride: " << stride(tensor) << std::endl;
    std::cout << "Layout: " << layout(tensor) << std::endl;
    std::cout << "Size: " << size(tensor) << std::endl;
}

// Helper function to print tensor values (CPU memory)
template <typename TensorType>
void print_tensor_values(const TensorType& tensor, const std::string& name, int max_elements = 16) {
    std::cout << name << ": ";
    int count = 0;
    for (int i = 0; i < size(tensor) && count < max_elements; ++i) {
        std::cout << tensor(i) << " ";
        count++;
    }
    if (size(tensor) > max_elements) {
        std::cout << "... (" << size(tensor) - max_elements << " more)";
    }
    std::cout << std::endl;
}


// use shape and seed to create random host data, return the pointer
template <typename T>
T* create_host_data(std::vector<int> shape, int seed = 0) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    T* data = (T*)malloc(size * sizeof(T));
    if (!data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    srand(seed);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<T>((rand() % 200 - 100) * 0.01f); // Random values between -1.0 and 1.0
    }
    return data;
}
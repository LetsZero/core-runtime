/**
 * @file basic_test.cpp
 * @brief Basic tests for Zero Core Runtime (no external dependencies)
 */

#include <zero/zero.hpp>
#include <cstdio>
#include <cmath>

using namespace zero;

#define ASSERT(cond, msg) \
    if (!(cond)) { \
        printf("FAIL: %s\n", msg); \
        return 1; \
    } else { \
        printf("PASS: %s\n", msg); \
    }

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_FLOAT_EQ(a, b, msg) ASSERT(std::abs((a) - (b)) < 1e-5f, msg)

int main() {
    printf("=== Zero Core Runtime Tests ===\n\n");
    
    // DType tests
    printf("--- DType Tests ---\n");
    ASSERT_EQ(dtype_size(DType::F32), 4u, "F32 size is 4");
    ASSERT_EQ(dtype_size(DType::I64), 8u, "I64 size is 8");
    ASSERT(dtype_is_float(DType::F32), "F32 is float");
    ASSERT(!dtype_is_float(DType::I32), "I32 is not float");
    
    // Tensor tests
    printf("\n--- Tensor Tests ---\n");
    int64_t shape[] = {2, 3};
    Tensor t = Tensor::alloc(shape, 2, DType::F32);
    ASSERT(t.data != nullptr, "Tensor allocation");
    ASSERT_EQ(t.numel(), 6, "Tensor numel");
    ASSERT_EQ(t.nbytes(), 24u, "Tensor nbytes");
    ASSERT(t.is_contiguous(), "Tensor is contiguous");
    
    // Fill tensor
    float* data = static_cast<float*>(t.data);
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    // Reshape test
    int64_t new_shape[] = {3, 2};
    Tensor reshaped = t.reshape(new_shape, 2);
    ASSERT_EQ(reshaped.shape[0], 3, "Reshape dim 0");
    ASSERT_EQ(reshaped.shape[1], 2, "Reshape dim 1");
    ASSERT(!reshaped.owns_data, "View doesn't own data");
    
    // Transpose test
    Tensor transposed = t.transpose();
    ASSERT_EQ(transposed.shape[0], 3, "Transpose dim 0");
    ASSERT_EQ(transposed.shape[1], 2, "Transpose dim 1");
    
    t.free();
    ASSERT(t.data == nullptr, "Tensor freed");
    
    // Scalar tests
    printf("\n--- Scalar Tests ---\n");
    Scalar s1 = Scalar(3.14f);
    ASSERT_EQ(s1.dtype, DType::F32, "Scalar dtype");
    ASSERT_FLOAT_EQ(s1.to_f32(), 3.14f, "Scalar value");
    ASSERT_EQ(s1.to_i64(), 3, "Scalar to int");
    
    // Ops tests
    printf("\n--- Operations Tests ---\n");
    int64_t vec_shape[] = {4};
    Tensor a = Tensor::alloc(vec_shape, 1, DType::F32);
    Tensor b = Tensor::alloc(vec_shape, 1, DType::F32);
    Tensor c = Tensor::alloc(vec_shape, 1, DType::F32);
    
    float* a_data = static_cast<float*>(a.data);
    float* b_data = static_cast<float*>(b.data);
    for (int i = 0; i < 4; ++i) {
        a_data[i] = static_cast<float>(i + 1);  // [1, 2, 3, 4]
        b_data[i] = static_cast<float>(i + 5);  // [5, 6, 7, 8]
    }
    
    ops::add(a, b, c);
    float* c_data = static_cast<float*>(c.data);
    ASSERT_FLOAT_EQ(c_data[0], 6.0f, "Add [0]");
    ASSERT_FLOAT_EQ(c_data[3], 12.0f, "Add [3]");
    
    ops::mul(a, b, c);
    ASSERT_FLOAT_EQ(c_data[0], 5.0f, "Mul [0]");
    ASSERT_FLOAT_EQ(c_data[3], 32.0f, "Mul [3]");
    
    // Reduce tests
    ASSERT_FLOAT_EQ(ops::sum_all(a), 10.0f, "Sum all");
    ASSERT_FLOAT_EQ(ops::max_all(a), 4.0f, "Max all");
    ASSERT_FLOAT_EQ(ops::mean_all(a), 2.5f, "Mean all");
    
    a.free();
    b.free();
    c.free();
    
    // Matmul test
    printf("\n--- Matmul Tests ---\n");
    int64_t A_shape[] = {2, 3};
    int64_t B_shape[] = {3, 2};
    int64_t C_shape[] = {2, 2};
    
    Tensor A = Tensor::alloc(A_shape, 2, DType::F32);
    Tensor B = Tensor::alloc(B_shape, 2, DType::F32);
    Tensor C = Tensor::alloc(C_shape, 2, DType::F32);
    
    float* A_data = static_cast<float*>(A.data);
    float* B_data = static_cast<float*>(B.data);
    for (int i = 0; i < 6; ++i) {
        A_data[i] = static_cast<float>(i + 1);
        B_data[i] = static_cast<float>(i + 1);
    }
    
    ops::matmul(A, B, C);
    float* C_data = static_cast<float*>(C.data);
    ASSERT_FLOAT_EQ(C_data[0], 22.0f, "Matmul [0,0]");
    ASSERT_FLOAT_EQ(C_data[1], 28.0f, "Matmul [0,1]");
    ASSERT_FLOAT_EQ(C_data[2], 49.0f, "Matmul [1,0]");
    ASSERT_FLOAT_EQ(C_data[3], 64.0f, "Matmul [1,1]");
    
    A.free();
    B.free();
    C.free();
    
    printf("\n=== All Tests Passed ===\n");
    return 0;
}

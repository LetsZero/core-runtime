/**
 * @file test_ops.cpp
 * @brief Tests for tensor operations
 */

#include <gtest/gtest.h>
#include <zero/zero.hpp>
#include <cmath>

using namespace zero;
using namespace zero::ops;

class TensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        int64_t shape[] = {4};
        a = Tensor::alloc(shape, 1, DType::F32);
        b = Tensor::alloc(shape, 1, DType::F32);
        c = Tensor::alloc(shape, 1, DType::F32);
        
        float* a_data = static_cast<float*>(a.data);
        float* b_data = static_cast<float*>(b.data);
        
        for (int i = 0; i < 4; ++i) {
            a_data[i] = static_cast<float>(i + 1);  // [1, 2, 3, 4]
            b_data[i] = static_cast<float>(i + 5);  // [5, 6, 7, 8]
        }
    }
    
    void TearDown() override {
        a.free();
        b.free();
        c.free();
    }
    
    Tensor a, b, c;
};

TEST_F(TensorOpsTest, Add) {
    add(a, b, c);
    
    float* result = static_cast<float*>(c.data);
    EXPECT_FLOAT_EQ(result[0], 6.0f);   // 1 + 5
    EXPECT_FLOAT_EQ(result[1], 8.0f);   // 2 + 6
    EXPECT_FLOAT_EQ(result[2], 10.0f);  // 3 + 7
    EXPECT_FLOAT_EQ(result[3], 12.0f);  // 4 + 8
}

TEST_F(TensorOpsTest, Mul) {
    mul(a, b, c);
    
    float* result = static_cast<float*>(c.data);
    EXPECT_FLOAT_EQ(result[0], 5.0f);   // 1 * 5
    EXPECT_FLOAT_EQ(result[1], 12.0f);  // 2 * 6
    EXPECT_FLOAT_EQ(result[2], 21.0f);  // 3 * 7
    EXPECT_FLOAT_EQ(result[3], 32.0f);  // 4 * 8
}

TEST_F(TensorOpsTest, Exp) {
    exp(a, c);
    
    float* a_data = static_cast<float*>(a.data);
    float* result = static_cast<float*>(c.data);
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], std::exp(a_data[i]));
    }
}

TEST_F(TensorOpsTest, Reduce) {
    EXPECT_FLOAT_EQ(sum_all(a), 10.0f);  // 1+2+3+4
    EXPECT_FLOAT_EQ(max_all(a), 4.0f);
    EXPECT_FLOAT_EQ(min_all(a), 1.0f);
    EXPECT_FLOAT_EQ(mean_all(a), 2.5f);
}

TEST(MatmulTest, Basic) {
    int64_t a_shape[] = {2, 3};
    int64_t b_shape[] = {3, 2};
    int64_t c_shape[] = {2, 2};
    
    Tensor A = Tensor::alloc(a_shape, 2, DType::F32);
    Tensor B = Tensor::alloc(b_shape, 2, DType::F32);
    Tensor C = Tensor::alloc(c_shape, 2, DType::F32);
    
    // A = [[1,2,3], [4,5,6]]
    float* a_data = static_cast<float*>(A.data);
    for (int i = 0; i < 6; ++i) a_data[i] = static_cast<float>(i + 1);
    
    // B = [[1,2], [3,4], [5,6]]
    float* b_data = static_cast<float*>(B.data);
    for (int i = 0; i < 6; ++i) b_data[i] = static_cast<float>(i + 1);
    
    matmul(A, B, C);
    
    float* c_data = static_cast<float*>(C.data);
    // C[0,0] = 1*1 + 2*3 + 3*5 = 22
    // C[0,1] = 1*2 + 2*4 + 3*6 = 28
    // C[1,0] = 4*1 + 5*3 + 6*5 = 49
    // C[1,1] = 4*2 + 5*4 + 6*6 = 64
    EXPECT_FLOAT_EQ(c_data[0], 22.0f);
    EXPECT_FLOAT_EQ(c_data[1], 28.0f);
    EXPECT_FLOAT_EQ(c_data[2], 49.0f);
    EXPECT_FLOAT_EQ(c_data[3], 64.0f);
    
    A.free();
    B.free();
    C.free();
}

TEST(ReshapeTest, BroadcastShape) {
    int64_t a_shape[] = {3, 1};
    int64_t b_shape[] = {1, 4};
    int64_t out_shape[MAX_DIMS];
    int8_t out_ndim;
    
    bool ok = broadcast_shape(a_shape, 2, b_shape, 2, out_shape, out_ndim);
    
    EXPECT_TRUE(ok);
    EXPECT_EQ(out_ndim, 2);
    EXPECT_EQ(out_shape[0], 3);
    EXPECT_EQ(out_shape[1], 4);
}

TEST(ReshapeTest, Squeeze) {
    int64_t shape[] = {1, 3, 1, 4, 1};
    int64_t strides[] = {48, 16, 16, 4, 4};
    
    Tensor t = Tensor::view(nullptr, shape, strides, 5, DType::F32);
    Tensor squeezed = squeeze(t);
    
    EXPECT_EQ(squeezed.ndim, 2);
    EXPECT_EQ(squeezed.shape[0], 3);
    EXPECT_EQ(squeezed.shape[1], 4);
}

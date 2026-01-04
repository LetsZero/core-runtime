/**
 * @file test_tensor.cpp
 * @brief Tests for Tensor primitive
 */

#include <gtest/gtest.h>
#include <zero/zero.hpp>

using namespace zero;

TEST(TensorTest, Allocation) {
    int64_t shape[] = {2, 3, 4};
    Tensor t = Tensor::alloc(shape, 3, DType::F32);
    
    EXPECT_NE(t.data, nullptr);
    EXPECT_EQ(t.ndim, 3);
    EXPECT_EQ(t.shape[0], 2);
    EXPECT_EQ(t.shape[1], 3);
    EXPECT_EQ(t.shape[2], 4);
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.nbytes(), 24 * 4);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_TRUE(t.owns_data);
    
    t.free();
    EXPECT_EQ(t.data, nullptr);
}

TEST(TensorTest, View) {
    int64_t shape[] = {4, 4};
    Tensor t = Tensor::alloc(shape, 2, DType::F32);
    
    // Fill with data
    float* data = static_cast<float*>(t.data);
    for (int i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Create view
    int64_t view_shape[] = {2, 8};
    Tensor v = t.reshape(view_shape, 2);
    
    EXPECT_EQ(v.data, t.data);  // Same underlying data
    EXPECT_EQ(v.shape[0], 2);
    EXPECT_EQ(v.shape[1], 8);
    EXPECT_FALSE(v.owns_data);  // View doesn't own data
    
    t.free();
}

TEST(TensorTest, Transpose) {
    int64_t shape[] = {3, 4};
    Tensor t = Tensor::alloc(shape, 2, DType::F32);
    
    Tensor transposed = t.transpose();
    
    EXPECT_EQ(transposed.shape[0], 4);
    EXPECT_EQ(transposed.shape[1], 3);
    EXPECT_FALSE(transposed.owns_data);
    
    t.free();
}

TEST(TensorTest, Slice) {
    int64_t shape[] = {10};
    Tensor t = Tensor::alloc(shape, 1, DType::F32);
    
    float* data = static_cast<float*>(t.data);
    for (int i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    Tensor s = t.slice(0, 2, 7);
    
    EXPECT_EQ(s.shape[0], 5);
    float* slice_data = static_cast<float*>(s.data);
    EXPECT_EQ(slice_data[0], 2.0f);
    
    t.free();
}

TEST(TensorTest, Scalar) {
    Tensor t = Tensor::empty();
    t.ndim = 0;
    
    EXPECT_TRUE(t.is_scalar());
    EXPECT_EQ(t.numel(), 1);
}

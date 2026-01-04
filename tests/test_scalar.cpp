/**
 * @file test_scalar.cpp
 * @brief Tests for Scalar primitive
 */

#include <gtest/gtest.h>
#include <zero/zero.hpp>

using namespace zero;

TEST(ScalarTest, Constructors) {
    Scalar f = Scalar(3.14f);
    EXPECT_EQ(f.dtype, DType::F32);
    EXPECT_FLOAT_EQ(f.to_f32(), 3.14f);
    
    Scalar i = Scalar(42);
    EXPECT_EQ(i.dtype, DType::I32);
    EXPECT_EQ(i.to_i64(), 42);
    
    Scalar b = Scalar(true);
    EXPECT_EQ(b.dtype, DType::Bool);
    EXPECT_TRUE(b.to_bool());
}

TEST(ScalarTest, Conversions) {
    Scalar f = Scalar(3.14f);
    EXPECT_EQ(f.to_i64(), 3);
    EXPECT_TRUE(f.to_bool());
    
    Scalar zero = Scalar(0.0f);
    EXPECT_FALSE(zero.to_bool());
}

TEST(ScalarTest, Constants) {
    EXPECT_FLOAT_EQ(constants::ZERO_F32.to_f32(), 0.0f);
    EXPECT_FLOAT_EQ(constants::ONE_F32.to_f32(), 1.0f);
    EXPECT_EQ(constants::ZERO_I32.to_i64(), 0);
    EXPECT_EQ(constants::ONE_I32.to_i64(), 1);
}

TEST(ScalarTest, ByteRoundtrip) {
    Scalar original = Scalar(123.456f);
    
    uint8_t buffer[8];
    original.to_bytes(buffer);
    
    Scalar restored = Scalar::from_bytes(buffer, DType::F32);
    EXPECT_FLOAT_EQ(restored.to_f32(), 123.456f);
}

/**
 * @file test_dtype.cpp
 * @brief Tests for dtype system
 */

#include <gtest/gtest.h>
#include <zero/zero.hpp>

using namespace zero;

TEST(DTypeTest, Sizes) {
    EXPECT_EQ(dtype_size(DType::F16), 2);
    EXPECT_EQ(dtype_size(DType::F32), 4);
    EXPECT_EQ(dtype_size(DType::F64), 8);
    EXPECT_EQ(dtype_size(DType::I8), 1);
    EXPECT_EQ(dtype_size(DType::I16), 2);
    EXPECT_EQ(dtype_size(DType::I32), 4);
    EXPECT_EQ(dtype_size(DType::I64), 8);
    EXPECT_EQ(dtype_size(DType::Bool), 1);
}

TEST(DTypeTest, TypeChecks) {
    EXPECT_TRUE(dtype_is_float(DType::F32));
    EXPECT_TRUE(dtype_is_float(DType::F64));
    EXPECT_FALSE(dtype_is_float(DType::I32));
    
    EXPECT_TRUE(dtype_is_signed(DType::I32));
    EXPECT_FALSE(dtype_is_signed(DType::U32));
    
    EXPECT_TRUE(dtype_is_unsigned(DType::U8));
    EXPECT_FALSE(dtype_is_unsigned(DType::I8));
}

TEST(DTypeTest, Names) {
    EXPECT_STREQ(dtype_name(DType::F32), "f32");
    EXPECT_STREQ(dtype_name(DType::I64), "i64");
    EXPECT_STREQ(dtype_name(DType::Bool), "bool");
}

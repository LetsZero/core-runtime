#pragma once

/**
 * @file dtype.hpp
 * @brief Zero Core Runtime — Data Type Definitions
 * 
 * All supported numeric types for Tensor and Scalar primitives.
 * This is the fundamental type system of Zero.
 */

#include <cstdint>
#include <cstddef>

namespace zero {

/**
 * @brief Enumeration of all supported data types
 */
enum class DType : uint8_t {
    // Floating point
    F16 = 0,    // 16-bit float (half precision)
    F32 = 1,    // 32-bit float (single precision)
    F64 = 2,    // 64-bit float (double precision)
    
    // Signed integers
    I8  = 3,    // 8-bit signed integer
    I16 = 4,    // 16-bit signed integer
    I32 = 5,    // 32-bit signed integer
    I64 = 6,    // 64-bit signed integer
    
    // Unsigned integers
    U8  = 7,    // 8-bit unsigned integer
    U16 = 8,    // 16-bit unsigned integer
    U32 = 9,    // 32-bit unsigned integer
    U64 = 10,   // 64-bit unsigned integer
    
    // Boolean
    Bool = 11,  // Boolean (stored as uint8_t)
    
    // Special
    BF16 = 12,  // Brain floating point (for ML)
};

/**
 * @brief Get the size in bytes of a data type
 */
constexpr size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::F16:  return 2;
        case DType::F32:  return 4;
        case DType::F64:  return 8;
        case DType::I8:   return 1;
        case DType::I16:  return 2;
        case DType::I32:  return 4;
        case DType::I64:  return 8;
        case DType::U8:   return 1;
        case DType::U16:  return 2;
        case DType::U32:  return 4;
        case DType::U64:  return 8;
        case DType::Bool: return 1;
        case DType::BF16: return 2;
    }
    return 0; // Unreachable
}

/**
 * @brief Get the alignment requirement of a data type
 */
constexpr size_t dtype_alignment(DType dtype) noexcept {
    return dtype_size(dtype); // Natural alignment
}

/**
 * @brief Check if dtype is a floating point type
 */
constexpr bool dtype_is_float(DType dtype) noexcept {
    return dtype == DType::F16 || dtype == DType::F32 || 
           dtype == DType::F64 || dtype == DType::BF16;
}

/**
 * @brief Check if dtype is a signed integer type
 */
constexpr bool dtype_is_signed(DType dtype) noexcept {
    return dtype == DType::I8 || dtype == DType::I16 || 
           dtype == DType::I32 || dtype == DType::I64;
}

/**
 * @brief Check if dtype is an unsigned integer type
 */
constexpr bool dtype_is_unsigned(DType dtype) noexcept {
    return dtype == DType::U8 || dtype == DType::U16 || 
           dtype == DType::U32 || dtype == DType::U64 || dtype == DType::Bool;
}

/**
 * @brief Get human-readable name for a dtype
 */
constexpr const char* dtype_name(DType dtype) noexcept {
    switch (dtype) {
        case DType::F16:  return "f16";
        case DType::F32:  return "f32";
        case DType::F64:  return "f64";
        case DType::I8:   return "i8";
        case DType::I16:  return "i16";
        case DType::I32:  return "i32";
        case DType::I64:  return "i64";
        case DType::U8:   return "u8";
        case DType::U16:  return "u16";
        case DType::U32:  return "u32";
        case DType::U64:  return "u64";
        case DType::Bool: return "bool";
        case DType::BF16: return "bf16";
    }
    return "unknown";
}

} // namespace zero

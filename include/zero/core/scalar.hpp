#pragma once

/**
 * @file scalar.hpp
 * @brief Zero Core Runtime — Scalar Primitive
 * 
 * A rank-0 tensor or immediate value.
 * Used for loop bounds, hyperparameters, and constants.
 */

#include "dtype.hpp"
#include <cstdint>
#include <cstring>

namespace zero {

/**
 * @brief Scalar value that can hold any dtype
 * 
 * Design principles:
 * - Union storage for type-punning
 * - Constexpr constructors for compile-time folding
 * - Easy conversion to/from Tensor
 */
struct Scalar {
    union {
        float    f32;
        double   f64;
        int8_t   i8;
        int16_t  i16;
        int32_t  i32;
        int64_t  i64;
        uint8_t  u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        bool     b;
        uint16_t f16_bits;  // F16/BF16 stored as bits
    } value;
    
    DType dtype;
    
    // ─────────────────────────────────────────────────────────────────
    // Constructors
    // ─────────────────────────────────────────────────────────────────
    
    constexpr Scalar() noexcept : value{.f32 = 0.0f}, dtype(DType::F32) {}
    
    constexpr Scalar(float v) noexcept : value{.f32 = v}, dtype(DType::F32) {}
    
    constexpr Scalar(double v) noexcept : value{.f64 = v}, dtype(DType::F64) {}
    
    constexpr Scalar(int32_t v) noexcept : value{.i32 = v}, dtype(DType::I32) {}
    
    constexpr Scalar(int64_t v) noexcept : value{.i64 = v}, dtype(DType::I64) {}
    
    constexpr Scalar(bool v) noexcept : value{.b = v}, dtype(DType::Bool) {}
    
    // ─────────────────────────────────────────────────────────────────
    // Accessors (type-checked)
    // ─────────────────────────────────────────────────────────────────
    
    float to_f32() const noexcept {
        switch (dtype) {
            case DType::F32:  return value.f32;
            case DType::F64:  return static_cast<float>(value.f64);
            case DType::I32:  return static_cast<float>(value.i32);
            case DType::I64:  return static_cast<float>(value.i64);
            case DType::Bool: return value.b ? 1.0f : 0.0f;
            default:          return 0.0f;
        }
    }
    
    double to_f64() const noexcept {
        switch (dtype) {
            case DType::F64:  return value.f64;
            case DType::F32:  return static_cast<double>(value.f32);
            case DType::I32:  return static_cast<double>(value.i32);
            case DType::I64:  return static_cast<double>(value.i64);
            case DType::Bool: return value.b ? 1.0 : 0.0;
            default:          return 0.0;
        }
    }
    
    int64_t to_i64() const noexcept {
        switch (dtype) {
            case DType::I64:  return value.i64;
            case DType::I32:  return static_cast<int64_t>(value.i32);
            case DType::I16:  return static_cast<int64_t>(value.i16);
            case DType::I8:   return static_cast<int64_t>(value.i8);
            case DType::F32:  return static_cast<int64_t>(value.f32);
            case DType::F64:  return static_cast<int64_t>(value.f64);
            case DType::Bool: return value.b ? 1 : 0;
            default:          return 0;
        }
    }
    
    bool to_bool() const noexcept {
        switch (dtype) {
            case DType::Bool: return value.b;
            case DType::I32:  return value.i32 != 0;
            case DType::I64:  return value.i64 != 0;
            case DType::F32:  return value.f32 != 0.0f;
            case DType::F64:  return value.f64 != 0.0;
            default:          return false;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Raw Access
    // ─────────────────────────────────────────────────────────────────
    
    /**
     * @brief Copy raw bytes to destination
     */
    void to_bytes(void* dst) const noexcept {
        std::memcpy(dst, &value, dtype_size(dtype));
    }
    
    /**
     * @brief Create scalar from raw bytes
     */
    static Scalar from_bytes(const void* src, DType dt) noexcept {
        Scalar s;
        s.dtype = dt;
        std::memcpy(&s.value, src, dtype_size(dt));
        return s;
    }
};

// ─────────────────────────────────────────────────────────────────────
// Compile-time Constants
// ─────────────────────────────────────────────────────────────────────

namespace constants {
    constexpr Scalar ZERO_F32 = Scalar(0.0f);
    constexpr Scalar ONE_F32  = Scalar(1.0f);
    constexpr Scalar ZERO_I32 = Scalar(0);
    constexpr Scalar ONE_I32  = Scalar(1);
    constexpr Scalar TRUE_VAL = Scalar(true);
    constexpr Scalar FALSE_VAL = Scalar(false);
}

} // namespace zero

#pragma once

/**
 * @file scalar.hpp
 * @brief Zero Core Runtime — Scalar Primitive
 * 
 * A rank-0 tensor or immediate value.
 * Used for loop bounds, hyperparameters, and constants.
 * 
 * NOTE: Scalar represents a value, not conversion policy.
 * All cross-dtype conversions are LOSSY and UNCHECKED.
 */

#include "dtype.hpp"
#include <cstdint>
#include <cstring>
#include <cstdio>

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
        uint16_t f16_bits;  // F16/BF16 stored as opaque bits (no decode support)
    } value;
    
    DType dtype;
    
    // ─────────────────────────────────────────────────────────────────
    // Constructors (full dtype coverage)
    // ─────────────────────────────────────────────────────────────────
    
    constexpr Scalar() noexcept : value{.f32 = 0.0f}, dtype(DType::F32) {}
    
    // Floating point
    constexpr Scalar(float v) noexcept : value{.f32 = v}, dtype(DType::F32) {}
    constexpr Scalar(double v) noexcept : value{.f64 = v}, dtype(DType::F64) {}
    
    // Signed integers
    constexpr Scalar(int8_t v) noexcept : value{.i8 = v}, dtype(DType::I8) {}
    constexpr Scalar(int16_t v) noexcept : value{.i16 = v}, dtype(DType::I16) {}
    constexpr Scalar(int32_t v) noexcept : value{.i32 = v}, dtype(DType::I32) {}
    constexpr Scalar(int64_t v) noexcept : value{.i64 = v}, dtype(DType::I64) {}
    
    // Unsigned integers
    constexpr Scalar(uint8_t v) noexcept : value{.u8 = v}, dtype(DType::U8) {}
    constexpr Scalar(uint16_t v) noexcept : value{.u16 = v}, dtype(DType::U16) {}
    constexpr Scalar(uint32_t v) noexcept : value{.u32 = v}, dtype(DType::U32) {}
    constexpr Scalar(uint64_t v) noexcept : value{.u64 = v}, dtype(DType::U64) {}
    
    // Boolean
    constexpr Scalar(bool v) noexcept : value{.b = v}, dtype(DType::Bool) {}
    
    // F16/BF16 from opaque bits (no math support)
    static Scalar from_f16_bits(uint16_t bits) noexcept {
        Scalar s;
        s.value.f16_bits = bits;
        s.dtype = DType::F16;
        return s;
    }
    
    static Scalar from_bf16_bits(uint16_t bits) noexcept {
        Scalar s;
        s.value.f16_bits = bits;
        s.dtype = DType::BF16;
        return s;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Type Queries (for graph attributes and dispatch)
    // ─────────────────────────────────────────────────────────────────
    
    bool is_integer() const noexcept {
        return dtype_is_signed(dtype) || dtype_is_unsigned(dtype);
    }
    
    bool is_floating() const noexcept {
        return dtype_is_float(dtype);
    }
    
    bool is_signed() const noexcept {
        return dtype_is_signed(dtype) || dtype_is_float(dtype);
    }
    
    bool is_logical() const noexcept {
        return dtype_is_logical(dtype);
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Accessors (LOSSY, UNCHECKED conversions)
    // Scalar does not decide conversion policy — these are convenience only.
    // ─────────────────────────────────────────────────────────────────
    
    float to_f32() const noexcept {
        switch (dtype) {
            case DType::F32:  return value.f32;
            case DType::F64:  return static_cast<float>(value.f64);
            case DType::I8:   return static_cast<float>(value.i8);
            case DType::I16:  return static_cast<float>(value.i16);
            case DType::I32:  return static_cast<float>(value.i32);
            case DType::I64:  return static_cast<float>(value.i64);
            case DType::U8:   return static_cast<float>(value.u8);
            case DType::U16:  return static_cast<float>(value.u16);
            case DType::U32:  return static_cast<float>(value.u32);
            case DType::U64:  return static_cast<float>(value.u64);
            case DType::Bool: return value.b ? 1.0f : 0.0f;
            default:          return 0.0f;
        }
    }
    
    double to_f64() const noexcept {
        switch (dtype) {
            case DType::F64:  return value.f64;
            case DType::F32:  return static_cast<double>(value.f32);
            case DType::I8:   return static_cast<double>(value.i8);
            case DType::I16:  return static_cast<double>(value.i16);
            case DType::I32:  return static_cast<double>(value.i32);
            case DType::I64:  return static_cast<double>(value.i64);
            case DType::U8:   return static_cast<double>(value.u8);
            case DType::U16:  return static_cast<double>(value.u16);
            case DType::U32:  return static_cast<double>(value.u32);
            case DType::U64:  return static_cast<double>(value.u64);
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
            case DType::U64:  return static_cast<int64_t>(value.u64);
            case DType::U32:  return static_cast<int64_t>(value.u32);
            case DType::U16:  return static_cast<int64_t>(value.u16);
            case DType::U8:   return static_cast<int64_t>(value.u8);
            case DType::F32:  return static_cast<int64_t>(value.f32);
            case DType::F64:  return static_cast<int64_t>(value.f64);
            case DType::Bool: return value.b ? 1 : 0;
            default:          return 0;
        }
    }
    
    bool to_bool() const noexcept {
        switch (dtype) {
            case DType::Bool: return value.b;
            case DType::I8:   return value.i8 != 0;
            case DType::I16:  return value.i16 != 0;
            case DType::I32:  return value.i32 != 0;
            case DType::I64:  return value.i64 != 0;
            case DType::U8:   return value.u8 != 0;
            case DType::U16:  return value.u16 != 0;
            case DType::U32:  return value.u32 != 0;
            case DType::U64:  return value.u64 != 0;
            case DType::F32:  return value.f32 != 0.0f;
            case DType::F64:  return value.f64 != 0.0;
            default:          return false;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Scalar Arithmetic (constexpr-friendly, for graph/compile time)
    // Same-class promotion: int+int→i64, float+float→f64
    // Mixed class (int↔float) NOT supported, returns zero.
    // ─────────────────────────────────────────────────────────────────
    
    Scalar add(const Scalar& other) const noexcept {
        if (is_floating() && other.is_floating()) {
            return Scalar(to_f64() + other.to_f64());
        }
        if (is_integer() && other.is_integer()) {
            return Scalar(to_i64() + other.to_i64());
        }
        return Scalar(); // Type mismatch
    }
    
    Scalar sub(const Scalar& other) const noexcept {
        if (is_floating() && other.is_floating()) {
            return Scalar(to_f64() - other.to_f64());
        }
        if (is_integer() && other.is_integer()) {
            return Scalar(to_i64() - other.to_i64());
        }
        return Scalar();
    }
    
    Scalar mul(const Scalar& other) const noexcept {
        if (is_floating() && other.is_floating()) {
            return Scalar(to_f64() * other.to_f64());
        }
        if (is_integer() && other.is_integer()) {
            return Scalar(to_i64() * other.to_i64());
        }
        return Scalar();
    }
    
    Scalar div(const Scalar& other) const noexcept {
        if (is_floating() && other.is_floating()) {
            double d = other.to_f64();
            if (d == 0.0) return Scalar();
            return Scalar(to_f64() / d);
        }
        if (is_integer() && other.is_integer()) {
            int64_t d = other.to_i64();
            if (d == 0) return Scalar();
            return Scalar(to_i64() / d);
        }
        return Scalar();
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
     * @brief Create scalar from raw bytes (UNSAFE)
     * 
     * WARNING: Low-level escape hatch. Bypasses constructors.
     * May create semantically invalid scalar states.
     * Intended for Tensor internals only.
     */
    static Scalar from_bytes(const void* src, DType dt) noexcept {
        Scalar s;
        s.dtype = dt;
        std::memcpy(&s.value, src, dtype_size(dt));
        return s;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Debug Utilities (guarded, zero-cost when off)
    // ─────────────────────────────────────────────────────────────────
    
#ifndef NDEBUG
    void debug_print() const noexcept {
        switch (dtype) {
            case DType::F32:  std::printf("Scalar(f32: %f)\n", value.f32); break;
            case DType::F64:  std::printf("Scalar(f64: %lf)\n", value.f64); break;
            case DType::I8:   std::printf("Scalar(i8: %d)\n", value.i8); break;
            case DType::I16:  std::printf("Scalar(i16: %d)\n", value.i16); break;
            case DType::I32:  std::printf("Scalar(i32: %d)\n", value.i32); break;
            case DType::I64:  std::printf("Scalar(i64: %lld)\n", static_cast<long long>(value.i64)); break;
            case DType::U8:   std::printf("Scalar(u8: %u)\n", value.u8); break;
            case DType::U16:  std::printf("Scalar(u16: %u)\n", value.u16); break;
            case DType::U32:  std::printf("Scalar(u32: %u)\n", value.u32); break;
            case DType::U64:  std::printf("Scalar(u64: %llu)\n", static_cast<unsigned long long>(value.u64)); break;
            case DType::Bool: std::printf("Scalar(bool: %s)\n", value.b ? "true" : "false"); break;
            case DType::F16:  std::printf("Scalar(f16: bits=0x%04x)\n", value.f16_bits); break;
            case DType::BF16: std::printf("Scalar(bf16: bits=0x%04x)\n", value.f16_bits); break;
        }
    }
#else
    void debug_print() const noexcept {}
#endif
};

// ─────────────────────────────────────────────────────────────────────
// Compile-time Constants (convenience only, not policy)
// ─────────────────────────────────────────────────────────────────────

namespace constants {
    constexpr Scalar ZERO_F32 = Scalar(0.0f);
    constexpr Scalar ONE_F32  = Scalar(1.0f);
    constexpr Scalar ZERO_I32 = Scalar(static_cast<int32_t>(0));
    constexpr Scalar ONE_I32  = Scalar(static_cast<int32_t>(1));
    constexpr Scalar TRUE_VAL = Scalar(true);
    constexpr Scalar FALSE_VAL = Scalar(false);
}

} // namespace zero

// ─────────────────────────────────────────────────────────────────────
// Tensor ↔ Scalar Bridge (after both types defined)
// ─────────────────────────────────────────────────────────────────────

#include "tensor.hpp"

namespace zero {

/**
 * @brief Create a rank-0 tensor from scalar value
 */
inline Tensor Tensor::from_scalar(const Scalar& s, Device device) noexcept {
    Tensor t = Tensor::empty();
    t.dtype = s.dtype;
    t.device = device;
    t.ndim = 0;  // Scalar tensor
    
    // Allocate single element
    size_t bytes = dtype_size(s.dtype);
    t.data = mem_alloc(bytes, dtype_alignment(s.dtype), device);
    t.owns_data = (t.data != nullptr);
    
    if (t.data != nullptr) {
        s.to_bytes(t.data);
    }
    
    return t;
}

/**
 * @brief Convert rank-0 tensor to Scalar
 */
inline Scalar Tensor::to_scalar() const noexcept {
    if (ndim != 0 || data == nullptr) {
        return Scalar();  // Invalid, return zero
    }
    return Scalar::from_bytes(data, dtype);
}

} // namespace zero

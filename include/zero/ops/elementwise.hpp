#pragma once

/**
 * @file elementwise.hpp
 * @brief Zero Core Runtime — Elementwise Operations
 * 
 * Elementwise operations on tensors.
 * 
 * NOTE: Activations (relu, sigmoid, tanh) are unary, shape-preserving ops.
 * Broadcasting is a frontend concern, not a runtime concern.
 */

#include "../core/tensor.hpp"
#include "../core/scalar.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>

namespace zero {
namespace ops {

// ─────────────────────────────────────────────────────────────────────
// Elementwise Operation Types
// ─────────────────────────────────────────────────────────────────────

enum class ElementwiseOp : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    NEG = 4,
    ABS = 5,
    EXP = 6,
    LOG = 7,
    SQRT = 8,
    SIN = 9,
    COS = 10,
    TANH = 11,
    RELU = 12,     // v1.1: max(0, x)
    SIGMOID = 13,  // v1.1: 1 / (1 + exp(-x))
};

// ─────────────────────────────────────────────────────────────────────
// Unary Operations (in-place capable)
// ─────────────────────────────────────────────────────────────────────

/**
 * @brief Apply unary operation to tensor
 * 
 * @param input  Input tensor
 * @param output Output tensor (can be same as input for in-place)
 * @param op     Operation to apply
 */
inline void unary_op(const Tensor& input, Tensor& output, ElementwiseOp op) noexcept {
    // CPU-only implementation
    if (input.device != Device::CPU || output.device != Device::CPU) return;
    
    // Debug assertions (release build: silent return)
#ifndef NDEBUG
    assert(input.data != nullptr && "unary_op: input.data is null");
    assert(output.data != nullptr && "unary_op: output.data is null");
    assert(input.numel() == output.numel() && "unary_op: shape mismatch");
    assert(input.ndim == output.ndim && "unary_op: ndim mismatch");
#endif
    
    if (input.dtype != DType::F32 || output.dtype != DType::F32) return;
    if (input.data == nullptr || output.data == nullptr) return;
    if (input.numel() != output.numel()) return;
    
    const float* in_ptr = static_cast<const float*>(input.data);
    float* out_ptr = static_cast<float*>(output.data);
    int64_t n = input.numel();
    
    switch (op) {
        case ElementwiseOp::NEG:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = -in_ptr[i];
            break;
        case ElementwiseOp::ABS:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = std::abs(in_ptr[i]);
            break;
        case ElementwiseOp::EXP:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = std::exp(in_ptr[i]);
            break;
        case ElementwiseOp::LOG:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = std::log(in_ptr[i]);
            break;
        case ElementwiseOp::SQRT:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = std::sqrt(in_ptr[i]);
            break;
        case ElementwiseOp::SIN:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = std::sin(in_ptr[i]);
            break;
        case ElementwiseOp::COS:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = std::cos(in_ptr[i]);
            break;
        case ElementwiseOp::TANH:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = std::tanh(in_ptr[i]);
            break;
        case ElementwiseOp::RELU:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
            break;
        case ElementwiseOp::SIGMOID:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = 1.0f / (1.0f + std::exp(-in_ptr[i]));
            break;
        default:
            break;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Binary Operations (broadcast-aware)
// ─────────────────────────────────────────────────────────────────────

/**
 * @brief Apply binary operation to two tensors
 * 
 * Assumes contiguous tensors with compatible shapes.
 */
inline void binary_op(
    const Tensor& a, 
    const Tensor& b, 
    Tensor& output, 
    ElementwiseOp op
) noexcept {
    // CPU-only implementation
    if (a.device != Device::CPU || b.device != Device::CPU || output.device != Device::CPU) return;
    
    if (a.dtype != DType::F32 || b.dtype != DType::F32) return;
    
    const float* a_ptr = static_cast<const float*>(a.data);
    const float* b_ptr = static_cast<const float*>(b.data);
    float* out_ptr = static_cast<float*>(output.data);
    int64_t n = output.numel();
    
    // Simple case: same shape
    if (a.numel() == b.numel()) {
        switch (op) {
            case ElementwiseOp::ADD:
                for (int64_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] + b_ptr[i];
                break;
            case ElementwiseOp::SUB:
                for (int64_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] - b_ptr[i];
                break;
            case ElementwiseOp::MUL:
                for (int64_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] * b_ptr[i];
                break;
            case ElementwiseOp::DIV:
                for (int64_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] / b_ptr[i];
                break;
            default:
                break;
        }
    }
    // Broadcast case: b is scalar
    else if (b.numel() == 1) {
        float b_val = b_ptr[0];
        switch (op) {
            case ElementwiseOp::ADD:
                for (int64_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] + b_val;
                break;
            case ElementwiseOp::SUB:
                for (int64_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] - b_val;
                break;
            case ElementwiseOp::MUL:
                for (int64_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] * b_val;
                break;
            case ElementwiseOp::DIV:
                for (int64_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] / b_val;
                break;
            default:
                break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Scalar Operations
// ─────────────────────────────────────────────────────────────────────

/**
 * @brief Apply scalar operation to tensor
 */
inline void scalar_op(
    const Tensor& input, 
    const Scalar& scalar, 
    Tensor& output, 
    ElementwiseOp op
) noexcept {
    // CPU-only implementation
    if (input.device != Device::CPU || output.device != Device::CPU) return;
    
    if (input.dtype != DType::F32) return;
    
    const float* in_ptr = static_cast<const float*>(input.data);
    float* out_ptr = static_cast<float*>(output.data);
    float s = scalar.to_f32();
    int64_t n = input.numel();
    
    switch (op) {
        case ElementwiseOp::ADD:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] + s;
            break;
        case ElementwiseOp::SUB:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] - s;
            break;
        case ElementwiseOp::MUL:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] * s;
            break;
        case ElementwiseOp::DIV:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] / s;
            break;
        default:
            break;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Convenience Functions
// ─────────────────────────────────────────────────────────────────────

inline void add(const Tensor& a, const Tensor& b, Tensor& out) noexcept {
    binary_op(a, b, out, ElementwiseOp::ADD);
}

inline void sub(const Tensor& a, const Tensor& b, Tensor& out) noexcept {
    binary_op(a, b, out, ElementwiseOp::SUB);
}

inline void mul(const Tensor& a, const Tensor& b, Tensor& out) noexcept {
    binary_op(a, b, out, ElementwiseOp::MUL);
}

inline void div(const Tensor& a, const Tensor& b, Tensor& out) noexcept {
    binary_op(a, b, out, ElementwiseOp::DIV);
}

inline void neg(const Tensor& input, Tensor& out) noexcept {
    unary_op(input, out, ElementwiseOp::NEG);
}

inline void exp(const Tensor& input, Tensor& out) noexcept {
    unary_op(input, out, ElementwiseOp::EXP);
}

inline void log(const Tensor& input, Tensor& out) noexcept {
    unary_op(input, out, ElementwiseOp::LOG);
}

inline void sqrt(const Tensor& input, Tensor& out) noexcept {
    unary_op(input, out, ElementwiseOp::SQRT);
}

inline void tanh(const Tensor& input, Tensor& out) noexcept {
    unary_op(input, out, ElementwiseOp::TANH);
}

inline void relu(const Tensor& input, Tensor& out) noexcept {
    unary_op(input, out, ElementwiseOp::RELU);
}

inline void sigmoid(const Tensor& input, Tensor& out) noexcept {
    unary_op(input, out, ElementwiseOp::SIGMOID);
}

} // namespace ops
} // namespace zero

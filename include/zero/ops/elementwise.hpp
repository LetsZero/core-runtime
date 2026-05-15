#pragma once

/**
 * @file elementwise.hpp
 * @brief Zero Core Runtime — Elementwise Operations
 *
 * Elementwise operations on tensors.
 *
 * NOTE: Activations (relu, sigmoid, tanh) are unary, shape-preserving ops.
 * Broadcasting is a frontend concern, not a runtime concern.
 *
 * Spec 002: every compute op returns Status. On error, the op writes
 * zero bytes to the output and returns the appropriate StatusCode.
 */

#include "../core/tensor.hpp"
#include "../core/scalar.hpp"
#include "../core/status.hpp"
#include "../device/sync.hpp"

#include <cmath>
#include <algorithm>

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

namespace detail {

// File-private validation helpers. Not part of the public API.
// Returns ok() if all preconditions met; otherwise the failing Status.
inline Status validate_unary(const Tensor& input, const Tensor& output) noexcept {
    if (input.data == nullptr || output.data == nullptr)
        return status::invalid_state("null data pointer");
    if (input.device != Device::CPU || output.device != Device::CPU)
        return status::invalid_argument("non-CPU device not supported");
    if (input.dtype != output.dtype)
        return status::type_mismatch("input/output dtype disagree");
    if (input.dtype != DType::F32)
        return status::type_mismatch("only F32 supported on CPU");
    if (input.ndim != output.ndim)
        return status::invalid_argument("ndim mismatch");
    if (input.numel() != output.numel())
        return status::invalid_argument("shape mismatch");
    return status::OK;
}

// Validates a binary op. Allows two shape patterns:
//   1. a.shape == b.shape == out.shape (elementwise)
//   2. b.numel() == 1 (scalar broadcast on the RHS); a.shape == out.shape
inline Status validate_binary(const Tensor& a, const Tensor& b, const Tensor& output) noexcept {
    if (a.data == nullptr || b.data == nullptr || output.data == nullptr)
        return status::invalid_state("null data pointer");
    if (a.device != Device::CPU || b.device != Device::CPU || output.device != Device::CPU)
        return status::invalid_argument("non-CPU device not supported");
    if (a.dtype != b.dtype || a.dtype != output.dtype)
        return status::type_mismatch("dtype disagreement among a, b, output");
    if (a.dtype != DType::F32)
        return status::type_mismatch("only F32 supported on CPU");
    if (a.numel() != output.numel())
        return status::invalid_argument("a and output must have matching numel");
    if (a.numel() != b.numel() && b.numel() != 1)
        return status::invalid_argument("b must match a or be a scalar (numel==1)");
    return status::OK;
}

inline Status validate_scalar_op(const Tensor& input, const Tensor& output) noexcept {
    // Scalar arg has no shape; reuse the unary check.
    return validate_unary(input, output);
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────
// Unary Operations (in-place capable)
// ─────────────────────────────────────────────────────────────────────

/**
 * @brief Apply unary operation to tensor.
 *
 * Returns ok() on success; on validation failure the output is not modified.
 */
inline Status unary_op(const Tensor& input, Tensor& output, ElementwiseOp op,
                       Stream* stream = nullptr) noexcept {
    (void)stream;  // Spec 003: parameter committed for GPU; CPU ignores.
    if (Status s = detail::validate_unary(input, output); s.is_error()) return s;

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
            return status::invalid_argument("unsupported unary op");
    }
    return status::OK;
}

// ─────────────────────────────────────────────────────────────────────
// Binary Operations
// ─────────────────────────────────────────────────────────────────────

/**
 * @brief Apply binary operation to two tensors.
 *
 * Supports: matching shapes, or b as a scalar (numel == 1).
 */
inline Status binary_op(
    const Tensor& a,
    const Tensor& b,
    Tensor& output,
    ElementwiseOp op,
    Stream* stream = nullptr
) noexcept {
    (void)stream;
    if (Status s = detail::validate_binary(a, b, output); s.is_error()) return s;

    const float* a_ptr = static_cast<const float*>(a.data);
    const float* b_ptr = static_cast<const float*>(b.data);
    float* out_ptr = static_cast<float*>(output.data);
    int64_t n = output.numel();

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
                return status::invalid_argument("unsupported binary op");
        }
    } else {
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
                return status::invalid_argument("unsupported binary op");
        }
    }
    return status::OK;
}

// ─────────────────────────────────────────────────────────────────────
// Scalar Operations
// ─────────────────────────────────────────────────────────────────────

inline Status scalar_op(
    const Tensor& input,
    const Scalar& scalar,
    Tensor& output,
    ElementwiseOp op,
    Stream* stream = nullptr
) noexcept {
    (void)stream;
    if (Status s = detail::validate_scalar_op(input, output); s.is_error()) return s;

    const float* in_ptr = static_cast<const float*>(input.data);
    float* out_ptr = static_cast<float*>(output.data);
    float s_val = scalar.to_f32();
    int64_t n = input.numel();

    switch (op) {
        case ElementwiseOp::ADD:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] + s_val;
            break;
        case ElementwiseOp::SUB:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] - s_val;
            break;
        case ElementwiseOp::MUL:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] * s_val;
            break;
        case ElementwiseOp::DIV:
            for (int64_t i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] / s_val;
            break;
        default:
            return status::invalid_argument("unsupported scalar op");
    }
    return status::OK;
}

// ─────────────────────────────────────────────────────────────────────
// Convenience Functions
// ─────────────────────────────────────────────────────────────────────

inline Status add(const Tensor& a, const Tensor& b, Tensor& out, Stream* stream = nullptr) noexcept {
    return binary_op(a, b, out, ElementwiseOp::ADD, stream);
}

inline Status sub(const Tensor& a, const Tensor& b, Tensor& out, Stream* stream = nullptr) noexcept {
    return binary_op(a, b, out, ElementwiseOp::SUB, stream);
}

inline Status mul(const Tensor& a, const Tensor& b, Tensor& out, Stream* stream = nullptr) noexcept {
    return binary_op(a, b, out, ElementwiseOp::MUL, stream);
}

inline Status div(const Tensor& a, const Tensor& b, Tensor& out, Stream* stream = nullptr) noexcept {
    return binary_op(a, b, out, ElementwiseOp::DIV, stream);
}

inline Status neg(const Tensor& input, Tensor& out, Stream* stream = nullptr) noexcept {
    return unary_op(input, out, ElementwiseOp::NEG, stream);
}

inline Status exp(const Tensor& input, Tensor& out, Stream* stream = nullptr) noexcept {
    return unary_op(input, out, ElementwiseOp::EXP, stream);
}

inline Status log(const Tensor& input, Tensor& out, Stream* stream = nullptr) noexcept {
    return unary_op(input, out, ElementwiseOp::LOG, stream);
}

inline Status sqrt(const Tensor& input, Tensor& out, Stream* stream = nullptr) noexcept {
    return unary_op(input, out, ElementwiseOp::SQRT, stream);
}

inline Status tanh(const Tensor& input, Tensor& out, Stream* stream = nullptr) noexcept {
    return unary_op(input, out, ElementwiseOp::TANH, stream);
}

inline Status relu(const Tensor& input, Tensor& out, Stream* stream = nullptr) noexcept {
    return unary_op(input, out, ElementwiseOp::RELU, stream);
}

inline Status sigmoid(const Tensor& input, Tensor& out, Stream* stream = nullptr) noexcept {
    return unary_op(input, out, ElementwiseOp::SIGMOID, stream);
}

} // namespace ops
} // namespace zero

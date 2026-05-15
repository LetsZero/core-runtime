#pragma once

/**
 * @file reduce.hpp
 * @brief Zero Core Runtime — Reduction Operations
 *
 * Sum, max, min, mean along tensor axes.
 *
 * Spec 002: tensor-output reductions (reduce_last_axis, sum, max, mean,
 * argmax) return Status. Scalar-result reductions (reduce_all, sum_all,
 * max_all, min_all, mean_all) are debug helpers and remain unchanged.
 */

#include "../core/tensor.hpp"
#include "../core/status.hpp"

#include <limits>
#include <cmath>

namespace zero {
namespace ops {

/**
 * @brief Reduction operation types
 */
enum class ReduceOp : uint8_t {
    SUM = 0,
    MAX = 1,
    MIN = 2,
    MEAN = 3,
    PROD = 4,
};

/**
 * @brief Full reduction (tensor to scalar). Debug helper.
 *
 * Returns 0.0f for empty input, non-CPU device, or non-F32 dtype.
 * No Status return (carved out of spec 002 §5).
 */
inline float reduce_all(const Tensor& input, ReduceOp op) noexcept {
    if (input.device != Device::CPU) return 0.0f;
    if (input.dtype != DType::F32) return 0.0f;

    const float* data = static_cast<const float*>(input.data);
    int64_t n = input.numel();

    if (n == 0) return 0.0f;

    switch (op) {
        case ReduceOp::SUM: {
            float sum = 0.0f;
            for (int64_t i = 0; i < n; ++i) sum += data[i];
            return sum;
        }
        case ReduceOp::MAX: {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t i = 0; i < n; ++i) {
                if (data[i] > max_val) max_val = data[i];
            }
            return max_val;
        }
        case ReduceOp::MIN: {
            float min_val = std::numeric_limits<float>::infinity();
            for (int64_t i = 0; i < n; ++i) {
                if (data[i] < min_val) min_val = data[i];
            }
            return min_val;
        }
        case ReduceOp::MEAN: {
            float sum = 0.0f;
            for (int64_t i = 0; i < n; ++i) sum += data[i];
            return sum / static_cast<float>(n);
        }
        case ReduceOp::PROD: {
            float prod = 1.0f;
            for (int64_t i = 0; i < n; ++i) prod *= data[i];
            return prod;
        }
    }
    return 0.0f;
}

namespace detail {

// Validate that `output` is `input` with the last axis dropped.
// Requires input.ndim >= 1.
inline Status validate_reduce_last(
    const Tensor& input,
    const Tensor& output,
    DType expected_output_dtype
) noexcept {
    if (input.data == nullptr || output.data == nullptr)
        return status::invalid_state("null data pointer");
    if (input.device != Device::CPU || output.device != Device::CPU)
        return status::invalid_argument("non-CPU device not supported");
    if (input.dtype != DType::F32)
        return status::type_mismatch("only F32 input supported on CPU");
    if (output.dtype != expected_output_dtype)
        return status::type_mismatch("output dtype does not match expected");
    if (input.ndim < 1)
        return status::invalid_argument("input must have rank >= 1");
    if (output.ndim != input.ndim - 1)
        return status::invalid_argument("output rank must be input rank - 1");
    for (int8_t i = 0; i < output.ndim; ++i) {
        if (output.shape[i] != input.shape[i])
            return status::invalid_argument("output leading-axis shape must match input");
    }
    return status::OK;
}

} // namespace detail

/**
 * @brief Reduce along last axis.
 *
 * input:  [..., N]
 * output: [...]
 */
inline Status reduce_last_axis(
    const Tensor& input,
    Tensor& output,
    ReduceOp op
) noexcept {
    if (Status s = detail::validate_reduce_last(input, output, DType::F32); s.is_error())
        return s;

    const float* in_ptr = static_cast<const float*>(input.data);
    float* out_ptr = static_cast<float*>(output.data);

    int64_t reduction_size = input.shape[input.ndim - 1];
    int64_t outer_size = (reduction_size > 0) ? input.numel() / reduction_size : 0;

    for (int64_t outer = 0; outer < outer_size; ++outer) {
        const float* row = in_ptr + outer * reduction_size;

        switch (op) {
            case ReduceOp::SUM: {
                float sum = 0.0f;
                for (int64_t i = 0; i < reduction_size; ++i) sum += row[i];
                out_ptr[outer] = sum;
                break;
            }
            case ReduceOp::MAX: {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < reduction_size; ++i) {
                    if (row[i] > max_val) max_val = row[i];
                }
                out_ptr[outer] = max_val;
                break;
            }
            case ReduceOp::MIN: {
                float min_val = std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < reduction_size; ++i) {
                    if (row[i] < min_val) min_val = row[i];
                }
                out_ptr[outer] = min_val;
                break;
            }
            case ReduceOp::MEAN: {
                float sum = 0.0f;
                for (int64_t i = 0; i < reduction_size; ++i) sum += row[i];
                out_ptr[outer] = sum / static_cast<float>(reduction_size);
                break;
            }
            case ReduceOp::PROD: {
                float prod = 1.0f;
                for (int64_t i = 0; i < reduction_size; ++i) prod *= row[i];
                out_ptr[outer] = prod;
                break;
            }
        }
    }
    return status::OK;
}

// ─────────────────────────────────────────────────────────────────────
// Scalar-result reductions (debug helpers, unchanged signature per spec 002 §5)
// ─────────────────────────────────────────────────────────────────────

inline float sum_all(const Tensor& input) noexcept {
    return reduce_all(input, ReduceOp::SUM);
}

inline float max_all(const Tensor& input) noexcept {
    return reduce_all(input, ReduceOp::MAX);
}

inline float min_all(const Tensor& input) noexcept {
    return reduce_all(input, ReduceOp::MIN);
}

inline float mean_all(const Tensor& input) noexcept {
    return reduce_all(input, ReduceOp::MEAN);
}

// ─────────────────────────────────────────────────────────────────────
// Tensor-output reductions (Status-returning per spec 002)
// ─────────────────────────────────────────────────────────────────────

inline Status sum(const Tensor& input, Tensor& output) noexcept {
    return reduce_last_axis(input, output, ReduceOp::SUM);
}

inline Status max(const Tensor& input, Tensor& output) noexcept {
    return reduce_last_axis(input, output, ReduceOp::MAX);
}

inline Status mean(const Tensor& input, Tensor& output) noexcept {
    return reduce_last_axis(input, output, ReduceOp::MEAN);
}

/**
 * @brief Argmax along last axis. Output dtype must be I32 or I64.
 */
inline Status argmax(const Tensor& input, Tensor& output) noexcept {
    if (input.data == nullptr || output.data == nullptr)
        return status::invalid_state("null data pointer");
    if (input.device != Device::CPU || output.device != Device::CPU)
        return status::invalid_argument("non-CPU device not supported");
    if (input.dtype != DType::F32)
        return status::type_mismatch("argmax input must be F32");
    if (output.dtype != DType::I64 && output.dtype != DType::I32)
        return status::type_mismatch("argmax output must be I32 or I64");
    if (input.ndim < 1)
        return status::invalid_argument("input must have rank >= 1");
    if (output.ndim != input.ndim - 1)
        return status::invalid_argument("output rank must be input rank - 1");
    for (int8_t i = 0; i < output.ndim; ++i) {
        if (output.shape[i] != input.shape[i])
            return status::invalid_argument("output leading-axis shape must match input");
    }

    const float* in_ptr = static_cast<const float*>(input.data);

    int64_t reduction_size = input.shape[input.ndim - 1];
    int64_t outer_size = (reduction_size > 0) ? input.numel() / reduction_size : 0;

    if (output.dtype == DType::I64) {
        int64_t* out_ptr = static_cast<int64_t*>(output.data);
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const float* row = in_ptr + outer * reduction_size;
            float max_val = -std::numeric_limits<float>::infinity();
            int64_t max_idx = 0;
            for (int64_t i = 0; i < reduction_size; ++i) {
                if (row[i] > max_val) { max_val = row[i]; max_idx = i; }
            }
            out_ptr[outer] = max_idx;
        }
    } else {  // I32
        int32_t* out_ptr = static_cast<int32_t*>(output.data);
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            const float* row = in_ptr + outer * reduction_size;
            float max_val = -std::numeric_limits<float>::infinity();
            int32_t max_idx = 0;
            for (int64_t i = 0; i < reduction_size; ++i) {
                if (row[i] > max_val) { max_val = row[i]; max_idx = static_cast<int32_t>(i); }
            }
            out_ptr[outer] = max_idx;
        }
    }
    return status::OK;
}

} // namespace ops
} // namespace zero

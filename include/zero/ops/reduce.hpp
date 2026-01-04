#pragma once

/**
 * @file reduce.hpp
 * @brief Zero Core Runtime — Reduction Operations
 * 
 * Sum, max, min, mean along tensor axes.
 */

#include "../core/tensor.hpp"

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
 * @brief Full reduction (tensor to scalar)
 */
inline float reduce_all(const Tensor& input, ReduceOp op) noexcept {
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

/**
 * @brief Reduce along last axis
 * 
 * input:  [..., N]
 * output: [...]
 */
inline void reduce_last_axis(
    const Tensor& input, 
    Tensor& output, 
    ReduceOp op
) noexcept {
    if (input.dtype != DType::F32 || output.dtype != DType::F32) return;
    if (input.ndim == 0) return;
    
    const float* in_ptr = static_cast<const float*>(input.data);
    float* out_ptr = static_cast<float*>(output.data);
    
    int64_t reduction_size = input.shape[input.ndim - 1];
    int64_t outer_size = input.numel() / reduction_size;
    
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
}

// ─────────────────────────────────────────────────────────────────────
// Convenience Functions
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

inline void sum(const Tensor& input, Tensor& output) noexcept {
    reduce_last_axis(input, output, ReduceOp::SUM);
}

inline void max(const Tensor& input, Tensor& output) noexcept {
    reduce_last_axis(input, output, ReduceOp::MAX);
}

inline void mean(const Tensor& input, Tensor& output) noexcept {
    reduce_last_axis(input, output, ReduceOp::MEAN);
}

/**
 * @brief Argmax along last axis
 */
inline void argmax(const Tensor& input, Tensor& output) noexcept {
    if (input.dtype != DType::F32) return;
    if (output.dtype != DType::I64 && output.dtype != DType::I32) return;
    
    const float* in_ptr = static_cast<const float*>(input.data);
    int64_t* out_ptr = static_cast<int64_t*>(output.data);
    
    int64_t reduction_size = input.shape[input.ndim - 1];
    int64_t outer_size = input.numel() / reduction_size;
    
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        const float* row = in_ptr + outer * reduction_size;
        float max_val = -std::numeric_limits<float>::infinity();
        int64_t max_idx = 0;
        
        for (int64_t i = 0; i < reduction_size; ++i) {
            if (row[i] > max_val) {
                max_val = row[i];
                max_idx = i;
            }
        }
        out_ptr[outer] = max_idx;
    }
}

} // namespace ops
} // namespace zero

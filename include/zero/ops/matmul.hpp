#pragma once

/**
 * @file matmul.hpp
 * @brief Zero Core Runtime — Matrix Multiplication
 *
 * GEMM: C = alpha * A @ B + beta * C
 *
 * Spec 002: returns Status. On validation failure C is not modified.
 */

#include "../core/tensor.hpp"
#include "../core/scalar.hpp"
#include "../core/status.hpp"
#include "../device/sync.hpp"

namespace zero {
namespace ops {

namespace detail {

inline Status validate_gemm(const Tensor& A, const Tensor& B, const Tensor& C) noexcept {
    if (A.data == nullptr || B.data == nullptr || C.data == nullptr)
        return status::invalid_state("null data pointer");
    if (A.device != Device::CPU || B.device != Device::CPU || C.device != Device::CPU)
        return status::invalid_argument("non-CPU device not supported");
    if (A.dtype != B.dtype || A.dtype != C.dtype)
        return status::type_mismatch("dtype disagreement among A, B, C");
    if (A.dtype != DType::F32)
        return status::type_mismatch("only F32 supported on CPU");
    if (A.ndim != 2 || B.ndim != 2 || C.ndim != 2)
        return status::invalid_argument("gemm requires rank-2 tensors");
    if (A.shape[1] != B.shape[0])
        return status::invalid_argument("inner dimension mismatch (A.cols != B.rows)");
    if (C.shape[0] != A.shape[0] || C.shape[1] != B.shape[1])
        return status::invalid_argument("output shape mismatch (must be [A.rows, B.cols])");
    return status::OK;
}

} // namespace detail

/**
 * @brief General matrix multiplication (GEMM): C = alpha * A @ B + beta * C
 */
inline Status gemm(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float alpha = 1.0f,
    float beta = 0.0f,
    Stream* stream = nullptr
) noexcept {
    (void)stream;  // Spec 003: parameter committed for GPU; CPU ignores.
    if (Status s = detail::validate_gemm(A, B, C); s.is_error()) return s;

    int64_t M = A.shape[0];
    int64_t K = A.shape[1];
    int64_t N = B.shape[1];

    const float* a_ptr = static_cast<const float*>(A.data);
    const float* b_ptr = static_cast<const float*>(B.data);
    float* c_ptr = static_cast<float*>(C.data);

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += a_ptr[m * K + k] * b_ptr[k * N + n];
            }
            c_ptr[m * N + n] = alpha * sum + beta * c_ptr[m * N + n];
        }
    }
    return status::OK;
}

/**
 * @brief Matrix multiplication (A @ B). Wraps gemm with alpha=1, beta=0.
 */
inline Status matmul(const Tensor& A, const Tensor& B, Tensor& C,
                     Stream* stream = nullptr) noexcept {
    return gemm(A, B, C, 1.0f, 0.0f, stream);
}

// NOTE: batched_matmul and matvec removed from core.
// - batched_matmul: violates "no hidden allocation" (view creation in loop)
// - matvec: derivable from matmul via reshape
// These should be handled by IR lowering if needed.

} // namespace ops
} // namespace zero

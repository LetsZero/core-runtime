#pragma once

/**
 * @file matmul.hpp
 * @brief Zero Core Runtime — Matrix Multiplication
 * 
 * GEMM: C = alpha * A @ B + beta * C
 */

#include "../core/tensor.hpp"
#include "../core/scalar.hpp"

namespace zero {
namespace ops {

/**
 * @brief General matrix multiplication (GEMM)
 * 
 * C = alpha * A @ B + beta * C
 * 
 * @param A      Left matrix [M, K]
 * @param B      Right matrix [K, N]
 * @param C      Output matrix [M, N]
 * @param alpha  Scaling factor for A @ B
 * @param beta   Scaling factor for existing C
 */
inline void gemm(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float alpha = 1.0f,
    float beta = 0.0f
) noexcept {
    // Explicit device check (CPU-only implementation)
    if (A.device != Device::CPU || B.device != Device::CPU || C.device != Device::CPU) return;
    
    // Validate dimensions
    if (A.ndim != 2 || B.ndim != 2 || C.ndim != 2) return;
    if (A.dtype != DType::F32 || B.dtype != DType::F32 || C.dtype != DType::F32) return;
    
    int64_t M = A.shape[0];
    int64_t K = A.shape[1];
    int64_t N = B.shape[1];
    
    if (B.shape[0] != K || C.shape[0] != M || C.shape[1] != N) return;
    
    const float* a_ptr = static_cast<const float*>(A.data);
    const float* b_ptr = static_cast<const float*>(B.data);
    float* c_ptr = static_cast<float*>(C.data);
    
    // Naive implementation (will be optimized via Zero IR lowering)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += a_ptr[m * K + k] * b_ptr[k * N + n];
            }
            c_ptr[m * N + n] = alpha * sum + beta * c_ptr[m * N + n];
        }
    }
}

/**
 * @brief Matrix multiplication (A @ B)
 */
inline void matmul(const Tensor& A, const Tensor& B, Tensor& C) noexcept {
    gemm(A, B, C, 1.0f, 0.0f);
}

// NOTE: batched_matmul and matvec removed from core.
// - batched_matmul: violates "no hidden allocation" (view creation in loop)
// - matvec: derivable from matmul via reshape
// These should be handled by IR lowering if needed.

} // namespace ops
} // namespace zero

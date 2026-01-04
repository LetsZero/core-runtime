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

/**
 * @brief Batched matrix multiplication
 * 
 * A: [batch, M, K]
 * B: [batch, K, N]
 * C: [batch, M, N]
 */
inline void batched_matmul(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
) noexcept {
    if (A.ndim != 3 || B.ndim != 3 || C.ndim != 3) return;
    
    int64_t batch = A.shape[0];
    int64_t M = A.shape[1];
    int64_t K = A.shape[2];
    int64_t N = B.shape[2];
    
    if (B.shape[0] != batch || B.shape[1] != K) return;
    if (C.shape[0] != batch || C.shape[1] != M || C.shape[2] != N) return;
    
    size_t a_batch_stride = static_cast<size_t>(M * K);
    size_t b_batch_stride = static_cast<size_t>(K * N);
    size_t c_batch_stride = static_cast<size_t>(M * N);
    
    for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
        // Create views for each batch slice
        int64_t a_shape[2] = {M, K};
        int64_t b_shape[2] = {K, N};
        int64_t c_shape[2] = {M, N};
        int64_t a_strides[2], b_strides[2], c_strides[2];
        
        calc_contiguous_strides(a_shape, 2, DType::F32, a_strides);
        calc_contiguous_strides(b_shape, 2, DType::F32, b_strides);
        calc_contiguous_strides(c_shape, 2, DType::F32, c_strides);
        
        float* a_ptr = static_cast<float*>(A.data) + b_idx * a_batch_stride;
        float* b_ptr = static_cast<float*>(B.data) + b_idx * b_batch_stride;
        float* c_ptr = static_cast<float*>(C.data) + b_idx * c_batch_stride;
        
        Tensor a_slice = Tensor::view(a_ptr, a_shape, a_strides, 2, DType::F32, A.device);
        Tensor b_slice = Tensor::view(b_ptr, b_shape, b_strides, 2, DType::F32, B.device);
        Tensor c_slice = Tensor::view(c_ptr, c_shape, c_strides, 2, DType::F32, C.device);
        
        matmul(a_slice, b_slice, c_slice);
    }
}

/**
 * @brief Matrix-vector multiplication
 * 
 * y = A @ x
 * A: [M, N]
 * x: [N]
 * y: [M]
 */
inline void matvec(const Tensor& A, const Tensor& x, Tensor& y) noexcept {
    if (A.ndim != 2 || x.ndim != 1 || y.ndim != 1) return;
    if (A.dtype != DType::F32 || x.dtype != DType::F32 || y.dtype != DType::F32) return;
    
    int64_t M = A.shape[0];
    int64_t N = A.shape[1];
    
    if (x.shape[0] != N || y.shape[0] != M) return;
    
    const float* a_ptr = static_cast<const float*>(A.data);
    const float* x_ptr = static_cast<const float*>(x.data);
    float* y_ptr = static_cast<float*>(y.data);
    
    for (int64_t m = 0; m < M; ++m) {
        float sum = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            sum += a_ptr[m * N + n] * x_ptr[n];
        }
        y_ptr[m] = sum;
    }
}

} // namespace ops
} // namespace zero

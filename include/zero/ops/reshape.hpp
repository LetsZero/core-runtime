#pragma once

/**
 * @file reshape.hpp
 * @brief Zero Core Runtime — Reshape and View Operations
 * 
 * Broadcast, reshape, transpose, squeeze, unsqueeze.
 * All O(1) metadata operations where possible.
 */

#include "../core/tensor.hpp"

namespace zero {
namespace ops {

/**
 * @brief Calculate broadcast output shape
 * 
 * @param a_shape  Shape of tensor A
 * @param a_ndim   Dims of tensor A
 * @param b_shape  Shape of tensor B
 * @param b_ndim   Dims of tensor B
 * @param out_shape Output shape (must have MAX_DIMS elements)
 * @param out_ndim  Output number of dims
 * @return True if shapes are broadcast-compatible
 */
inline bool broadcast_shape(
    const int64_t* a_shape, int8_t a_ndim,
    const int64_t* b_shape, int8_t b_ndim,
    int64_t* out_shape, int8_t& out_ndim
) noexcept {
    out_ndim = (a_ndim > b_ndim) ? a_ndim : b_ndim;
    
    // Work backwards from the last dimension
    for (int8_t i = 0; i < out_ndim; ++i) {
        int8_t a_idx = a_ndim - 1 - i;
        int8_t b_idx = b_ndim - 1 - i;
        int8_t out_idx = out_ndim - 1 - i;
        
        int64_t a_dim = (a_idx >= 0) ? a_shape[a_idx] : 1;
        int64_t b_dim = (b_idx >= 0) ? b_shape[b_idx] : 1;
        
        if (a_dim == b_dim) {
            out_shape[out_idx] = a_dim;
        } else if (a_dim == 1) {
            out_shape[out_idx] = b_dim;
        } else if (b_dim == 1) {
            out_shape[out_idx] = a_dim;
        } else {
            return false; // Incompatible shapes
        }
    }
    
    return true;
}

/**
 * @brief Check if shapes can be broadcast together
 */
inline bool can_broadcast(
    const int64_t* a_shape, int8_t a_ndim,
    const int64_t* b_shape, int8_t b_ndim
) noexcept {
    int64_t out_shape[MAX_DIMS];
    int8_t out_ndim;
    return broadcast_shape(a_shape, a_ndim, b_shape, b_ndim, out_shape, out_ndim);
}

/**
 * @brief Squeeze: remove dimensions of size 1
 */
inline Tensor squeeze(const Tensor& input) noexcept {
    Tensor out = input;
    out.owns_data = false;
    
    int8_t new_ndim = 0;
    for (int8_t i = 0; i < input.ndim; ++i) {
        if (input.shape[i] != 1) {
            out.shape[new_ndim] = input.shape[i];
            out.strides[new_ndim] = input.strides[i];
            new_ndim++;
        }
    }
    out.ndim = new_ndim;
    
    return out;
}

/**
 * @brief Squeeze along specific dimension
 */
inline Tensor squeeze_dim(const Tensor& input, int8_t dim) noexcept {
    if (dim < 0 || dim >= input.ndim || input.shape[dim] != 1) {
        return input;
    }
    
    Tensor out = input;
    out.owns_data = false;
    
    for (int8_t i = dim; i < input.ndim - 1; ++i) {
        out.shape[i] = input.shape[i + 1];
        out.strides[i] = input.strides[i + 1];
    }
    out.ndim = input.ndim - 1;
    
    return out;
}

/**
 * @brief Unsqueeze: add dimension of size 1 at position
 */
inline Tensor unsqueeze(const Tensor& input, int8_t dim) noexcept {
    if (dim < 0 || dim > input.ndim || input.ndim >= MAX_DIMS) {
        return input;
    }
    
    Tensor out = input;
    out.owns_data = false;
    out.ndim = input.ndim + 1;
    
    // Shift dimensions after insertion point
    for (int8_t i = out.ndim - 1; i > dim; --i) {
        out.shape[i] = input.shape[i - 1];
        out.strides[i] = input.strides[i - 1];
    }
    
    // Insert new dimension
    out.shape[dim] = 1;
    out.strides[dim] = (dim < input.ndim) ? input.strides[dim] : static_cast<int64_t>(dtype_size(input.dtype));
    
    return out;
}

/**
 * @brief Permute tensor dimensions
 * 
 * @param input Input tensor
 * @param perm  Permutation array (e.g., [1, 0] for transpose)
 */
inline Tensor permute(const Tensor& input, const int8_t* perm) noexcept {
    Tensor out = input;
    out.owns_data = false;
    
    for (int8_t i = 0; i < input.ndim; ++i) {
        out.shape[i] = input.shape[perm[i]];
        out.strides[i] = input.strides[perm[i]];
    }
    
    return out;
}

/**
 * @brief Expand tensor to new shape (broadcast without copy)
 */
inline Tensor expand(const Tensor& input, const int64_t* new_shape, int8_t new_ndim) noexcept {
    Tensor out = Tensor::empty();
    out.data = input.data;
    out.dtype = input.dtype;
    out.device = input.device;
    out.ndim = new_ndim;
    out.owns_data = false;
    
    // Calculate strides for expanded dimensions
    int8_t input_offset = new_ndim - input.ndim;
    
    for (int8_t i = 0; i < new_ndim; ++i) {
        out.shape[i] = new_shape[i];
        
        int8_t input_idx = i - input_offset;
        if (input_idx < 0) {
            // New dimension, stride is 0 (broadcast)
            out.strides[i] = 0;
        } else if (input.shape[input_idx] == 1) {
            // Broadcast existing dim
            out.strides[i] = 0;
        } else {
            out.strides[i] = input.strides[input_idx];
        }
    }
    
    return out;
}

/**
 * @brief Flatten tensor to 1D
 */
inline Tensor flatten(const Tensor& input) noexcept {
    int64_t new_shape[1] = {input.numel()};
    return input.reshape(new_shape, 1);
}

/**
 * @brief View tensor with new shape (must be contiguous)
 */
inline Tensor view(const Tensor& input, const int64_t* new_shape, int8_t new_ndim) noexcept {
    if (!input.is_contiguous()) {
        return Tensor::empty(); // Cannot view non-contiguous tensor
    }
    return input.reshape(new_shape, new_ndim);
}

} // namespace ops
} // namespace zero

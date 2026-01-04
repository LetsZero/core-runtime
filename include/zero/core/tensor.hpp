#pragma once

/**
 * @file tensor.hpp
 * @brief Zero Core Runtime — Tensor Primitive
 * 
 * The ONLY real data container in Zero.
 * Tensors are first-class primitives, not library objects.
 */

#include "dtype.hpp"
#include "memory.hpp"
#include "../device/device.hpp"

#include <array>
#include <cstdint>

namespace zero {

/// Maximum number of tensor dimensions
constexpr int8_t MAX_DIMS = 8;

/**
 * @brief Core Tensor structure
 * 
 * Design principles:
 * - No virtual methods, no inheritance
 * - Fixed-size arrays for shape/strides (no heap allocation for metadata)
 * - Views are O(1) metadata operations
 */
struct Tensor {
    void* data;                              ///< Raw memory pointer
    DType dtype;                             ///< Element data type
    Device device;                           ///< Device location
    int8_t ndim;                             ///< Number of dimensions
    std::array<int64_t, MAX_DIMS> shape;     ///< Size of each dimension
    std::array<int64_t, MAX_DIMS> strides;   ///< Byte stride for each dimension
    bool owns_data;                          ///< True if tensor owns its memory
    
    // ─────────────────────────────────────────────────────────────────
    // Factory Functions
    // ─────────────────────────────────────────────────────────────────
    
    /**
     * @brief Create an empty tensor (no allocation)
     */
    static Tensor empty() noexcept {
        Tensor t{};
        t.data = nullptr;
        t.dtype = DType::F32;
        t.device = Device::CPU;
        t.ndim = 0;
        t.shape.fill(0);
        t.strides.fill(0);
        t.owns_data = false;
        return t;
    }
    
    /**
     * @brief Allocate a new tensor with given shape
     */
    static Tensor alloc(
        const int64_t* shape_ptr, 
        int8_t ndim, 
        DType dtype, 
        Device device = Device::CPU
    ) noexcept {
        Tensor t = empty();
        t.dtype = dtype;
        t.device = device;
        t.ndim = ndim;
        
        for (int8_t i = 0; i < ndim; ++i) {
            t.shape[i] = shape_ptr[i];
        }
        
        calc_contiguous_strides(shape_ptr, ndim, dtype, t.strides.data());
        
        size_t bytes = calc_tensor_bytes(shape_ptr, ndim, dtype);
        t.data = mem_alloc(bytes, dtype_alignment(dtype), device);
        t.owns_data = (t.data != nullptr);
        
        return t;
    }
    
    /**
     * @brief Create a tensor view (no copy, shared memory)
     */
    static Tensor view(
        void* data_ptr,
        const int64_t* shape_ptr,
        const int64_t* strides_ptr,
        int8_t ndim,
        DType dtype,
        Device device = Device::CPU
    ) noexcept {
        Tensor t = empty();
        t.data = data_ptr;
        t.dtype = dtype;
        t.device = device;
        t.ndim = ndim;
        t.owns_data = false;
        
        for (int8_t i = 0; i < ndim; ++i) {
            t.shape[i] = shape_ptr[i];
            t.strides[i] = strides_ptr[i];
        }
        
        return t;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Properties
    // ─────────────────────────────────────────────────────────────────
    
    /**
     * @brief Total number of elements
     */
    int64_t numel() const noexcept {
        if (ndim == 0) return 1; // Scalar
        int64_t n = 1;
        for (int8_t i = 0; i < ndim; ++i) {
            n *= shape[i];
        }
        return n;
    }
    
    /**
     * @brief Total bytes of data
     */
    size_t nbytes() const noexcept {
        return static_cast<size_t>(numel()) * dtype_size(dtype);
    }
    
    /**
     * @brief Check if tensor is contiguous in memory
     */
    bool is_contiguous() const noexcept {
        if (ndim == 0) return true;
        
        int64_t expected_stride = static_cast<int64_t>(dtype_size(dtype));
        for (int8_t i = ndim - 1; i >= 0; --i) {
            if (strides[i] != expected_stride) return false;
            expected_stride *= shape[i];
        }
        return true;
    }
    
    /**
     * @brief Check if this is a scalar (0-dimensional tensor)
     */
    bool is_scalar() const noexcept {
        return ndim == 0;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // View Operations (O(1) metadata changes)
    // ─────────────────────────────────────────────────────────────────
    
    /**
     * @brief Reshape tensor (must have same numel)
     */
    Tensor reshape(const int64_t* new_shape, int8_t new_ndim) const noexcept {
        Tensor t = *this;
        t.ndim = new_ndim;
        t.owns_data = false; // View doesn't own data
        
        for (int8_t i = 0; i < new_ndim; ++i) {
            t.shape[i] = new_shape[i];
        }
        
        // If contiguous, recalculate strides
        if (is_contiguous()) {
            calc_contiguous_strides(new_shape, new_ndim, dtype, t.strides.data());
        }
        
        return t;
    }
    
    /**
     * @brief Transpose last two dimensions
     */
    Tensor transpose() const noexcept {
        if (ndim < 2) return *this;
        
        Tensor t = *this;
        t.owns_data = false;
        
        // Swap last two dims
        int8_t last = ndim - 1;
        int8_t second_last = ndim - 2;
        
        std::swap(t.shape[last], t.shape[second_last]);
        std::swap(t.strides[last], t.strides[second_last]);
        
        return t;
    }
    
    /**
     * @brief Slice along a dimension
     */
    Tensor slice(int8_t dim, int64_t start, int64_t end) const noexcept {
        Tensor t = *this;
        t.owns_data = false;
        
        // Adjust data pointer
        t.data = static_cast<uint8_t*>(data) + start * strides[dim];
        t.shape[dim] = end - start;
        
        return t;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Memory Management
    // ─────────────────────────────────────────────────────────────────
    
    /**
     * @brief Free owned memory
     */
    void free() noexcept {
        if (owns_data && data != nullptr) {
            mem_free(data, device);
            data = nullptr;
            owns_data = false;
        }
    }
};

} // namespace zero

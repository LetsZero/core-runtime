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
#include <cstdio>

namespace zero {

// Forward declare Scalar for Tensor ↔ Scalar bridge
struct Scalar;

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
    
    /**
     * @brief Wrap external memory (non-owning)
     * 
     * For interop with CUDA, ONNX, user buffers, etc.
     */
    static Tensor wrap(
        void* external,
        const int64_t* shape_ptr,
        int8_t ndim,
        DType dtype,
        Device device = Device::CPU
    ) noexcept {
        Tensor t = empty();
        t.data = external;
        t.dtype = dtype;
        t.device = device;
        t.ndim = ndim;
        t.owns_data = false;  // Never owns external memory
        
        for (int8_t i = 0; i < ndim; ++i) {
            t.shape[i] = shape_ptr[i];
        }
        calc_contiguous_strides(shape_ptr, ndim, dtype, t.strides.data());
        
        return t;
    }
    
    /**
     * @brief Create a rank-0 tensor from scalar value (Scalar ↔ Tensor bridge)
     * 
     * NOTE: Defined after Scalar in scalar.hpp to avoid circular dependency.
     */
    static Tensor from_scalar(const Scalar& s, Device device = Device::CPU) noexcept;
    
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
     * @brief Convert to Scalar (only valid for ndim == 0)
     */
    Scalar to_scalar() const noexcept;
    
    // ─────────────────────────────────────────────────────────────────
    // Validation (for compiler/runtime assertions)
    // ─────────────────────────────────────────────────────────────────
    
    /**
     * @brief Check if tensor is in a valid state
     */
    bool valid() const noexcept {
        // Check ndim bounds
        if (ndim < 0 || ndim > MAX_DIMS) return false;
        
        // Device check
        if (!device_available(device)) return false;
        
        // Shape checks
        for (int8_t i = 0; i < ndim; ++i) {
            if (shape[i] < 0) return false;
        }
        
        // Data consistency
        if (data == nullptr) {
            // null data is only valid for empty tensor or zero numel
            if (numel() > 0 && owns_data) return false;
        }
        
        // Stride checks for non-scalar
        if (ndim > 0) {
            for (int8_t i = 0; i < ndim; ++i) {
                if (shape[i] > 1 && strides[i] == 0) return false;
            }
        }
        
        return true;
    }
    
    /**
     * @brief Check if reshape is valid
     */
    bool can_reshape(const int64_t* new_shape, int8_t new_ndim) const noexcept {
        if (new_ndim < 0 || new_ndim > MAX_DIMS) return false;
        if (!is_contiguous()) return false;
        
        int64_t new_numel = 1;
        for (int8_t i = 0; i < new_ndim; ++i) {
            if (new_shape[i] < 0) return false;
            new_numel *= new_shape[i];
        }
        return new_numel == numel();
    }
    
    /**
     * @brief Check if slice is valid
     */
    bool can_slice(int8_t dim, int64_t start, int64_t end) const noexcept {
        if (dim < 0 || dim >= ndim) return false;
        if (start < 0 || end < start) return false;
        if (end > shape[dim]) return false;
        return true;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Shape Semantics (for ML dispatch)
    // ─────────────────────────────────────────────────────────────────
    
    bool is_scalar() const noexcept { return ndim == 0; }
    bool is_vector() const noexcept { return ndim == 1; }
    bool is_matrix() const noexcept { return ndim == 2; }
    bool is_batch() const noexcept { return ndim >= 1 && shape[0] > 1; }
    
    // ─────────────────────────────────────────────────────────────────
    // Layout Introspection
    // ─────────────────────────────────────────────────────────────────
    
    /**
     * @brief Check if tensor is contiguous (row-major)
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
    
    bool is_row_major() const noexcept {
        return is_contiguous();
    }
    
    bool is_column_major() const noexcept {
        if (ndim == 0) return true;
        
        int64_t expected_stride = static_cast<int64_t>(dtype_size(dtype));
        for (int8_t i = 0; i < ndim; ++i) {
            if (strides[i] != expected_stride) return false;
            expected_stride *= shape[i];
        }
        return true;
    }
    
    /**
     * @brief Check if strides are dense (monotonic, no gaps)
     */
    bool is_dense() const noexcept {
        return is_contiguous() || is_column_major();
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Shape Algebra
    // ─────────────────────────────────────────────────────────────────
    
    bool same_shape(const Tensor& other) const noexcept {
        if (ndim != other.ndim) return false;
        for (int8_t i = 0; i < ndim; ++i) {
            if (shape[i] != other.shape[i]) return false;
        }
        return true;
    }
    
    /**
     * @brief Check if tensors are broadcastable (NumPy rules)
     */
    bool broadcastable_with(const Tensor& other) const noexcept {
        int8_t max_ndim = (ndim > other.ndim) ? ndim : other.ndim;
        
        for (int8_t i = 0; i < max_ndim; ++i) {
            int8_t idx_a = ndim - 1 - i;
            int8_t idx_b = other.ndim - 1 - i;
            
            int64_t dim_a = (idx_a >= 0) ? shape[idx_a] : 1;
            int64_t dim_b = (idx_b >= 0) ? other.shape[idx_b] : 1;
            
            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                return false;
            }
        }
        return true;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Device Semantics
    // ─────────────────────────────────────────────────────────────────
    
    bool on(Device d) const noexcept {
        return device == d;
    }
    
    /**
     * @brief Copy tensor to another device (explicit, deep copy)
     */
    Tensor to(Device target_device) const noexcept {
        if (device == target_device) {
            return clone();  // Same device, just clone
        }
        
        // Cross-device copy (only CPU↔CPU supported in core)
        if (device != Device::CPU || target_device != Device::CPU) {
            return empty();  // Not implemented
        }
        
        return clone();
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
    // Copy Operations
    // ─────────────────────────────────────────────────────────────────
    
    /**
     * @brief Deep copy (allocates new memory, copies data)
     */
    Tensor clone() const noexcept {
        Tensor t = alloc(shape.data(), ndim, dtype, device);
        if (t.data != nullptr && data != nullptr) {
            mem_copy_cpu(t.data, data, nbytes());
        }
        return t;
    }
    
    /**
     * @brief Shallow copy (non-owning view of same data)
     */
    Tensor view_like() const noexcept {
        Tensor t = *this;
        t.owns_data = false;
        return t;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Memory Management
    // ─────────────────────────────────────────────────────────────────
    
    /**
     * @brief Reset to empty state (frees if owning)
     */
    void reset() noexcept {
        free();
        *this = empty();
    }
    
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
    
    // ─────────────────────────────────────────────────────────────────
    // Debug Utilities (zero-cost when NDEBUG)
    // ─────────────────────────────────────────────────────────────────
    
#ifndef NDEBUG
    void dump_meta() const noexcept {
        std::printf("Tensor @ %p\n", data);
        std::printf("  dtype: %s\n", dtype_name(dtype));
        std::printf("  device: %s\n", device_name(device));
        std::printf("  ndim: %d\n", ndim);
        std::printf("  shape: [");
        for (int8_t i = 0; i < ndim; ++i) {
            std::printf("%lld%s", static_cast<long long>(shape[i]), i < ndim-1 ? ", " : "");
        }
        std::printf("]\n");
        std::printf("  strides: [");
        for (int8_t i = 0; i < ndim; ++i) {
            std::printf("%lld%s", static_cast<long long>(strides[i]), i < ndim-1 ? ", " : "");
        }
        std::printf("]\n");
        std::printf("  owns_data: %s\n", owns_data ? "true" : "false");
        std::printf("  valid: %s\n", valid() ? "true" : "false");
    }
#else
    void dump_meta() const noexcept {}
#endif
};

} // namespace zero
